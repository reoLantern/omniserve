#!/usr/bin/env python3
import argparse, json, importlib, math, os, sys
import torch

# ---------------- dtype 映射 ----------------
DTYPE = {
    "float16": torch.float16, "float32": torch.float32, "float64": torch.float64,
    "bfloat16": torch.bfloat16, "int8": torch.int8, "uint8": torch.uint8,
    "int16": torch.int16, "int32": torch.int32, "int64": torch.int64, "bool": torch.bool,
}

def make_tensor(spec, device, *, fill=None):
    """按照日志 spec 造一个空张量；可选填充值策略 fill: None/'zero'/'one'/'rand'。"""
    if spec is None: return None
    t = torch.empty(tuple(spec["shape"]), dtype=DTYPE[spec["dtype"]], device=device)
    if fill == "zero":
        return t.zero_()
    if fill == "one":
        # 只对浮点/整型有效；bool/uint8 也能工作
        return t.fill_(1)
    if fill == "rand":
        if t.dtype.is_floating_point or t.dtype == torch.bfloat16:
            return t.uniform_(-1, 1)
        if t.dtype == torch.int8:
            return t.random_(-8, 8)  # 量化输入取个小范围
        if t.dtype == torch.int16 or t.dtype == torch.int32 or t.dtype == torch.int64:
            return t.random_(0, 7)
        if t.dtype == torch.uint8 or t.dtype == torch.bool:
            return t.zero_()
    return t  # 默认不填充

# ---------------- extension 装载 ----------------
def load_backend_modules():
    mods = {}
    mods["fused_attention_pure_dense"] = importlib.import_module("omniserve_backend.fused_attention_pure_dense")
    mods["layernorm_ops"]              = importlib.import_module("omniserve_backend.layernorm_ops")
    try:
        mods["fused_kernels_ops"]      = importlib.import_module("omniserve_backend.fused_kernels_ops")
    except ModuleNotFoundError:
        mods["fused_kernels_ops"]      = None
    try:
        mods["activation_ops"]         = importlib.import_module("omniserve_backend.activation_ops")
    except ModuleNotFoundError:
        mods["activation_ops"]         = None
    try:
        mods["qgemm_w4a8_per_group"]   = importlib.import_module("omniserve_backend.qgemm_w4a8_per_group")
    except ModuleNotFoundError:
        mods["qgemm_w4a8_per_group"]   = None
    try:
        mods["qgemm_w8a8"]             = importlib.import_module("omniserve_backend.qgemm_w8a8")
    except ModuleNotFoundError:
        mods["qgemm_w8a8"]             = None
    return mods

# ---------------- KV 指针表（pure dense 用） ----------------
def build_kv_pointer_table_pure_dense(ev, device):
    B = ev["q"]["shape"][0]
    M = ev["kv_pointers"]["shape"][2]
    tpb = int(ev["tokens_per_block"])
    spt = int(ev["size_per_token"])
    bytes_per_block = tpb * spt

    ptrs = torch.empty((B, 2, M), dtype=torch.int64, device=device)
    for b in range(B):
        k_pool = torch.empty(bytes_per_block * M, dtype=torch.uint8, device=device)
        v_pool = torch.empty(bytes_per_block * M, dtype=torch.uint8, device=device)
        base_k, base_v = k_pool.data_ptr(), v_pool.data_ptr()
        for m in range(M):
            ptrs[b, 0, m] = base_k + m * bytes_per_block
            ptrs[b, 1, m] = base_v + m * bytes_per_block
    return ptrs

def build_length_per_sample(ev, device):
    B = ev["q"]["shape"][0]
    max_len = int(ev["memory_max_seqlen"])
    tl = int(ev.get("timestep", max_len))
    tl = min(tl, max_len)
    return torch.full((B,), tl, dtype=torch.int32, device=device)

# ---------------- 重放实现 ----------------
def replay_qgemm_w4a8_per_group(ev, mods, device):
    """
    需要满足关系：
      in :  [N, Cin]   int8
      W  :  [Cout, Cin/2]  int8   (4bit pack → /2)
      zeros/scales_i8 : [Cin/G, Cout]  int8    (G=128)
      wscales : [Cout]    fp16
      ascales : [N]       fp16
      out: [N, Cout] fp16
    """
    if mods["qgemm_w4a8_per_group"] is None:
        raise RuntimeError("Extension omniserve_backend.qgemm_w4a8_per_group 未找到。")

    G = 128  # 内核里用到的 group size 常数
    a_spec  = ev["_in_feats"];   w_spec  = ev["_kernel"]
    z_spec  = ev["_zeros"];      s8_spec = ev["_scales_i8"]
    ws_spec = ev["_wscales"];    as_spec = ev["_ascales"]
    o_spec  = ev["_out_feats"]

    # 先按“安全值”构造张量
    a   = make_tensor(a_spec,  device, fill="rand")    # 激活 int8 随机
    w   = make_tensor(w_spec,  device, fill="rand")    # 权重量化 int8 随机
    z   = make_tensor(z_spec,  device, fill="zero")    # per-group zero 置零
    s8  = make_tensor(s8_spec, device, fill="zero")    # per-group scales_i8 置零（实现里会配合 wscales 使用）
    ws  = make_tensor(ws_spec, device, fill="one")     # per-output 通道尺度 = 1
    asc = make_tensor(as_spec, device, fill="one")     # per-token 尺度 = 1
    out = make_tensor(o_spec,  device, fill=None)      # 输出只要分配

    # --- 形状约束校验（避免非法访问） ---
    N, Cin = a.shape
    Cout   = out.shape[-1]
    assert w.shape[0] == Cout,           f"weight[0]({w.shape[0]}) != Cout({Cout})"
    assert w.shape[1] * 2 == Cin,        f"weight[1]*2({w.shape[1]*2}) != Cin({Cin})  (4bit pack关系)"
    assert z.shape[0] == Cin // G,       f"zeros[0]({z.shape[0]}) != Cin/G({Cin//G})"
    assert z.shape[1] == Cout,           f"zeros[1]({z.shape[1]}) != Cout({Cout})"
    assert s8.shape == z.shape,          f"scales_i8{tuple(s8.shape)} != zeros{tuple(z.shape)}"
    assert ws.numel() == Cout,           f"wscales numel {ws.numel()} != Cout {Cout}"
    assert asc.numel() == N,             f"ascales numel {asc.numel()} != N {N}"

    # 运行
    mod = mods["qgemm_w4a8_per_group"]
    return mod.gemm_forward_cuda(a, w, z, s8, ws, asc, out)

def replay_layernorm_general(ev, mods, device):
    ln = mods["layernorm_ops"]
    out     = make_tensor(ev["out"], device)
    x       = make_tensor(ev["input"], device, fill="rand")
    weight  = make_tensor(ev["weight"], device, fill="one")
    scaling = make_tensor(ev["scaling"], device, fill="one")
    eps = 1e-6
    return ln.rms_norm_general(out, x, weight, scaling, eps, bool(ev["use_per_token_quant"]))

def replay_attention_pure_dense(ev, mods, device):
    mod = mods["fused_attention_pure_dense"]
    q  = make_tensor(ev["q"], device, fill="rand")
    k  = make_tensor(ev["k"], device, fill="rand")
    v  = make_tensor(ev["v"], device, fill="rand")
    kv = build_kv_pointer_table_pure_dense(ev, device)
    lps = build_length_per_sample(ev, device)
    alibi = make_tensor(ev.get("alibi_slopes"), device, fill="one") if ev.get("alibi_slopes") else None
    return mod.single_query_attention(
        q, k, v, kv, lps, alibi,
        int(ev["memory_max_seqlen"]), int(ev["tokens_per_block"]), int(ev["size_per_token"]),
        int(ev["timestep"]),
        int(ev["rotary_embedding_dim"]), float(ev["rotary_base"]),
        bool(ev["neox_rotary_style"]), bool(ev["int4_kv_cache"]), bool(ev["kv_cache_with_zeros"])
    )

# 统一分发
def replay_call(ev, mods, device):
    m, f = ev["module"], ev["name"]
    if m == "qgemm_w4a8_per_group" and f == "gemm_forward_cuda":
        return replay_qgemm_w4a8_per_group(ev, mods, device)
    if m == "layernorm" and f == "rms_norm_general":
        return replay_layernorm_general(ev, mods, device)
    if m == "fused_attention_pure_dense" and f == "single_query_attention":
        return replay_attention_pure_dense(ev, mods, device)
    raise NotImplementedError(f"Replay mapping missing for: {m}.{f}")

# ---------------- 事件选择 / 配套 launch 打印 ----------------
def pick_event(events, module, name, prefer_decode=True, index=None):
    cands = [e for e in events if e.get("kind") == "call" and e["module"] == module and e["name"] == name]
    if prefer_decode:
        # 按 timestep 升序（decode 更小），再按输入规模排序
        def keyfun(e):
            ts = e.get("timestep", 1 << 30)
            sz = 0
            for k in ("q", "input", "_in_feats"):
                if k in e:
                    sh = e[k].get("shape", [])
                    if sh:
                        sz = max(sz, int(sh[0]) * int(sh[-1]))
            return (ts, sz)
        cands = sorted(cands, key=keyfun)
    if not cands:
        raise RuntimeError(f"No call event for {module}.{name}")
    return cands[index if index is not None else 0]

def find_next_launch(events, call_idx):
    """找出紧随其后的同模块 launch 事件，方便核对 grid/block/smem。"""
    m = events[call_idx]["module"]
    for j in range(call_idx + 1, min(call_idx + 20, len(events))):
        e = events[j]
        if e.get("kind") == "launch" and e.get("module") == m:
            return e
    return None

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--module", required=True)
    ap.add_argument("--name",   required=True)
    ap.add_argument("--index",  type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--show-launch", dest="show_launch", action="store_true",
                help="打印紧随其后的 launch 配置用于核对")
    args = ap.parse_args()

    with open(args.jsonl, "r") as f:
        raw = [line for line in f if line.strip().startswith("{")]
        events = [json.loads(line) for line in raw]

    # 记录 call 在 events 中的下标，以便找它后面的 launch
    call_positions = [i for i,e in enumerate(events) if e.get("kind")=="call" and e.get("module")==args.module and e.get("name")==args.name]
    if not call_positions:
        raise RuntimeError(f"No call event for {args.module}.{args.name}")

    idx_in_cands = args.index if args.index is not None else 0
    if idx_in_cands >= len(call_positions):
        raise RuntimeError(f"--index 超界：共有 {len(call_positions)} 个匹配事件")

    call_pos = call_positions[idx_in_cands]
    ev = events[call_pos]
    print(f"[replay] {ev['module']}.{ev['name']} (pos {call_pos}/{len(events)})")

    if args.show_launch:
        ln = find_next_launch(events, call_pos)
        if ln:
            g, b, sm = ln["grid"], ln["block"], ln.get("smem_bytes", 0)
            print(f"[launch] should be: grid=({g['x']},{g['y']},{g['z']}), block=({b['x']},{b['y']},{b['z']}), smem={sm}")
        else:
            print("[launch] (not found nearby)")

    device = torch.device(args.device)
    torch.cuda.init()
    mods = load_backend_modules()

    torch.cuda.synchronize()
    out = replay_call(ev, mods, device)
    torch.cuda.synchronize()

    if out is not None:
        if isinstance(out, torch.Tensor):
            # 触发下 lazy，在 host 打印形状即可
            print("Output shape:", tuple(out.shape), out.dtype)
        else:
            print("Return:", type(out))

if __name__ == "__main__":
    main()
