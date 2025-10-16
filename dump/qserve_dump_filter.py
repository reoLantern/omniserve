#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QServe dump(JSONL) 精简脚本（修订版）

仅保留：
1) 第一条 {"event":"stage_begin","stage":"prefill"} 之前的所有原始行。
2) {"event":"stage_begin","stage":"prefill"} 这一行；{"event":"stage_end","stage":"prefill"} 这一行。
3) prefill 阶段内：仅 layer==0 的
   - {"event":"executing layer"} 这一行；
   - 紧随其后的所有 kernel 的 {"kind":"call"} / {"kind":"launch"} 行；
   - 以及 {"event" 以 "flash_attn_varlen_func called" 开头} 的行；
   直到出现下一次 {"event":"executing layer"}（其它 layer）或下一次 {"event":"stage_begin"}。
4) prefill 之后的 decode 阶段：按 step 采样（第 1、1+step、1+2*step、… 个 decode）。
   对于“采样到的” decode：
   - 保留该阶段的 {"event":"stage_begin","stage":"decode"} 与 {"event":"stage_end","stage":"decode"}；
   - 同上，仅保留 layer==0 的执行块（executing layer + call/launch + flash_attn_varlen_func called...）。
5) 输出：与输入同目录，文件名为 <原名>.filtered_step{step}.jsonl

用法示例：
    python3 qserve_dump_filter.py /path/to/input.jsonl --step 50
"""

import argparse
import json
from pathlib import Path


FLASH_ATTN_PREFIX = "flash_attn_varlen_func called"


def _stage_of(obj) -> str | None:
    """获取 stage 类型：'prefill' / 'decode' / None（若无法判断）"""
    stage = obj.get("stage")
    if stage in ("prefill", "decode"):
        return stage
    # 兼容某些日志写法
    if obj.get("is_prefill") is True:
        return "prefill"
    if obj.get("is_prefill") is False:
        return "decode"
    return None


def filter_qserve_dump(in_path: Path, step: int, target_layer: int = 0) -> Path:
    in_path = Path(in_path)
    if step < 1:
        raise ValueError("step 必须为 >= 1 的整数")

    out_path = in_path.with_name(in_path.name + f".filtered_step{step}.jsonl")

    before_first_prefill_begin = True  # 是否仍处于第一条 prefill stage_begin 之前
    current_stage: str | None = None   # None / 'prefill' / 'decode'
    decode_idx = 0                      # prefill 之后第几个 decode（遇到 decode 的 stage_begin 时 +1，从 1 开始）
    want_this_stage = False             # 当前阶段是否需要保留内容（prefill 恒 True；decode 由采样决定）
    inside_layer0_block = False         # 是否正在 layer0 的“kernel 区段”（仅在 want_this_stage=True 时可能为 True）

    with in_path.open("r", encoding="utf-8") as rf, out_path.open("w", encoding="utf-8") as wf:
        for raw_line in rf:
            line = raw_line.strip()
            if not line:
                continue

            # 解析 JSON；若解析失败且仍在第一条 prefill stage_begin 之前，则原样保留
            try:
                obj = json.loads(line)
            except Exception:
                if before_first_prefill_begin:
                    wf.write(raw_line)
                continue

            event = obj.get("event")

            # --- 处理 stage_begin ---
            if event == "stage_begin":
                stage_type = _stage_of(obj)

                # 第一条 prefill stage_begin 之前的内容结束
                if before_first_prefill_begin and stage_type == "prefill":
                    before_first_prefill_begin = False

                # 进入新阶段：阶段切换会终止上一阶段的 layer0 区段
                current_stage = stage_type
                inside_layer0_block = False

                if stage_type == "prefill":
                    # prefill 必保留
                    want_this_stage = True
                    # 必须保留 prefill 的 stage_begin
                    wf.write(raw_line)

                elif stage_type == "decode":
                    # 计数第几个 decode，并决定是否采样保留（1、1+step、…）
                    decode_idx += 1
                    want_this_stage = ((decode_idx - 1) % step == 0)
                    if want_this_stage:
                        # 采样到的 decode：保留 stage_begin
                        wf.write(raw_line)

                else:
                    # 未知阶段：不保留
                    want_this_stage = False

                continue

            # --- 第一条 prefill stage_begin 之前：全部原样输出 ---
            if before_first_prefill_begin:
                wf.write(raw_line)
                continue

            # --- 处理 stage_end ---
            if event == "stage_end":
                stage_type = _stage_of(obj)

                # 对于 prefill，必须保留 stage_end
                if stage_type == "prefill":
                    wf.write(raw_line)

                # 对于 decode：仅当该阶段被采样到时保留
                elif stage_type == "decode" and want_this_stage:
                    wf.write(raw_line)

                # 结束当前阶段（稳妥起见，同时关闭 layer0 区段）
                inside_layer0_block = False
                current_stage = None
                want_this_stage = False
                continue

            # --- 非阶段的事件处理 ---
            # 处理 "executing layer"：界定 layer 切换与 layer0 区段起点
            if event == "executing layer":
                # 只有当当前阶段被保留时才可能开始/切换 layer0 区段
                if want_this_stage and obj.get("layer") == target_layer:
                    inside_layer0_block = True
                    # 需要保留这一行（layer0 的 executing layer）
                    wf.write(raw_line)
                else:
                    # 进入其它 layer，终止 layer0 区段
                    if inside_layer0_block:
                        inside_layer0_block = False
                continue

            # 在 layer0 区段内，额外保留以 FLASH_ATTN_PREFIX 开头的事件行
            if inside_layer0_block and isinstance(event, str) and event.startswith(FLASH_ATTN_PREFIX):
                wf.write(raw_line)
                continue

            # 其它 event 行：不保留
            if event is not None:
                continue

            # --- 非 event 行（kernel 记录等） ---
            # 仅在 layer0 区段内保留 {"kind":"call"} / {"kind":"launch"}
            if inside_layer0_block:
                kind = obj.get("kind")
                if kind in ("call", "launch"):
                    wf.write(raw_line)
                # 其它非 event 的 JSON 行忽略
                continue

            # 走到这里：当前行既不是我们要的 event，也不是 layer0 区段内的 call/launch —— 忽略
            continue

    return out_path


def main():
    parser = argparse.ArgumentParser(description="精简 QServe dump(JSONL)：保留 prefill 及采样到的 decode 的 layer0 核心信息。")
    parser.add_argument("input", type=Path, help="输入 JSONL 文件路径")
    parser.add_argument("--step", "-s", type=int, required=True, help="decode 采样步长（1 表示全部保留；5 表示保留第 1、6、11、… 个 decode）")
    parser.add_argument("--layer", "-l", type=int, default=0, help="关注的 layer（默认 0）")
    args = parser.parse_args()

    out_path = filter_qserve_dump(args.input, step=args.step, target_layer=args.layer)
    print(str(out_path))


if __name__ == "__main__":
    main()
