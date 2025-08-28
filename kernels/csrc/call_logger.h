// kernels/csrc/fused_attention/common/call_logger.h
#pragma once

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#endif

namespace qsrvlog {

// ---- runtime switches -------------------------------------------------------
inline bool enabled() {
  return true;
}

inline const char* logfile_path() {
  static std::string p = [](){
    const char* e = std::getenv("QSRV_DUMP_KERNEL_FILE");
    return e ? std::string(e) : std::string("~/work/omniserve/dump/kernel_calls.jsonl");
  }();
  return p.c_str();
}

// ---- utils ------------------------------------------------------------------
inline uint64_t now_us() {
  using clock = std::chrono::steady_clock;
  return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
      clock::now().time_since_epoch()).count();
}

inline uint32_t pid() {
#ifdef _WIN32
  return (uint32_t)_getpid();
#else
  return (uint32_t)getpid();
#endif
}

inline uint64_t tid() {
  auto tid = std::this_thread::get_id();
  return std::hash<std::thread::id>{}(tid);
}

inline std::string json_escape(const std::string& s) {
  std::string o; o.reserve(s.size() + 16);
  for (char c : s) {
    switch (c) {
      case '"':  o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\b': o += "\\b";  break;
      case '\f': o += "\\f";  break;
      case '\n': o += "\\n";  break;
      case '\r': o += "\\r";  break;
      case '\t': o += "\\t";  break;
      default:
        if ((unsigned char)c < 0x20) {
          char buf[8]; std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned)c);
          o += buf;
        } else {
          o += c;
        }
    }
  }
  return o;
}

// at::ScalarType -> string
inline const char* scalarTypeToStr(c10::ScalarType t) {
  switch (t) {
    case c10::ScalarType::Float:     return "float32";
    case c10::ScalarType::Half:      return "float16";
    case c10::ScalarType::BFloat16:  return "bfloat16";
    case c10::ScalarType::Double:    return "float64";
    case c10::ScalarType::Int:       return "int32";
    case c10::ScalarType::Long:      return "int64";
    case c10::ScalarType::Short:     return "int16";
    case c10::ScalarType::Byte:      return "uint8";
    case c10::ScalarType::Char:      return "int8";
    case c10::ScalarType::Bool:      return "bool";
    default:                         return "unknown";
  }
}

inline std::string vec_i64_to_json(const c10::IntArrayRef& a) {
  std::string s; s += "[";
  for (size_t i = 0; i < a.size(); ++i) {
    s += std::to_string((long long)a[i]);
    if (i + 1 < a.size()) s += ",";
  }
  s += "]";
  return s;
}

// format pointer as hex string
inline std::string ptr_hex(const void* p) {
  std::ostringstream oss;
  oss << "0x" << std::hex << (uintptr_t)p;
  return oss.str();
}

#ifdef __CUDACC__
inline std::string dim3_to_json(const dim3& d) {
  return std::string("{\"x\":") + std::to_string(d.x) +
         ",\"y\":" + std::to_string(d.y) +
         ",\"z\":" + std::to_string(d.z) + "}";
}
#endif

// ---- sink (thread-safe) -----------------------------------------------------
inline void write_jsonl_line(const std::string& line) {
  static std::mutex m;
  static std::ofstream ofs;
  std::lock_guard<std::mutex> g(m);
  if (!ofs.is_open()) {
    ofs.open(logfile_path(), std::ios::out | std::ios::app);
  }
  ofs << line << "\n";
  ofs.flush();
}

// ---- Event builder ----------------------------------------------------------
struct Event {
  std::string buf;
  bool firstKV = true;
  bool active = false;

  Event(const char* kind, const char* module, const char* name) {
    active = enabled();
    if (!active) return;
    buf.reserve(1024);
    buf += "{";
    addK("ts_us", (int64_t)now_us());
    addK("pid",  (int64_t)pid());
    addK("tid",  (int64_t)tid());
    addK("kind", kind);
    addK("module", module);
    addK("name", name);
  }

  // low-level KV adders
  void addComma() { if (!firstKV) buf += ","; firstKV = false; }
  void addKV(const char* k, const std::string& vjson) {
    if (!active) return;
    addComma();
    buf += "\""; buf += json_escape(k); buf += "\":";
    buf += vjson;
  }
  void addK(const char* k, const char* v)            { addKV(k, std::string("\"")+json_escape(v)+"\""); }
  void addK(const char* k, const std::string& vstr)  { addK(k, vstr.c_str()); }
  void addK(const char* k, bool v)                   { addKV(k, v ? "true" : "false"); }
  void addK(const char* k, int64_t v)                { addKV(k, std::to_string(v)); }
  void addK(const char* k, double v) {
    char buff[64]; std::snprintf(buff, sizeof(buff), "%.6f", v);
    addKV(k, buff);
  }

  // pointer helpers
  void addPtr(const char* k, const void* p) {
    std::string j = std::string("{\"addr\":\"") + ptr_hex(p) + "\"}";
    addKV(k, j);
  }
  void addPtrSize(const char* k, const void* p, uint64_t nbytes) {
    std::string j = std::string("{\"addr\":\"") + ptr_hex(p) + "\",\"bytes\":" + std::to_string(nbytes) + "}";
    addKV(k, j);
  }

  // high-level helpers
  void addTensor(const char* k, const at::Tensor& t) {
    if (!active) return;
    std::string j = "{";
    j += "\"dtype\":\""; j += scalarTypeToStr(t.scalar_type()); j += "\"";
    j += ",\"shape\":"; j += vec_i64_to_json(t.sizes());
    // j += ",\"device\":\""; j += t.device().str(); j += "\"";
    // j += ",\"contig\":"; j += (t.is_contiguous()?"true":"false");
    // j += ",\"ptr\":"; { // 用 uintptr_t 记录 data_ptr 便于重放
    //   uintptr_t p = (uintptr_t)t.data_ptr();
    //   j += std::to_string((unsigned long long)p);
    // }
    j += "}";
    addKV(k, j);
  }

  template <typename OptionalTensor>
  void addOptionalTensor(const char* k, const OptionalTensor& opt) {
    if (!active) return;
    if (opt.has_value()) addTensor(k, opt.value());
    else addKV(k, "null");
  }

#ifdef __CUDACC__
  void addDim3(const char* k, const dim3& d) {
    addKV(k, dim3_to_json(d));
  }
  void addStream(const char* k, cudaStream_t s) {
    // 用数值形式记录 stream 标识
    addKV(k, std::to_string((unsigned long long)(s)));
  }
#endif

  void end() {
    if (!active) return;
    buf += "}";
    write_jsonl_line(buf);
    active = false;
  }
};

// ---- macros (统一一套 API) --------------------------------------------------
// 通用事件（函数调用/marker等）
#define QSRV_EVENT_BEGIN(_kind, _module, _name) \
  if (::qsrvlog::enabled()) { ::qsrvlog::Event _qsrv_e((_kind), (_module), (_name));

#define QSRV_EVENT_END() \
  _qsrv_e.end(); }

// 语义化别名：pybind 入口用
#define QSRV_CALL_BEGIN(_module, _name) QSRV_EVENT_BEGIN("call", (_module), (_name))
#define QSRV_CALL_END()                 QSRV_EVENT_END()

// kernel 发射前用：自动把 grid/block/smem/stream 记进去
#ifdef __CUDACC__
#define QSRV_LAUNCH_BEGIN(_module, _kname, _grid, _block, _smem, _stream)  \
  QSRV_EVENT_BEGIN("launch", (_module), (_kname))                          \
  _qsrv_e.addDim3("grid",  (_grid));                                       \
  _qsrv_e.addDim3("block", (_block));                                      \
  _qsrv_e.addK("smem_bytes", (int64_t)(_smem));                            \
  _qsrv_e.addStream("stream", (_stream));
#define QSRV_LAUNCH_BEGIN_LIGHT(_module, _kname, _grid, _block, _smem)           \
  QSRV_EVENT_BEGIN("launch", (_module), (_kname))                          \
  _qsrv_e.addDim3("grid",  (_grid));                                       \
  _qsrv_e.addDim3("block", (_block));                                      \
  _qsrv_e.addK("smem_bytes", (int64_t)(_smem));
#else
#define QSRV_LAUNCH_BEGIN(_module, _kname, _grid, _block, _smem, _stream) \
  QSRV_EVENT_BEGIN("launch", (_module), (_kname))                          \
  _qsrv_e.addK("smem_bytes", (int64_t)(_smem));
#endif
#define QSRV_LAUNCH_END() QSRV_EVENT_END()

// 统一的参数追加宏（两边都能用）
#define QSRV_ARG_TENSOR(key, tensor)           _qsrv_e.addTensor((key), (tensor))
#define QSRV_ARG_OPT_TENSOR(key, opt_tensor)   _qsrv_e.addOptionalTensor((key), (opt_tensor))
#define QSRV_ARG_I(key, val)                   _qsrv_e.addK((key), (int64_t)(val))
#define QSRV_ARG_F(key, val)                   _qsrv_e.addK((key), (double)(val))
#define QSRV_ARG_B(key, val)                   _qsrv_e.addK((key), (bool)(val))
#define QSRV_ARG_S(key, cstr)                  _qsrv_e.addK((key), (cstr))
// 指针/缓冲区
#define QSRV_ARG_PTR(key, ptr)                 _qsrv_e.addPtr((key), (const void*)(ptr))
#define QSRV_ARG_PTR_SIZE(key, ptr, nbytes)    _qsrv_e.addPtrSize((key), (const void*)(ptr), (uint64_t)(nbytes))

// 可选：模板/类型名（编译期）辅助（简单映射）
template<typename X> constexpr const char* type_name() {
#if defined(__CUDACC__)
  if      (std::is_same<X, at::Half>::value)      return "half";
  else if (std::is_same<X, float>::value)         return "fp32";
  else if (std::is_same<X, double>::value)        return "fp64";
  else if (std::is_same<X, uint16_t>::value)      return "u16";
  else if (std::is_same<X, int8_t>::value)        return "int8";
#ifdef ENABLE_BF16
  else if (std::is_same<X, __nv_bfloat16>::value) return "bf16";
#endif
#endif
  return "unknown";
}

} // namespace qsrvlog
