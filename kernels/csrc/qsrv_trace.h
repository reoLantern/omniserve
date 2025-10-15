#pragma once
#include <cstdio>
#include <cstdlib>
#include <string>

inline bool qsrv_trace_on() {
  static int on = [](){
    const char* e = std::getenv("QSRV_TRACE_PYBIND");
    return (e && std::string(e) != "0");
  }();
  return on;
}

#define QSRV_TRACE_HIT(mod, fn)                                                     \
  do {                                                                              \
    if (qsrv_trace_on()) {                                                          \
      std::fprintf(stderr, "[QSRV][HIT] %s::%s\n", (mod), (fn));                    \
      std::fflush(stderr);                                                          \
    }                                                                               \
  } while (0)
