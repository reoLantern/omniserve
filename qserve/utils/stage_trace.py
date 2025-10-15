import json, os, time, threading, sys

_path = os.environ.get("QSRV_DUMP_KERNEL_FILE",
                       "~/work/omniserve/dump/kernel_calls.jsonl")
_lock = threading.Lock()
_enabled = os.environ.get("QSRV_TRACE_STAGES", "1") != "0"

def emit(event: dict):
    if not _enabled: 
        return
    event = dict(event)
    event.setdefault("ts", time.time())
    # event.setdefault("pid", os.getpid())
    with _lock:
        with open(_path, "a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            f.flush()

class span:
    """上下文/装饰用：with span('prefill', {...}): ..."""
    def __init__(self, name, meta=None):
        self.name = name
        self.meta = meta or {}
    def __enter__(self):
        self.t0 = time.time()
        emit({"event": "stage_begin", "stage": self.name, **self.meta})
        return self
    def __exit__(self, exc_type, exc, tb):
        emit({"event": "stage_end", "stage": self.name, "dur_ms": (time.time()-self.t0)*1000,
              "ok": exc is None, **self.meta})