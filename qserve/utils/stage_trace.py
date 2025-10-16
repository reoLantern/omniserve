import json, os, time, threading, sys

_path = os.environ.get("QSRV_DUMP_KERNEL_FILE")
_lock = threading.Lock()
_enabled = os.environ.get("QSRV_ENABLE_CALL_LOGGER") == "1"
if _enabled and not _path:
    sys.stderr.write("[trace] QSRV_ENABLE_CALL_LOGGER=1 but no QSRV_DUMP_KERNEL_FILE is specified; we disable stage_trace\n")
    _enabled = False

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