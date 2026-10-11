"""torch.profiler traces of a few sampler steps, armed from the API and recorded on the worker thread.

Kineto collects CPU ops only on the thread that started it, so the API arms a recording and the next
sampler step starts it. Unarmed, a step costs one global read.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from difflab import config

DIR = config.OUTPUT_DIR / "profiles"
TABLE_ROWS = 40

_lock = threading.Lock()
_wanted: tuple[int, bool] | None = None
_stop = False
_running: Any = None
_left = 0
_last: list[str] = []


def arm(steps: int, stack: bool) -> dict[str, Any]:
    global _wanted, _stop
    with _lock:
        _wanted, _stop = (steps, stack), False
    return state()


def disarm() -> dict[str, Any]:
    global _wanted, _stop
    with _lock:
        _wanted, _stop = None, _running is not None
    return state()


def state() -> dict[str, Any]:
    return {
        "armed": None if _wanted is None else {"steps": _wanted[0], "stack": _wanted[1]},
        "recording": _running is not None,
        "steps_left": _left if _running is not None else 0,
        "last": list(_last),
    }


def step() -> None:
    """A sampler step boundary."""
    global _left
    if _wanted is None and _running is None:
        return
    if _running is not None:
        _left -= 1
        if _left <= 0 or _stop:
            done()
    if _wanted is not None and _running is None:
        _start()


def done() -> None:
    """The sampler ended: a recording still going is written as it stands."""
    global _running, _stop
    if _running is None:
        return
    profiler, _running, _stop = _running, None, False
    stem = DIR / f"torch-{time.strftime('%Y%m%d-%H%M%S')}"
    try:
        profiler.stop()
        DIR.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(f"{stem}.json.gz")
        table = profiler.key_averages().table(sort_by="self_device_time_total", row_limit=TABLE_ROWS)
        Path(f"{stem}.txt").write_text(table)
    except Exception as error:  # a profile is never worth the render it rides on
        print(f"difflab: the profile was not written ({type(error).__name__}: {error})", flush=True)
        return
    _last.insert(0, f"{stem}.json.gz")
    del _last[5:]


def _start() -> None:
    global _wanted, _running, _left
    import torch.profiler

    with _lock:
        if _wanted is None:
            return
        (steps, stack), _wanted = _wanted, None
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    profiler = torch.profiler.profile(activities=activities, with_stack=stack)
    profiler.start()
    _running, _left = profiler, steps
