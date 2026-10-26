"""Host RAM that keeps evicted models while it is free, and gives it back the moment anything else wants it.

Measured 2026-10-26 on the WD SN770: cold reads 3.4-4.0 GB/s, a warm btrfs page cache only 6.4 GB/s and an
mmap copy out of it 0.3-3.7 GB/s, so the page cache is not worth keeping; the copies here are parked CPU
tensors. They can be swapped to zram, which gives them back slower than the NVMe would, so the watcher frees
them once the machine swaps or feels pressure instead of letting the kernel choose.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from difflab import config


@dataclass
class _Entry:
    size: int
    release: Callable[[], None]


_entries: OrderedDict[str, _Entry] = OrderedDict()  # least recently used first
_calm_until = 0.0
_lock = threading.RLock()
_watcher: threading.Thread | None = None


def available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def rss() -> int:
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def held() -> int:
    with _lock:
        return sum(entry.size for entry in _entries.values())


def calm() -> bool:
    """Whether a new copy may be kept: not for a while after one was freed under pressure, since keeping 16 GB
    of a 58 GB machine set it swapping and the copy was made and freed again within a minute."""
    return time.time() >= _calm_until


def room(needed: int, keep: str | None = None) -> bool:
    """Frees the oldest copies until `needed` more bytes leave the reserve free; whether they do."""
    with _lock:
        for key in [key for key in _entries if key != keep]:
            if available() - needed >= config.HOST_RESERVE_BYTES:
                break
            _release(key, "to make room")
        return available() - needed >= config.HOST_RESERVE_BYTES


def hold(key: str, size: int, release: Callable[[], None]) -> None:
    global _watcher
    with _lock:
        if key in _entries:
            _release(key, "replaced")
        _entries[key] = _Entry(size, release)
        if _watcher is None:
            _watcher = threading.Thread(target=_watch, name="difflab-hostcache", daemon=True)
            _watcher.start()


def take(key: str) -> bool:
    """Whether `key` was held; it is the caller's from here, not the cache's to free."""
    with _lock:
        return _entries.pop(key, None) is not None


def touch(key: str) -> bool:
    with _lock:
        if key in _entries:
            _entries.move_to_end(key)
            return True
        return False


def release_all(reason: str) -> list[str]:
    with _lock:
        keys = list(_entries)
        for key in keys:
            _release(key, reason)
        return keys


def _release(key: str, reason: str) -> None:
    entry = _entries.pop(key)
    try:
        entry.release()
    finally:
        print(f"difflab: released {key} from RAM, {entry.size / 1024**3:.1f} GB ({reason})", flush=True)


def _pressure() -> float:
    try:
        line = Path("/proc/pressure/memory").read_text().splitlines()[0]
    except OSError:
        return 0.0
    return float(line.split()[1].split("=")[1])


def _swapped() -> int:
    for line in Path("/proc/vmstat").read_text().splitlines():
        if line.startswith("pswpout "):
            return int(line.split()[1])
    return 0


def _watch() -> None:
    global _calm_until
    swapped = _swapped()
    while True:
        time.sleep(config.HOST_WATCH_SECONDS)
        now = _swapped()
        reason = (
            "swapping"
            if now - swapped > config.HOST_SWAP_PAGES
            else "pressure"
            if _pressure() > config.HOST_PRESSURE
            else "low memory"
            if available() < config.HOST_RESERVE_BYTES
            else None
        )
        swapped = now
        if reason is None:
            continue
        with _lock:
            if _entries:
                _calm_until = time.time() + config.HOST_CALM_SECONDS
                _release(next(iter(_entries)), reason)
