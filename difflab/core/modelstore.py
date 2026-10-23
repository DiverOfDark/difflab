"""A plain copy of every fetched checkpoint on the RunPod global volume, copied whole onto the container disk
when a later pod needs it, so the Hub is asked once per volume rather than once per pod.

Nothing loads from the volume itself: it is an object store, and the Hub cache's symlinks and renames and
the loaders' mmap reads there handed back an empty config.json and read a 16.5 GB VLM in 889 s (2026-10-16).
Here files are written front to back once and read back in large parallel ranges.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(root) if (root := os.environ.get("DIFFLAB_MODEL_STORE")) else None
RANGE = 32 << 20  # READERS of these in memory at once
READERS = 16
WRITERS = 4

_lock = threading.Lock()
_storing: set[Path] = set()


def storing() -> bool:
    with _lock:
        return bool(_storing)


def restore(snapshot: Path, marker: Path, label: str) -> bool:
    """Fills `snapshot` from the store and writes `marker`, if the store holds every file at its size."""
    if ROOT is None:
        return False
    stored = _place(snapshot)
    sizes = _sizes(stored / marker.name)
    if not sizes or any(_size(stored / name) != size for name, size in sizes.items()):
        return False
    from difflab.core import ui  # not at the top: config names checkpoints, and ui imports config

    done, count, fds = [0], threading.Lock(), []
    started = time.perf_counter()

    def part(source: int, target: int, offset: int, length: int) -> None:
        while length:
            data = os.pread(source, length, offset)
            if not data:
                raise OSError(f"the volume ended {length} bytes early")
            os.pwrite(target, data, offset)
            offset, length = offset + len(data), length - len(data)
            with count:
                done[0] += len(data)

    try:
        with (
            ui.transfer(f"Copying {label} from the volume", sum(sizes.values()), lambda: done[0]),
            ThreadPoolExecutor(READERS) as pool,
        ):
            parts = []
            for name, size in sizes.items():
                target = snapshot / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.unlink(missing_ok=True)  # a Hub symlink would be written through into its blob
                fds.append(source := os.open(stored / name, os.O_RDONLY))
                fds.append(written := os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644))
                os.ftruncate(written, size)
                parts += [
                    pool.submit(part, source, written, offset, min(RANGE, size - offset))
                    for offset in range(0, size, RANGE)
                ]
            for future in parts:
                future.result()
    except OSError as problem:
        # snapshot_download skips any file already in the snapshot, whatever its size.
        for name in sizes:
            (snapshot / name).unlink(missing_ok=True)
        print(f"difflab: could not copy {label} from the volume ({problem}); fetching it from the Hub", flush=True)
        return False
    finally:
        for fd in fds:
            os.close(fd)
    marker.write_text(json.dumps(sizes))
    seconds = time.perf_counter() - started
    print(f"difflab: copied {label} from the volume, {done[0] / 1e9:.1f} GB in {seconds:.0f} s", flush=True)
    return True


def keep(snapshot: Path, marker: Path, label: str) -> None:
    """Copies a fetched snapshot to the store in the background, unless the store has it already."""
    if ROOT is None:
        return
    stored = _place(snapshot)
    key = stored / marker.name
    sizes = _sizes(marker)
    if not sizes or key.exists():
        return
    with _lock:
        if key in _storing:
            return
        _storing.add(key)
    threading.Thread(target=_store, args=(snapshot, key, sizes, label), name="difflab-modelstore", daemon=True).start()


def _store(snapshot: Path, key: Path, sizes: dict[str, int], label: str) -> None:
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(WRITERS) as pool:
            list(pool.map(lambda name: _copy(snapshot / name, key.parent / name), sizes))
        # The marker goes last: a pod deleted mid-copy leaves files the next pod will not trust.
        if all(_size(key.parent / name) == size for name, size in sizes.items()):
            key.write_text(json.dumps(sizes))
            seconds = time.perf_counter() - started
            print(
                f"difflab: kept {label} on the volume, {sum(sizes.values()) / 1e9:.1f} GB in {seconds:.0f} s",
                flush=True,
            )
    except OSError as problem:
        print(f"difflab: could not keep {label} on the volume ({problem})", flush=True)
    finally:
        with _lock:
            _storing.discard(key)


def _copy(source: Path, target: Path) -> None:
    """Front to back in one stream: the global volume is unreliable at writing into the middle of a file."""
    if _size(target) == source.stat().st_size:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as read, target.open("wb") as write:
        shutil.copyfileobj(read, write, 16 << 20)


def _place(snapshot: Path) -> Path:
    """`.../models--org--name/snapshots/<commit>` -> `ROOT/models--org--name/<commit>`."""
    assert ROOT is not None
    return ROOT / snapshot.parent.parent.name / snapshot.name


def _sizes(marker: Path) -> dict[str, int]:
    try:
        sizes: dict[str, int] = json.loads(marker.read_text() or "{}")
        return sizes
    except (OSError, ValueError):
        return {}


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return -1
