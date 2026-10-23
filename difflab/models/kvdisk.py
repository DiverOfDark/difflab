"""The VLM's prompt cache on disk: KV pages and the linear-attention checkpoints that make them resumable, kept
past the model's eviction and the server's restart.

Plugs into ExLlamaV3 where its pinned-RAM tier (`CPUPageCache`) does, and as its `RecurrentCache`. On this hybrid
model a cached page is only worth anything up to the last checkpoint before it, so whole chains are written at
chosen checkpoints (`keep`) and nothing else is: a page the GPU evicts is not written, which would be gigabytes
per judged clip.
"""

import hashlib
import os
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from exllamav3.cache.recurrent import RecurrentCache
from exllamav3.constants import PAGE_SIZE
from exllamav3.generator.pagetable import tensor_hash_checksum
from safetensors.torch import load_file, save_file


def anchors(ids: torch.Tensor, pages: int) -> list[bytes]:
    """The page hashes ExLlamaV3 gives the first `pages` pages of `ids`."""
    chain: list[bytes] = []
    for page in range(pages):
        tokens = ids[:, page * PAGE_SIZE : (page + 1) * PAGE_SIZE].contiguous()
        chain.append(tensor_hash_checksum(tokens, chain[-1] if chain else None))
    return chain


def _forget(path: Path) -> None:
    handle = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(handle, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(handle)


class Shelf:
    """The index and the byte budget both halves share. Spent by checkpoint: the least recently used goes,
    then every page no remaining checkpoint reaches."""

    def __init__(self, root: Path, budget: int) -> None:
        self.root, self.budget = root, budget
        for folder in ("pages", "states"):
            (root / folder).mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(root / "index.db", isolation_level=None, check_same_thread=False)
        self.db.executescript(
            "CREATE TABLE IF NOT EXISTS pages (hash BLOB PRIMARY KEY, prev BLOB, tokens BLOB, bytes INTEGER);"
            "CREATE TABLE IF NOT EXISTS states (hash BLOB PRIMARY KEY, position INTEGER, bytes INTEGER, used REAL);"
        )

    def page_path(self, phash: bytes) -> Path:
        return self.root / "pages" / phash.hex()

    def state_path(self, phash: bytes) -> Path:
        return self.root / "states" / f"{phash.hex()}.safetensors"

    def spend(self) -> tuple[set[bytes], set[bytes]]:
        """The checkpoints and pages it deleted."""
        (held,) = self.db.execute(
            "SELECT (SELECT COALESCE(SUM(bytes), 0) FROM pages) + (SELECT COALESCE(SUM(bytes), 0) FROM states)"
        ).fetchone()
        if held <= self.budget:
            return set(), set()
        states = set()
        for phash, size in self.db.execute("SELECT hash, bytes FROM states ORDER BY used").fetchall():
            self.db.execute("DELETE FROM states WHERE hash = ?", (phash,))
            self.state_path(phash).unlink(missing_ok=True)
            states.add(phash)
            held -= size
            if held <= self.budget * 0.9:
                break
        links = dict(self.db.execute("SELECT hash, prev FROM pages").fetchall())
        reached = set()
        for (phash,) in self.db.execute("SELECT hash FROM states"):
            while phash is not None and phash not in reached:
                reached.add(phash)
                phash = links.get(phash)
        pages = set(links) - reached
        for phash in pages:
            self.db.execute("DELETE FROM pages WHERE hash = ?", (phash,))
            self.page_path(phash).unlink(missing_ok=True)
        return states, pages


class DiskPages:
    """`PageTable.cpu_tier`: what it reads (`entries`, `in`, `fetch`, `metrics`) and a `store` that only refreshes."""

    def __init__(self, shelf: Shelf, segments: list[torch.Tensor]) -> None:
        self.shelf, self.segments = shelf, segments
        self.page_bytes = sum(t[0].numel() * t.element_size() for t in self.segments)
        self.max_slots = shelf.budget // self.page_bytes
        self.pagetable: Any = None
        self.entries: dict[bytes, dict[str, Any]] = {
            phash: {"prev_hash": prev, "tokens": tokens}
            for phash, prev, tokens in shelf.db.execute("SELECT hash, prev, tokens FROM pages")
        }
        self.metrics = {"pushes": 0, "dedup_hits": 0, "restores": 0, "evictions": 0, "cold_allocs": 0}

    def attach(self, pagetable: Any) -> None:
        self.pagetable = pagetable

    def __contains__(self, phash: bytes) -> bool:
        return phash in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def store(self, page: Any, serial: int, protect: set[bytes] | None = None) -> None:
        pass

    def keep(self, page: Any) -> None:
        if page.phash in self.entries:
            return
        path = self.shelf.page_path(page.phash)
        with path.with_suffix(".part").open("wb") as out:
            for t in self.segments:
                out.write(t[page.page_index].contiguous().view(torch.uint8).cpu().numpy().tobytes())
        path.with_suffix(".part").rename(path)
        tokens = page.sequence.numpy().tobytes()
        self.shelf.db.execute(
            "INSERT OR REPLACE INTO pages VALUES (?, ?, ?, ?)", (page.phash, page.prev_hash, tokens, self.page_bytes)
        )
        self.entries[page.phash] = {"prev_hash": page.prev_hash, "tokens": tokens}
        self.metrics["pushes"] += 1

    def fetch(self, phash: bytes, page_index: int, serial: int) -> dict[str, Any]:
        entry = self.entries[phash]
        path = self.shelf.page_path(phash)
        data = torch.from_file(str(path), shared=False, size=self.page_bytes, dtype=torch.uint8)
        offset = 0
        for t in self.segments:
            size = t[0].numel() * t.element_size()
            t[page_index].copy_(data[offset : offset + size].view(t.dtype).view(t.shape[1:]))
            offset += size
        del data
        _forget(path)
        self.metrics["restores"] += 1
        tokens = entry["tokens"]
        if isinstance(tokens, bytes):
            entry["tokens"] = torch.frombuffer(bytearray(tokens), dtype=torch.long).view(1, PAGE_SIZE)
        return entry

    def close(self) -> None:
        self.entries = {}


class DiskStates(RecurrentCache):  # type: ignore[misc]
    """`Generator.recurrent_cache`, which finds a checkpoint on disk as if it were held, and reads it in when used."""

    def __init__(self, model: Any, max_size: int, cache: Any, shelf: Shelf) -> None:
        super().__init__(model, max_size)
        self.cache, self.shelf = cache, shelf
        self.on_disk: dict[bytes, int] = dict(shelf.db.execute("SELECT hash, position FROM states").fetchall())

    def get(self, key: bytes, default: Any = None) -> Any:
        if key in self:
            return super().get(key)
        if key in self.on_disk:
            return {"position": self.on_disk[key]}
        return default

    def has(self, key: bytes) -> bool:
        return key in self or key in self.on_disk

    def get_stashed(self, key: bytes, default: Any = None) -> Any:
        if key not in self and key in self.on_disk:
            self._read(key)
        return super().get_stashed(key, default)

    def _layers(self) -> Iterator[Any]:
        return iter(self.cache.get_all_recurrent_layers())

    def _read(self, key: bytes) -> None:
        path = self.shelf.state_path(key)
        tensors = load_file(path)
        _forget(path)
        stashed: dict[Any, Any] = {"position": self.on_disk[key]}
        for index, layer in enumerate(self._layers()):
            stashed[layer] = tuple(
                tensors[f"{index}.{part}"] for part in range(len(tensors)) if f"{index}.{part}" in tensors
            )
        stashed["checkpoint_size"] = sum(t.numel() * t.element_size() for t in tensors.values())
        while self.update_total_size() + stashed["checkpoint_size"] > self.max_size and len(self):
            self.popitem(last=False)
        self[key] = stashed
        self.update_total_size()
        self.shelf.db.execute("UPDATE states SET used = ? WHERE hash = ?", (time.time(), key))

    def keep(self, key: bytes) -> None:
        if key in self.on_disk:
            return
        stashed = super().get_stashed(key)
        tensors = {
            f"{index}.{part}": tensor.contiguous()
            for index, layer in enumerate(self._layers())
            for part, tensor in enumerate(stashed[layer])
        }
        path = self.shelf.state_path(key)
        save_file(tensors, path.with_suffix(".part"))
        path.with_suffix(".part").rename(path)
        _forget(path)
        size = sum(t.numel() * t.element_size() for t in tensors.values())
        self.shelf.db.execute(
            "INSERT OR REPLACE INTO states VALUES (?, ?, ?, ?)", (key, stashed["position"], size, time.time())
        )
        self.on_disk[key] = stashed["position"]


class DiskCache:
    """Both halves over one folder per model and cache layout, so a checkpoint never meets pages of another shape."""

    def __init__(self, root: Path, budget: int, generator: Any) -> None:
        caches = [generator.cache, *([generator.draft_cache] if generator.draft_cache is not None else [])]
        segments = [t for cache in caches for layer in cache.layers.values() for t in layer.get_tensors()]
        layout = repr([(tuple(t.shape[1:]), str(t.dtype)) for t in segments])
        self.shelf = Shelf(root / hashlib.sha256(layout.encode()).hexdigest()[:12], budget)
        self.pages = DiskPages(self.shelf, segments)
        self.states = DiskStates(generator.model, generator.recurrent_cache_size, generator.cache, self.shelf)
        self.pagetable = generator.pagetable
        self.pages.attach(self.pagetable)
        self.states.pagetable = self.pagetable
        self.pagetable.cpu_tier = generator.cpu_page_cache = self.pages
        generator.recurrent_cache = self.states

    def keep(self, chain: list[bytes]) -> str | None:
        """Writes the checkpoint at the end of `chain` and every page of it, all still live on the card; or says
        why it could not."""
        live = [self.pagetable.get_live_page(phash) for phash in chain]
        gone = [index for index, page in enumerate(live) if page is None]
        held = [index for index, phash in enumerate(chain) if self.states.has(phash)]
        if gone or not self.states.has(chain[-1]):
            return f"{len(chain)} pages, not live {gone[:8]}, checkpoints at pages {held[-8:]}"
        for page in live:
            self.pages.keep(page)
        self.states.keep(chain[-1])
        states, pages = self.shelf.spend()
        for phash in states:
            self.states.on_disk.pop(phash, None)
        for phash in pages:
            self.pages.entries.pop(phash, None)
        return None
