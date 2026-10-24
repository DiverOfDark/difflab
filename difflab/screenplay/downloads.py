"""Every model file on this disk, with what names it, and taking one off again.

Hub deletions go through huggingface_hub's own cache strategies: since 1.x a repo's blobs are links into
one deduplicated store shared by every repo, with `.refs` manifests a hand-rolled unlink would leave stale.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import re
import shutil
import sys
from fnmatch import fnmatch
from typing import Any

from huggingface_hub import scan_cache_dir

from difflab import store, tools
from difflab.core import hub
from difflab.screenplay import settings

# Pins the code makes outside any setting. Only modules already imported are scanned besides these, and
# qwen_image21 is not imported by the API process.
BUILT_IN = (
    "difflab.config",
    "difflab.models.minimax_h3.config",
    "difflab.models.ltx25.config",
    "difflab.models.taeltx",
    "difflab.models.qwen_image21",
)
MARKER = ".difflab-fetched"
_SHA = re.compile(r"^[0-9a-f]{64}$")
PINNED = "pinned:"

# (repo or PINNED, revision or sha256, glob patterns; none -> every file)
Pin = tuple[str, str, tuple[str, ...]]


def _pin(value: Any) -> Pin | None:
    if isinstance(value, hub.HubFile):
        return value.repo, value.revision, (value.filename,)
    if isinstance(value, hub.Checkpoint):
        return value.repo, value.revision, value.files
    if isinstance(value, hub.UrlFile | hub.CivitaiFile):
        return PINNED, value.sha256, ()
    text = str(value).strip()
    try:
        return _pin(hub.CivitaiFile.parse(text) if text.startswith("civitai:") else hub.Checkpoint.parse(text))
    except ValueError:
        return None


def _uses() -> list[tuple[Pin, str, str]]:
    """(pin, role, where): every model field as this machine resolves it, and as each production does where
    it differs; `role` is "" for the code's own pins and `where` is "remembered" for picker history."""
    out: list[tuple[Pin, str, str]] = []

    def add(value: str, role: str, where: str) -> None:
        if (pin := _pin(value)) is not None:
            out.append((pin, role, where))

    with store.connect() as connection:
        productions = store.productions(connection)
        for tool in tools.TOOLS:
            if tool.settings is None:
                continue
            fields = [f for f in dataclasses.fields(tool.settings) if f.metadata.get("role")]
            if not fields:
                continue
            machine = store.overrides(connection, settings.MACHINE, "", tool.name)
            here = tool.settings.resolve(machine)[0]
            layers = [("this machine", here)] + [
                (p.title, tool.settings.resolve(machine, own)[0])
                for p in productions
                if (own := store.overrides(connection, settings.PRODUCTION, str(p.id), tool.name))
            ]
            for where, resolved in layers:
                for f in fields:
                    value = str(getattr(resolved, f.name))
                    if where != "this machine" and value == str(getattr(here, f.name)):
                        continue
                    for model in settings._models_in(value) if f.metadata.get("multiple") else [value]:
                        add(model, f.metadata["role"], where)
        for role, values in store.model_choices(connection).items():
            for value in values:
                add(value, role, "remembered")
    for name in BUILT_IN:
        importlib.import_module(name)
    for name, module in list(sys.modules.items()):
        if name.startswith("difflab."):
            for value in list(vars(module).values()):
                if isinstance(value, hub.Checkpoint | hub.HubFile | hub.UrlFile | hub.CivitaiFile):
                    out.append((_pin(value), "", "built in"))  # type: ignore[arg-type]
    return out


def _named_by(uses: list[tuple[Pin, str, str]], repo: str, revision: str, name: str) -> list[dict[str, str]]:
    found = {
        (role, where)
        for (r, rev, patterns), role, where in uses
        if r == repo and rev == revision and (not patterns or any(fnmatch(name, p) for p in patterns))
    }
    return [{"role": role, "where": where} for role, where in sorted(found)]


def listing() -> list[dict[str, Any]]:
    """One entry per Hub snapshot and per pinned file, biggest first, each file with what names it."""
    uses = _uses()
    out = []
    for repo in scan_cache_dir(hub.HF_HUB_CACHE).repos:
        if repo.repo_type != "model":
            continue
        for revision in repo.revisions:
            files = [
                {"name": rel, "size": f.size_on_disk, "uses": _named_by(uses, repo.repo_id, revision.commit_hash, rel)}
                for f in revision.files
                if not (rel := f.file_path.relative_to(revision.snapshot_path).as_posix()).startswith(MARKER)
            ]
            out.append(
                {
                    "id": f"{repo.repo_id}@{revision.commit_hash}",
                    "source": "huggingface",
                    "name": repo.repo_id,
                    "revision": revision.commit_hash,
                    "size": sum(f["size"] for f in files),
                    "files": sorted(files, key=lambda f: -f["size"]),
                }
            )
    civitai = {pin[1] for v in _remembered() if v.startswith("civitai:") and (pin := _pin(v))}
    if hub.URL_CACHE.is_dir():
        for folder in hub.URL_CACHE.iterdir():
            if not (_SHA.match(folder.name) and folder.is_dir()):
                continue
            named = _named_by(uses, PINNED, folder.name, "")
            files = [
                {"name": p.name, "size": p.stat().st_blocks * 512, "uses": named}
                for p in folder.iterdir()
                if p.is_file() and p.name != hub.PINNED_MARKER
            ]
            if not files:
                continue
            out.append(
                {
                    "id": PINNED + folder.name,
                    "source": "civitai" if folder.name in civitai else "url",
                    "name": files[0]["name"],
                    "revision": folder.name,
                    "size": sum(f["size"] for f in files),
                    "files": files,
                }
            )
    return sorted(out, key=lambda e: -e["size"])


def _remembered() -> list[str]:
    with store.connect() as connection:
        return [v for values in store.model_choices(connection).values() for v in values]


def in_use(entry_id: str, names: list[str]) -> bool:
    """Whether a setting or the code names any of `names` in the entry (all of it when empty)."""
    entry = next((e for e in listing() if e["id"] == entry_id), None)
    if entry is None:
        raise KeyError(entry_id)
    return any(u["where"] != "remembered" for f in entry["files"] if not names or f["name"] in names for u in f["uses"])


def delete(entry_id: str, names: list[str]) -> None:
    """The entry, or only `names` inside a Hub snapshot. Raises KeyError for anything `listing` would not show."""
    if entry_id.startswith(PINNED):
        sha = entry_id.removeprefix(PINNED)
        if not _SHA.match(sha) or not (hub.URL_CACHE / sha).is_dir():
            raise KeyError(entry_id)
        shutil.rmtree(hub.URL_CACHE / sha)
        return
    repo_id, _, commit = entry_id.rpartition("@")
    cache = scan_cache_dir(hub.HF_HUB_CACHE)
    revision = next(
        (r for repo in cache.repos if repo.repo_id == repo_id for r in repo.revisions if r.commit_hash == commit), None
    )
    if revision is None:
        raise KeyError(entry_id)
    snapshot = revision.snapshot_path
    chosen = [f for f in revision.files if f.file_path.relative_to(snapshot).as_posix() in names]
    if len(chosen) != len(set(names)):
        raise KeyError(entry_id)
    rest = [f for f in revision.files if f not in chosen and not f.file_name.startswith(MARKER)]
    if not names or not rest:
        cache.delete_revisions(commit).execute()
        return
    cache.delete_files(*chosen).execute()
    # A marker that lists nothing passes `_intact`, so it would hide the gap from the next fetch.
    for marker in snapshot.glob(MARKER + "*"):
        try:
            listed = json.loads(marker.read_text() or "{}")
        except (OSError, ValueError):
            listed = {}
        if not listed or any(name in listed for name in names):
            marker.unlink(missing_ok=True)
