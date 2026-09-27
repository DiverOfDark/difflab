"""Every downloaded model by name: a Hugging Face repo at a pinned commit, or a pinned URL.

A checkpoint's place on disk is known from its name alone (`path`), so a fingerprint, a page-cache
drop or an output's filename can use it before anything is downloaded.

This is the only place difflab touches the network: every `from_pretrained` passes local_files_only.
HF's own progress bars stay off (`difflab/__init__.py`): a download reports its bytes as events instead
(`ui.transfer`), sized from the Hub's file list and fed by the bytes landing in the cache.
"""

from __future__ import annotations

import hashlib
import os
import re
import urllib.request
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.file_download import repo_folder_name

URL_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "difflab"


def _bytes_in(folder: Path) -> int:
    """Counted by allocated blocks, not size, since xet writes chunks at their offsets in parallel and a
    file's size can reach its end long before the bytes in between do."""
    if not folder.is_dir():
        return 0
    return sum(entry.stat().st_blocks * 512 for entry in folder.iterdir() if entry.is_file())


_REVISION = re.compile(r"^[0-9a-f]{40}$")


def _split(text: str) -> tuple[str, str, str]:
    """`repo@commit[:rest]` -> (repo, commit, rest). A commit sha, because `Checkpoint.path` is
    `snapshots/<revision>` and a branch name would point at a folder the Hub never writes."""
    head, _, rest = text.strip().partition(":")
    repo, at, revision = head.partition("@")
    if not repo or not at or not _REVISION.match(revision):
        raise ValueError(f"{text!r}: write it as org/repo@<40-hex commit>[:file]")
    return repo, revision, rest


@dataclass(frozen=True)
class Checkpoint:
    repo: str  # "Qwen/Qwen3.8-27B"
    revision: str  # a commit sha, so the folder is known before the download and never moves
    files: tuple[str, ...] = ()  # glob patterns inside the repo; none -> the whole repo

    def __str__(self) -> str:
        return f"{self.repo}@{self.revision}" + (":" + ",".join(self.files) if self.files else "")

    @classmethod
    def parse(cls, text: str) -> Checkpoint:
        repo, revision, rest = _split(text)
        return cls(repo, revision, tuple(part for part in rest.split(",") if part))

    @property
    def path(self) -> Path:
        """The snapshot folder, downloaded or not."""
        return Path(HF_HUB_CACHE) / repo_folder_name(repo_id=self.repo, repo_type="model") / "snapshots" / self.revision

    def fetch(self) -> Path:
        """The snapshot folder, with every wanted file in it: downloaded now if a fetch never completed."""
        marker = self.path / f".difflab-fetched-{hashlib.sha256(repr(self.files).encode()).hexdigest()[:8]}"
        if not marker.exists():
            from difflab.core import ui  # not at the top: config names checkpoints, and ui imports config

            listed = HfApi().model_info(self.repo, revision=self.revision, files_metadata=True).siblings or []
            wanted = [f for f in listed if not self.files or any(fnmatch(f.rfilename, p) for p in self.files)]
            missing = sum(f.size or 0 for f in wanted if not (self.path / f.rfilename).exists())
            blobs = self.path.parent.parent / "blobs"
            before = _bytes_in(blobs)
            with ui.transfer(f"Fetching {self.repo}", missing, lambda: _bytes_in(blobs) - before):
                snapshot_download(self.repo, revision=self.revision, allow_patterns=list(self.files) or None)
            marker.touch()
        return self.path


@dataclass(frozen=True)
class HubFile:
    repo: str
    revision: str  # a commit sha
    filename: str  # its path inside the repo: "vae/minimax_h3_video_vae_fp16.safetensors"

    def __str__(self) -> str:
        return f"{self.repo}@{self.revision}:{self.filename}"

    @classmethod
    def parse(cls, text: str) -> HubFile:
        repo, revision, filename = _split(text)
        if not filename:
            raise ValueError(f"{text!r}: a file needs its path after the colon")
        return cls(repo, revision, filename)

    @property
    def path(self) -> Path:
        """The file in its snapshot folder, downloaded or not."""
        return Checkpoint(self.repo, self.revision).path / self.filename

    def fetch(self) -> Path:
        return Checkpoint(self.repo, self.revision, (self.filename,)).fetch() / self.filename


@dataclass(frozen=True)
class UrlFile:
    """A file that lives outside the Hub, at a URL pinned to a commit and checked by its sha256."""

    url: str
    sha256: str

    @property
    def path(self) -> Path:
        return URL_CACHE / self.sha256 / self.url.rsplit("/", 1)[-1]

    def fetch(self) -> Path:
        if not self.path.exists():
            from difflab.core import ui

            self.path.parent.mkdir(parents=True, exist_ok=True)
            partial = self.path.with_suffix(self.path.suffix + ".partial")
            digest = hashlib.sha256()
            with urllib.request.urlopen(self.url) as response, partial.open("wb") as out:
                total = int(response.headers.get("Content-Length") or 0)
                with ui.transfer(f"Fetching {self.path.name}", total, lambda: out.tell()):
                    while block := response.read(1 << 20):
                        digest.update(block)
                        out.write(block)
            if digest.hexdigest() != self.sha256:
                partial.unlink()
                raise RuntimeError(f"{self.url} has sha256 {digest.hexdigest()}, not the pinned {self.sha256}")
            partial.rename(self.path)
        return self.path
