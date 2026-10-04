"""Every downloaded model by name: a Hugging Face repo at a pinned commit, a Civitai file pinned by its
sha256, or a pinned URL.

A checkpoint's place on disk is known from its name alone (`path`), so a fingerprint, a page-cache
drop or an output's filename can use it before anything is downloaded.

This is the only place difflab touches the network: every `from_pretrained` passes local_files_only.
HF's own progress bars stay off (`difflab/__init__.py`): a download reports its bytes as events instead
(`ui.transfer`), sized from the Hub's file list and fed by the bytes landing in the cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

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

            listed = (
                HfApi(token=token(HUGGINGFACE))
                .model_info(self.repo, revision=self.revision, files_metadata=True)
                .siblings
                or []
            )
            wanted = [f for f in listed if not self.files or any(fnmatch(f.rfilename, p) for p in self.files)]
            missing = sum(f.size or 0 for f in wanted if not (self.path / f.rfilename).exists())
            blobs = self.path.parent.parent / "blobs"
            before = _bytes_in(blobs)
            with ui.transfer(f"Fetching {self.repo}", missing, lambda: _bytes_in(blobs) - before):
                snapshot_download(
                    self.repo,
                    revision=self.revision,
                    allow_patterns=list(self.files) or None,
                    token=token(HUGGINGFACE),
                )
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
            _download(self.url, self.path, self.sha256)
        return self.path


# civitai.com and civitai.red answer the same API; mature files download only from .red.
CIVITAI_URL = "https://civitai.red"
_CIVITAI_REF = re.compile(r"^civitai:(\d+)@([0-9a-f]{64}):(.+)$")


@dataclass(frozen=True)
class CivitaiFile:
    """One file of a Civitai model version. Civitai has no commits, so the sha256 it lists is the pin."""

    version: int
    sha256: str
    filename: str

    def __str__(self) -> str:
        return f"civitai:{self.version}@{self.sha256}:{self.filename}"

    @classmethod
    def parse(cls, text: str) -> CivitaiFile:
        match = _CIVITAI_REF.match(text.strip())
        if match is None:
            raise ValueError(f"{text!r}: write it as civitai:<version id>@<sha256>:<file>")
        return cls(int(match[1]), match[2], match[3])

    @property
    def path(self) -> Path:
        return URL_CACHE / self.sha256 / self.filename

    def fetch(self) -> Path:
        if not self.path.exists():
            files = _civitai(f"/api/v1/model-versions/{self.version}")["files"]
            url = next((f["downloadUrl"] for f in files if f["name"] == self.filename), None)
            if url is None:
                raise FileNotFoundError(f"Civitai version {self.version} no longer lists {self.filename}")
            if key := token(CIVITAI):
                url += ("&" if "?" in url else "?") + urllib.parse.urlencode({"token": key})
            try:
                _download(url, self.path, self.sha256)
            except urllib.error.HTTPError as problem:
                if problem.code not in (401, 403):
                    raise
                raise PermissionError(
                    f"Civitai refused {self.filename} ({problem.code}): most files need a logged-in download. Save "
                    "an API key from civitai.com/user/account under Settings → Accounts."
                ) from None
        return self.path


ModelFile = HubFile | CivitaiFile

HUGGINGFACE, CIVITAI = "huggingface", "civitai"
# Read when nothing is saved on the settings page. Hugging Face then falls back to its own login file.
TOKEN_ENV = {HUGGINGFACE: "HF_TOKEN", CIVITAI: "CIVITAI_TOKEN"}


def token(service: str) -> str | None:
    from difflab import store  # not at the top: store imports config, which names checkpoints

    with store.connect() as connection:
        saved = store.tokens(connection).get(service)
    return saved[0] if saved else os.environ.get(TOKEN_ENV[service]) or None


def account(service: str, key: str) -> str:
    """Whom the service says `key` belongs to; raises when it refuses the key."""
    if service == HUGGINGFACE:
        return str(HfApi(token=key).whoami()["name"])
    try:
        me = _civitai("/api/v1/me", key)
    except urllib.error.HTTPError as problem:
        raise PermissionError(f"Civitai refused the key ({problem.code})") from None
    return str(me.get("username") or me.get("id") or "")


def parse_file(text: str) -> ModelFile:
    return CivitaiFile.parse(text) if text.strip().startswith("civitai:") else HubFile.parse(text)


def check(text: str) -> None:
    """Refuses what no model field can load."""
    if text.strip().startswith("civitai:"):
        CivitaiFile.parse(text)
    else:
        Checkpoint.parse(text)


_HF_URL = re.compile(
    r"^https?://(?:www\.)?(?:huggingface\.co|hf\.co)/([\w.-]+/[\w.-]+)(?:/(?:blob|resolve|tree)/([^/]+)(?:/(.+))?)?/?$"
)
_HF_LOOSE = re.compile(r"^([\w.-]+/[\w.-]+)(?:@([^:]+))?(?::(.+))?$")


def resolve(text: str) -> str:
    """A pasted page URL, a branch name or a bare Civitai id, as the pinned reference a field stores."""
    text = text.strip().split("#", 1)[0]
    if text.startswith("civitai:") and "@" not in text:
        return _civitai_ref(int(text.removeprefix("civitai:").split(":", 1)[0]), None)
    if "civitai." in text:
        return _civitai_url(text)
    if (match := _HF_URL.match(text)) or (match := _HF_LOOSE.match(text)):
        repo, revision, path = match[1], match[2] or "main", match[3]
        if not _REVISION.match(revision):
            revision = (
                HfApi(token=token(HUGGINGFACE)).model_info(repo, revision=urllib.parse.unquote(revision)).sha
                or revision
            )
        return f"{repo}@{revision}" + (f":{urllib.parse.unquote(path)}" if path else "")
    check(text)
    return text


def _civitai_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    file_id = int(query["fileId"][0]) if "fileId" in query else None
    if "modelVersionId" in query:
        return _civitai_ref(int(query["modelVersionId"][0]), file_id)
    if match := re.search(r"/(?:api/download/models|model-versions)/(\d+)", parsed.path):
        return _civitai_ref(int(match[1]), file_id)
    if match := re.search(r"/models/(\d+)", parsed.path):
        return _civitai_ref(int(_civitai(f"/api/v1/models/{match[1]}")["modelVersions"][0]["id"]), file_id)
    raise ValueError(f"{url!r}: not a Civitai model or version page")


def _civitai_ref(version: int, file_id: int | None) -> str:
    files = [f for f in _civitai(f"/api/v1/model-versions/{version}")["files"] if f.get("hashes", {}).get("SHA256")]
    chosen = next((f for f in files if f["id"] == file_id), None) if file_id is not None else None
    chosen = (
        chosen
        or next((f for f in files if f.get("primary")), None)
        or next((f for f in files if f.get("type") == "Model"), None)
    )
    if chosen is None:
        raise ValueError(f"Civitai version {version} lists no model file with a hash")
    return str(CivitaiFile(version, chosen["hashes"]["SHA256"].lower(), chosen["name"]))


def _civitai(path: str, key: str | None = None) -> dict[str, Any]:
    headers = {"User-Agent": "difflab"} | ({"Authorization": f"Bearer {key}"} if key else {})
    request = urllib.request.Request(CIVITAI_URL + path, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        found: dict[str, Any] = json.load(response)
    return found


def _download(url: str, path: Path, sha256: str) -> None:
    from difflab.core import ui  # not at the top: config names checkpoints, and ui imports config

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "difflab"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        with ui.transfer(f"Fetching {path.name}", total, lambda: out.tell()):
            while block := response.read(1 << 20):
                digest.update(block)
                out.write(block)
    if digest.hexdigest() != sha256:
        partial.unlink()
        raise RuntimeError(f"{path.name} has sha256 {digest.hexdigest()}, not the pinned {sha256}")
    partial.rename(path)
