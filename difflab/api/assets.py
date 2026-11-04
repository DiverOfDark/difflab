"""Pictures the browser sent, kept where a job can name them: over HTTP the photos are on a laptop and
the card is in a pod, so a photo has to arrive before it can be part of a take.

An upload is stored under its own content hash, which makes the same picture sent twice the same
file - and a job that names it keeps naming the same bytes, so the director's fingerprint does not
move because a file was re-uploaded. Each one gets a small thumbnail beside it, because the form
shows five of them and the originals are several megabytes each.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

import av
from PIL import Image

from difflab import config

ASSETS_DIR = config.OUTPUT_DIR / "assets"
THUMB = 320  # the long edge of the picture the form shows
PICTURES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLIPS = {".mp4", ".mov", ".mkv", ".webm"}
KINDS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
}


def store(data: bytes, filename: str) -> dict[str, Any]:
    suffix = Path(filename).suffix.lower()
    if suffix not in KINDS:
        raise ValueError(f"{filename}: not a picture, a clip or a sound ({', '.join(sorted(KINDS))})")
    digest = hashlib.sha256(data).hexdigest()[:16]
    name = f"{digest}-{_safe(Path(filename).stem)}{suffix}"
    path = ASSETS_DIR / name
    if not path.exists():
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return describe(path)


def describe(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path),
        "name": _shown(path.name),
        "url": f"/uploads/{path.name}",
        "bytes": path.stat().st_size,
    }
    thumb = _thumbnail(path)
    if thumb is not None:
        info["thumb"] = f"/uploads/{thumb.name}"
    if path.suffix.lower() in PICTURES:
        with Image.open(path) as picture:
            info["width"], info["height"] = picture.size
    return info


def listing() -> list[dict[str, Any]]:
    if not ASSETS_DIR.is_dir():
        return []
    files = (f for f in ASSETS_DIR.iterdir() if f.is_file() and ".thumb" not in f.suffixes)
    return [describe(f) for f in sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)]


def preview_of(path: Path) -> Path | None:
    """A thumbnail for a picture the pod already holds, for a job made on the machine the card is in
    and naming photos under `sources/` or `output/`. The roots are an allow-list, because the
    alternative is a route that reads any file on the pod. Relative is under output/, as a job reads it."""
    path = (path if path.is_absolute() else config.OUTPUT_DIR / path).resolve()
    roots = (ASSETS_DIR.resolve(), (config.PROJECT_DIR / "sources").resolve(), config.OUTPUT_DIR.resolve())
    if not path.is_file() or not any(path.is_relative_to(root) for root in roots):
        return None
    return _thumbnail(path)


def resolve(name: str) -> Path | None:
    """The file behind a URL, refusing anything that tries to leave the folder."""
    path = (ASSETS_DIR / name).resolve()
    return path if path.is_file() and path.is_relative_to(ASSETS_DIR.resolve()) else None


def _thumbnail(path: Path) -> Path | None:
    """A small copy beside the original, drawn once; a clip's is its first frame. `None` for a sound."""
    suffix = path.suffix.lower()
    if suffix not in PICTURES and suffix not in CLIPS:
        return None
    thumb = path.with_suffix(".thumb.jpg" if suffix in CLIPS else f".thumb{path.suffix}")
    if thumb.exists():
        return thumb
    # Drawn to a private name and renamed: the gallery asks for one thumbnail from several
    # connections at once, and a half-written file is served as a broken picture.
    partial = thumb.with_name(f".{os.getpid()}-{os.urandom(4).hex()}-{thumb.name}")
    try:
        small = _first_frame(path) if suffix in CLIPS else _opened(path)
        small.thumbnail((THUMB, THUMB))
        small.save(partial, quality=85)
        partial.replace(thumb)
    except (OSError, ValueError, av.FFmpegError):
        partial.unlink(missing_ok=True)
        return None
    return thumb


def _opened(path: Path) -> Image.Image:
    with Image.open(path) as picture:
        return picture.convert("RGB")


def _first_frame(path: Path) -> Image.Image:
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            return frame.to_image().convert("RGB")
    raise ValueError(f"{path.name} has no video frames")


def _safe(stem: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)[:64] or "photo"


def _shown(name: str) -> str:
    """The name without the hash that was put in front of it."""
    return name.split("-", 1)[1] if "-" in name else name
