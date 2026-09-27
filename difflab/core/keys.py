"""What identifies a made thing: the tool that makes it, that tool's key version, and its inputs by
name. Two calls with equal parts share one key and one artifact.

The parts are listed by name at the call site, so a key says what it was made of. Adding a part
changes every key of that tool, which is what the version is for: bump it, and the change is
deliberate and visible. Nothing here reflects over a class, so renaming one moves nothing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import torch

from difflab.core.settings import Settings


@dataclass(frozen=True)
class Key:
    tool: str
    version: int
    parts: dict[str, Any]

    @property
    def token(self) -> str:
        digest = hashlib.sha256()
        digest.update(f"{self.tool}/v{self.version}".encode())
        for name in sorted(self.parts):
            digest.update(b"\x00" + name.encode() + b"=" + canonical(self.parts[name]).encode())
        return digest.hexdigest()[:20]

    def __str__(self) -> str:
        return f"{self.tool}/v{self.version}/{self.token}"

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Key) and str(self) == str(other)


def key(tool: str, version: int, **parts: Any) -> Key:
    return Key(tool, version, parts)


def canonical(value: Any) -> str:
    """Any input a tool takes, rendered so equal inputs give equal text. Pixels and latents by their
    bytes, settings by their keyed fields, everything else as sorted JSON."""
    if isinstance(value, torch.Tensor):
        return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    if isinstance(value, Settings):
        return canonical(value.key_part())
    if is_dataclass(value) and not isinstance(value, type):
        return canonical({f.name: getattr(value, f.name) for f in fields(value)})
    if isinstance(value, dict):
        return json.dumps({str(k): canonical(v) for k, v in sorted(value.items())}, sort_keys=True)
    if isinstance(value, list | tuple):
        return json.dumps([canonical(item) for item in value])
    if isinstance(value, Path):
        return str(value)
    return json.dumps(value, default=str, sort_keys=True)
