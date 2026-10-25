"""The Take tab's jobs as an MCP server, for agents: Streamable HTTP at /mcp, answered as plain JSON with
no session and no server-sent stream, which the spec allows a server that never calls the client."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from difflab.api import assets, runs
from difflab.screenplay import settings
from difflab.work import queue
from difflab.work.quick import Job

VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
WAIT_MOST = 600.0  # seconds; a client's own tool timeout is usually shorter, and the run goes on either way
POLL = 1.0
TEXT_MOST = 4000
PARTS = {
    "video": "render_chunk",
    "ltx": "render_ltx",
    "edit": "edit_image",
    "vlm": "write_instruction",
    "video_vlm": "write_video_prompt",
}
SETTLED = ("done", "failed", "cancelled", "stopped")

INSTRUCTIONS = (
    "difflab renders on one GPU behind a queue. A make_* tool queues a run and returns its id at once; "
    "call `run` with wait_seconds to follow it. Photos and clips are paths the server holds: send yours "
    "with `upload` first. `take_settings` lists every knob the `settings` argument accepts."
)

_WORDS = {
    "idea": {"type": "string", "description": "what to make, in plain words; the VLM writes the prompt"},
    "prompt": {"type": "string", "description": "the prompt exactly as the model gets it; give this or idea"},
}
_SETTINGS = {
    "type": "object",
    "description": 'per part, fields over this machine\'s settings: {"edit": {"seed": 7}}; see take_settings',
    "properties": {part: {"type": "object"} for part in PARTS},
    "additionalProperties": False,
}
_FRAMING = {
    "aspect": {"type": "string", "description": '"W:H" such as "3:2"; empty keeps the photo\'s own'},
    "fit": {"type": "string", "enum": ["crop", "extend"], "description": "how a photo is brought to aspect"},
}
_FINISH = {
    "finish_megapixels": {"type": "number", "description": "the size to deliver at, enlarged after rendering"},
    "grade_strength": {
        "type": "number",
        "description": "0-1: how far the clip is pulled back to its opening colour; 0.5 closes half the gap",
    },
    "retime_fps": {"type": "integer", "description": "deliver at this frame rate, above 24, by interpolation"},
}
_PHOTOS = {"type": "array", "items": {"type": "string"}}
_RUN_ID = {"run_id": {"type": "string"}}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "upload",
        "description": "Store a picture, clip or sound on the server; returns the path a job names it by.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "its name, for the suffix: photo.jpg, clip.mp4"},
                "data": {"type": "string", "description": "the file, base64"},
            },
            "required": ["filename", "data"],
        },
    },
    {
        "name": "make_picture",
        "description": "Queue a picture: drawn from words and reference photos, or an edit of a start photo.",
        "inputSchema": {
            "type": "object",
            "properties": {
                **_WORDS,
                "photos": {**_PHOTOS, "description": "references, in order; the first is edited when start is set"},
                "start": {"type": "boolean", "description": "edit photos[0] rather than draw a fresh canvas"},
                "engine": {"type": "string", "enum": ["qwen-image-edit", "qwen-image-2.1", "minimax-h3"]},
                **_FRAMING,
                "settings": _SETTINGS,
            },
        },
    },
    {
        "name": "make_clip",
        "description": "Queue a video clip that opens on photos[0]; the rest of the photos are references.",
        "inputSchema": {
            "type": "object",
            "properties": {
                **_WORDS,
                "photos": {**_PHOTOS, "minItems": 1},
                "engine": {"type": "string", "enum": ["minimax-h3", "ltx-2.5"]},
                "duration": {"type": "number", "description": "seconds"},
                **_FRAMING,
                **_FINISH,
                "settings": _SETTINGS,
            },
            "required": ["photos"],
        },
    },
    {
        "name": "upscale",
        "description": "Queue an upscale of a finished clip to finish_megapixels.",
        "inputSchema": {
            "type": "object",
            "properties": {"clip": {"type": "string"}, **_FINISH, "settings": _SETTINGS},
            "required": ["clip", "finish_megapixels"],
        },
    },
    {
        "name": "run",
        "description": "A run's state, its jobs and what it made; waits up to wait_seconds for it to settle.",
        "inputSchema": {
            "type": "object",
            "properties": {**_RUN_ID, "wait_seconds": {"type": "number", "maximum": WAIT_MOST}},
            "required": ["run_id"],
        },
    },
    {
        "name": "list_runs",
        "description": "The newest runs, with their state.",
        "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer"}}},
    },
    {
        "name": "cancel_run",
        "description": "Cancel a queued or running run.",
        "inputSchema": {"type": "object", "properties": _RUN_ID, "required": ["run_id"]},
    },
    {
        "name": "take_settings",
        "description": "Each settings part's fields, what they do, and the value this machine runs them at.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class UnknownToolError(LookupError):
    pass


class ToolError(ValueError):
    """A call refused with a reason the agent can act on, answered as a tool result with isError."""


async def answer(message: Any, base_url: str, gone: Callable[[], Awaitable[bool]]) -> dict[str, Any] | None:
    """The response to one JSON-RPC message, or None for a notification."""
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
        return _error(None, -32600, "not a JSON-RPC 2.0 request")
    if "id" not in message:
        return None
    ask_id, method, params = message["id"], message["method"], message.get("params") or {}
    if method == "initialize":
        wanted = params.get("protocolVersion")
        return _result(
            ask_id,
            {
                "protocolVersion": wanted if wanted in VERSIONS else VERSIONS[0],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "difflab", "version": "1"},
                "instructions": INSTRUCTIONS,
            },
        )
    if method == "ping":
        return _result(ask_id, {})
    if method == "tools/list":
        return _result(ask_id, {"tools": TOOLS})
    if method == "tools/call":
        name, given = params.get("name"), params.get("arguments") or {}
        try:
            content = await _call(str(name), dict(given), base_url, gone)
        except UnknownToolError as problem:
            return _error(ask_id, -32602, str(problem))
        except KeyError as missing:
            return _result(ask_id, {"content": [{"type": "text", "text": f"missing {missing}"}], "isError": True})
        except (ToolError, TypeError, ValueError) as problem:
            return _result(ask_id, {"content": [{"type": "text", "text": str(problem)}], "isError": True})
        return _result(ask_id, {"content": content, "isError": False})
    return _error(ask_id, -32601, f"no method {method!r}")


def _result(ask_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": ask_id, "result": result}


def _error(ask_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": ask_id, "error": {"code": code, "message": message}}


async def _call(
    name: str, given: dict[str, Any], base_url: str, gone: Callable[[], Awaitable[bool]]
) -> list[dict[str, Any]]:
    if name == "run":
        return await _follow(str(given["run_id"]), float(given.get("wait_seconds") or 0), base_url, gone)
    calls: dict[str, Callable[[dict[str, Any]], Any]] = {
        "upload": _upload,
        "make_picture": _picture,
        "make_clip": _clip,
        "upscale": _upscale,
        "list_runs": _list_runs,
        "cancel_run": _cancel,
        "take_settings": _settings,
    }
    if name not in calls:
        raise UnknownToolError(f"no tool {name!r}")
    return [_json(await asyncio.to_thread(calls[name], given))]


def _json(data: Any) -> dict[str, Any]:
    return {"type": "text", "text": json.dumps(data, ensure_ascii=False, indent=1)}


# --- the tools -----------------------------------------------------------------------------------
def _upload(given: dict[str, Any]) -> dict[str, Any]:
    stored = assets.store(base64.b64decode(given["data"]), str(given["filename"]))
    return {"path": stored["path"], "width": stored.get("width"), "height": stored.get("height")}


def _picture(given: dict[str, Any]) -> dict[str, Any]:
    fields = _framed(given, _worded(given))
    fields |= {"mode": "picture", "photos": list(given.get("photos") or []), "start": bool(given.get("start"))}
    knobs = dict(given.get("settings") or {})
    if given.get("engine"):
        knobs["edit"] = {**knobs.get("edit", {}), "engine": given["engine"]}
    return _queue(fields, knobs)


def _clip(given: dict[str, Any]) -> dict[str, Any]:
    fields = _finished(given, _framed(given, _worded(given)))
    engine = str(given.get("engine") or "minimax-h3")
    fields |= {"mode": "video", "photos": list(given["photos"]), "video_engine": engine}
    knobs = dict(given.get("settings") or {})
    if given.get("duration") is not None:
        part = "ltx" if engine == "ltx-2.5" else "video"
        knobs[part] = {**knobs.get(part, {}), "duration": float(given["duration"])}
    return _queue(fields, knobs)


def _upscale(given: dict[str, Any]) -> dict[str, Any]:
    return _queue(_finished(given, {"mode": "upscale", "clip": str(given["clip"])}), dict(given.get("settings") or {}))


def _worded(given: dict[str, Any]) -> dict[str, Any]:
    return {"idea": str(given.get("idea") or ""), "prompt": str(given.get("prompt") or "")}


def _framed(given: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
    return fields | {name: given[name] for name in ("aspect", "fit") if given.get(name)}


def _finished(given: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
    return fields | {name: given[name] for name in _FINISH if given.get(name) is not None}


def _queue(fields: dict[str, Any], knobs: dict[str, Any]) -> dict[str, Any]:
    """The job as the page would send it: each part at this machine's settings, the caller's fields over them."""
    unknown = sorted(set(knobs) - set(PARTS))
    if unknown:
        raise ToolError(f"settings has no part {', '.join(unknown)} (parts: {', '.join(PARTS)})")
    for part, tool in PARTS.items():
        fields[part] = settings.effective(tool)[0].data() | dict(knobs.get(part) or {})
    job = Job.of(fields)
    problems = job.check()
    if problems:
        raise ToolError("; ".join(problems))
    return {"run_id": queue.enqueue(job, asked_by="mcp"), "title": job.title}


def _list_runs(given: dict[str, Any]) -> list[dict[str, Any]]:
    keep = ("id", "title", "status", "asked_by", "created", "ended", "jobs_done", "jobs_total", "error")
    return [{name: run[name] for name in keep} for run in runs.runs()[: int(given.get("limit") or 20)]]


def _cancel(given: dict[str, Any]) -> dict[str, Any]:
    problem = runs.control(Path(str(given["run_id"])).name, {"cmd": "cancel"})
    if problem:
        raise ToolError(problem)
    return {"cancelled": True}


def _settings(_: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for part, tool in PARTS.items():
        values = settings.effective(tool)[0].data()
        schema = settings.settings_class(tool).schema()
        out[part] = {
            "tool": tool,
            "fields": [
                {
                    "name": f["name"],
                    "type": f["type"],
                    "value": values.get(f["name"]),
                    **({"choices": f["choices"]} if f["choices"] else {}),
                    **{k: f[k] for k in ("hint", "unit", "low", "high") if f.get(k) not in (None, "")},
                }
                for f in schema
            ],
        }
    return out


async def _follow(run_id: str, wait: float, base_url: str, gone: Callable[[], Awaitable[bool]]) -> list[dict[str, Any]]:
    run_id = Path(run_id).name
    until = time.monotonic() + min(max(wait, 0.0), WAIT_MOST)
    while True:
        found = await asyncio.to_thread(runs.one, run_id)
        if found is None:
            raise ToolError(f"no run {run_id!r}")
        if found["status"] in SETTLED or time.monotonic() >= until or await gone():
            break
        await asyncio.sleep(POLL)
    return await asyncio.to_thread(_described, found, base_url)


def _doing(tasks: list[dict[str, Any]]) -> str:
    task = next((t for t in reversed(tasks) if t["state"] == "running"), None)
    if task is None:
        return ""
    counted = f" {task['done']}/{task['total']} {task['unit']}".rstrip() if task["total"] else ""
    return f"{task['name']}{counted}"


def _described(found: dict[str, Any], base_url: str) -> list[dict[str, Any]]:
    made = [
        {"path": a["path"], "kind": a["kind"], "seed": a["seed"], "url": f"{base_url}output/{a['path']}"}
        for a in found["artifacts"]
    ]
    jobs = [
        {
            "tool": job["tool"],
            "state": job["state"],
            **({"doing": doing} if (doing := _doing(job["tasks"])) else {}),
            **({"error": job["error"]} if job["error"] else {}),
            **{
                name: value[:TEXT_MOST]
                for name, value in (job["outputs"] or {}).items()
                if name == "text" and isinstance(value, str)
            },
        }
        for job in found["jobs"]
    ]
    summary = {name: found[name] for name in ("id", "title", "status", "error", "jobs_done", "jobs_total")}
    content = [_json(summary | {"jobs": jobs, "made": made})]
    for artifact in made:
        if artifact["kind"] == "image":
            small = assets.preview_of(Path(artifact["path"]))
            if small is not None:
                content.append(
                    {
                        "type": "image",
                        "data": base64.b64encode(small.read_bytes()).decode(),
                        "mimeType": mimetypes.guess_type(small.name)[0] or "image/png",
                    }
                )
    return content
