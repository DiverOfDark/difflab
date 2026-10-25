"""Productions over MCP. Each tool is one route of `app.py`, called in-process so its staling and its
refusals are the page's; a production in an answer is digested, since in full one runs to 200 KB."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import re
from collections import Counter
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from difflab import config
from difflab.api import assets
from difflab.screenplay import planner, production

PICTURES_MOST = 6
FINISHED = ("made", "approved", "pinned")
FLAGGED = ("needs_review", "failed")
LIVE = ("queued", "running", "paused")
NUMBER = re.compile(r"\d+(?:\.\d+)?")
VIEWS = {
    "timeline": "/api/productions/{production_id}/timeline",
    "planning": "/api/productions/{production_id}/ingesting",
    "render_queue": "/api/productions/{production_id}/queue",
    "assembled": "/api/productions/{production_id}/assembled",
    "preflight": "/api/productions/{production_id}/preflight",
    "story_rounds": "/api/productions/{production_id}/ideas",
    "timings": "/api/timings",
}


class DirectorError(ValueError):
    """A route's refusal, or a call that has to be confirmed first."""


@dataclass(frozen=True)
class Route:
    name: str
    description: str
    method: str
    path: str
    properties: dict[str, Any] = field(default_factory=dict)
    required: tuple[str, ...] = ()
    query: tuple[str, ...] = ()
    changes: bool = False  # the `changes` argument is the whole body
    costly: Callable[[dict[str, Any]], str] | None = None  # what an unconfirmed call is told instead
    prepare: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    def tool(self) -> dict[str, Any]:
        named = set(re.findall(r"{(\w+)}", self.path))
        properties = {name: {"type": "integer"} for name in sorted(named)} | self.properties
        if self.costly:
            properties["confirm"] = {"type": "boolean", "description": "true to queue it; without, what it costs"}
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {"type": "object", "properties": properties, "required": sorted(named | set(self.required))},
        }


def _str(description: str = "") -> dict[str, Any]:
    return {"type": "string", **({"description": description} if description else {})}


def _num(description: str = "") -> dict[str, Any]:
    return {"type": "number", **({"description": description} if description else {})}


def _int(description: str = "") -> dict[str, Any]:
    return {"type": "integer", **({"description": description} if description else {})}


def _bool(description: str = "") -> dict[str, Any]:
    return {"type": "boolean", **({"description": description} if description else {})}


def _paths(description: str = "") -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}, **({"description": description} if description else {})}


def _changes(description: str) -> dict[str, Any]:
    return {"changes": {"type": "object", "description": description}}


CHAPTER = {"chapter": _int("a chapter's position, from 0; absent, every chapter")}
P = "/api/productions/{production_id}"


def _render_cost(given: dict[str, Any]) -> str:
    found = production.describe(int(given["production_id"]))
    if found is None:
        raise DirectorError("no such production")
    chapter = given.get("chapter")
    shots = [
        shot
        for one in found["chapters"]
        if chapter is None or one["position"] == int(chapter)
        for shot in one["shots"]
        if shot["clip"]["state"] not in FINISHED
    ]
    rate = (production.timings_now().get("render_shot") or {}).get("per_second") or 0
    seconds = sum(shot["seconds"] for shot in shots)
    clips = f"{len(shots)} clip{'' if len(shots) == 1 else 's'}"
    problems = {k: v for k, v in planner.preflight(int(given["production_id"])).items() if v}
    return (
        f"{clips} to render, {seconds:.0f} s of film, about {seconds * rate / 60:.0f} min on this card"
        + (f"; preflight refuses: {json.dumps(problems, ensure_ascii=False)}" if problems else "")
        + ". Call again with confirm: true to queue it."
    )


def _reroll_cost(given: dict[str, Any]) -> str:
    return (
        f"rerolls shot {given['shot_id']}'s clip and renders its chapter again: the shots chained after it follow. "
        "Call again with confirm: true to queue it."
    )


def _finish_cost(given: dict[str, Any]) -> str:
    scope = f"chapter {given['chapter']}" if given.get("chapter") is not None else "every chapter and the film"
    return f"enlarges {scope} to 4K and retimes it, the longest pass after render. Call again with confirm: true."


def _shot_lengths(given: dict[str, Any]) -> dict[str, Any]:
    """What the ingest form sends: the longest shot as the render's duration, at its megapixels and steps."""
    shortest, longest = float(given.pop("shortest", None) or 6), float(given.pop("longest", None) or 8)
    video = {"duration": longest, "megapixels": 0.4, "steps": 8} | dict(given.get("video") or {})
    return given | {"shortest": shortest, "video": video}


ROUTES: tuple[Route, ...] = (
    # --- a story, before and as it becomes a production
    Route(
        "draft_story",
        "The director rewrites an idea into a story it renders well, as a run; its text is the run's result. "
        "previous with feedback revises a round.",
        "POST",
        "/api/ideas",
        {
            "idea": _str(),
            "photos": _paths(),
            "take_seconds": _num("the longest shot"),
            "seconds": _num("the whole film"),
            "feedback": _str(),
            "previous": _str("the story being revised"),
            "round": _int(),
        },
        ("idea",),
    ),
    Route(
        "guess_references",
        "Each uploaded picture described by the VLM as a run: its kind and what it shows.",
        "POST",
        "/api/ingest/guess",
        {"paths": _paths()},
        ("paths",),
    ),
    Route(
        "reference_frames",
        "Sharp frames spread over an uploaded video, to use as references.",
        "POST",
        "/api/ingest/frames",
        {"path": _str(), "frames": _int(), "min_sharpness": _num()},
        ("path",),
    ),
    Route(
        "start_production",
        "A production from a story and its references. The bible is built at once as a run; plan: true plans "
        "every chapter too, stopping before render.",
        "POST",
        "/api/ingest",
        {
            "title": _str(),
            "story": _str(),
            "references": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": _str(),
                        "kind": {"type": "string", "enum": ["character", "location", "prop", "style", "video"]},
                        "entity": _str("who or what it shows, one name"),
                        "guess": _str("what it shows, as guess_references said"),
                    },
                    "required": ["path", "kind"],
                },
            },
            "target_seconds": _num("the whole film"),
            "shortest": _num("the shortest shot, seconds; 6 unless given"),
            "longest": _num("the longest shot, seconds, at most 15; 8 unless given"),
            "aspect": _num("width over height; absent, the first photo's"),
            "fit": {"type": "string", "enum": ["crop", "extend"]},
            "engine": {"type": "string", "enum": ["minimax-h3", "ltx-2.5"]},
            "plan": _bool(),
            "video": {"type": "object", "description": "render_chunk settings over the defaults"},
            "continuity": {"type": "object"},
            "ingest": {"type": "object"},
        },
        ("title", "story"),
        prepare=_shot_lengths,
    ),
    Route(
        "start_video_production",
        "A production that redoes a video: shots that each open on the source's frame and re-create that "
        "stretch with instruction as the one difference.",
        "POST",
        "/api/ingest/video",
        {
            "title": _str(),
            "source": _str("an uploaded video"),
            "instruction": _str(),
            "photos": _paths(),
            "clip_scale": _num(),
            "longest": _num("the longest shot, seconds, at most 15; 8 unless given"),
            "video": {"type": "object"},
            "continuity": {"type": "object"},
        },
        ("title", "source", "instruction"),
        prepare=lambda given: {k: v for k, v in _shot_lengths(given).items() if k != "shortest"},
    ),
    Route(
        "steer_planning",
        "skip: hand on what a planning step has not done yet; cancel: stop the plan.",
        "POST",
        P + "/ingesting",
        {"skip": _str("the step's name, as production_view planning shows it"), "cancel": _bool()},
    ),
    Route(
        "edit_production",
        "title, story, opening (the first frame in words, each member as {id}), aspect, fit, engine, keep. "
        "A changed story, opening, aspect or fit makes prompts, keyframes and clips stale until a re-plan.",
        "PATCH",
        P,
        {
            "title": _str(),
            "story": _str(),
            "opening": _str(),
            "aspect": _num(),
            "fit": {"type": "string", "enum": ["crop", "extend"]},
            "engine": _str('"" for minimax-h3, or ltx-2.5'),
            "keep": _bool("spares it from the cleanup"),
        },
    ),
    Route(
        "improve_story",
        "A round of the director rewriting the production's story, as a run. from_round with feedback "
        "revises that round; without, a fresh rewrite of idea.",
        "POST",
        P + "/ideas",
        {"feedback": _str(), "from_round": _int(), "idea": _str()},
    ),
    Route(
        "adopt_story",
        "A round's story becomes the production's; stale says what a re-plan has to redo.",
        "POST",
        P + "/ideas/{round}/adopt",
    ),
    Route(
        "add_photos",
        "Pictures given after ingest; they wait in the bible to be attached, and the next plan reads them.",
        "POST",
        P + "/photos",
        {"paths": _paths()},
        ("paths",),
    ),
    Route(
        "attach_photo",
        'An unmatched photo held as member to\'s picture (as its garment when named), or to "ignore".',
        "POST",
        P + "/unmatched",
        {"photo": _str(), "to": _str("a member's slug, or ignore"), "garment": _str()},
        ("photo", "to"),
    ),
    # --- the bible
    Route(
        "add_member",
        "A member decided in words, pinned: a re-plan casts it where the story needs it. It needs at least one "
        "trait (part: hair, coat, markings, face, top, bottom, footwear, accessory, tail, ears, component, material, "
        "landmark, furniture or surface), and a place at least one light.",
        "POST",
        P + "/members",
        {
            "slug": _str("one lower-case word, or words joined by _"),
            "kind": _str("person, animal, object or environment"),
            "noun": _str(),
            "possessive": _str(),
            "find": _str("how to find it in a picture"),
            "traits": {"type": "array", "items": {"type": "object"}, "description": "{part, phrase, short, name}"},
            "lights": _paths(),
        },
        ("slug", "kind"),
    ),
    Route(
        "edit_member",
        "changes: noun, kind, possessive, find, traits, lights, photographed, appearance, allowed (pins it and "
        "stales its shots); voice, delivery, reference_audio; aliases; plate; {pinned: false} alone gives it "
        "back to the director.",
        "PATCH",
        "/api/members/{member_id}",
        _changes("the fields to change"),
        ("changes",),
        changes=True,
    ),
    Route("delete_member", "The member, out of the production.", "DELETE", "/api/members/{member_id}"),
    Route(
        "draw_member",
        "The member's picture drawn from its words as a run; again, a new seed unless seed is given. "
        "describe writes a place's words from the story first.",
        "POST",
        "/api/members/{member_id}/reference",
        {"seed": _int(), "describe": _bool()},
    ),
    Route(
        "draw_member_variants", "Three drawings on new seeds, as a run.", "POST", "/api/members/{member_id}/variants"
    ),
    Route(
        "pick_member_variant",
        "One of the variants, by its seed, as the member's picture.",
        "POST",
        "/api/members/{member_id}/variants/pick",
        {"seed": _int()},
        ("seed",),
    ),
    Route(
        "give_member_picture",
        "An uploaded picture held as the member's role picture: face or body for a person, face or view for an "
        "animal, object, environment.",
        "POST",
        "/api/members/{member_id}/given",
        {"path": _str(), "role": _str()},
        ("path", "role"),
    ),
    Route(
        "dress_member",
        "An uploaded garment the member can change into, called name; no path takes it out.",
        "POST",
        "/api/members/{member_id}/wardrobe",
        {"name": _str(), "path": _str()},
        ("name",),
    ),
    Route(
        "describe_member",
        "The member read again from its photographs as a run.",
        "POST",
        "/api/members/{member_id}/describe",
    ),
    Route("detach_reference", "A reference that does not show its member, let go of.", "DELETE", "/api/refs/{ref_id}"),
    Route(
        "draw_references",
        "Every member with no usable picture, drawn from its words as one run.",
        "POST",
        P + "/references",
    ),
    Route(
        "resolve_conflict",
        "A conflict settled in words: which wording is kept; empty opens it again.",
        "POST",
        "/api/conflicts/{conflict_id}/resolve",
        {"resolution": _str()},
        ("resolution",),
    ),
    Route(
        "keep_conflict",
        "reference rewrites the story's sentences to the photograph; novel takes the story's trait.",
        "POST",
        "/api/conflicts/{conflict_id}/keep",
        {"keep": {"type": "string", "enum": ["reference", "novel"]}},
        ("keep",),
    ),
    # --- shots
    Route(
        "plan",
        "Plan a chapter, or every one, a run each. fresh asks for new shots though nothing changed; "
        "shots: false only builds the bible again.",
        "POST",
        P + "/plan",
        {**CHAPTER, "fresh": _bool(), "shots": _bool()},
        query=("chapter", "fresh", "shots"),
    ),
    Route(
        "edit_shot",
        'changes: prompt, keyframe (a path, composed, opening, or "" to chain again), opening (keyframe, '
        "chain or refs), landed, seed, closing, cast, camera_move, camera_toward, dialogue, sound, seconds. "
        "A plan edit pins the shot and recompiles its prompt.",
        "PATCH",
        "/api/shots/{shot_id}",
        _changes("the fields to change"),
        ("changes",),
        changes=True,
    ),
    Route("delete_shot", "The shot, out of its chapter.", "DELETE", "/api/shots/{shot_id}"),
    Route(
        "rewrite_shot",
        "The shot's prose and prompt written again as a run; a hand-written prompt is let go.",
        "POST",
        "/api/shots/{shot_id}/rewrite",
    ),
    Route(
        "reroll_shot",
        "One more attempt at the shot's clip on a new seed, unless seed is given; its chapter renders again.",
        "POST",
        "/api/shots/{shot_id}/reroll",
        {"seed": _int()},
        costly=_reroll_cost,
    ),
    Route(
        "redo_keyframe",
        "The shot's keyframe composed again on a new seed, note added to what the editor is told; only the "
        "chapter's keyframes are made, not its clips.",
        "POST",
        "/api/shots/{shot_id}/keyframe",
        {"note": _str()},
    ),
    Route(
        "draw_keyframes",
        "The composed keyframes and their verdicts, without clips.",
        "POST",
        P + "/keyframes",
        CHAPTER,
        query=("chapter",),
    ),
    Route(
        "add_casting",
        "A member (its slug) into the shot, as it stands unless parent, amend or label say otherwise.",
        "POST",
        "/api/shots/{shot_id}/castings",
        {
            "member": _str(),
            "parent": {"description": 'null its last look, "base", or an earlier casting id'},
            "amend": _str("what this look has beyond its parent"),
            "label": _str(),
        },
        ("member",),
    ),
    Route(
        "edit_casting",
        "changes: parent, amend, label, pictured, form, form_kind (what the look turns the member into).",
        "PATCH",
        "/api/castings/{casting_id}",
        _changes("the fields to change"),
        ("changes",),
        changes=True,
    ),
    Route("remove_casting", "The member, out of the shot.", "DELETE", "/api/castings/{casting_id}"),
    Route(
        "give_look_picture",
        "An uploaded picture as the look's own instead of a drawn one, approved.",
        "POST",
        "/api/castings/{casting_id}/upload",
        {"path": _str()},
        ("path",),
    ),
    Route(
        "draw_look",
        "The look drawn from its source's picture and judged, as a run; again, a new seed.",
        "POST",
        "/api/castings/{casting_id}/draw",
    ),
    Route(
        "draw_shot_looks",
        "Every look of the shot with no picture, or a failed or stale one, as one run.",
        "POST",
        "/api/shots/{shot_id}/looks",
    ),
    Route(
        "draw_looks",
        "Every undrawn look of a chapter, or of every one, as one run.",
        "POST",
        P + "/looks",
        CHAPTER,
        query=("chapter",),
    ),
    # --- render and after
    Route(
        "render",
        "Render a chapter, or every one, a run each in chapter order.",
        "POST",
        P + "/render",
        CHAPTER,
        query=("chapter",),
        costly=_render_cost,
    ),
    Route(
        "steer_render",
        "pause: every live render run of the production after its clip in flight, or false to go on; "
        "auto_retry: whether the judge retakes a failing clip.",
        "POST",
        P + "/queue",
        {"pause": _bool(), "auto_retry": _bool()},
    ),
    Route(
        "assemble",
        "A chapter, or every one and the film, joined, levelled and subtitled from the clips it has.",
        "POST",
        P + "/assemble",
        CHAPTER,
        query=("chapter",),
    ),
    Route(
        "finish",
        "A chapter, or every one and the film, enlarged to 4K and retimed.",
        "POST",
        P + "/finish",
        CHAPTER,
        query=("chapter",),
        costly=_finish_cost,
    ),
)
BY_NAME = {route.name: route for route in ROUTES}

APPROVABLE = ("bible", "shot", "keyframe", "prompt", "clip", "reference", "casting")
READS: list[dict[str, Any]] = [
    {
        "name": "list_productions",
        "description": "Every production with its stages' phases and where it stopped.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "production",
        "description": "A production digested: its stages, cast, every shot in a line, what needs review, live runs.",
        "inputSchema": {"type": "object", "properties": {"production_id": _int()}, "required": ["production_id"]},
    },
    {
        "name": "shot",
        "description": "One shot in full: its plan, prompt, castings, keyframe and clip, pictures inline.",
        "inputSchema": {"type": "object", "properties": {"shot_id": _int()}, "required": ["shot_id"]},
    },
    {
        "name": "member",
        "description": "One member in full: its card, pictures, conflicts, wardrobe and looks, pictures inline.",
        "inputSchema": {"type": "object", "properties": {"member_id": _int()}, "required": ["member_id"]},
    },
    {
        "name": "production_view",
        "description": "One page of a production: timeline, planning (its progress), render_queue, assembled, "
        "preflight, story_rounds, or this machine's timings.",
        "inputSchema": {
            "type": "object",
            "properties": {"production_id": _int(), "view": {"type": "string", "enum": list(VIEWS)}, **CHAPTER},
            "required": ["production_id", "view"],
        },
    },
    {
        "name": "approve",
        "description": "Approve as a person would, so the judge no longer overrules it and a run paused on a clip "
        "goes on. bible: every made picture of the production (id is the production); shot: its prompt, "
        "keyframe and looks (id is the shot); keyframe, prompt, clip (id is the shot), reference, casting.",
        "inputSchema": {
            "type": "object",
            "properties": {"what": {"type": "string", "enum": list(APPROVABLE)}, "id": _int()},
            "required": ["what", "id"],
        },
    },
]
TOOLS: list[dict[str, Any]] = READS + [route.tool() for route in ROUTES]
NAMES = {tool["name"] for tool in TOOLS}


# --- calling a route ----------------------------------------------------------------------------
async def call(name: str, given: dict[str, Any], base_url: str) -> list[dict[str, Any]]:
    if name in ("list_productions", "production", "shot", "member"):
        return await asyncio.to_thread(_read, name, given, base_url)
    if name == "production_view":
        view = str(given["view"])
        if view not in VIEWS:
            raise DirectorError(f"view is one of {', '.join(VIEWS)}")
        if view == "timeline":
            return [await asyncio.to_thread(_timeline, int(given["production_id"]), given.get("chapter"))]
        query = (
            {"chapter": given["chapter"]}
            if given.get("chapter") is not None and view in ("timeline", "render_queue")
            else {}
        )
        return [_text(await _route("GET", VIEWS[view].format(**given), query))]
    if name == "approve":
        return [_text(_shaped(await _approve(str(given["what"]), int(given["id"]))))]
    route = BY_NAME[name]
    if route.costly and not given.get("confirm"):
        return [_text({"queued": False, "cost": await asyncio.to_thread(route.costly, given)})]
    named = set(re.findall(r"{(\w+)}", route.path))
    missing = sorted(n for n in named | set(route.required) if given.get(n) is None)
    if missing:
        raise DirectorError(f"missing {', '.join(missing)}")
    path = route.path.format(**{n: int(given[n]) for n in named})
    rest = {k: v for k, v in given.items() if k not in named and k != "confirm"}
    if route.prepare:
        rest = route.prepare(rest)
    query = {k: rest.pop(k) for k in route.query if k in rest and rest[k] is not None}
    body = rest.get("changes") if route.changes else rest
    return [_text(_shaped(await _route(route.method, path, query, body if route.method != "GET" else None)))]


def _read(name: str, given: dict[str, Any], base_url: str) -> list[dict[str, Any]]:
    if name == "list_productions":
        keep = ("id", "title", "chapters", "shots", "seconds", "stages", "resume", "live", "paused")
        return [_text([{k: p[k] for k in keep} for p in production.listing()])]
    if name == "shot":
        return _shot(int(given["shot_id"]), base_url)
    if name == "member":
        return _member(int(given["member_id"]), base_url)
    found = production.describe(int(given["production_id"]))
    if found is None:
        raise DirectorError("no such production")
    return [_text(digest(found))]


async def _approve(what: str, made_id: int) -> Any:
    if what == "bible":
        return await _route("POST", f"/api/productions/{made_id}/bible/approve")
    if what == "shot":
        return await _route("POST", f"/api/shots/{made_id}/approve")
    if what in APPROVABLE:
        return await _route("POST", f"/api/made/{what}/{made_id}/approve")
    raise DirectorError(f"what is one of {', '.join(APPROVABLE)}")


async def _route(method: str, path: str, query: dict[str, Any] | None = None, body: Any = None) -> Any:
    """The route as the page reaches it, without a socket: the token middleware is passed like any client."""
    from difflab.api.app import app

    sent = json.dumps(body if body is not None else {}).encode()
    headers = [(b"content-type", b"application/json")]
    if config.TOKEN:
        headers.append((b"authorization", f"Bearer {config.TOKEN}".encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": urlencode(
            {k: str(v).lower() if isinstance(v, bool) else v for k, v in (query or {}).items()}
        ).encode(),
        "headers": headers,
        "client": ("mcp", 0),
        "server": ("mcp", 0),
    }
    delivered = False
    status = 500
    chunks: list[bytes] = []

    async def receive() -> dict[str, Any]:
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": sent, "more_body": False}

    async def send(message: MutableMapping[str, Any]) -> None:
        nonlocal status
        if message["type"] == "http.response.start":
            status = message["status"]
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    await app(scope, receive, send)
    raw = b"".join(chunks)
    try:
        answer = json.loads(raw) if raw else None
    except ValueError:
        answer = raw.decode(errors="replace")
    if status >= 400:
        detail = answer.get("detail") if isinstance(answer, dict) else answer
        raise DirectorError(detail if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False))
    return answer


def _text(data: Any) -> dict[str, Any]:
    return {"type": "text", "text": json.dumps(data, ensure_ascii=False, indent=1)}


def _shaped(answer: Any) -> Any:
    if isinstance(answer, dict) and "chapters" in answer and "members" in answer:
        return digest(answer) | {k: answer[k] for k in ("stale", "left", "run", "runs") if k in answer}
    return answer


# --- reading a production -------------------------------------------------------------------------
def digest(found: dict[str, Any]) -> dict[str, Any]:
    listed = next((p for p in production.listing() if p["id"] == found["id"]), {})
    counted = ("clips_done", "clips_review", "bible_review", "shots_review", "to_draw")
    review: list[str] = []
    cast = []
    for m in found["members"]:
        unapproved = [
            r
            for r in m["references"]
            if r["artifact"] and not (r["state"] == "approved" and r["reviewed_by"] == "person")
        ]
        review += [
            f"member {m['id']} ({m['slug']}): reference {r['id']} {r['role']} is {r['state']}"
            + (f" by the {r['reviewed_by']}" if r["reviewed_by"] else "")
            + ", not approved by a person"
            for r in unapproved
        ]
        conflicts = [c for c in m["conflicts"] if not c.get("resolution")]
        review += [f"member {m['id']} ({m['slug']}): conflict {c.get('id')} open" for c in conflicts]
        cast.append(
            {
                "id": m["id"],
                "slug": m["slug"],
                "kind": m["kind"],
                "noun": m["noun"],
                "pinned": m["pinned"],
                "pictures": [f"{r['id']} {r['role']} {r['state']}" for r in m["references"]],
                **({"conflicts": conflicts} if conflicts else {}),
                **({"gaps": m["gaps"]} if m["gaps"] else {}),
                **({"wardrobe": len(m["wardrobe"])} if m["wardrobe"] else {}),
            }
        )
    review += [f"unmatched photo {photo}" for photo in found.get("unmatched") or []]
    chapters = []
    # Checks differ shot to shot only in their numbers: a 78-shot plan's were 29k characters of 84k, grouped 2.5k.
    checks: dict[tuple[str, str, str], list[tuple[int, str]]] = {}
    for chapter in found["chapters"]:
        shots: list[str] = []
        scene = action = None
        for s in chapter["shots"]:
            for part in ("keyframe", "prompt", "clip"):
                if s[part]["state"] in FLAGGED:
                    review.append(f"shot {s['id']}: {part} {s[part]['state']}")
            for check in s["prompt"].get("checks") or []:
                kind = (check["severity"], check["code"], NUMBER.sub("#", check["message"]))
                checks.setdefault(kind, []).append((s["id"], check["message"]))
            looks = [c for c in s["castings"] if c.get("amended")]
            review += [
                f"shot {s['id']}: look casting {c['id']} ({c['member']}) {c['state']}"
                for c in looks
                if c["state"] in FLAGGED
            ]
            if s["scene"] and s["scene"] != scene:
                shots.append(s["scene"])
                scene = s["scene"]
            seen = [c["member"] + (f" · {c['label']}" if c.get("amended") else "") for c in s["castings"]]
            takes = s["clip"].get("takes") or {}
            states = "/".join(s[part]["state"] for part in ("keyframe", "prompt", "clip"))
            extra = [f"cast {', '.join(seen)}"] if seen else []
            extra += [f"opens on {s['opening']}"] if s["opening"] != "keyframe" else []
            extra += [] if s["chained"] else ["not chained"]
            extra += [", ".join(f"{n} {k}" for k, n in takes.items() if n)] if any(takes.values()) else []
            extra += [f"says {json.dumps(s['dialogue'], ensure_ascii=False)}"] if s["dialogue"] else []
            said = "as above" if s["action"] == action else s["action"]
            action = s["action"]
            shots.append(
                f"{s['id']} · {s['seconds']:.1f} s · {states}" + "".join(f" · {one}" for one in extra) + f" · {said}"
            )
        chapters.append({"position": chapter["position"], "title": chapter["title"], "shots": shots})
    for (severity, code, _), where in checks.items():
        times = Counter(shot_id for shot_id, _ in where)
        ids = ", ".join(f"{shot_id}" + (f" ({n} times)" if n > 1 else "") for shot_id, n in times.items())
        said = where[0][1] if len(times) == 1 else f"as on shot {where[0][0]}: {where[0][1]}"
        review.append(f"shot{'s' if len(times) > 1 else ''} {ids}: prompt {severity} {code}, {said}")
    return {
        "id": found["id"],
        "title": found["title"],
        "story": found["story"],
        "stages": listed.get("stages"),
        "resume": listed.get("resume"),
        **{k: listed[k] for k in counted if k in listed},
        "cast": cast,
        "shot_line": "id · seconds · keyframe/prompt/clip · cast, opening, takes, dialogue when there are any · action",
        "chapters": chapters,
        "review": review,
        # The ingest route's answer puts the ids it queued where the production's runs go.
        "live_runs": [r for r in found["runs"] if isinstance(r, dict) and r["state"] in LIVE],
        **({"missing_references": found["missing_references"]} if found.get("missing_references") else {}),
    }


def _timeline(production_id: int, chapter: int | None) -> dict[str, Any]:
    """A line a shot and a line a member: the page's timeline, a chapter of 48 shots, was 19.5k characters."""
    found = production.timeline(production_id, None if chapter is None else int(chapter))
    if found is None:
        raise DirectorError("no such production")
    shots = found["shots"]
    lines = ["shot id · position · seconds · keyframe/prompt/clip · action"]
    scene = action = None
    for i, s in enumerate(shots):
        if i == 0 or s["chapter"] != shots[i - 1]["chapter"]:
            lines.append(f"chapter {s['chapter']}")
            scene = None
        if s["scene"] and s["scene"] != scene:
            lines.append(f"  {s['scene']}")
            scene = s["scene"]
        extra = [f"opens on {s['opening']}"] if s["opening"] != "keyframe" else []
        extra += [f"{s['waiting']} looks waiting"] if s["waiting"] else []
        said = "as above" if s["action"] == action else s["action"]
        action = s["action"]
        lines.append(
            f"  {s['id']} · #{s['position']} · {s['seconds']:.1f} s · {'/'.join(s['pipe'])}"
            + "".join(f" · {one}" for one in extra)
            + f" · {said}"
        )
    for track in found["tracks"]:
        runs, worn = [], None
        for run in track["runs"]:
            if run["gap"]:
                continue
            span = f"{shots[run['a']]['id']}" + (f"-{shots[run['b']]['id']}" if run["b"] > run["a"] else "")
            label = run["label"] + (" (back)" if run["revert"] else "") + (" (amended)" if run["mark"] else "")
            said = span if label == worn else f"{label} {span}"
            worn = label
            runs.append(said + (f" [{run['image']}]" if run["image"] != "done" else ""))
        lines.append(f"{track['member']} ({track['kind']}): {', '.join(runs)}")
    return {"type": "text", "text": "\n".join(lines)}


def _shot(shot_id: int, base_url: str) -> list[dict[str, Any]]:
    owner = production.production_of_shot(shot_id)
    found = production.describe(owner) if owner is not None else None
    shot = next((s for c in (found or {}).get("chapters", []) for s in c["shots"] if s["id"] == shot_id), None)
    if shot is None:
        raise DirectorError(f"no shot {shot_id}")
    if shot["clip"].get("artifact"):
        shot["clip"]["url"] = f"{base_url}output/{shot['clip']['artifact']}"
    pictured = [("keyframe", shot["keyframe"].get("artifact"))]
    pictured += [(f"look {c['id']} {c['member']}", c.get("artifact")) for c in shot["castings"] if c.get("amended")]
    return [_text(shot), *_pictures(pictured)]


def _member(member_id: int, base_url: str) -> list[dict[str, Any]]:
    owner = production.production_of_member(member_id)
    found = production.describe(owner) if owner is not None else None
    member = next((m for m in (found or {}).get("members", []) if str(m["id"]) == str(member_id)), None)
    if member is None:
        raise DirectorError(f"no member {member_id}")
    pictured = [(f"reference {r['id']} {r['role']}", r.get("artifact")) for r in member["references"]]
    pictured += [(f"variant seed {v.get('seed')}", v.get("artifact")) for v in member.get("variants") or []]
    return [_text(member), *_pictures(pictured)]


def _pictures(named: list[tuple[str, str | None]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label, path in named:
        if len(out) >= 2 * PICTURES_MOST:
            break
        small = assets.preview_of(Path(path)) if path else None
        if small is None:
            continue
        out.append({"type": "text", "text": label})
        out.append(
            {
                "type": "image",
                "data": base64.b64encode(small.read_bytes()).decode(),
                "mimeType": mimetypes.guess_type(small.name)[0] or "image/png",
            }
        )
    return out
