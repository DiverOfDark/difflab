"""What each stage of a manga plan run does: pages read, the cast named, panels laid out as shots, cleaned,
and each character drawn once in colour. Keyed like the other plan stages, by everything they read."""

from __future__ import annotations

import dataclasses
from typing import Any

from PIL import Image

from difflab import store
from difflab.core import context, ui
from difflab.core.cache import cache
from difflab.core.imaging import load_image, megapixel_size
from difflab.core.keys import key
from difflab.core.settings import EditSettings
from difflab.core.types import Frame
from difflab.models.minimax_h3 import caption
from difflab.models.minimax_h3.settings import VideoSettings
from difflab.screenplay import compiler, drawn, manga, settings, words
from difflab.screenplay import config as knobs
from difflab.screenplay import production as production_module
from difflab.tools import direct, edit_image, fill_shots, panels, read_page
from difflab.work.plan_runners import _fingerprint, _path, _start, _take, _taken, _voices, _write_and_compile

_READING = {name: value for name, value in vars(words).items() if name.startswith("MANGA_")}


def _manga() -> dict[str, Any]:
    """The model and the manga's own words and knobs: what every manga stage keys by beside its inputs."""
    return {
        "model": direct.MODEL.get(),
        "manga_words": _READING,
        "manga_knobs": {
            k: v for k, v in vars(knobs).items() if k.startswith(("MANGA_", "DIRECTOR_PHOTO", "DIRECTOR_SURVEY"))
        },
    }


def _out(job: int | None, name: str) -> dict[str, Any]:
    assert job is not None
    return {"from_job": job, "output": name}


def _saved(image: Image.Image, made: Any) -> str:
    target = cache.path(made, ".png")
    if not target.is_file():
        image.convert("RGB").save(target)
    return store.relative(target)


# --- reading -------------------------------------------------------------------------------------------
def run_read_page(inputs: dict[str, Any], settings_: dict[str, Any]) -> dict[str, Any]:
    path, reading = str(inputs["page"]), str(inputs["reading"])
    page = load_image(_path(path)).convert("RGB")
    after = load_image(_path(inputs["next"])).convert("RGB") if inputs.get("next") else None
    names = [str(n) for n in inputs.get("names") or []]
    pixels = Frame.from_pil(page).pixels
    made = key(
        "read_page", 1, page=pixels, after=Frame.from_pil(after).pixels if after is not None else None,
        reading=reading, names=names, **_manga(),
    )  # fmt: skip
    if (previous := _start("read_page", made)) is not None:
        return previous
    cut = panels.cut(page, reading)
    cut = read_page.confirm_bands(page, cut) if reading == "strip" else read_page.confirm(page, cut, reading)
    following = panels.cut(after, reading).panels[0] if after is not None else None
    found: list[dict[str, Any]] = []
    known = list(names)
    for index, panel in enumerate(cut.panels):
        shown = page.crop(panel.read_box)
        if index + 1 < len(cut.panels):
            nxt: Image.Image | None = page.crop(cut.panels[index + 1].read_box)
        elif after is not None and following is not None:
            nxt = after.crop(following.read_box)
        else:
            nxt = None
        script = read_page.read(shown, None if reading == "strip" else page, nxt, known, reading)
        for name in [*script.who, *(b.speaker for b in script.balloons)]:
            if name and name not in known:
                known.append(name)
        found.append({
            "box": list(panel.box), "read_box": list(panel.read_box), "confirmed": panel.framed,
            "picture": _saved(page.crop(panel.box), key("manga_panel", 1, page=pixels, box=list(panel.box))),
            "read": _saved(shown, key("manga_panel", 1, page=pixels, box=list(panel.read_box))),
            "script": script.data(),
        })  # fmt: skip
    ui.panel(f"page {int(inputs.get('index') or 0) + 1}", {
        f"panel {i + 1}": f"{len(p['script']['balloons'])} balloon(s) · {p['script']['into_next']} · "
                          f"{p['script']['action'][:60]}" for i, p in enumerate(found)
    })  # fmt: skip
    return {"page": path, "size": list(cut.size), "panels": found, "names": known, "settled": cut.settled}


def run_cast_panels(inputs: dict[str, Any], settings_: dict[str, Any]) -> dict[str, Any]:
    """Every name the reader gave, as the characters they mean: one text-only question for the whole comic."""
    scripts = [read_page.Script.of(p["script"]) for page in inputs["pages"] for p in page]
    names = list(dict.fromkeys(n for s in scripts for n in [*s.who, *(b.speaker for b in s.balloons)] if n))
    made = key("cast_panels", 1, names=names, scripts=[s.data() for s in scripts], **_manga())
    if (previous := _start("cast_panels", made)) is not None:
        return previous
    if not names:
        return {"characters": [], "ids": {}}
    lines = "\n".join(
        f"{n}. {', '.join(s.who) or '-'} | {s.action} | "
        + (", ".join(dict.fromkeys(b.speaker for b in s.balloons if b.speaker)) or "-")
        for n, s in enumerate(scripts, 1)
    )

    def gate(data: Any) -> list[caption.Problem]:
        out = []
        ids = [str(c["id"]) for c in data["characters"]]
        out += [direct.problem(f'"{i}" is not a valid id: one lowercase word, or two joined by "_"')
                for i in ids if not direct.ID.match(i)]  # fmt: skip
        if len(set(ids)) != len(ids):
            out.append(direct.problem("every character has a different id"))
        out += [direct.problem(f'{c["id"]}: "kind" is "person" or "animal"')
                for c in data["characters"] if c["kind"] not in ("person", "animal")]  # fmt: skip
        given = [read_page._name(n) for c in data["characters"] for n in c["names"]]
        out += [direct.problem(f'"{n}" is in no character\'s "names"') for n in names if n not in given]
        out += [direct.problem(f'"{n}" is in two characters\' "names"') for n in set(given) if given.count(n) > 1]
        out += [direct.problem(f'"{n}" is not one of the NAMES') for n in set(given) - set(names)]
        return out

    data = direct.ask_json(
        words.MANGA_CAST_SYSTEM, words.MANGA_CAST_USER.format(names=", ".join(names), panels=lines), [], gate,
        "name the cast", knobs.MANGA_TOKENS["cast"],
    )  # fmt: skip
    characters = [
        {"id": str(c["id"]), "kind": str(c["kind"]), "names": [read_page._name(n) for n in c["names"]]}
        for c in data["characters"]
    ]
    return {"characters": characters, "ids": {n: c["id"] for c in characters for n in c["names"]}}


def _as_ids(script: read_page.Script, ids: dict[str, str]) -> read_page.Script:
    def one(name: str) -> str:
        return ids.get(name, name) if name else ""

    return dataclasses.replace(
        script,
        balloons=tuple(dataclasses.replace(b, speaker=one(b.speaker)) for b in script.balloons),
        who=tuple(dict.fromkeys(one(w) for w in script.who)),
    )


# --- the panels as shots ---------------------------------------------------------------------------------
def run_lay_panels(inputs: dict[str, Any], settings_: dict[str, Any]) -> dict[str, Any]:
    """One shot per panel in every chapter; a panel already laid keeps its row, its keyframe and what was
    edited by hand. Then the rest of the run, which needed the panels counted."""
    production_id = int(inputs["production"])
    characters = list(inputs["cast"])
    ids = {n: str(c["id"]) for c in characters for n in c["names"]}
    pages = [list(page) for page in inputs["pages"]]
    with store.connect() as connection:
        production = store.production(connection, production_id)
        assert production is not None
        chapters = store.chapters_of(connection, production_id)
    groups = manga.chapter_pages(production.manifest)
    video, _ = settings.effective("render_chunk", production_id)
    assert isinstance(video, VideoSettings)
    laid: list[dict[str, Any]] = []  # every panel in film order, with its shot
    stories = []
    with store.connect() as connection, store.transaction(connection):
        at = 0
        for chapter, group in zip(chapters, groups, strict=False):
            panels_ = [(index, p) for index in group for p in pages[at + group.index(index)]]
            scripts = [_as_ids(read_page.Script.of(p["script"]), ids) for _, p in panels_]
            pages_of = [str(production.manifest["pages"][i]) for i, _ in panels_]
            laid += _lay(connection, chapter, panels_, scripts, pages_of, video)
            story = manga.story_of(scripts)
            stories.append(story)
            connection.execute("UPDATE chapters SET story = ? WHERE id = ?", (story, chapter.id))
            at += len(group)
        store.update_production(connection, production_id, story="\n\n".join(stories))
        production = store.production(connection, production_id)
        assert production is not None
    _queue_rest(inputs, production, laid, characters)
    return {"shots": len(laid), "characters": characters}


def _lay(
    connection: Any, chapter: store.Chapter, panels_: list[tuple[int, dict[str, Any]]],
    scripts: list[read_page.Script], pages: list[str], video: VideoSettings,
) -> list[dict[str, Any]]:  # fmt: skip
    existing = store.shots_of(connection, chapter.id)
    out = []
    for position, ((index, panel), script, page) in enumerate(zip(panels_, scripts, pages, strict=True)):
        seconds, fits = manga.seconds_for(script, video.duration)
        before = scripts[position - 1].into_next if position else "cut"
        given = {
            "page": page, "page_index": index, "box": panel["box"], "read_box": panel["read_box"],
            "picture": panel["picture"], "read": panel["read"], "confirmed": bool(panel.get("confirmed")),
            "script": script.data(), "landed_by_reader": before == "land", **({} if fits else {"too_long": True}),
        }  # fmt: skip
        shot = existing[position] if position < len(existing) else None
        held = shot.plan.get("panel") if shot is not None else None
        if shot is not None and held and held.get("picture") == panel["picture"]:
            kept = {k: held[k] for k in ("clean", "clean_review", "edited") if k in held}
            if held.get("edited"):
                kept["script"] = held["script"]
            mine = {**given, **kept}
            script_now = read_page.Script.of(mine["script"])
            cast = list(dict.fromkeys(w for w in script_now.who))
            store.update_shot(connection, shot.id, plan={**shot.plan, "panel": mine,
                              **({} if shot.plan.get("beats") else {"cast": cast})})  # fmt: skip
            if held.get("landed_by_reader") != given["landed_by_reader"] and position:
                store.land_keyframe(connection, shot.id, given["landed_by_reader"])
            source = store.keyframe_of(connection, shot.id).source
            if position and not script_now.drawn and source == store.COMPOSED:
                store.chain_keyframe(connection, shot.id)
            elif script_now.drawn and source == store.CHAINED:
                store.compose_keyframe(connection, shot.id)
            if not shot.plan.get("beats"):
                store.update_shot(connection, shot.id, seconds=seconds, action=script_now.action)
        else:
            if shot is None:
                shot = store.add_shot(connection, chapter.id, seconds, action=script.action)
                store.set_clip_seed(connection, shot.id, video.seed + shot.position)
            store.update_shot(
                connection, shot.id, plan={"panel": given, "cast": list(script.who)}, seconds=seconds,
                action=script.action, dialogue=" / ".join(f"{b.speaker}: {b.text}" for b in script.balloons
                                                          if b.kind in read_page.SPOKEN),
            )  # fmt: skip
            if position and not script.drawn:
                store.chain_keyframe(connection, shot.id)  # lettering alone: it goes on from the shot before
            else:
                store.compose_keyframe(connection, shot.id)
            store.land_keyframe(connection, shot.id, bool(position) and before == "land")
        out.append({"shot": shot.id, "chapter": chapter.id, "picture": panel["picture"], "script": script,
                    "box": panel["box"]})  # fmt: skip
    for extra in existing[len(panels_) :]:
        store.delete_shot(connection, extra.id)
    return out


def _queue_rest(
    inputs: dict[str, Any], production: store.Production, laid: list[dict[str, Any]], characters: list[dict[str, Any]]
) -> None:
    """Cleaning, the bible, the colour sheets and the fill, once per run: a lay reused by key still adds them."""
    ctx = context.current()
    production_id = production.id
    with store.connect() as connection, store.transaction(connection):
        if ctx.run_id is None or any(j.tool == "clean_panel" for j in store.jobs_of(connection, ctx.run_id)):
            return
        run_id = ctx.run_id

        def add(tool: str, given: dict[str, Any], label: str, chosen: dict[str, Any] | None = None) -> int:
            if chosen is None:
                chosen = settings.for_job(tool, production_id)
            return store.add_job(
                connection, run_id, tool, {"production": production_id, **given}, label=label, settings=chosen
            ).id

        editor = settings.for_job("edit_image", production_id)
        # A panel of lettering alone is never cleaned: asked to fill a balloon on black, the editor drew a cast
        # of its own (four faces of another manga's hero, One-Punch Man p6).
        cleaned = [
            add("clean_panel", {"shot": p["shot"], "picture": p["picture"]}, f"clean panel {n + 1}", editor)
            if p["script"].drawn
            else None
            for n, p in enumerate(laid)
        ]
        for n, (p, job) in enumerate(zip(laid, cleaned, strict=True)):
            if job is not None:
                add("judge_clean", {"shot": p["shot"], "picture": p["picture"], "clean": _out(job, "frame")},
                    f"check panel {n + 1}")  # fmt: skip
        chosen, labels = _survey_panels([p for p in laid if p["script"].drawn], characters)
        drawn = [i for i, p in enumerate(laid) if p["script"].drawn]
        chosen = [drawn[i] for i in chosen]
        photos = add("manga_photos", {"photos": [_out(cleaned[i], "frame") for i in chosen], "labels": labels},
                     "the cleaned panels the bible reads")  # fmt: skip
        surveyed = add(
            "survey",
            {"idea": production.story, "photos": _out(photos, "photos"), "labels": labels, "chapter_position": 0,
             "last": True},
            "survey the panels",
        )  # fmt: skip
        cast = {"members": _out(surveyed, "members"), "shows": _out(surveyed, "shows")}
        grounds = [
            add("ground", {"photo": _out(cleaned[i], "frame"), "index": n, **cast}, f"find the cast in panel {i + 1}")
            for n, i in enumerate(chosen)
        ]
        found = [_out(job, "found") for job in grounds]
        video, _ = settings.effective("render_chunk", production_id)
        assert isinstance(video, VideoSettings)
        aspect = production.manifest.get("aspect")
        add("cut_crops", {"photos": _out(photos, "photos"), "found": found, "aspect": aspect, **cast},
            "cut the references", {"video": video.data()})  # fmt: skip
        carded = [
            add("cards", {"photo": _out(cleaned[i], "frame"), "index": n, **cast}, f"read the cast in panel {i + 1}")
            for n, i in enumerate(chosen)
        ]
        tidied = add(
            "tidy",
            {"idea": production.story, "cards": [_out(job, "card") for job in carded], "found": found, **cast},
            "reconcile the cards with the story",
        )
        for character in characters:
            add("colour_sheet", {"member": character["id"], "cards": _out(tidied, "cards")},
                f"draw {character['id']} in colour", editor)  # fmt: skip
        if inputs.get("shots"):
            assert ctx.job_id is not None
            filled = {"chapter": int(inputs["chapter"]), "cards": _out(tidied, "cards"),
                      "characters": _out(ctx.job_id, "characters"), **_taken(inputs)}  # fmt: skip
            add("fill_shots", filled, "fill the shots")


def _survey_panels(
    laid: list[dict[str, Any]], characters: list[dict[str, Any]]
) -> tuple[list[int], list[dict[str, str]]]:
    """The first panel, for the place the comic opens in, then each character's clearest panel: the largest
    with the fewest others in it."""
    chosen: dict[int, str] = {0: ""}
    for character in characters:
        shown = [i for i, p in enumerate(laid) if character["id"] in p["script"].who]
        if not shown:
            continue

        def clear(i: int) -> tuple[int, float]:
            box = laid[i]["box"]
            return -len(laid[i]["script"].who), (box[2] - box[0]) * (box[3] - box[1])

        best = max((i for i in shown if i not in chosen or not chosen[i]), key=clear, default=None)
        if best is not None:
            chosen[best] = character["id"]
    order = sorted(chosen, key=lambda i: (i != 0, i))
    return order, [{"kind": "character" if chosen[i] else "", "entity": chosen[i]} for i in order]


def run_manga_photos(inputs: dict[str, Any], settings_: dict[str, Any]) -> dict[str, Any]:
    photos = [str(p) for p in inputs["photos"]]
    labels = {p: dict(label) for p, label in zip(photos, inputs.get("labels") or [], strict=False)}
    with store.connect() as connection, store.transaction(connection):
        production = store.production(connection, int(inputs["production"]))
        assert production is not None
        store.update_production(
            connection, production.id, manifest={**production.manifest, "photos": photos, "labels": labels}
        )
    return {"photos": photos}


def run_clean_panel(inputs: dict[str, Any], settings_: dict[str, Any]) -> dict[str, Any]:
    chosen = EditSettings(**settings_)
    picture = load_image(_path(inputs["picture"])).convert("RGB")
    frame = Frame.from_pil(picture)
    said = words.COMPOSE_MANGA_CLEAN
    made = key("clean_panel", 1, picture=frame.pixels, said=said, settings=chosen)
    target = cache.path(made, ".png")
    if not target.is_file():
        width, height = megapixel_size(frame.width, frame.height, chosen.megapixels)
        work = (max(16, width // 16 * 16), max(16, height // 16 * 16))
        edit_image.edit(frame, said, chosen, size=work).pil().save(target)
    path = store.relative(target)
    _panel(int(inputs["shot"]), clean=path)
    return {"frame": path, "key": str(made)}


def run_judge_clean(inputs: dict[str, Any], settings_: dict[str, Any]) -> dict[str, Any]:
    before = load_image(_path(inputs["picture"])).convert("RGB")
    after = load_image(_path(inputs["clean"])).convert("RGB")
    made = key("judge_clean", 1, before=Frame.from_pil(before).pixels, after=Frame.from_pil(after).pixels, **_manga())
    if (previous := _start("judge_clean", made)) is None:
        shown = [read_page._shrunk(before, knobs.DIRECTOR_SURVEY_PIXELS),
                 read_page._shrunk(after, knobs.DIRECTOR_SURVEY_PIXELS)]  # fmt: skip
        data = direct.ask_json(
            words.MANGA_CLEAN_SYSTEM, words.MANGA_CLEAN_USER, shown, lambda data: [], "check the cleaned panel",
            knobs.MANGA_TOKENS["confirm"],
        )  # fmt: skip
        outputs = {"lettering": bool(data["lettering"]), "changed": bool(data["changed"]), "why": str(data["why"])}
    else:
        outputs = dict(previous)
    flagged = outputs["lettering"] or outputs["changed"]
    _panel(int(inputs["shot"]), clean_review=outputs["why"] if flagged else "")
    return outputs


def _panel(shot_id: int, **fields: Any) -> None:
    with store.connect() as connection, store.transaction(connection):
        shot = store.shot(connection, shot_id)
        if shot is not None and shot.plan.get("panel"):
            store.update_shot(connection, shot_id, plan={**shot.plan, "panel": {**shot.plan["panel"], **fields}})


# --- colour --------------------------------------------------------------------------------------------
def run_colour_sheet(inputs: dict[str, Any], settings_: dict[str, Any]) -> dict[str, Any]:
    """A character drawn once in full colour over its clearest crop: what every keyframe takes its colours from.
    A sheet already drawn under the same key keeps its row, and with it a person's approval."""
    production_id = int(inputs["production"])
    with store.connect() as connection:
        production = store.production(connection, production_id)
        member = next((m for m in store.members_of(connection, production_id) if m.slug == inputs["member"]), None)
        own = [] if member is None else store.refs_of(connection, member.id)
    if production is None or member is None or member.kind not in ("person", "animal"):
        ui.note(f"{inputs['member']}: not a person or animal in the bible, so no colour sheet")
        return {"frame": None}
    crops = [r for r in own if r.artifact and r.role != manga.SHEET and not r.generated]
    crops.sort(key=lambda r: (r.role not in ("body", "view"), r.role == "face", -r.quality))
    if not crops:
        ui.note(f"{member.slug}: no crop of it to draw a colour sheet over")
        return {"frame": None}
    crop = crops[0]
    chosen = EditSettings(**settings_)
    source = Frame.from_pil(load_image(_path(crop.artifact)).convert("RGB"))
    traits = "; ".join(str(t["phrase"]) for t in member.identity.get("traits", [])) or member.find
    colours = " ".join(member.appearance.split())
    said = words.COMPOSE_MANGA_SHEET.format(
        noun=member.noun, traits=traits.rstrip("."),
        colours=words.COMPOSE_MANGA_SHEET_COLOURS.format(colours=colours.rstrip("."))
        if colours
        else words.COMPOSE_MANGA_SHEET_KEEP
        if production.manifest.get("coloured")
        else "",
    )  # fmt: skip
    size = drawn.size_for(member.kind, production.manifest)
    made = key("colour_sheet", 1, source=source.pixels, said=said, settings=chosen, size=size)
    target = cache.path(made, ".png")
    if not target.is_file():
        width, height = megapixel_size(size[0], size[1], drawn.MEGAPIXELS)
        work = (max(16, width // 16 * 16), max(16, height // 16 * 16))
        edit_image.edit(source, said, chosen, size=work).resized(*size).pil().save(target)
    path = store.relative(target)
    with store.connect() as connection, store.transaction(connection):
        held = [r for r in store.refs_of(connection, member.id) if r.role == manga.SHEET]
        if not any(r.wanted_key == str(made) for r in held):
            for old in held:
                connection.execute("DELETE FROM refs WHERE id = ?", (old.id,))
            store.add_ref(
                connection, member.id, manga.SHEET, crop.photo, crop.box, size, (size[0] // 32) * (size[1] // 32),
                [member.slug], 0.0, artifact=path, key=str(made),
            )  # fmt: skip
    sheet_colours(path, member.noun)
    return {"frame": path, "key": str(made)}


COLOUR_WORDS = 40


def sheet_colours(path: str, noun: str) -> str:
    """A sheet's colours in words: what a keyframe is told, since handed the sheet itself the editor draws it in."""
    image = read_page._shrunk(load_image(_path(path)), knobs.DIRECTOR_PHOTO_PIXELS)

    def gate(data: Any) -> list[caption.Problem]:
        said = str(data["colours"]).split()
        if not said:
            return [direct.problem('"colours" names the colour of every part that shows')]
        if len(said) > COLOUR_WORDS:
            return [direct.problem(f'"colours" has {len(said)} words: at most {COLOUR_WORDS}')]
        return []

    data = direct.ask_json(
        words.COMPOSE_MANGA_COLOURS_SYSTEM.format(words=COLOUR_WORDS),
        words.COMPOSE_MANGA_COLOURS_USER.format(noun=noun), [image], gate, "sheet colours", 512,
    )  # fmt: skip
    return " ".join(str(data["colours"]).split()).rstrip(".")


def panel_shows(panel: str, noun: str, colours: str) -> tuple[str, str]:
    """What of a character a panel shows, and the colours of only that, or two "" when it shows none."""
    image = read_page._shrunk(load_image(_path(panel)), knobs.DIRECTOR_PHOTO_PIXELS)

    def gate(data: Any) -> list[caption.Problem]:
        shown, kept = str(data["shown"]).split(), str(data["colours"]).split()
        if bool(shown) != bool(kept):
            return [direct.problem('"shown" and "colours" are both "" or both said')]
        if len(kept) > COLOUR_WORDS:
            return [direct.problem(f'"colours" has {len(kept)} words: at most {COLOUR_WORDS}')]
        return []

    data = direct.ask_json(
        words.COMPOSE_MANGA_SHOWN_SYSTEM.format(words=COLOUR_WORDS),
        words.COMPOSE_MANGA_SHOWN_USER.format(noun=noun, colours=colours), [image], gate, "panel shows", 512,
    )  # fmt: skip
    return tuple(" ".join(str(data[k]).split()).rstrip(".") for k in ("shown", "colours"))  # type: ignore[return-value]


# --- filling -------------------------------------------------------------------------------------------
KEPT = ("panel", "captions", "motion_comic")  # a manga shot's own: what no planner or writer rewrites


def run_fill_shots(inputs: dict[str, Any], settings_: dict[str, Any]) -> dict[str, Any]:
    production_id, chapter_id = int(inputs["production"]), int(inputs["chapter"])
    with store.connect() as connection:
        production = store.production(connection, production_id)
        chapter = store.chapter(connection, chapter_id)
        assert production is not None and chapter is not None
        shots = store.shots_of(connection, chapter_id)
        keyframes = [store.keyframe_of(connection, shot.id) for shot in shots]
        rows = store.members_of(connection, production_id)
    cards = {m.slug: compiler.card_of(m) for m in rows}
    aliases = {str(c["id"]): ", ".join(c["names"]) for c in inputs.get("characters") or [] if c["id"] in cards}
    given = []
    for index, shot in enumerate(shots):
        script = read_page.Script.of(shot.plan.get("panel", {}).get("script", {}))
        drawn = (store.COMPOSED, store.PROVIDED)
        lands = index + 1 < len(shots) and keyframes[index + 1].landed and keyframes[index + 1].source in drawn
        given.append(fill_shots.Given(
            shot.seconds, tuple(m for m in script.who if m in cards and cards[m].kind in compiler.LIVING),
            script.action, tuple(b for b in script.balloons if not b.speaker or b.speaker in cards), lands,
            bool(index) and keyframes[index].landed and keyframes[index].source in drawn,
            bool(index) and keyframes[index].source == store.CHAINED,
        ))  # fmt: skip
    video, _ = settings.effective("render_chunk", production_id)
    assert isinstance(video, VideoSettings)
    made = key(
        "fill_shots", 1, production=production_id, chapter=chapter_id, given=[dataclasses.asdict(g) for g in given],
        cards={k: dataclasses.asdict(c) for k, c in cards.items()}, aliases=aliases, **_taken(inputs),
        **{k: v for k, v in _manga().items() if k != "model"}, **_fingerprint(),
    )  # fmt: skip
    if (previous := _start("fill_shots", made)) is None:
        base = compiler.Take(
            title=production.title, members=tuple(m.slug for m in rows), cards=cards, refs={}, opening="",
            ambience="", seconds=video.duration, tail=True, budget=knobs.DIRECTOR_REFERENCE_ROWS,
        )  # fmt: skip
        with _take(inputs):
            take = fill_shots.fill(base, given, aliases)
        outputs: dict[str, Any] = {
            "chunks": [compiler.plan_of(chunk) for chunk in take.chunks], "opening": take.opening,
            "ambience": take.ambience, "voices": dict(take.voices),
        }  # fmt: skip
    else:
        outputs = dict(previous)
    _write_fill(production_id, chapter_id, outputs, given)
    _voices(production_id, dict(outputs.get("voices") or {}))
    _write_and_compile(inputs, len(outputs["chunks"]))
    return {**outputs, "count": len(outputs["chunks"])}


def _write_fill(production_id: int, chapter_id: int, outputs: dict[str, Any], given: list[fill_shots.Given]) -> None:
    with store.connect() as connection, store.transaction(connection):
        production = store.production(connection, production_id)
        assert production is not None
        plans = {**production.manifest.get("chapter_plans", {}),
                 str(chapter_id): {"ambience": outputs["ambience"], "story": []}}  # fmt: skip
        chapters = store.chapters_of(connection, production_id)
        first = {"opening_arrangement": outputs["opening"]} if chapters and chapters[0].id == chapter_id else {}
        store.update_production(connection, production_id, manifest={
            **production.manifest, **first, "chapter_plans": plans, "budget": knobs.DIRECTOR_REFERENCE_ROWS,
        })  # fmt: skip
        for shot, plan, panel in zip(store.shots_of(connection, chapter_id), outputs["chunks"], given, strict=False):
            if shot.pinned:
                continue
            captions = _captions(panel)
            plan = {**plan, **{k: shot.plan[k] for k in KEPT if k in shot.plan}, "captions": captions}
            dialogue = " / ".join(f"{line['who']}: {line['line']}" for line in plan.get("dialogue", []))
            store.update_shot(
                connection, shot.id, plan=plan, action=str(plan["purpose"]), dialogue=dialogue,
                camera=f"{plan['camera_move']} → {plan['camera_toward']}", scene=str(plan.get("scene") or ""),
            )  # fmt: skip
            production_module._sync_castings(connection, shot, [str(m) for m in plan.get("cast", [])])


def _captions(panel: fill_shots.Given) -> list[dict[str, Any]]:
    """The thoughts and narration of a panel, subtitled in turn from its start: no one says them aloud."""
    out, at = [], 0.0
    for balloon in panel.balloons:
        if balloon.kind in read_page.SPOKEN:
            continue
        out.append({"start": round(at, 2), "line": balloon.text, "kind": balloon.kind})
        at = min(at + len(balloon.text.split()) / knobs.MANGA_READ_WORDS_PER_SECOND, max(panel.seconds - 1.0, 0.0))
    return out


MANGA_RUNNERS = {
    "read_page": run_read_page,
    "cast_panels": run_cast_panels,
    "lay_panels": run_lay_panels,
    "clean_panel": run_clean_panel,
    "judge_clean": run_judge_clean,
    "manga_photos": run_manga_photos,
    "colour_sheet": run_colour_sheet,
    "fill_shots": run_fill_shots,
}
