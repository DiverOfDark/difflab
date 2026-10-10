"""The gate every director prompt passes before it may render.

The format's own rules live in `caption` (the Prompt IDE port and the label check), because a
hand-written take needs them too; what is here is what the *director* needs on top. A problem marked
`repairable` goes back to the model that made it with the message as the instruction; anything else
is a bug in the compiler or in the plan.
"""

from __future__ import annotations

import itertools
import re

from difflab.models.minimax_h3.caption import LABEL, Problem, body, label_rules, sections
from difflab.screenplay import compiler as compose
from difflab.screenplay import config, words

VISUAL_MARKERS = ("fully_preserved", "partially_preserved", "attribute_transfer", "weak_reference")
AUDIO_MARKERS = ("fully_copy", "partially_copy", "reference", "weak_reference")
HYPE = re.compile(
    r"\b(photorealistic|hyper-?real\w*|4k|8k|hdr|masterpiece|best quality|ultra[- ]realistic|cgi|octane|"
    r"unreal engine|award[- ]winning|stunning|breathtaking)\b",
    re.I,
)
META = re.compile(r"\b(chunk|segment|previous clip|next clip|stitch\w*|this prompt|part \d)\b", re.I)
NEGATION = re.compile(r"\b(no|not|never|without|nothing|nobody|don't|doesn't)\b", re.I)
PERSISTENCE = re.compile(r"\b(must not reappear|do not restore|do not regenerate|must not return|no cuts?)\b", re.I)
WRITER_TIMING = re.compile(
    r"\bAt \d|\[Shot|\b\d+(\.\d+)?\s*(s|sec|seconds)\b|\b(early in the clip|in the middle seconds|"
    r"later in the clip|toward the end of the clip)\b"
    # The compiler's own "For the first two seconds" and "in the final second", not "the first meow".
    r"|\b(for the first|in the final)(\s+\w+)?\s+(seconds?|moments?)\b",
    re.I,
)
# A side from a person's own point of view: an 8B model mirrors it, and a take planned on a mirrored
# room passes the hose across a table on the other side. "On his left wrist" is a place on his body,
# not a side of the room, so BODY is excluded.
BODY = (
    r"(?:hands?|arm|forearm|wrists?|paws?|legs?|knee|foot|feet|shoulder|elbow|ear|eye|cheek|hip|thigh|palm|"
    r"fingers?|sleeve)"
)
SIDE = re.compile(
    rf"\b(?:to|on|at) (?:his|her|its|their) (?:left|right)(?: side)?\b(?! {BODY}\b)"
    r"|\b(?:left|right) of (?:him|her|it|them)\b",
    re.I,
)
PHASES = ("airlock", "carry", "build", "primary", "handoff", "reaction", "settle")
QUIET = ("airlock", "settle")  # beats in which nothing new happens


def chunk_rules(text: str, labels: list[str], directive: str) -> list[Problem]:
    """What difflab needs beyond the IDE for one compiled chunk."""
    out = label_rules(text, labels)
    definitions, main = body(text, "subject_definitions"), body(text, "detailed_description")
    numbers = sorted({int(n) for n in re.findall(r"<Subject (\d+)>", definitions)})
    if numbers != list(range(1, len(numbers) + 1)):
        out.append(Problem("error", "subjects", f"<Subject N> must run 1..K without gaps, got {numbers}"))
    if not body(text, "summary").startswith(directive):
        out.append(Problem("error", "directive", f"summary must start with {directive}"))
    for line in body(text, "retention_analysis").split("\n"):
        visual = re.match(r"^<(?:Subject|Picture|Video)\s+\d+>[^:]*:\s*(\w+) - ", line)
        audio = re.match(r"^<Audio\s+\d+>:\s*(\w+) - ", line)
        good = (visual and visual.group(1) in VISUAL_MARKERS) or (audio and audio.group(1) in AUDIO_MARKERS)
        if not good or re.search(r"\(S\d", line):
            out.append(Problem("error", "retention", f"malformed retention line: {line[:60]}"))
    for name, start, _, _ in sections(text)[1:]:
        if not text[:start].endswith("\n\n"):
            out.append(Problem("error", "layout", f"{name}: needs a blank line before it"))
    for match in re.finditer(r"<d>", text):
        if not re.search(r"\(S\d+\)[^<]*$", text[max(0, match.start() - 200) : match.start()]):
            out.append(Problem("error", "speaker", "a <d> line needs its speaker's (Sx) ID before it"))
    if re.search(r"\[Shot (?:[2-9]|\d\d+)\]", text):
        out.append(Problem("error", "one_take", "a [Shot N>1] is a cut; one take is a lone [Shot 1]"))
    for match in re.finditer(r"\b\d\d:\d\d\.\d\d\d\b", text):
        out.append(Problem("error", "timestamp", f"{match.group(0)}: one take times its beats with words, not cuts"))
    count = len(main.split())
    (low, high), (floor, ceiling) = config.DIRECTOR_WORDS, config.DIRECTOR_WORDS_HARD
    if not floor <= count <= ceiling:
        out.append(Problem("error", "length", f"detailed_description has {count} words; write {low}-{high}", True))
    elif not low <= count <= high:
        out.append(Problem("warning", "length", f"detailed_description has {count} words; aim for {low}-{high}", True))
    sound = body(text, "overall_soundscape")
    sentences = len([s for s in re.split(r"(?<=[.!?])\s+", sound) if s.strip()])
    if not 1 <= sentences <= 4 or "<d>" in sound:
        out.append(
            Problem("error", "soundscape", f"overall_soundscape: 1-4 sentences, no speech (has {sentences})", True)
        )
    music = body(text, "non_diegetic_music")
    if music != "N/A" and not 1 <= len(re.split(r"(?<=[.!?])\s+", music)) <= 3:
        out.append(Problem("error", "music", "non_diegetic_music is N/A or 1-3 sentences"))
    for pattern, code, severity in (
        (HYPE, "hype", "warning"),
        (META, "meta", "error"),
        (PERSISTENCE, "negation", "error"),
    ):
        out += [
            Problem(severity, code, f"'{m.group(0)}' in detailed_description", True) for m in pattern.finditer(main)
        ]
    if len(NEGATION.findall(main)) > config.DIRECTOR_MAX_NEGATIONS:
        out.append(
            Problem(
                "warning",
                "negation",
                f"more than {config.DIRECTOR_MAX_NEGATIONS} negations; say what happens instead",
                True,
            )
        )
    # With the words either side of it: the prompt is two thousand words across six sections, and
    # "a placeholder survived" alone leaves the reader hunting for the brace that escaped compilation.
    for found in re.finditer(r"\{[a-z_]+\}", text):
        around = text[max(0, found.start() - 40) : found.end() + 40].replace("\n", " ")
        out.append(Problem("error", "placeholder", f"{found.group(0)} survived compilation: ...{around}..."))
    return out


def writer_rules(prose: str, cast: set[str], actors: set[str] | None = None) -> list[Problem]:
    """The writer's own mistakes, checked on its JSON before compiling. All repairable.

    Each of `actors` must be named as its {id} in the beat's prose: that binds the action to its
    <Subject N> and catches prose written for some other beat."""
    out = [
        Problem("error", "placeholder", f"{{{name}}} is not in this clip's cast", True)
        for name in re.findall(r"\{([a-z_]+)\}", prose)
        if name not in cast and name != "say"
    ]
    for actor in sorted((actors or set()) - set(re.findall(r"\{([a-z_]+)\}", prose))):
        out.append(
            Problem("error", "actor", f"this beat's action belongs to {{{actor}}}: name it as {{{actor}}}", True)
        )
    if LABEL.search(prose):
        out.append(Problem("error", "labels", "write cast members as {id}; the labels are added for you", True))
    if message := possessive(prose):
        out.append(Problem("error", "possessive", message, True))
    if WRITER_TIMING.search(prose):
        out.append(
            Problem("error", "timing", "no timing words, seconds or shot markers; timing is added for you", True)
        )
    if side := SIDE.search(prose):
        out.append(Problem("error", "sides", f"say sides as sides of the frame, not '{side.group(0)}'", True))
    return out


def possessive(text: str) -> str | None:
    """A possessive after braces, quoted with its rewrite: "write 'the fur of {cat}', not '{cat}'s fur'"."""
    match = re.search(r"\{([a-z_]+)\}'s(?: (\w+))?", text)
    if match is None:
        return None
    thing = match.group(2) or "hand"
    return f"no possessive after braces: write 'the {thing} of {{{match.group(1)}}}', not '{match.group(0)}'"


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


NUMBERED = re.compile(r"^\s*\d+[.)]\s+(.*)$", re.M)


def actions_of(idea: str) -> str:
    """The part of an IDEA that tells events. A story the idea editor wrote opens with a paragraph on the
    place and the first frame and ends on a Sound line; only its numbered beats happen. On production 10
    the place's "sits", "falls" and "holds" were asked of clip 1 as events, and its main beat became the
    desk sitting by the window."""
    numbered = NUMBERED.findall(idea)
    if numbered:
        return "\n".join(numbered)
    return "\n".join(line for line in idea.splitlines() if not re.match(r"\s*sound\s*:", line, re.I))


def story_bounds(
    idea: str, story: list[tuple[str, int]], take: compose.Take
) -> tuple[list[Problem], list[tuple[str, int]]]:
    """The events quoted from the IDEA that point at a clip that exists, and a problem for every other."""
    out, kept = [], []
    vocabulary = set(tokens(idea))
    for event, clip in story:
        if not set(tokens(event)) <= vocabulary:
            out.append(Problem("error", "story", f"story: '{event}' is not in the IDEA's own words; quote them", True))
        elif not 1 <= clip <= len(take.chunks):
            out.append(Problem("error", "story", f"story: '{event}' points to clip {clip}, which does not exist", True))
        else:
            kept.append((event, clip))
    return out, kept


def plan_rules(take: compose.Take) -> list[Problem]:
    """Timing, continuity and staging of the whole plan, before any prose is written. All repairable."""
    out: list[Problem] = []
    living = {m for m in take.members if take.cards[m].kind in compose.LIVING}

    def fail(code: str, message: str) -> None:
        out.append(Problem("error", code, message, True))

    for index, chunk in enumerate(take.chunks):
        clip = f"clip {index + 1}"
        beats = list(chunk.beats)
        seconds = compose.length(take, index)
        unknown = {m for beat in beats for m in beat.who} | set(chunk.cast) | {chunk.camera_toward}
        if unknown - set(take.members):
            fail("cast", f"{clip}: {sorted(unknown - set(take.members))} are not cast ids {list(take.members)}")
            continue
        for beat in beats:
            if beat.phase not in PHASES:
                fail("phase", f"{clip}: '{beat.phase}' is not a phase; use one of {', '.join(PHASES)}")
            if set(beat.who) - set(chunk.cast):
                fail("cast", f"{clip}: every id in a beat's 'who' is in the clip's 'cast'")
        texts = (*(b.action for b in beats), chunk.closing)
        named_in_text = {m for text in texts for m in compose.PLACEHOLDER.findall(text)}
        for member in sorted(named_in_text - set(chunk.cast)):
            fail("cast", f"{clip}: {{{member}}} is named in its beats or closing, so it is in its 'cast'")
        places = [m for m in chunk.cast if take.cards[m].kind == "environment"]
        if len(places) != 1:
            fail("place", f"{clip}: its 'cast' names exactly one place, the one it happens in, not {places}")
            continue
        if chunk.cut not in compose.CUTS or (index == 0 and chunk.cut):
            fail("cut", f'{clip}: \'cut\' is "" (it continues), "back" or "new", and clip 1 does not cut')
        elif chunk.cut == "new" and not compose.PLACEHOLDER.search(chunk.opening):
            fail("cut", f"{clip}: a \"new\" cut writes its 'opening', the first frame, naming every member in it")
        for member, change in chunk.looks.items():
            if member not in chunk.cast:
                fail("looks", f"{clip}: 'looks' changes {{{member}}}, who is not in its cast")
            elif len(change.split()) > config.DIRECTOR_LOOK_WORDS:
                fail("looks", f"{clip}: the look of {{{member}}} is {len(change.split())} words; at most "
                     f"{config.DIRECTOR_LOOK_WORDS}")  # fmt: skip
            fewest, most = config.DIRECTOR_LOOK_NAME_WORDS
            called = len(chunk.look_names.get(member, "").split())
            if member in chunk.cast and not fewest <= called <= most:
                fail("looks", f'{clip}: the look of {{{member}}} has a "name" of {fewest} to {most} words, '
                     f'not {called}: "blue robe", "soaked shirt"')  # fmt: skip
        if len(beats) < 3:
            fail("beats", f"{clip}: write 3 to 5 beats")
            continue
        for beat in beats:
            if beat.end - beat.start < config.DIRECTOR_BEAT_MIN - 0.01:
                span = f"{beat.phase} beat {beat.start:g}-{beat.end:g} s"
                fail(
                    "beats",
                    f"{clip}: the {span} is under {config.DIRECTOR_BEAT_MIN:g} second; lengthen it or merge it into "
                    "the next one",
                )
        present = set(re.findall(r"\{([a-z_]+)\}", compose.opening(take, index)))
        first = set(beats[0].who) | set(re.findall(r"\{([a-z_]+)\}", beats[0].action))
        for member in sorted(first & (living - present)):
            message = "is outside the frame when the clip starts (its opening does not name it), so it takes no part"
            fail("entrance", f"{clip}: {{{member}}} {message} in the first beat and enters in a later one")
        if abs(beats[0].start) > 0.01 or abs(beats[-1].end - seconds) > 0.3:
            fail("coverage", f"{clip}: beats must run from 0 to {seconds:.2f}")
        for a, b in itertools.pairwise(beats):
            if abs(a.end - b.start) > 0.01:
                fail("coverage", f"{clip}: beats must be contiguous ({a.end} then {b.start})")
        airlock, settle = config.DIRECTOR_AIRLOCK[1], config.DIRECTOR_SETTLE[1]
        following = take.chunks[index + 1] if index + 1 < len(take.chunks) else None
        continued = following is not None and not following.cut
        if chunk.cut and (beats[0].phase != "airlock" or beats[0].end - beats[0].start < airlock):
            fail("airlock", f"{clip}: a cut starts with an airlock beat of at least {airlock:g} seconds")
        elif index > 0 and not chunk.cut and beats[0].phase != "carry":
            fail("carry", f"{clip}: it continues the clip before, so it starts with a carry beat, that clip's last "
                 "motion going on")  # fmt: skip
        if continued and beats[-1].phase != "carry":
            fail("carry", f"{clip}: the next clip continues it, so it ends with a carry beat, the next movement "
                 "already under way")  # fmt: skip
        elif not continued and (beats[-1].phase != "settle" or beats[-1].end - beats[-1].start < settle):
            fail("settle", f"{clip}: it ends the take or comes before a cut, so it ends with a settle of at least "
                 f"{settle:g} seconds")  # fmt: skip
        if any(beat.phase == "carry" for beat in beats[1:-1]):
            fail("carry", f"{clip}: a carry beat comes only first or last")
        for beat in beats:
            # 0.36 s: `plan_shots.on_grid` moves the last beat's end by up to half a 17-frame cycle.
            if beat.phase in (*QUIET, "carry") and beat.end - beat.start > config.DIRECTOR_QUIET_MAX + 0.36:
                fail(beat.phase, f"{clip}: the {beat.phase} beat {beat.start:g}-{beat.end:g} s lasts over "
                     f"{config.DIRECTOR_QUIET_MAX:g} seconds; shorten it and the clip with it")  # fmt: skip
        if any(beat.phase == "settle" for beat in beats[:-1]):
            fail("settle", f"{clip}: only the last beat is a settle")
        main = [beat for beat in beats if beat.phase in ("primary", "handoff")]
        if not main:
            fail("main_beat", f"{clip}: mark the beat that carries the clip's one main event as primary or handoff")
        if len(main) > 1:
            listed = ", ".join(f"{b.phase} {b.start:g}-{b.end:g} s" for b in main)
            fail("main_beat", f"{clip} has {len(main)} main beats ({listed}): keep one as its main event and make the "
                 "others build or reaction beats, or move one event into a neighbouring clip")  # fmt: skip
        window, handoff = config.DIRECTOR_MAIN_BEAT_WINDOW, config.DIRECTOR_HANDOFF_MIN
        for beat in main:
            if not window[0] <= beat.start / seconds <= window[1]:
                low, high = (fraction * seconds for fraction in window)
                fail("main_beat", f"{clip}: its {beat.phase} beat starts at {beat.start:g} s; start it between "
                     f"{low:.1f} and {high:.1f} s")  # fmt: skip
            if beat.phase == "handoff" and beat.end - beat.start < handoff:
                fail("handoff", f"{clip}: the handoff beat {beat.start:g}-{beat.end:g} s lasts under {handoff:g} "
                     "seconds; give the reach, grip and release at least that long")  # fmt: skip
        if chunk.camera_move not in words.CAMERA_MOVES:
            fail("camera", f"{clip}: the camera move is one of {', '.join(words.CAMERA_MOVES)}")
        if len(set(chunk.cast) & living) > config.DIRECTOR_MAX_LIVING:
            fail("crowd", f"{clip}: at most {config.DIRECTOR_MAX_LIVING} people and animals on screen")
        # A camera following a member away leaves the room's things behind: they need not be cast.
        things = {m for m in take.members if take.cards[m].kind not in compose.LIVING}
        behind = things if chunk.camera_move == "follow" else set()
        for member in set(re.findall(r"\{([a-z_]+)\}", compose.opening(take, index))) - set(chunk.cast) - behind:
            fail("join", f"{clip}: {{{member}}} is in its opening arrangement, so it must be in its cast")
        held = compose.references(take, index)
        # A member no photograph shows has no crop by construction: its <Subject N> line carries it.
        unseen = {m for m in chunk.cast if not take.cards[m].photographed}
        for member in sorted(set(chunk.cast) - unseen - {m for key in held for m in take.refs[key].shows}):
            fail("reference", f"{clip}: {{{member}}} has no clean picture here (a picture of it also shows someone "
                 "who is not in this clip); keep that member on screen or leave the other one in the cast")  # fmt: skip
        closing = set(re.findall(r"\{([a-z_]+)\}", chunk.closing))
        # Each on-screen person or animal is restated with its traits at its first {id} in the chunk:
        # the opening, a beat it moves in (the writer must name it there), or the closing.
        named = present | closing | {m for beat in compose.written(chunk) for m in beat.who}
        for member in sorted(set(chunk.cast) & living - named):
            fail("drift", f"{clip}: {{{member}}} is on screen, but no beat moves it and neither the opening nor the "
                 "closing names it: give it a part in a beat's 'who', or take it out of the cast")  # fmt: skip
        after = set(take.chunks[index + 1].cast) if index + 1 < len(take.chunks) else set()
        # An object the clip leaves behind - the blanket below a cat flying off into the sky - is nowhere
        # in the next frame, so nothing is doubled at the join; the last clip places all of its own.
        objects = {m for m in chunk.cast if take.cards[m].kind == "object" and (following is None or m in after)}
        for member in sorted(objects - closing):
            fail("closing", f"{clip}: the closing says where {{{member}}} is and who holds it, naming it as "
                 f"{{{member}}}")  # fmt: skip
        for member in sorted(set(chunk.cast) & after & living - closing):
            fail("closing", f"{clip}: {{{member}}} is on screen in this clip and the next one, so the closing says "
                 f"where it is, as {{{member}}}")  # fmt: skip
        if closing - set(chunk.cast):
            fail("closing", f"{clip}: the closing names {sorted(closing - set(chunk.cast))}, who are not in its cast")
        leaving = {chunk.camera_toward} if chunk.camera_move == "stay" else set()  # its back to the camera
        if not closing & living - leaving <= set(chunk.faces_at_close):
            fail("faces", f"{clip}: every person and animal in the closing needs its face visible")
        # The breach and the count, not the rule: told it is nine words over, the writer cuts nine
        # words; told the rule again, it writes the same sentence again, three times over.
        words_in_closing = len(chunk.closing.split())
        if words_in_closing > config.DIRECTOR_CLOSING_WORDS:
            fail("closing", f"{clip}: the closing is {words_in_closing} words; at most "
                 f"{config.DIRECTOR_CLOSING_WORDS}, so cut "
                 f"{words_in_closing - config.DIRECTOR_CLOSING_WORDS} and keep every {{id}}")  # fmt: skip
        if not closing:
            fail("closing", f"{clip}: the closing names nobody; name each cast member on screen as {{id}}")
        for text in (chunk.closing, *(beat.action for beat in beats)):
            if wrong := possessive(text):
                fail("possessive", f"{clip}: {wrong}")
                break
        for text in (chunk.closing, *(beat.action for beat in beats), *chunk.looks.values()):
            if side := SIDE.search(text):
                fail("sides", f"{clip}: say sides as sides of the frame ('at the left of the frame'), not "
                     f"'{side.group(0)}'")  # fmt: skip
                break
        if not chunk.irreversible_change.strip():
            fail("change", f"{clip}: name the irreversible change")
        for line in chunk.dialogue:
            ends = line.start + len(line.line.split()) / config.DIRECTOR_WORDS_PER_SECOND
            if line.start < airlock or ends > seconds - config.DIRECTOR_DIALOGUE_TAIL:
                fail("dialogue", f"{clip}: a line starts after the airlock and ends "
                     f"{config.DIRECTOR_DIALOGUE_TAIL:g} seconds before the clip does")  # fmt: skip
    return out


def cross_chunk_rules(take: compose.Take, prompts: list[str]) -> list[Problem]:
    """What must not drift between chunks: the style, the room and its light, the room sound, the cast."""
    out: list[Problem] = []
    for index, (chunk, prompt) in enumerate(zip(take.chunks, prompts, strict=True)):
        room = take.cards[compose.place(take, chunk)]
        lit = f", lit by {compose.join(room.lights)}." if room.lights else "."
        heard = compose.ambience(take, index).strip().rstrip(".")[1:]
        fixed = [words.STYLE.split("{room}")[0], f"with {room.short()}{lit}", heard]
        wanted = fixed + [
            f"the {take.cards[m].noun} with {take.cards[m].short()}"
            for m in chunk.cast
            if take.cards[m].kind in compose.LIVING
        ]
        out += [Problem("error", "drift", f"clip {index + 1} lost '{x[:48]}'") for x in wanted if x not in prompt]
    return out
