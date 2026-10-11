"""The H3-Context-IR stand-in's deterministic half: a chapter's plan in, six-section Ref2VA prompts out.

MiniMax trained H3 on captions written by its own rewriter (Context-IR, not released). A local model
asked to write those whole gets the prose right and the bookkeeping wrong - label numbers, the task
directive, which picture anchors a frame and which only donates a face, the sentences that must be
identical in every chunk - so the models write only prose and facts and this module writes the rest.

Labels come from the chunk's own reference list in difflab's presentation order: `<Picture 1>` is the
opening (or handoff) frame, `<Picture 2>` the frame it ends on when it lands on the next shot's
keyframe, the crops follow, then `<Audio 1>` and `<Video 1>` for the tail (see `text_encoder.encode`).
Chunk k+1 opens on chunk k's closing arrangement word for word, because a contradiction at a join
renders as the union of both arrangements.

No model is loaded here, so a plan recompiles instantly.
"""

from __future__ import annotations

import dataclasses
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from difflab import store
from difflab.models.minimax_h3 import caption
from difflab.screenplay import config, looks, words

PLACEHOLDER = re.compile(r"\{(?!say\})([a-z_]+)\}")  # {say} marks where a dialogue line goes
LIVING = ("person", "animal")
# What a chunk short of rows holds first: each member's best face, body, view and object crop, the room;
# then second-best crops, views and bodies before faces, since a second angle on a coat says more than a
# second portrait.
PRIORITY = {"face": 0, "body": 1, "view": 2, "object": 3, "environment": 4}
SECOND = {"view": 5, "body": 6, "face": 7, "object": 8, "environment": 9}
# Who is made sure of a picture first: the room's picture shows whoever sits in it, and may already
# cover the objects on its tables.
GUARANTEE_ORDER = {"environment": 0, "person": 1, "animal": 2, "object": 3}
# A drawn picture shows its member whole and alone, so it plays the part of that kind's whole-body crop.
DRAWN_ROLE = {"person": "body", "animal": "view", "object": "object", "environment": "environment"}
NUMBERS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")


@dataclass(frozen=True)
class Trait:
    part: str  # hair | face | top | ... (config.DIRECTOR_TRAIT_ORDER): decides the order they are said in
    phrase: str  # "a black crew-neck t-shirt with a white line-art print across the chest"
    short: str  # "a black line-art-print t-shirt": restated in the description and the retention line
    name: str  # "t-shirt": how a picture's contribution is named ("whose clothing, smartwatch and build")


@dataclass(frozen=True)
class Card:
    """One cast member as the VLM saw it, merged over its photographs. Visible traits only."""

    id: str
    kind: str  # person | animal | object | environment
    noun: str
    possessive: str  # his | her | its
    traits: tuple[Trait, ...]
    lights: tuple[str, ...] = ()  # the environment's visible light sources, each one named
    # A person or animal the idea needs and no photograph shows ("a man enters"): its traits were
    # decided rather than seen, it has no crop of its own, and its sentence cites no picture.
    photographed: bool = True

    def ordered(self) -> list[Trait]:
        rank = {part: index for index, part in enumerate(config.DIRECTOR_TRAIT_ORDER)}
        return sorted(self.traits, key=lambda trait: rank.get(trait.part, len(rank)))

    def full(self) -> str:
        return join([trait.phrase for trait in self.ordered()])

    def short(self) -> str:
        """What is restated at the member's first mention in each chunk's description."""
        return join([trait.short for trait in self.ordered()[: config.DIRECTOR_SHORT_TRAITS[self.kind]]])

    def retain(self) -> str:
        """The retention clause, in the hand-tuned "his face, swept-back light-brown hair, ..." form."""
        traits = [t for t in self.ordered() if self.kind != "person" or t.name != "face"]  # "his face" says it
        bare = [_bare(trait.short) for trait in traits[: config.DIRECTOR_RETAIN_TRAITS[self.kind]]]
        if self.kind == "person":
            return f"{self.possessive} face, {join(bare)}"
        if self.kind == "environment":
            return "the " + ", ".join(bare) + " and the photo's color rendering"
        return f"the {join(bare)}"

    def looks(self) -> str:
        """What a person's body pictures give: "hair, clothing, smartwatch and build"."""
        parts = ["hair"] if any(trait.part == "hair" for trait in self.traits) else []
        parts.append("clothing")
        parts += [trait.name for trait in self.ordered() if trait.part == "accessory" and trait.name not in parts]
        return join([*parts, "build"])


@dataclass(frozen=True)
class Ref:
    """A clean crop of an original photo, or the editor's picture of a member no photo shows, and what
    it shows. Never a rendered frame."""

    key: str  # refs/<key>.png
    member: str  # whose crop it is
    role: str  # face | body | view | object | environment
    shows: tuple[str, ...]  # every member it shows, its own first
    rows: int  # what holding it costs a chunk: (width/32) * (height/32) reference rows
    quality: float  # among crops of one member and role, the larger wins (the head's or the box's size in px)
    photo: int = 0  # which photo it is cut from; -1 for a drawn one
    look: bool = False  # the picture of an amended look: held only by a chunk its member has that look in


@dataclass(frozen=True)
class Line:
    who: str
    start: float
    language: str
    line: str


@dataclass(frozen=True)
class Beat:
    start: float
    end: float
    phase: str  # airlock | carry | build | primary | handoff | reaction | settle
    who: tuple[str, ...]
    action: str  # the planner's note
    prose: str = ""  # the writer's expansion: {id} placeholders, no labels, no timing words


@dataclass(frozen=True)
class Now:
    """A member's look beyond its card in one chunk, from `looks.resolve`."""

    before: str  # what earlier shots changed, as the chunk starts; "" at the bible's look
    after: str  # what it has by the chunk's end: `before`, unless this chunk's casting amends it
    picture: str = ""  # the ref key of `after`'s own picture once approved; "" while it has none
    start: str = ""  # the ref key of `before`'s own picture, when it was picked beside `after`'s
    held: tuple[str, ...] = ()  # the look pictures handed over: `picture` alone unless picked by hand
    bible: bool = True  # whether the bible's pictures, which show none of its looks, are handed over


@dataclass(frozen=True)
class Chunk:
    purpose: str
    cast: tuple[str, ...]
    beats: tuple[Beat, ...]
    camera_move: str
    camera_toward: str
    mouths: tuple[str, ...]  # "his slow draw on the mouthpiece": every way a mouth moves in this chunk
    closing: str  # {id} placeholders; the next chunk's opening, verbatim
    irreversible_change: str
    dialogue: tuple[Line, ...] = ()
    summary: str = ""  # writer
    sounds: str = ""  # writer
    seconds: float = 0.0  # on H3's frame grid; 0 in a plan from before the director chose lengths: `Take.seconds`
    cut: str = ""  # "" continues the chunk before; "back" returns to the opening photo; "new" opens on `opening`
    opening: str = ""  # a "new" cut's first frame, {id} placeholders
    looks: dict[str, str] = dataclasses.field(default_factory=dict)  # the planner's: member -> its look from here on
    look_names: dict[str, str] = dataclasses.field(default_factory=dict)  # the planner's: member -> that look's name
    look_from: dict[str, str] = dataclasses.field(default_factory=dict)  # member -> the version it starts from
    ambience: str = ""  # the sound of a place other than the take's first; "" is `Take.ambience`
    scene: str = ""  # the slug line the chunk plays under, "INT lantern room · dusk"
    states: dict[str, Now] = dataclasses.field(default_factory=dict)  # the castings' looks; never stored in a plan


@dataclass(frozen=True)
class Take:
    title: str
    members: tuple[str, ...]  # global cast order, by first appearance: fixes <Subject N> numbering
    cards: dict[str, Card]
    refs: dict[str, Ref]  # every clean crop, in the order a chunk presents the ones it holds
    opening: str  # chunk 1's opening arrangement, from the opening frame
    ambience: str  # a noun phrase of room sound, the same in every chunk
    seconds: float  # the longest chunk, on H3's 17n+5 frame grid (12 s -> 12.25); `length` is one chunk's
    tail: bool  # Continuity.tail_seconds > 0: chunks after the first carry <Audio 1> and <Video 1>
    budget: int  # reference rows a chunk may hold (config.DIRECTOR_REFERENCE_ROWS)
    chunks: tuple[Chunk, ...] = ()
    story: tuple[tuple[str, int], ...] = ()  # the IDEA's actions: (its words, the clip from 1)
    tails: tuple[bool, ...] = ()  # per chunk, whether it carries the one before's tail; empty is `tail` after the first
    lands: tuple[bool, ...] = ()  # per chunk, whether it ends on the next one's keyframe
    unframed: tuple[bool, ...] = ()  # per chunk, whether it opens on no picture, from its references alone
    scenes: tuple[bool, ...] = ()  # per chunk, whether it may hold the chapter's opening still (`scened`)
    scene_rows: int = 0  # reference rows the opening still costs, at the canvas size
    voices: dict[str, str] = dataclasses.field(default_factory=dict)  # member -> how it sounds, from the bible
    look: str = ""  # the style members' words: the grade and lens every shot keeps


@dataclass(frozen=True)
class Presentation:
    """The labels difflab's text encoder announces for one chunk, in order."""

    pictures: tuple[str, ...]  # a ref key per <Picture i>: the keyframes (KEYFRAMES) first, then crops
    tail: bool

    def picture(self, key: str) -> str:
        return f"<Picture {self.pictures.index(key) + 1}>"

    def labels(self) -> list[str]:
        return caption.labels(len(self.pictures), self.tail)


@dataclass(frozen=True)
class Compiled:
    prompt: str
    presentation: Presentation
    references: tuple[str, ...]  # ref keys for `long_video.render(beat_references=...)`, in order
    subjects: dict[str, str]  # member id -> <Subject N>


def _bare(phrase: str) -> str:
    return re.sub(r"^(a|an|the) ", "", phrase)


def join(items: list[str] | tuple[str, ...]) -> str:
    items = list(items)
    return "".join(items) if len(items) <= 1 else ", ".join(items[:-1]) + " and " + items[-1]


def length(take: Take, index: int) -> float:
    return take.chunks[index].seconds or take.seconds


CUTS = ("", "back", "new")


def opening(take: Take, index: int) -> str:
    chunk = take.chunks[index]
    if index == 0 or chunk.cut == "back":
        return take.opening
    return chunk.opening if chunk.cut == "new" else take.chunks[index - 1].closing


def ambience(take: Take, index: int) -> str:
    return take.chunks[index].ambience or take.ambience


def written(chunk: Chunk) -> list[Beat]:
    """The beats the writer writes: all but a closing settle or carry, which the compiler says from the closing."""
    last = chunk.beats[-1] if chunk.beats else None
    return [beat for beat in chunk.beats if not (beat is last and beat.phase in ("settle", "carry"))]


def place(take: Take, chunk: Chunk) -> str:
    return next(m for m in chunk.cast if take.cards[m].kind == "environment")


KEYFRAMES = ("opening", "handoff", "closing", "scene")  # a Presentation's keys that are frames, not crops


def carried(take: Take, index: int) -> bool:
    if take.chunks and take.chunks[index].cut:
        return False  # a cut opens on its own keyframe: the clip before's motion stops there
    return take.tails[index] if take.tails else take.tail and index > 0


def scenes(keyframes: list[store.Keyframe], shots: list[store.Shot]) -> tuple[bool, ...]:
    """Per shot, whether it opens on the shot before's handoff frame while the chapter opens on a clean still."""
    clean = bool(keyframes) and keyframes[0].source in (store.COMPOSED, store.PROVIDED)
    return tuple(
        clean and index > 0 and keyframe.source == store.CHAINED and not shot.plan.get("cut")
        for index, (keyframe, shot) in enumerate(zip(keyframes, shots, strict=True))
    )


def scened(take: Take, index: int) -> bool:
    """A chained chunk holds the chapter's opening still, the one clean picture of the scene as it is lit,
    since its own first frame is the chunk before's last, softer and drifted at every join: production 13's chain
    lost 80% of its Laplacian variance over four shots, and got it back only where a clean look picture
    joined the references. Never where the still would bring back someone off screen, another place, or the
    look someone in it has since changed out of."""
    if not (take.scenes and take.scenes[index]):
        return False
    chunk, first = take.chunks[index], take.chunks[0]
    shown = {m for m in PLACEHOLDER.findall(take.opening) if m in take.cards and take.cards[m].kind in LIVING}
    changed = any(state.before or state.after for m, state in chunk.states.items() if m in shown)
    return place(take, chunk) == place(take, first) and shown <= set(chunk.cast) and not changed


def lands(take: Take, index: int) -> bool:
    return bool(take.lands) and take.lands[index]


def unframed(take: Take, index: int) -> bool:
    return bool(take.unframed) and take.unframed[index]


def joins(
    keyframes: list[store.Keyframe], shots: list[store.Shot], tail: bool
) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    """`(tails, lands)` for a chapter. A shot ends on the next one's composed or provided keyframe
    unless that keyframe is set not to be landed on; the motion then runs into it, so the tail crosses
    too. Into a keyframe nothing landed on it would drag the old motion into a picture it never reaches.
    A shot opened from its references alone lands on nothing and is landed on by nothing: its caption
    numbers its references from <Picture 1>, where a landing's last frame would have to come first."""
    pictured = (store.COMPOSED, store.PROVIDED)
    landed = tuple(
        index + 1 < len(keyframes)
        and keyframes[index].source != store.REFS
        and keyframes[index + 1].source in pictured
        and keyframes[index + 1].landed
        for index in range(len(keyframes))
    )
    tails = tuple(
        index > 0 and tail and shot.chained and (keyframes[index].source == store.CHAINED or landed[index - 1])
        for index, shot in enumerate(shots)
    )
    return tails, landed


def references(take: Take, index: int) -> tuple[str, ...]:
    """The crops chunk `index` holds (<Picture 2> onward), in `take.refs` order.

    A crop that shows a person or animal who is not on screen in this chunk is never held: it would
    bring them back, a union at the join. Of the rest, every on-screen member is first made sure of
    its best picture; then the others are added by role (faces, bodies, animal views, objects, the
    room, then second-best crops) while the chunk's reference rows stay within `take.budget`. The room
    is held once: the whole first photo, or, when someone in it is off screen, the cut clear of them.
    An animal seen in several photos is held at most once per photo: its look is its coat, and photos
    in different light disagree about the coat's colour, so no one photo may outvote the others.
    `lint.plan_rules` fails a plan that leaves an on-screen member with no picture at all.

    A member with a look of its own is held as that look's picture and its own face, nothing else that
    shows it: every other picture of it shows the bible's look. The room's picture stays.
    """
    chunk = take.chunks[index]
    budget = take.budget - (take.scene_rows if scened(take, index) else 0)
    on = set(chunk.cast)
    looked = {m for m, now in chunk.states.items() if not now.bible and m in on}
    picked = {m: now.held for m, now in chunk.states.items() if m in on}

    def kind(member: str) -> str:
        return take.cards[member].kind

    def held(ref: Ref) -> bool:
        if ref.look:
            return ref.key in picked.get(ref.member, ())
        if ref.role == "environment":
            return ref.member not in looked
        amended = set(ref.shows) & looked
        return not amended or (ref.role == "face" and amended == {ref.member})

    # A picture of a place the chunk is not in brings that place back: the room photo, which also shows
    # the cat, pulled production 10's sky chase back onto the desk.
    usable = [
        ref
        for ref in take.refs.values()
        if on & set(ref.shows) and all(m in on for m in ref.shows if kind(m) in (*LIVING, "environment")) and held(ref)
    ]
    best: dict[tuple[str, str], Ref] = {}
    for ref in usable:
        if (ref.member, ref.role) not in best or ref.quality > best[ref.member, ref.role].quality:
            best[ref.member, ref.role] = ref
    firsts = {ref.key for ref in best.values()}
    usable = [ref for ref in usable if ref.role != "environment" or ref.key in firsts]
    photos: dict[str, set[int]] = {}
    for ref in usable:
        photos.setdefault(ref.member, set()).add(ref.photo)

    def spent(ref: Ref, chosen: list[Ref]) -> bool:
        """An animal's second crop from a photo it is already held from, while it has other photos."""
        if kind(ref.member) != "animal" or len(photos[ref.member]) < 2:
            return False
        return any(held.member == ref.member and held.photo == ref.photo for held in chosen)

    ranked = sorted(
        usable,
        key=lambda ref: (
            PRIORITY[ref.role] if ref.key in firsts else SECOND[ref.role],
            take.members.index(ref.member),
            -ref.quality,
        ),
    )
    chosen: list[Ref] = []
    for member in sorted(on, key=lambda m: (GUARANTEE_ORDER[kind(m)], take.members.index(m))):
        chosen += [ref for ref in ranked if ref.look and ref.key in picked.get(member, ()) and ref not in chosen]
        if any(member in ref.shows for ref in chosen):
            continue
        own = [ref for ref in ranked if ref.member == member] or [ref for ref in ranked if member in ref.shows]
        if own:
            chosen.append(own[0])
    rows = sum(ref.rows for ref in chosen)
    for ref in ranked:
        if ref not in chosen and rows + ref.rows <= budget and not spent(ref, chosen):
            chosen.append(ref)
            rows += ref.rows
    return tuple(key for key, ref in take.refs.items() if ref in chosen)


def rows(take: Take, keys: tuple[str, ...]) -> int:
    return sum(take.refs[key].rows for key in keys)


class CastError(ValueError):
    """Text names a member the chunk does not cast: a message the model that wrote it can act on."""


class Expander:
    """{id} -> <Subject N>, restating the member's short traits - and a lasting change to its look - at
    its first mention in the chunk."""

    def __init__(
        self, take: Take, subjects: dict[str, str], speakers: dict[str, str], looks: dict[str, str] | None = None
    ) -> None:
        self.take, self.subjects, self.speakers, self.looks = take, subjects, speakers, looks or {}
        self.seen: set[str] = set()

    def __call__(self, text: str, expand: bool = True, behind: bool = False) -> str:
        """`behind`: the text is an opening, the clip before's own words, so a member the clip does not cast
        is named plainly rather than failing a writer who cannot change them; `lint.plan_rules` asks the
        plan to cast a person or animal there."""

        def label(match: re.Match[str]) -> str:
            member = match.group(1)
            card = self.take.cards.get(member)
            if member not in self.subjects and behind and card is not None:
                return f"the {card.noun}"
            if member not in self.subjects:
                cast = ", ".join(f"{{{m}}}" for m in self.subjects)
                raise CastError(f"{{{member}}} is named, but this clip's cast is {cast}: leave it out or cast it")
            card, subject = self.take.cards[member], self.subjects[member]
            follow = text[match.end() : match.end() + 7]
            if follow.startswith(" {say}"):
                return f"{subject} ({self.speakers[member]})"  # a vocal event takes the speaker ID, no appositive
            if not expand or member in self.seen or card.kind == "environment" or follow.startswith("'s"):
                return subject
            self.seen.add(member)
            comma = "" if follow[:1] in tuple(".,;:!?") else ","
            now = f" and {self.looks[member]}" if self.looks.get(member) else ""
            return f"{subject}, the {card.noun} with {card.short()}{now}{comma}"

        return PLACEHOLDER.sub(label, text)


def spoken(seconds: float) -> str:
    """4.0 -> "four seconds", 1.2 -> "second": how long a beat lasts, in the words a caption uses."""
    count = max(1, round(seconds))
    if count == 1:
        return "second"
    return f"{NUMBERS[count] if count < len(NUMBERS) else count} seconds"


def timing(beat: Beat, index: int, seconds: float, used: set[str]) -> str:
    """The phrase in front of a beat's prose. The settle is phrased by `compile_chunk` itself."""
    length = spoken(beat.end - beat.start)
    if index == 0:
        return words.TIMING[beat.phase if beat.phase in ("airlock", "carry") else "first"].format(seconds=length)
    if beat.phase == "handoff":
        phrase = words.TIMING["handoff"]
    else:
        phrase = words.TIMING_BY_POSITION[0][1]
        for threshold, text in words.TIMING_BY_POSITION:
            if beat.start / seconds >= threshold:
                phrase = text
    if phrase in used:
        phrase = words.TIMING_REPEAT
    used.add(phrase)
    return phrase.format(seconds="a second" if length == "second" else length)


def _lower(text: str) -> str:
    """The first letter lowered, for text that continues a sentence - unless it is a label or an {id}."""
    return text if text.startswith(("{", "<")) else text[:1].lower() + text[1:]


def _sentence(text: str) -> str:
    return text.strip().rstrip(".")


def _upper(text: str) -> str:
    return text[:1].upper() + text[1:]


def definition(take: Take, member: str, subject: str, shown: Presentation) -> str:
    """One `<Subject N>` line: the member, the pictures it comes from and what each gives, its traits."""
    card = take.cards[member]
    held = [key for key in shown.pictures if key not in KEYFRAMES and member in take.refs[key].shows]

    def cite(keys: list[str]) -> str:
        return join([shown.picture(key) for key in keys])

    def that(keys: list[str]) -> str:
        return "that photo" if len(keys) == 1 else "those photos"

    say = words.SENTENCES
    if not card.photographed and not held:
        group = "unseen_group" if card.possessive == "their" else "unseen"
        unseen = "unseen_place" if card.kind == "environment" else group
        return say[unseen].format(subject=subject, noun=card.noun, traits=card.full())
    if card.kind == "environment":
        own = [key for key in held if take.refs[key].role == "environment"] or held
        return say["environment"].format(subject=subject, noun=card.noun, pictures=cite(own), traits=card.full())
    if card.kind == "person":
        faces = [key for key in held if take.refs[key].role == "face" and take.refs[key].member == member]
        bodies = [key for key in held if take.refs[key].role != "face"]
        if faces and bodies:
            return say["person"].format(
                subject=subject,
                noun=card.noun,
                faces=cite(faces),
                parts=card.looks(),
                bodies=cite(bodies),
                possessive=card.possessive,
                traits=card.full(),
            )
        if faces:
            return say["person_face"].format(
                subject=subject,
                noun=card.noun,
                faces=cite(faces),
                possessive=card.possessive,
                that=that(faces),
                traits=card.full(),
            )
    what = f"{card.possessive} appearance" if card.kind == "person" else f"the {card.noun}"
    return say["member"].format(
        subject=subject, noun=card.noun, pictures=cite(held), what=what, that=that(held), traits=card.full()
    )


def compile_chunk(take: Take, index: int) -> Compiled:
    say = words.SENTENCES
    chunk = take.chunks[index]
    refs = references(take, index)
    tail = carried(take, index)
    landing = lands(take, index)
    framed = not unframed(take, index)
    first = ("opening" if index == 0 else "handoff",) if framed else ()
    scene = ("scene",) if scened(take, index) else ()
    shown = Presentation((*first, *(("closing",) if landing else ()), *scene, *refs), tail)
    cast = [m for m in take.members if m in chunk.cast]
    subject = {m: f"<Subject {i + 1}>" for i, m in enumerate(cast)}
    speakers = {line.who: f"S{i + 1}" for i, line in enumerate(chunk.dialogue)}
    env = place(take, chunk)
    living = [m for m in cast if take.cards[m].kind in LIVING]
    labels_only = Expander(take, subject, speakers)
    states = {m: chunk.states[m] for m in cast if m in chunk.states}
    now = {m: state.before for m, state in states.items() if state.before}
    changes = {m: state for m, state in states.items() if state.after != state.before}

    def cited(state: Now, key: str | None = None) -> str:
        key = state.picture if key is None else key
        return shown.picture(key) if key and key in shown.pictures else ""

    # subject_definitions: identity-only pictures are cited inside the <Subject N> lines (ref-en 2.2)
    definitions = [definition(take, member, subject[member], shown) for member in cast]
    for m in cast:
        if m in now and m not in changes and cited(states[m]):
            definitions.append(say["look_pictured"].format(subject=subject[m], look=now[m], pictures=cited(states[m])))
        elif m in now and m in changes and cited(states[m], states[m].start):
            definitions.append(
                say["look_start_pictured"].format(
                    subject=subject[m], look=now[m], pictures=cited(states[m], states[m].start)
                )
            )
        elif m in now:
            definitions.append(say["look"].format(subject=subject[m], look=now[m]))
    for m, state in changes.items():
        if cited(state):
            definitions.append(
                say["look_change_pictured"].format(subject=subject[m], look=state.after, pictures=cited(state))
            )
        else:
            definitions.append(say["look_change"].format(subject=subject[m], look=state.after))
    if framed:
        definitions.append(
            say["first_frame"].format(arrangement=_lower(_sentence(labels_only(opening(take, index), False, True))))
        )
    if landing:
        definitions.append(say["last_frame"].format(arrangement=_lower(_sentence(labels_only(chunk.closing, False)))))
    if scene:
        definitions.append(say["scene_frame"].format(picture=shown.picture("scene")))
    if tail:
        definitions += [say["video"], say["audio"]]

    # summary
    directive = caption.directive(tail, framed)
    lead = "lead_continued" if tail else "lead" if framed else "lead_refs"
    summary = " ".join(
        part
        for part in (
            directive,
            say[lead].format(room=subject[env]),
            say["lead_lands"] if landing else "",
            labels_only(chunk.summary.strip(), False),
            say["audio_role"] if tail else "",
        )
        if part
    )

    # retention_analysis
    retention = [
        say["retain_look" if m in now else "retain"].format(
            subject=subject[m], traits=take.cards[m].retain(), look=now.get(m, "")
        )
        for m in cast
    ]
    retention += ([say["retain_frame"]] if framed else []) + ([say["retain_last"]] if landing else [])
    retention += [say["retain_scene"].format(picture=shown.picture("scene"))] if scene else []
    moving = index > 0 and any(beat.phase == "carry" for beat in take.chunks[index - 1].beats[-1:])
    retention += [say["retain_video_moving" if moving else "retain_video"], say["retain_audio"]] if tail else []

    # detailed_description
    expand = Expander(take, subject, speakers, now)
    room = take.cards[env]
    body = [
        say["begin_continued" if tail else "begin" if framed else "begin_refs"].format(
            arrangement=_lower(_sentence(expand(opening(take, index), behind=True)))
        ),
        say["room" if room.lights else "room_unlit"].format(
            room=subject[env], traits=room.short(), lights=join(room.lights)
        ),
    ]
    used: set[str] = set()
    pending = list(chunk.dialogue)  # taken in the writer's own order, one per {say}
    camera_said = False
    camera = words.CAMERA_MOVES[chunk.camera_move].format(toward=subject.get(chunk.camera_toward, subject[env]))
    for number, beat in enumerate(written(chunk)):
        prose = beat.prose.strip() or beat.action.strip()  # a beat edited by hand is said as planned
        text = expand(f"{timing(beat, number, length(take, index), used)} {_lower(prose)}")
        # A dialogue line lands where the writer put its {say}, taking them in order - not where the
        # line's own `start` second happens to fall.
        #
        # Matching by time window looks tidier and is a trap. The writer chooses `start` and the
        # placement of {say} as two separate decisions, and nothing tells it they have to agree; when
        # they disagree the line is dropped, the {say} survives compilation, and the gate rejects the
        # whole take with "an {id} placeholder survived" - which names neither the cause nor the
        # chunk. It is also unrepairable by the writer, because the rule it broke is not one it can
        # see. Takes already on disk are unaffected: a take only exists if its {say}s and its starts
        # agreed, and where they agree both readings give the same sentence in the same place.
        while "{say}" in text and pending:
            line = pending.pop(0)
            head, rest = text.split("{say}", 1)
            sentence = say["line"].format(voice=voice(take, line.who), language=line.language, words=line.line)
            # The speaker's (Sx) has to sit immediately before the <d>, and it only gets there by
            # itself when the writer put {say} right after that member's braces. It often does not -
            # "laughs, saying {say}" reads more naturally to a writer than "{right_girl} {say}" - and
            # the result was a take rejected for a marker it had no way to know about. The compiler
            # knows who speaks, so it says so: the dangling speech verb comes off and the speaker is
            # named where the line begins.
            if not re.search(r"\(S\d+\)[^<]*$", head):
                head = re.sub(r"[,\s]*\b(?:saying|says|said|speaking|and say)\s*$", "", head)
                head = f"{head}, " if head and not head.endswith((" ", ",")) else head
                # The subject label only comes along when the beat has not already said who this is;
                # otherwise the (Sx) alone carries it, and the sentence does not name the same person
                # twice in a row.
                who = speakers[line.who]
                tag = f"({who})" if subject[line.who] in head else f"{subject[line.who]} ({who})"
                sentence = f"{tag} {sentence}"
            # The template ends in a full stop and the writer's clause usually carries on with a
            # comma, which butts them together as ".,". Joined as one sentence instead.
            text = re.sub(r"\.,\s*", ", ", f"{head}{sentence}{rest}")
        body.append(text)
        # One camera phrase per chunk: during its main beat, or a reveal as its member comes into view.
        reveal = chunk.camera_move.startswith("reveal") and chunk.camera_toward in beat.who
        if (reveal or beat.phase in ("primary", "handoff")) and not camera_said:
            body.append(camera)
            camera_said = True
    if not camera_said:
        body.append(camera)
    # A line the prose has no {say} for - one added by hand to a shot already written - is said after
    # the beats by its speaker; otherwise nothing would say it at all.
    for line in pending:
        spoken_line = say["line"].format(voice=voice(take, line.who), language=line.language, words=line.line)
        body.append(f"{subject[line.who]} ({speakers[line.who]}) {spoken_line}")
    # Through the expander like every other line the plan writes. `mouths` was the one field that
    # reached the prompt raw, so a plan that named its speakers the way the rules ask - "the laugh of
    # {right_girl}" - put a live placeholder into the finished caption, and the gate rejected the
    # take for it. Labels only, as `summary` and `sounds` are: the traits are said once, elsewhere.
    mouths = labels_only(join(chunk.mouths), False) if chunk.mouths else ""
    body.append(say["mouths"].format(actions=mouths) if mouths else say["mouths_none"])
    settle = chunk.beats[-1]
    ending = "carry" if settle.phase == "carry" else "settle_lands" if landing else "settle"
    body.append(
        say[ending].format(
            timing=words.TIMING["settle"].format(seconds=spoken(settle.end - settle.start)),
            arrangement=_lower(_sentence(expand(chunk.closing))),
            shot="two-shot" if len(living) == 2 else "shot",
        )
    )
    # A member on screen that nothing names still gets its traits said once, after the room's.
    named = " ".join(body)
    body[2:2] = [
        say["present"].format(subject=subject[m], noun=take.cards[m].noun, traits=take.cards[m].short())
        for m in cast
        if m != env and subject[m] not in named
    ]
    style = words.STYLE.format(room=subject[env])
    if take.look:
        style = f"{style} {words.STYLE_LOOK.format(look=take.look.rstrip('.'))}"

    # sound
    heard = ambience(take, index)
    # The tail's room tone carries on only into the same place's sound.
    same = tail and heard == ambience(take, index - 1)
    room_sound = (
        say["room_sound_continued"].format(ambience=_sentence(heard))
        if same
        else say["room_sound"].format(ambience=_upper(_sentence(heard)))
    )
    soundscape = f"{room_sound} {labels_only(chunk.sounds.strip(), False)}".strip()

    # A {say} with no dialogue behind it marks speech the writer never wrote. Dropped here, on the
    # finished text, rather than beat by beat: the closing and the soundscape are assembled after the
    # beats are, and cleaning only the beats left a marker in the closing to be reported two stages
    # later as "an {id} placeholder survived compilation" - which names neither the marker nor where.
    prompt = "\n\n".join(
        [
            "subject_definitions:\n" + "\n".join(definitions),
            "summary:\n" + summary,
            "retention_analysis:\n" + "\n".join(retention),
            "detailed_description:\n" + style + "\n" + " ".join(body),
            "overall_soundscape:\n" + soundscape,
            "non_diegetic_music:\n" + words.MUSIC,
        ]
    )
    prompt = prompt.replace(" {say}", "").replace("{say}", "")
    return Compiled(prompt, shown, refs, subject)


def voice(take: Take, member: str) -> str:
    """How a member sounds in its lines: the bible's voice and delivery, or its noun's when none is set."""
    return take.voices.get(member) or f"a natural {take.cards[member].noun}'s voice"


# --- a chapter as the rows say it --------------------------------------------------------------
def voice_of(member: store.Member) -> str:
    return ", ".join(part for part in (member.voice.strip(), member.delivery.strip()) if part)


def card_of(member: store.Member) -> Card:
    """Words written over the traits stand in for all of them, as one trait: a card of traits alone keys as before."""
    identity = member.identity
    said = " ".join(member.appearance.split()) if member.kind != "style" else ""
    traits = (
        (Trait("appearance", said, said, "appearance"),)
        if said
        else tuple(
            Trait(str(t["part"]), str(t["phrase"]), str(t["short"]), str(t["name"])) for t in identity.get("traits", [])
        )
    )
    return Card(
        member.slug,
        member.kind,
        member.noun,
        member.possessive,
        traits,
        tuple(str(light) for light in identity.get("lights", [])),
        bool(identity.get("photographed", True)),
    )


def chunk_of(plan: dict[str, Any]) -> Chunk:
    return Chunk(
        purpose=str(plan.get("purpose", "")),
        cast=tuple(str(m) for m in plan.get("cast", [])),
        beats=tuple(
            Beat(
                float(b["start"]),
                float(b["end"]),
                str(b["phase"]),
                tuple(str(w) for w in b["who"]),
                str(b["action"]),
                str(b.get("prose", "")),
            )
            for b in plan.get("beats", [])
        ),
        camera_move=str(plan.get("camera_move", "")),
        camera_toward=str(plan.get("camera_toward", "")),
        mouths=tuple(str(m) for m in plan.get("mouths", [])),
        closing=str(plan.get("closing", "")),
        irreversible_change=str(plan.get("irreversible_change", "")),
        dialogue=tuple(
            Line(str(d["who"]), float(d["start"]), str(d["language"]), str(d["line"])) for d in plan.get("dialogue", [])
        ),
        summary=str(plan.get("summary", "")),
        sounds=str(plan.get("sounds", "")),
        seconds=float(plan.get("seconds", 0.0)),
        cut=str(plan.get("cut", "")),
        opening=str(plan.get("opening", "")),
        ambience=str(plan.get("ambience", "")),
        scene=str(plan.get("scene", "")),
    )


_PLANNER_ONLY = ("looks", "look_names", "look_from", "states")


def plan_of(chunk: Chunk) -> dict[str, Any]:
    """The chunk as the shot row keeps it: its looks are its castings'."""
    return {k: v for k, v in dataclasses.asdict(chunk).items() if k not in _PLANNER_ONLY}


def ref_keys(references: list[store.Reference], slugs: dict[int, str]) -> list[str]:
    """How the compiler names each reference, in row order: the member, its role, and a count."""
    out: list[str] = []
    for reference in references:
        member = slugs[reference.member_id]
        name = member if reference.role == "environment" else f"{member}_{reference.role}"
        out.append(name if name not in out else next(f"{name}{n}" for n in range(2, 100) if f"{name}{n}" not in out))
    return out


def pictured(references: list[store.Reference]) -> list[store.Reference]:
    """The references with a picture: a drawn one is a row before the editor has made it."""
    return [reference for reference in references if reference.artifact]


def refs_of(references: list[store.Reference], members: list[store.Member], photos: list[str]) -> dict[str, Ref]:
    """`take.refs`, keyed as the compiler names pictures; `photos` is the production's list, so a
    reference knows which photo it was cut from."""
    slugs = {member.id: member.slug for member in members}
    kinds = {member.id: member.kind for member in members}
    references = pictured(references)
    out: dict[str, Ref] = {}
    for name, reference in zip(ref_keys(references, slugs), references, strict=True):
        member = slugs[reference.member_id]
        shows: tuple[str, ...]
        if reference.generated:
            role, shows, photo = DRAWN_ROLE[kinds[reference.member_id]], (member,), -1
        else:
            role, shows = reference.role, tuple(reference.shows)
            photo = photos.index(reference.photo) if reference.photo in photos else 0
        out[name] = Ref(name, member, role, shows, reference.rows, reference.quality, photo)
    return out


def look_key(look: looks.Look) -> str:
    return f"{look.member}_look{look.id}"


def look_refs(resolved: looks.Resolved, members: list[store.Member]) -> dict[str, Ref]:
    """The picture of every approved look, as a crop of its member alone; held only where it is worn."""
    kinds = {member.slug: member.kind for member in members}
    out: dict[str, Ref] = {}
    for look in resolved.looks.values():
        if look.approved and look.casting is not None and look.casting.size:
            width, height = look.casting.size
            name = look_key(look)
            # Larger than any crop, so it is the member's best picture of its role wherever it is held.
            out[name] = Ref(name, look.member, DRAWN_ROLE[kinds[look.member]], (look.member,),
                            (width // 32) * (height // 32), float(width * height) * 1e6, -1, look=True)  # fmt: skip
    return out


def states_of(resolved: looks.Resolved, shot_id: int) -> dict[str, Now]:
    """`casting.pictured` names, by "base" or look id, the looks whose pictures the shot hands over."""

    def key(look: looks.Look) -> str:
        return look_key(look) if look.approved and look.casting is not None and look.casting.size else ""

    out: dict[str, Now] = {}
    for member, placed in resolved.placed.get(shot_id, {}).items():
        picked = placed.casting.pictured
        if placed.source is None and placed.look is None and picked is None:
            continue
        before, after = resolved.look(member, placed.source), resolved.look(member, placed.look)
        picture = key(after)
        if picked is None:
            out[member] = Now(before.text, after.text, picture, held=(picture,) if picture else (), bible=not picture)
            continue
        chosen = [look for look in (before, after) if look.id is not None and str(look.id) in picked]
        held = tuple(dict.fromkeys(k for look in chosen if (k := key(look))))
        start = key(before) if before.id != after.id and key(before) in held else ""
        out[member] = Now(before.text, after.text, picture, start, held, store.BASE in picked)
    return out


def pictures_of(connection: sqlite3.Connection, production: store.Production) -> dict[str, str]:
    """Each reference's file by the name the compiler gives it, for the clip job to be handed."""
    members = store.members_of(connection, production.id)
    references = pictured(store.refs_of_production(connection, production.id))
    slugs = {member.id: member.slug for member in members}
    owners = {member.id: member for member in members}
    resolved = looks.resolve(connection, production)
    worn = {
        look_key(look): str(look.casting.artifact)
        for look in resolved.looks.values()
        if look.approved and look.casting is not None
    }
    return {
        **{
            name: str(owners[ref.member_id].picture(ref))
            for name, ref in zip(ref_keys(references, slugs), references, strict=True)
        },
        **worn,
    }


def take_of(connection: sqlite3.Connection, production: store.Production, chapter: store.Chapter) -> Take:
    """The production's plan for one chapter, as the compiler and the writer read it."""
    members = store.members_of(connection, production.id)
    references = store.refs_of_production(connection, production.id)
    manifest = production.manifest
    planned = {**manifest, **manifest.get("chapter_plans", {}).get(str(chapter.id), {})}
    shots = store.shots_of(connection, chapter.id)
    tail = bool(manifest.get("tail", True))
    canvas = manifest.get("canvas") or {}
    keyframes = [store.keyframe_of(connection, shot.id) for shot in shots]
    tails, landed = joins(keyframes, shots, tail)
    first = keyframes[0].provided if keyframes else None
    resolved = looks.resolve(connection, production)
    chunks = [
        dataclasses.replace(chunk_of(shot.plan), seconds=shot.seconds, states=states_of(resolved, shot.id))
        for shot in shots
    ]
    # A later shot handed the first photo by hand cuts back to it, whatever its plan says.
    chunks = [
        dataclasses.replace(chunk, cut="back") if index and first and keyframe.provided == first else chunk
        for index, (chunk, keyframe) in enumerate(zip(chunks, keyframes, strict=True))
    ]
    return Take(
        title=production.title,
        members=tuple(member.slug for member in members),
        cards={member.slug: card_of(member) for member in members},
        refs={
            **refs_of(references, members, [str(p) for p in manifest.get("photos", [])]),
            **look_refs(resolved, members),
        },
        opening=str(manifest.get("opening_arrangement", "")),
        ambience=str(planned.get("ambience", "")),
        seconds=float(manifest.get("seconds", 0.0)),
        tail=tail,
        budget=int(manifest.get("budget", config.DIRECTOR_REFERENCE_ROWS)),
        chunks=tuple(chunks),
        story=tuple((str(event), int(clip)) for event, clip in planned.get("story", [])),
        tails=tails,
        lands=landed,
        unframed=tuple(keyframe.source == store.REFS for keyframe in keyframes),
        scenes=scenes(keyframes, shots),
        scene_rows=(int(canvas.get("width", 0)) // 32) * (int(canvas.get("height", 0)) // 32),
        voices={member.slug: said for member in members if (said := voice_of(member))},
        look=look_of(connection, production.id),
    )


def look_of(connection: sqlite3.Connection, production_id: int) -> str:
    """What the style members say of the film's look, one after another."""
    styles = [m for m in store.members_of(connection, production_id, style=True) if m.kind == "style"]
    return "; ".join(said.rstrip(".") for m in styles if (said := " ".join(m.appearance.split())))
