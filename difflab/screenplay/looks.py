"""How each member looks in each shot, carried forward until an amendment, a revert or a cut back."""

from __future__ import annotations

import dataclasses
import itertools
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from difflab import store
from difflab.core.settings import JudgeSettings
from difflab.screenplay import drawn, settings

SCOPE = "looks:{production}"


@dataclass(frozen=True)
class Look:
    """The bible's look of a member (`casting` None) or one an amended casting made from `source`."""

    member: str
    casting: store.Casting | None
    source: int | None
    words: tuple[str, ...]  # every amendment from the base on, oldest first

    @property
    def id(self) -> int | None:
        return None if self.casting is None else self.casting.id

    @property
    def label(self) -> str:
        if self.casting is None:
            return "base"
        return self.casting.label.strip() or "changed"

    @property
    def text(self) -> str:
        return ", ".join(self.words)

    @property
    def approved(self) -> bool:
        """Judged past the gate or approved by a person, with its picture there; the base always is."""
        return self.casting is None or (self.casting.state == store.APPROVED and self.casting.artifact is not None)


@dataclass(frozen=True)
class Placed:
    """A member in one shot: the look it starts from, the one it has, and the one it had last before."""

    casting: store.Casting
    shot: int
    member: str
    source: int | None
    look: int | None
    latest: int | None
    back: bool  # the source was decided by a cut back, not by the casting
    carried_into: tuple[int, ...] = ()

    @property
    def revert(self) -> bool:
        return not self.casting.amended and self.source != self.latest


@dataclass(frozen=True)
class Resolved:
    order: tuple[int, ...]  # every shot id, in film order
    placed: dict[int, dict[str, Placed]]  # shot id -> member slug -> its placement
    looks: dict[int, Look]  # amended casting id -> the look it makes, in film order
    gaps: dict[str, tuple[tuple[int, ...], ...]]  # member -> each stretch of shots it is off screen between two

    def look(self, member: str, look: int | None) -> Look:
        return Look(member, None, None, ()) if look is None else self.looks[look]

    def shown(self, shot: int) -> list[Look]:
        """Every look the shot shows, the ones it starts from included."""
        out: dict[tuple[str, int | None], Look] = {}
        for member, placed in self.placed.get(shot, {}).items():
            for look in (placed.source, placed.look):
                out[member, look] = self.look(member, look)
        return list(out.values())

    def waiting(self, shot: int) -> list[Look]:
        return [look for look in self.shown(shot) if not look.approved]


def resolve(connection: Any, production: store.Production) -> Resolved:
    """A cut back returns each member to the look it had when the chapter's opening scene was left."""
    slugs = {member.id: member.slug for member in store.members_of(connection, production.id)}
    by_shot: dict[int, list[store.Casting]] = {}
    for casting in store.castings_of_production(connection, production.id):
        by_shot.setdefault(casting.shot_id, []).append(casting)
    latest: dict[str, int | None] = {}
    looks: dict[int, Look] = {}
    placed: dict[int, dict[str, Placed]] = {}
    order: list[int] = []
    for chapter in store.chapters_of(connection, production.id):
        shots = store.shots_of(connection, chapter.id)
        first = store.keyframe_of(connection, shots[0].id).provided if shots else None
        opening: dict[str, int | None] = {}
        in_opening = True
        for shot in shots:
            cut = str(shot.plan.get("cut", ""))
            if shot.position and first and store.keyframe_of(connection, shot.id).provided == first:
                cut = "back"
            if shot.position and (cut == "new" or shot.scene != shots[0].scene):
                in_opening = False
            order.append(shot.id)
            here: dict[str, Placed] = {}
            for casting in by_shot.get(shot.id, []):
                member = slugs[casting.member_id]
                before = latest.get(member)
                back = False
                parent = casting.parent
                if parent == store.BASE:
                    source = None
                # A parent naming a look that is gone, or another member's, counts as none.
                elif parent and parent.isdigit() and int(parent) in looks and looks[int(parent)].member == member:
                    source = int(parent)
                elif cut == "back" and shot.position and member in opening:
                    source, back = opening[member], True
                else:
                    source = before
                look = source
                if casting.amended:
                    words = looks[source].words if source is not None else ()
                    looks[casting.id] = Look(member, casting, source, (*words, " ".join(casting.amend.split())))
                    look = casting.id
                here[member] = Placed(casting, shot.id, member, source, look, before, back)
                latest[member] = look
                if in_opening:
                    opening[member] = look
            placed[shot.id] = here
    for index, shot_id in enumerate(order):
        for member, one in placed[shot_id].items():
            if one.casting.amended:
                later = tuple(s for s in order[index + 1 :] if (p := placed[s].get(member)) and p.look == one.look)
                placed[shot_id][member] = dataclasses.replace(one, carried_into=later)
    return Resolved(tuple(order), placed, looks, _gaps(order, placed))


def moved(before: Resolved, after: Resolved) -> set[int]:
    """Every shot where a member starts from or shows a different look, or one whose words changed."""

    def shown(resolved: Resolved, shot: int) -> dict[str, tuple[Any, ...]]:
        return {
            member: tuple((look, resolved.look(member, look).words) for look in (one.source, one.look))
            for member, one in resolved.placed.get(shot, {}).items()
        }

    return {shot for shot in {*before.placed, *after.placed} if shown(before, shot) != shown(after, shot)}


def _gaps(order: list[int], placed: dict[int, dict[str, Placed]]) -> dict[str, tuple[tuple[int, ...], ...]]:
    seen: dict[str, list[int]] = {}
    for index, shot_id in enumerate(order):
        for member in placed[shot_id]:
            seen.setdefault(member, []).append(index)
    return {
        member: tuple(tuple(order[a + 1 : b]) for a, b in itertools.pairwise(at) if b - a > 1)
        for member, at in seen.items()
    }


# --- drawing a look ------------------------------------------------------------------------------
def base_picture(connection: Any, member: store.Member) -> store.Reference | None:
    """What a look is drawn from when it starts from the bible: a body before a face, a cut before a drawing."""
    own = [r for r in store.refs_of(connection, member.id) if r.artifact]
    own.sort(key=lambda r: (r.role == "face", r.generated, -r.quality))
    return own[0] if own else None


def face_of(connection: Any, member: store.Member) -> store.Reference | None:
    """A person's best face, handed to every look of it so a chain of edits keeps the face it began with."""
    if member.kind != "person":
        return None
    faces = [r for r in store.refs_of(connection, member.id) if r.artifact and r.role == "face"]
    return max(faces, key=lambda r: r.quality) if faces else None


def draw(production_id: int, casting_ids: list[int], asked_by: str = "person") -> str:
    """A run drawing and judging these looks in film order, one whose source is in the run drawn from its job."""
    with store.connect() as connection, store.transaction(connection):
        production = store.production(connection, production_id)
        if production is None:
            raise LookupError(f"no production {production_id}")
        scope = SCOPE.format(production=production_id)
        busy = next((r for r in store.runs(connection) if r.scope == scope and not r.settled), None)
        if busy is not None:
            raise ValueError(f"looks are being drawn already, in run {busy.id}")
        resolved = resolve(connection, production)
        unknown = [c for c in casting_ids if c not in resolved.looks]
        if unknown:
            raise LookupError(f"casting {unknown[0]} makes no look of its own: give it an amendment first")
        chosen = [c for c in resolved.looks if c in casting_ids]
        if not chosen:
            raise ValueError("no look to draw: each has its picture, or is being drawn")
        members = {m.slug: m for m in store.members_of(connection, production_id)}
        editor = drawn.editor_for(production_id)
        verdicts, _ = settings.effective("judge", production_id)
        assert isinstance(verdicts, JudgeSettings)
        run_id = store.next_run_id(connection, datetime.now().strftime("%Y-%m-%d-%H%M%S"))
        store.add_run(
            connection, run_id, title=f"{production.title} · looks", asked_by=asked_by, scope=scope,
            production_id=production_id,
            spec={"production": production_id, "items": [f"{resolved.looks[c].member} · {resolved.looks[c].label}"
                                                          for c in chosen]},
        )  # fmt: skip
        jobs: dict[int, tuple[int, list[int]]] = {}
        for casting_id in chosen:
            look = resolved.looks[casting_id]
            assert look.casting is not None
            member = members[look.member]
            source: Any
            if look.source is None:
                base = base_picture(connection, member)
                if base is None:
                    raise ValueError(f"{member.slug} has no picture to draw “{look.label}” from: give or draw one")
                source, size = member.picture(base), base.size
            elif look.source in jobs:
                job_id, size = jobs[look.source]
                source = {"from_job": job_id, "output": "frame"}
            else:
                parent = resolved.looks[look.source].casting
                assert parent is not None
                if not parent.artifact:
                    raise ValueError(
                        f"“{look.label}” starts from “{resolved.looks[look.source].label}”: draw that first"
                    )
                source, size = parent.artifact, parent.size
            face = face_of(connection, member)
            seed = random.randrange(2**32) if look.casting.artifact else (look.casting.seed or editor.seed)
            store.update_casting(connection, casting_id, seed=seed)
            job = store.add_job(
                connection, run_id, "draw_look",
                {"production": production_id, "casting": casting_id, "source": source,
                 "face": member.picture(face) if face else None, "kind": member.kind, "noun": member.noun,
                 "amend": " ".join(look.casting.amend.split()), "size": list(size), "seed": seed,
                 **({"allowed": member.allowed} if member.allowed.strip() else {})},
                label=f"draw {member.slug} · {look.label}", settings=editor.data(), for_kind="casting",
                for_id=casting_id,
            )  # fmt: skip
            jobs[casting_id] = (job.id, list(size))
            store.set_made(connection, "casting", casting_id, store.QUEUED, current_job=job.id)
            base = base_picture(connection, member)
            store.add_job(
                connection, run_id, "judge",
                {"production": production_id, "kind": "look picture",
                 "path": {"from_job": job.id, "output": "frame"},
                 "purpose": f"a reference picture of the {member.noun} as in its own reference photo, changed only "
                            f"in this: {look.text}",
                 "arrangement": f"{{{member.slug}}} alone", "cast": [member.slug],
                 "pictures": {member.slug: member.picture(base)} if base is not None else {},
                 "looks": {member.slug: look.text}},
                label=f"judge {member.slug} · {look.label}", settings=verdicts.data(), for_kind="casting",
                for_id=casting_id,
            )  # fmt: skip
    return run_id
