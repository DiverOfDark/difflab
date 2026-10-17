# PRD: Manga to anime video

## Summary

Add a fourth way to start a production: **from manga pages**. A chapter of English (already translated)
manga goes in. An anime-style video comes out, with every panel as a keyframe, every speech balloon
spoken, and the joins between panels animated by MiniMax H3.

The work reuses the director pipeline end to end (survey → ground → cut_crops → cards → keyframes →
render → assemble). What is new: reading pages into panels and a script, redrawing panels as anime
keyframes, building fixed shots from panels at ingest, and a smaller director pass that fills shots it
is given instead of planning a timeline.

## Decisions already made

These are settled. Do not reopen them.

| Question | Decision |
|---|---|
| Target look | **Anime**: coloured, cel-shaded. Not black-and-white manga, not live action. |
| How faithful | **Every panel is a keyframe.** One shot per panel, in reading order. The director never adds, drops, merges or reorders panels. |
| Language | **English.** The input is already-translated manga. No translation step. Balloon text is spoken word for word. |

## Goals

1. A user drops in the pages of one manga chapter and gets a production in which each panel is a shot
   with an anime keyframe, the dialogue attributed and timed, and a landing-or-cut choice between each
   pair of panels.
2. The production renders and assembles through the existing Render and Assemble stages with no special
   handling.
3. Every stage keeps the repo's cache contract: a re-run with nothing changed makes nothing, and editing
   one panel's script or keyframe re-renders that shot and the shots chained after it, nothing more.
4. Every model output goes through a gate (`direct.ask_json`) or a judge, and anything that failed shows
   in the UI as needing review rather than failing silently.

## Non-goals

- Translating text, or any language other than English.
- Left-to-right Western comics as a separate feature. Reading direction is a setting, defaulting to
  right-to-left, and nothing more.
- ~~Webtoons (long vertical strips). Panel detection assumes pages.~~ Reversed 2026-10-17: the reading
  setting has a third value, `strip`, for a webtoon (see "As built").
- Lip-sync quality beyond what H3 gives today.
- Music or score.
- Telling different translations of the same manga apart.

## Background: what the codebase already gives you

Read these before designing anything. The plan is built around reusing them.

| Piece | Where | Why it matters here |
|---|---|---|
| Ways to start a production | `difflab/screenplay/ingest.py`: `from_story`, `from_video` | `from_video` already makes **fixed shots at ingest**, each opening on a given frame. `from_manga` follows that model, not `from_story`. |
| API endpoints for those | `difflab/api/app.py` (around lines 530 and 588) | Add a matching endpoint for manga. |
| Gated VLM calls | `difflab/tools/direct.py` `ask_json` | Every new VLM question goes through it: cached by exact question, repaired with the gate's own messages, settled after `DIRECTOR_REPAIRS`. |
| Cast from pictures | `tools/survey.py`, `tools/ground.py`, `tools/cut_crops.py`, `tools/cards.py` | These run on "photos". Hand them the cleaned panels and labels (`survey.Label(entity=...)`) from the page reader. |
| Keyframe kinds | `store/screenplay.py`: `provide_keyframe`, `compose_keyframe`, `chain_keyframe`, `refs_keyframe`, `land_keyframe` | Panels become `COMPOSED` keyframes. `landed` decides landing versus cut. |
| Landing on the next keyframe | `screenplay/compiler.py` `joins` | Shot N ends on shot N+1's keyframe when that keyframe is `COMPOSED` or `PROVIDED` and `landed` is true. This is what animates the gap between two panels. |
| Composed keyframes | `tools/compose_keyframe.py` | Redraws a base picture with each member passed in as "Picture 2 onward", plus an optional style picture. This is the anime redraw. |
| What a composed keyframe is drawn over | `screenplay/planner.py` `base()` | Today it is the place's reference or the chapter's opening frame. **It has to learn a per-shot base: the shot's own cleaned panel.** |
| Image editing | `tools/edit_image.py` (Qwen-Image Edit / Qwen-Image 2.1) | Balloon removal and the anime redraw. |
| Drawing from words | `tools/draw_reference.py` | Character colour sheets. |
| Judging | `tools/judge.py`, `screenplay/made.py` | Keyframe verdicts and the redo loop. |
| Writing a shot's prose | `tools/write_shot.py` | Unchanged. It runs after the new fill pass. |
| Steps on the ingesting page | `screenplay/planner.py` `STEPS`, `screenplay/ingesting.py` `ORDER`/`NAMES` | Add a "Read pages" step. |
| Shot length rules | `screenplay/config.py`: `DIRECTOR_CLIP_MIN = 6.0`, `DIRECTOR_WORDS_PER_SECOND = 2.5`, `DIRECTOR_DIALOGUE_TAIL = 2.0` | The 6 s minimum is too long for manga. See "Open questions". |
| H3 frame grid | `tools/plan_shots.py` `on_grid` | Every shot length snaps to 17n+5 frames. |

Repo conventions in `CLAUDE.md` apply, especially on comments: only what the code cannot say (measured
numbers, cache-key consequences, workarounds), docstrings one line or absent. Checks:
`pixi run ruff format && pixi run ruff check && pixi run mypy`.

## The pipeline

```
pages ──► read_page ──► panel script (per page JSON)
              │
              ├──► clean_panel (balloons and SFX removed) ──► survey / ground / cut_crops / cards
              │                                                   │
              │                                        colour sheet per character (approved)
              │                                                   │
              └──► from_manga: one shot per panel ─────► COMPOSED keyframe over its cleaned panel
                         │                                 (anime redraw, character sheets as donors)
                         │
                   fill_shots (director): beats, camera, dialogue timing per fixed shot
                         │
                   write_shot ─► compiler ─► render_shot (existing) ─► assemble (existing)
```

## Requirements

### R1. `read_page`: a page into panels and a script

A new tool in `difflab/tools/read_page.py`, one job per page.

**Panel detection is code, not the VLM.** Find gutters and borders: threshold, connected components,
bounding boxes. The VLM is asked only when code finds something it can't settle (borderless panels,
panels that overlap or break the frame), and then only to confirm or correct boxes.

**Reading order is code.** Rows top to bottom, panels within a row right to left. A per-production
setting `reading` (`rtl` default, `ltr`).

**One gated VLM call per panel** via `direct.ask_json`, shown the panel crop (and the whole page at low
resolution for context). It answers:

```json
{
  "balloons": [{"kind": "speech|thought|narration|shout", "speaker": "<member id or ''>", "text": "..."}],
  "sfx": ["..."],
  "who": ["<member id>", "..."],
  "action": "one sentence: what happens in this panel",
  "into_next": "land|cut",
  "why": "one short phrase"
}
```

- `balloons` in reading order within the panel. `text` exactly as lettered: case and punctuation
  normalised, nothing reworded.
- `into_next`: `land` when the next panel continues the same action in roughly the same framing;
  `cut` when it is a new angle, a reaction close-up, a new place, or a time skip. The last panel of a
  page is judged against the first panel of the next page.
- The speaker ids come from the cast the survey found. On the first pass, before the survey, the reader
  may give descriptive names; reconcile them after the survey with one text-only call per chapter.

**Gate:** every speaker is a known id or empty for narration; `who` contains every speaker of a speech
or thought balloon; `into_next` is one of the two values; no balloon is empty.

**Output** stored per page: panel boxes (pixels on the page), the panel crops as files under `output/`,
and the script above. Cache key: the page's pixels, the reading setting, and the question text, so a
page is read once.

### R2. `clean_panel`: balloons and text removed

A new tool on `edit_image`, one job per panel. Removes balloons, sound effects and narration boxes,
filling what was behind them. Panel boxes from R1 are the crop; balloon positions, if the reader gives
them, may be passed as a hint.

Judged with the existing judge: a cleaned panel that still shows lettering, or whose figures changed,
is flagged for review. Cache key: the panel's pixels and the edit settings.

### R3. A bible from panels

- Run `survey`, `ground`, `cut_crops` and `cards` on the **cleaned** panels as the photos, with labels
  from R1's speakers so the survey doesn't have to guess identities.
- One **colour reference sheet per character**, made with `draw_reference` (or `edit_image` over their
  best crop): anime style, front view, full colour. It is shown on the Bible stage and must be
  approved before any keyframe uses it. Colours are decided **once**, here; keyframes take them from
  the sheet, not from the editor's memory.
- A `style` member for the anime look, from a style picture the user supplies (an original picture,
  not frames from an existing show).

### R4. Anime keyframes

Each shot's keyframe is `COMPOSED`, drawn by `compose_keyframe` with:

- **Base:** the shot's own cleaned panel (R2), extended to the render aspect ratio.
- **Donors:** the colour sheet of each character in the panel's `who`, as Picture 2 onward, then the
  style picture.
- **Instruction:** keep the composition, poses and line work; redraw as coloured anime cel shading; each
  character as their Picture. Put the words in `screenplay/words.py` next to the existing `COMPOSE_*`
  sentences.

`planner.base()` must return the shot's own panel for a manga shot (for example from
`shot.plan["panel"]`) before it falls back to the place reference. A panel with no character in it
(scenery, an object) is redrawn with the style picture alone.

The existing keyframe judge and redo (`production.redo_keyframe`) apply as they are.

### R5. `ingest.from_manga`: fixed shots from panels

A new function next to `from_story` and `from_video`, and an endpoint in `api/app.py`.

- Input: page images (a folder, or a CBZ unpacked under `output/`), a title, an optional style picture,
  the reading direction, and the usual video and continuity settings.
- Pages are grouped into chapters by the user (one chapter by default).
- One shot per panel, in reading order, with:
  - `plan["panel"]`: the panel crop and the cleaned panel's paths, the page, and the box;
  - `plan["cast"]`: the panel's `who`;
  - a `COMPOSED` keyframe (R4);
  - `landed` from the **previous** panel's `into_next` (`land_keyframe(shot, into_next == "land")`);
  - a length from R6.
- `story_kind = "manga"`; the chapter's story text is the panels' `action` sentences and their
  dialogue, so the existing chapter view and the survey have text to read.
- A "Read pages" step in `planner.STEPS` and `ingesting.ORDER`/`NAMES`, showing pages read, panels
  found, and balloons per panel.

### R6. Shot length

Per panel: `max(floor, spoken_words / DIRECTOR_WORDS_PER_SECOND + DIRECTOR_DIALOGUE_TAIL)`, then snapped
with `on_grid`, capped at the render length. `floor` is a new knob, `MANGA_SHOT_MIN`, for silent
panels. A panel whose dialogue cannot fit in the longest shot is flagged for review (split the balloons
across a held shot of the same panel; that is a later improvement, not this PRD).

Manga productions use `MANGA_SHOT_MIN` in place of `DIRECTOR_CLIP_MIN`. Its value comes from the spike
(Phase 0), not a guess.

### R7. `fill_shots`: the director fills the shots it is given

A variant of `plan_shots` in `difflab/tools/fill_shots.py`, one call per chapter like `plan_shots`, so
the model sees every join. The number of clips, their keyframes, their cast and their length are
**given**, not chosen. The model writes, per clip:

- `beats` (start, end, phase, who, action), the motion that takes panel N toward panel N+1 when landed;
- `camera` (`move`, `toward`), from `words.CAMERA_MOVES`;
- `dialogue`: each balloon of the panel as a `compose.Line` with a `start` time and a `voice` on first
  appearance; narration as a line with `who` set to a narrator member, or left to subtitles only
  (see "Open questions");
- `closing`, `mouths`, `irreversible_change`, `ambience`, as `plan_shots` writes them.

**Gates** (new rules next to `checks.plan_rules`, plus the existing `ruling.mouths` and
`ruling.leaving` where they apply):

1. Every balloon of panel N is a dialogue line of shot N, **word for word**, once, in reading order.
   No line appears in another shot. No invented lines.
2. A shot whose next keyframe is landed has a `closing` that describes the next panel's `action` and
   cast (checked against R1's words with the existing token matching in `screenplay/checks.py`).
3. Dialogue plus `DIRECTOR_DIALOGUE_TAIL` fits inside the shot.
4. Beats cover the shot without gaps and stay on its given length.
5. The cast of every beat is a subset of the shot's given cast.

The result is turned into a `compose.Take` exactly as `plan_shots._take_from_plan` does, so
`write_shot` and the compiler run unchanged.

### R8. Fallback: motion-comic shot

A shot that has failed its render judge after the automatic retries can be switched to a
**motion-comic shot**: a slow pan and zoom over the keyframe, with the shot's dialogue audio (H3 audio
from a still, or text-to-speech if that's simpler) and subtitles. No DiT. A new keyframe-and-render
mode on the shot, chosen by the user in the Shots stage. It exists so one impossible panel (crowds,
abstract splash pages) never blocks a chapter.

### R9. Frontend

- A "From manga" option on the new-production form in `frontend/src/production.ts`: drop a folder or
  CBZ, the title, the style picture, the reading direction.
- The Shots stage shows each shot's original panel next to its anime keyframe, the balloons, and the
  land/cut toggle (which already maps to `landed`).
- A panel's script (balloons, speakers, `into_next`) is editable. An edit makes that shot's prompt
  stale through the existing staleness mechanism.

## Phases and acceptance criteria

Each phase ends with something the user can look at. Don't start a phase before the one before it is
accepted.

### Phase 0. Spike, no new code

Use 1 page (6–8 panels) from the user's sample. Clean and redraw each panel by hand with the existing
edit tool. Make a `from_story` production with those as provided keyframes, landing on for continuing
panels and off for cuts, dialogue typed in. Render it.

**Report**, with the clips:

- Does H3 hold each character's anime face across chained shots?
- Do landed shots look like motion or like a morph?
- The shortest landed shot that still looks right. This sets `MANGA_SHOT_MIN`.
- Do colours stay put between keyframes when each is redrawn separately?

**Stop and report to the user if** identities drift badly or landings morph. The rest of the plan may
need to change (for example, fewer landings, or an anime LoRA for H3 through Settings).

### Phase 1. `read_page` (R1)

**Accepted when**, on 3–4 pages the user has marked up by hand: every panel found with the right
boxes; reading order matches; at least 95 % of balloons read exactly; speakers right for at least 90 %
of balloons; `into_next` agrees with the user for at least 80 % of joins.

### Phase 2. `clean_panel` and the bible (R2, R3)

**Accepted when** cleaned panels show no lettering and unchanged figures on the sample pages, and every
character has an approved colour sheet.

### Phase 3. Anime keyframes and `from_manga` (R4, R5, R6)

**Accepted when** a sample chapter ingests into one shot per panel, each with an anime keyframe that
keeps the panel's composition and the characters' sheet colours, and `landed` set from the reader.
Re-ingesting the same pages makes nothing new (check the cache).

### Phase 4. `fill_shots` (R7)

**Accepted when** the sample chapter plans with all gates passing or settled, every balloon spoken
word for word in its own shot, and it renders and assembles through the existing stages.

### Phase 5. Fallback and frontend (R8, R9)

**Accepted when** the user can start from a CBZ in the app, review panels against keyframes, toggle
land/cut, edit a balloon, and switch a failed shot to a motion-comic shot.

## Open questions

These can be decided during the phase that needs them. Ask the user rather than guessing.

1. **`MANGA_SHOT_MIN`**: from Phase 0. Expected around 2–3 s.
2. **Narration boxes**: a narrator voice, or subtitles only? Subtitles only is the simpler default.
3. **Sound effects**: only into the shot's `sounds`, or also shown on screen as stylised text? Default:
   sounds only.
4. **A panel too long to speak in one shot**: split into two shots on the same panel, or let the user
   trim? Default: flag for review.

## Risks

| Risk | Mitigation |
|---|---|
| H3 drifts on anime faces across chained shots | Phase 0 finds out first. Options after: an anime LoRA in Settings, fewer landings, more references per shot. |
| Colours drift between keyframes | One approved colour sheet per character, passed into every redraw. Judge flags drift. |
| Landing between panels from different angles looks like a morph | `into_next` per join; the user can flip it in the UI. |
| The VLM misreads stylised lettering or the speaker | Gates, per-panel review on the Shots stage, editable script. |
| Panel detection fails on unusual layouts | VLM confirms only what code could not settle; boxes editable as a last resort. |
| Rights | Only run this on manga the user has the right to adapt. Test data comes from the user. |

## Test data

The user supplies 3–5 pages of the translated manga they plan to use, placed under `output/`, plus a
hand-made answer key for Phase 1 (panel boxes, reading order, balloons with speakers, land/cut per
join). Don't commit the pages to the repository.

## As built (2026-10-17)

Where the implementation departs from the text above, and why.

- **Webtoons are in.** `reading = "strip"` cuts a vertical strip on full-width bands of flat background;
  a band that is mostly background is lettering, read with the picture before it and drawn with none. A
  band with a lot of background around its picture is tightened by the VLM. A coloured source (measured at
  ingest, `manifest.coloured`) keeps its own colours on the colour sheets. Strips default to a 9:16 canvas.
- **Shots are laid out by the first plan run, not at ingest.** Panel boxes are code, but unsure pages are
  confirmed by the VLM and `landed` and the length need the reading, so `lay_panels` makes one shot per panel
  after `read_page`, keeping a panel's row, keyframe and hand edits across re-plans by its crop's path.
- **The fill never retypes a balloon.** `fill_shots` times each spoken balloon by its number; the program puts
  the comic's words in, so gate 1 holds by construction.
- **Thoughts and narration are subtitles only** (`plan.captions`); speech and shouts are spoken. Sound effects
  stay in the script and are not voiced or drawn (open questions 2 and 3, the defaults).
- **A landing shot's cast includes the next panel's characters**, since its closing describes that panel and
  they come into frame during it.
- **H3's references for a manga shot are the colour sheets only**: the cut crops are of black-and-white
  drawing. A member with no sheet, and every place, is said from its words.
- **Motion-comic audio** is the soundtrack of the shot's last H3 render when it had one; there is no
  text-to-speech in the repo. Without one the shot is silent apart from its subtitles.
- **`MANGA_SHOT_MIN` is 3.04 s** (73 frames), the PRD's estimate, until the Phase 0 spike measures it.
