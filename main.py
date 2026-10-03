"""A playground: runnable experiments, one per function, called from the `__main__` block.

Run serves the app instead (`serve`, at the bottom), because a render belongs in a process of its own.
"""

from pathlib import Path

from difflab import config, prompts
from difflab.core import media as io
from difflab.core import ui
from difflab.core.settings import EditSettings
from difflab.models.minimax_h3 import prompts as video_prompts
from difflab.models.minimax_h3.settings import VideoSettings
from difflab.screenplay import production
from difflab.tools import (
    ask_vlm,
    edit_image,
    enlarge_latents,
    enlarge_pixels,
    grade,
    render_chunk,
    retime,
    write_prompt,
)
from difflab.work.worker import (  # noqa: F401  (render from here: `play_production(parity_chain())`)
    play,
    play_production,
)


# --- image edit ------------------------------------------------------------------------------
def manual_prompt() -> None:
    """Manual mode: skip the VLM, use a hand-written instruction."""
    frame = io.frame_from_image(config.INPUT_IMAGE)
    io.save(edit_image.edit(frame, prompts.SAMPLE_PROMPT), "edit", EditSettings().seed)


def full_workflow() -> None:
    """The original graph: the VLM turns the rough idea into an instruction, then the edit runs."""
    frame = io.frame_from_image(config.INPUT_IMAGE)
    instruction = write_prompt.write_edit_instruction(frame, prompts.SAMPLE_IDEA)
    io.save(edit_image.edit(frame, instruction), "edit", EditSettings().seed)


def two_seeds() -> None:
    """One instruction, two renders: the tools make the middle of a workflow reusable."""
    frame = io.frame_from_image(config.INPUT_IMAGE)
    instruction = write_prompt.write_edit_instruction(frame, prompts.SAMPLE_IDEA)
    for seed in (EditSettings().seed, EditSettings().seed + 1):
        out = edit_image.edit(frame, instruction, EditSettings(seed=seed))
        io.save(out, "edit", seed)


def ask_the_vlm() -> None:
    """Ask the VLM about the image (answers stream into panels)."""
    frame = io.frame_from_image(config.INPUT_IMAGE)
    ask_vlm.ask(frame, "Describe each person: position, clothing and colors, hair length and color.", "People")


# --- video -----------------------------------------------------------------------------------
def video_manual_prompt() -> None:
    """A finished H3 prompt -> MP4 with native audio. No VLM is loaded."""
    frame = io.frame_from_image(config.INPUT_IMAGE)
    io.save(render_chunk.render(frame, video_prompts.SAMPLE_PROMPT), "clip", VideoSettings().seed)


def video_full_workflow() -> None:
    """The original graph: the VLM writes the H3 prompt from a rough idea, then H3 renders it."""
    frame = io.frame_from_image(config.INPUT_IMAGE)
    prompt = write_prompt.write_video_prompt(frame, video_prompts.SAMPLE_IDEA)
    io.save(render_chunk.render(frame, prompt), "clip", VideoSettings().seed)


def video_full_length() -> None:
    """15 s at 0.8 MP on the 4-step preset: the geometry ComfyUI was benchmarked at."""
    frame = io.frame_from_image(config.INPUT_IMAGE)
    clip = render_chunk.render(frame, video_prompts.SAMPLE_PROMPT, VideoSettings.full_length())
    io.save(clip, "clip", VideoSettings().seed)


# --- piping models together ------------------------------------------------------------------
def edit_then_animate() -> None:
    """Qwen Edit and H3 in one pass: nothing round-trips through disk, and the registry evicts the editor
    before the DiT loads."""
    frame = io.frame_from_image(config.INPUT_IMAGE)
    restyled = edit_image.edit(frame, "make it golden hour, warm low sun from the left")
    prompt = write_prompt.write_video_prompt(restyled, "she turns toward the camera and smiles")
    clip = render_chunk.render(restyled, prompt)
    io.save(clip, "edited_then_animated", VideoSettings().seed)


def reference_to_video() -> None:
    """ref2va: a reference is not a frame of the clip but sits in the model's past, so it steers identity
    without deciding where the shot starts."""
    subject = io.frame_from_image(config.INPUT_IMAGE)
    clip = render_chunk.render(subject, video_prompts.SAMPLE_PROMPT, references=[subject])
    io.save(clip, "from_reference", VideoSettings().seed)


def video_from_reference_clip() -> None:
    """ref2va with motion. The DiT gets every latent frame the VAE made and those rows ride through every
    step, so a reference clip costs real time - keep it short."""
    scene = io.frame_from_image(config.INPUT_IMAGE)
    reference = io.clip_from_video(config.INPUT_VIDEO, limit=56)
    clip = render_chunk.render(None, video_prompts.SAMPLE_PROMPT, reference_clips=[reference], canvas_from=scene)
    io.save(clip, "from_reference_clip", VideoSettings().seed)


def generate_small_upscale_large() -> None:
    """Render at 0.4 MP and enlarge the latent to 2 MP before it is ever decoded, so H3's own VAE draws the
    extra pixels - on a tensor 256 times smaller than the frames it becomes."""
    frame = io.frame_from_image(config.INPUT_IMAGE)
    settings = VideoSettings(duration=2.0, megapixels=0.4)
    geometry = render_chunk.plan(frame, settings)

    first_latent, canvas = render_chunk.condition(frame, settings, geometry)
    conditioning = render_chunk.encode(video_prompts.SAMPLE_PROMPT, canvas, settings=settings)
    latents = render_chunk.sample(conditioning, first_latent, geometry, settings, video_prompts.SAMPLE_PROMPT)

    clip = enlarge_latents.upscaled(2.0)(latents)  # <- the explicit step: enlarge, then decode
    io.save(clip, "small_then_upscaled", VideoSettings().seed)


def retime_an_existing_clip() -> None:
    """24 -> 60 fps on a finished clip: the same seconds and the same soundtrack, moving more smoothly."""
    clip = io.clip_from_video(config.INPUT_VIDEO)
    io.save(retime.retime(clip, fps=60), "retimed")


def grade_a_long_take() -> None:
    """A LUT over a finished take, against the colour a chain inherits from itself at every seam. The cause
    is `Continuity.color_anchor`, which fixes it upstream at the price of re-rendering."""
    graded = grade.grade(Path("output/videos/.../take.mp4"), strength=0.5)
    print(graded)


def upscale_an_existing_clip() -> None:
    """Enlarge a finished mp4. Pixel-space, so it can only sharpen what the decoder already drew - the
    weaker route, and the only one left once the latents are gone."""
    clip = io.clip_from_video(config.INPUT_VIDEO)
    bigger = enlarge_pixels.upscale(clip, megapixels=2.0)
    io.save(bigger, "upscaled")


# --- productions -----------------------------------------------------------------------------------
# A production is data in the store: shots with a pinned prompt each, chained on one provided keyframe.
# `play_production` plans its chapter into jobs and plays them here; the page renders the same rows.
def parity_chain() -> int:
    """Three hand-written shots on one keyframe at the dev settings: the smallest production there is.
    Rendering it twice samples nothing; a re-roll of shot 2 re-renders shots 2 and 3 and nothing else."""
    made = production.create("parity", "parity", VideoSettings(duration=4, megapixels=0.4, seed=5, steps=8))
    light = "Soft window light, still camera, natural motion."
    production.add_shot(
        made.id,
        f"A young woman in a sunlit room turns slowly toward the camera and smiles. {light}",
        4,
        keyframe="assets/parity_opening.png",
    )
    production.add_shot(made.id, f"She tilts her head and laughs quietly, glancing to the side. {light}", 4)
    production.add_shot(made.id, f"She looks back at the camera, calm, and slowly closes her eyes. {light}", 4)
    return made.id


def replay_rulings() -> None:
    """The 27B's rulings over every stored plan. Loads the VLM: not beside a render."""
    import json

    from difflab import store
    from difflab.screenplay import compiler
    from difflab.tools import ruling

    with store.connect() as connection:
        jobs = connection.execute(
            "SELECT inputs FROM jobs WHERE tool = 'plan_shots' AND state = 'done' ORDER BY id DESC"
        )
        ideas: dict[tuple[int, int], str] = {}
        for row in jobs:
            asked = json.loads(row["inputs"])
            ideas.setdefault((int(asked["production"]), int(asked["chapter"])), str(asked["idea"]))
        takes = {}
        for (production_id, chapter_id), idea in ideas.items():
            found = store.production(connection, production_id)
            chapter = next((c for c in store.chapters_of(connection, production_id) if c.id == chapter_id), None)
            if found is not None and chapter is not None:
                takes[production_id, chapter_id] = (idea, compiler.take_of(connection, found, chapter))
    for (production_id, chapter_id), (idea, take) in takes.items():
        problems = ruling.story(idea, list(take.story), take) + ruling.mouths(take)
        for chunk in take.chunks:
            prose = [beat.prose for beat in compiler.written(chunk) if beat.prose]
            if prose:
                problems += ruling.voices([*prose, chunk.summary], chunk.sounds, chunk.mouths)
        rows = {f"{problem.code} {i}": problem.message for i, problem in enumerate(problems)} or {"rulings": "nothing"}
        ui.panel(f"production {production_id} · chapter {chapter_id}", rows, color="yellow" if problems else "cyan")


def serve() -> None:
    """No GPU. What Run does, and what a pod runs (`pixi run python -m difflab.api`): it holds the port, so
    every render started anywhere else records into this page rather than serving one of its own."""
    from difflab.api import serve
    from difflab.core import events

    events.disable()
    serve.serve_forever()


def describe_settings() -> None:
    """No GPU: every knob and model file is a settings object you pass in."""
    ui.panel(
        "settings",
        {
            "edit": str(EditSettings()),
            "video (dev)": str(VideoSettings()),
            "video (full)": str(VideoSettings.full_length()),
            "vlm (video)": str(write_prompt.VIDEO_VLM),
        },
    )


if __name__ == "__main__":
    # Swap in `play_production(parity_chain())` or any function above to render straight from here.
    serve()
