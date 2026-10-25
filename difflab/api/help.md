# What difflab is
Pictures, clips and whole films on one GPU, and where each part lives.

difflab makes pictures and video clips with audio from photos and words, and builds films shot by shot.
Everything renders on one card behind one queue: a job waits its turn, and nothing is refused for being busy.

The page has six tabs:
- **Take**: one-off jobs from a form. A picture, one clip, or an upscale of a finished clip.
- **Production**: films. A story and photos go in; five stages (Ingest, Bible, Shots, Render, Assemble) bring them out as a film.
- **Chat**: talk to the VLM, which makes pictures and clips, finds references on the web and works on productions.
- **Gallery**: every finished picture and video.
- **Runs**: every job, queued, live or ended. A live render shows a preview after each sampling step.
- **Settings**: models, LoRAs and knobs, for this machine or one production.

Over MCP, `make_picture`, `make_clip` and `upscale` are the Take tab; `start_production`, `production` and the tools
after them are the Production tab. Every tool that queues work returns a run id: follow it with `run`.

# Make a picture from words
Draw a fresh picture from a description, with optional reference photos.

In the page:
1. Open **Take**, pick mode **picture**.
2. Leave every photo unmarked as start (or drop none). Photos you drop are references.
3. Pick model **Qwen 2.1**, the experimental engine that draws without a start photo (about 30 s).
4. Under **Words**, choose **IDEA** to have the VLM write the prompt, or **PROMPT** to give it exactly.
5. Optional: set the aspect ("3:2") in the settings groups on the right. Press the render button.

Over MCP: `make_picture` with `idea` or `prompt`, `engine: "qwen-image-2.1"`, optional `photos` and `aspect`.
Then `run` with `wait_seconds` until it is done; the picture comes back inline.

# Edit a photo
Change something in a photo while keeping the rest.

In the page:
1. **Take** → **picture**. Drop the photo and mark it the start; any photos after it are references (a garment, a prop).
2. Model **Qwen Edit** (the default editor) or **Qwen 2.1**. **MiniMax H3** plays the edit out as a short clip and keeps its last frame, slower (about 2 min) but good at motion and pose changes.
3. Under **Words**, IDEA ("make it night, keep her pose") or PROMPT (the edit instruction as written).
4. Render. Change the seed in the settings to get another attempt at the same words.

Over MCP: `upload` the photo, then `make_picture` with `photos: [path, ...refs]`, `start: true`, `idea` or `prompt`,
and `engine`. A new seed: `settings: {"edit": {"seed": 7}}`.

# Make a clip from a photo
One video clip with audio that opens on a photo.

In the page:
1. **Take** → mode **video**. Drop the photo the clip opens on; it must be first.
2. Model **MiniMax H3** (default) or **LTX**.
3. **IDEA** ("she turns and laughs, waves rustle") lets the VLM write the caption; **PROMPT** is used as written.
4. Duration is in the **Video** settings group. Render.

Over MCP: `make_clip` with `photos: [opening, ...]`, `idea` or `prompt`, optional `engine` and `duration`.
`run` with `wait_seconds` follows it; the result is a video URL.

# Use reference photos in a clip
Keep a person, garment or prop looking right in a clip.

Every photo after the first is a reference, in the order given.
- **MiniMax H3** steers by them as references in its own caption format.
- **LTX** places up to 5 references after the opening photo on 2.5 (called Image 1…), 4 on 2.3 (Figure 1…).
An IDEA is written to name them; with PROMPT, refer to them yourself in the model's format.

Over MCP: `make_clip` with `photos: [opening, ref1, ref2]` and `engine: "ltx-2.5"` for LTX's reference slots.

# Upscale, grade or retime a clip
Enlarge a finished clip, pull its colour back, or deliver it at a higher frame rate.

In the page: **Take** → **upscale**, give the clip (an mp4 under output/ or its full path) and the finish size.
The **Delivery** group has the colour grade (0–1: how far the clip is pulled back to its opening colour;
0.5 closes half the gap) and the frame rate (above 24, by interpolation). A video take can carry the same
delivery settings, so it is graded and retimed as it finishes.

Over MCP: `upscale` with `clip` and `finish_megapixels`, optional `grade_strength` and `retime_fps`. `make_clip`
accepts the same three.

# Make many takes at once
The same words and settings over many photos, one run each.

In the page: in **References**, switch **ONE** to **BATCH** and drop the photos; each opens a take of its own.
Photos under "shared references" are handed to every take after its own photo. For upscale, drop mp4s instead.

Over MCP: there is no batch tool; call `make_picture` or `make_clip` once per photo. Each call queues its own
run, and they render in the order queued.

# Follow, cancel or find results
Where a job is, how to stop it, and where its files are.

In the page:
- **Runs** lists every job with its state. Open one to watch it live; **Cancel run** stops it.
- A finished run shows what it made and can send its job back to the Take form to run again.
- **Gallery** has every finished picture and video, for download or deletion.

Over MCP: `list_runs`, `run` (with `wait_seconds` to block until it settles, at most 600 s; call again if
it is still going), `cancel_run`. Results come back as URLs under /output/.

# Start a film
A production from an idea or a story and photos, planned shot by shot.

In the page:
1. **Production** → **+ new production**. Story kind **STORY**.
2. Title, the story (a rough idea for one chapter, or a text with headings that become chapters), the target length,
   the shot length range (6–15 s), aspect, fit and the model (MiniMax H3, or LTX with up to 5 references a shot).
3. Drop the references and give each a kind (character, location, prop, style, or video for motion) and who it shows.
4. For a rough idea, **Improve with director** rewrites it into a story the planner renders well; take a round or ask again with feedback.
5. Tick "run all plan stages, stop before render" and press **Ingest and plan**, or leave it unticked to stop at the bible for review.

Over MCP:
1. `upload` each photo; optionally `guess_references` to have the VLM say what each shows.
2. Optionally `draft_story` to turn an idea into a story; its text is in the run's result.
3. `start_production` with `title`, `story`, `references`, `target_seconds`, optional `shortest`/`longest`, and `plan: true`.
4. `run` on the returned run ids, then `production` to see each stage and what needs review.

# Redo a video with one change
Re-create a video shot by shot with one difference applied.

In the page: **Production** → **+ new production** → **REDO VIDEO**. Give the source video and **WHAT CHANGES**
("make it look like a watercolour"), then **Begin**. Each shot opens on the source's frame and re-creates that stretch.

Over MCP: `upload` the video, then `start_video_production` with `title`, `source` and `instruction`.

# Review the bible
Check the cast's standing look, and approve it, before shots are planned on it.

The bible is the standing look of every character, place, prop and style: what a shot starts from the first time
a member appears. Its review count is pictures not yet approved, open conflicts, and photos placed with no one.

In the page (**Bible** stage):
- Open a member with **edit** to change its words; a member edited by hand is pinned, and a re-plan leaves it as written.
- A **conflict** is the story and a photo disagreeing: keep the photo's version, keep the story's (**keep novel**), or write the resolution.
- A member no photo shows is drawn from its words; **generate 3 views** draws variants to pick from, and **describe again from photos** re-reads it.
- **+ add a member in words** adds one the director missed.
- **Approve bible** approves every made picture at once.

Over MCP: `production` lists the review items. `member` shows one with its pictures. Then `edit_member`,
`keep_conflict` or `resolve_conflict`, `draw_member`, `draw_member_variants` + `pick_member_variant`,
`give_member_picture` (a photo you have), `add_member`, `attach_photo` for unmatched photos, and `approve` with
`what: "bible"`.

# Fix a shot
A shot does the wrong thing, says the wrong words, or opens on the wrong frame.

In the page (**Shots** stage), open the shot on the timeline:
- **ACTION**, **DIALOGUE**, **CAMERA**, **SECONDS**, **SOUND** edit the plan; an edit pins the shot and its prompt is compiled again.
- **OPENING**: open on a keyframe, on the shot before's last frame (chained), or on the references alone.
- The keyframe can be composed again; "add the judge's issues" passes the judge's complaints to the editor.
- **rewrite** writes the shot's prose and prompt again; **H3 PROMPT** can be written by hand.
- **ADD FROM BIBLE** casts another member; a cast member's look can be amended for this shot (wet clothes, a torn sleeve) and drawn.
- **Approve · next** approves the shot and moves on; **Unpin** gives it back to the director.

Over MCP: `shot` for the full shot with pictures. Then `edit_shot` (`changes` with action, dialogue, seconds, opening…),
`rewrite_shot`, `redo_keyframe` with a `note`, `add_casting` / `edit_casting` / `remove_casting`, `draw_look`
or `draw_shot_looks`, and `approve` with `what: "shot"`.

# Why something is stale
What "stale" means and what to do about it.

Everything made is cached by what it was made from. When an input changes (the story, a member's look, a shot's
plan, a setting), what was made from it goes **stale** and is made again the next time it is needed. Edits answer
with what they made stale. Pinned shots and hand-written prompts stay as they are through a re-plan.

In the page: the STALE banner names what moved; **re-plan** the chapter to bring the shots up to date.
**Re-plan fresh** asks the director for new shots even when nothing changed; a plain re-plan with nothing
changed comes straight from the cache.

Over MCP: every edit tool returns `stale`. `production_view` with `view: "preflight"` says what would stop each
chapter rendering. `plan` re-plans (`fresh: true` for new shots).

# Render the film's shots
Turn a production's planned shots into clips, and deal with flagged ones.

In the page (**Render** stage):
1. **Dry run · CPU preflight** checks every chapter without queueing anything.
2. **Render** the chapter or the production. Each clip is judged; with "auto-retry below gate" a failing clip is retaken on its own.
3. **Pause after this clip** holds the queue; **Resume queue** goes on.
4. A flagged clip can be approved as it is, **re-roll**ed on a new seed (the shots chained after it follow), or sent **back to keyframe**.

Rendering is the expensive stage: `production_view` with `view: "timings"` has this machine's seconds per second of film.

Over MCP: `render` first answers with how many clips and roughly how long; call it again with `confirm: true`.
`production_view` `render_queue` shows progress and ETA. `steer_render` pauses or sets auto-retry. `reroll_shot`
(also gated by `confirm`) or `approve` with `what: "clip"` and the shot id settles a flagged clip.

# Assemble and finish the film
Join the clips into chapters and the film, then the 4K finish.

In the page (**Assemble** stage): **Assemble chapter** or **Assemble film** joins the clips there are, levels the
loudness, adds crossfades at scene changes, subtitles and chapter markers; nothing renders. **Preview** plays it.
**Finish** enlarges to 4K and retimes to 60 fps; it is the longest pass after render, so review the native cut first.

Over MCP: `assemble` (optional `chapter`), `production_view` with `view: "assembled"` for the results,
then `finish` with `confirm: true`.

# Change models, LoRAs and settings
Pick the checkpoints, add LoRAs, and set knobs for the machine or one production.

In the page: **Settings**. **This machine** changes the defaults every job starts from; **A production** overrides
them for one production only. Models and LoRAs come from Hugging Face or Civitai; the tokens for gated or private
files are saved on the same page. Each change shows what it would make stale before it is saved.

Over MCP: `take_settings` lists every field of the Take jobs with this machine's value; a single job can override
any of them through its `settings` argument. Saving machine or production settings is done in the page.

# Free disk space
See what takes space besides the models, and clear the caches.

In the page: **Settings**, then **Disk**. Each row is one kind of file with its size: **Models**, **Results**,
**Production shots**, **Step cache**, **VLM prompt cache**, **Logs and profiles**, **Database and run records**
and **Other**. **clear**, then **free … ?** to confirm, empties the step cache (renders nothing shown points at),
the VLM prompt cache, or the old logs. A clear waits for the job in flight and says **after this job** until then.
Models are deleted one by one under **Downloads**; results in the **Gallery**.

Over MCP: none; clearing is done in the page.

# Have the chat make pictures, clips and changes
Ask in the Chat tab in plain words, and the model does the work with difflab's own tools.

In the page:
1. Open **Chat**, then **New chat**. To work on a film, name it in your message, by its title or its number.
2. Ask: "make a picture of…", "a clip from this photo where…", "make shot 3 two seconds shorter", "find a reference
   photo of … on the web". Attach photos with **＋**, by pasting, or by dropping them; a picture dragged from
   another browser tab is fetched by its address.
3. Each tool call is a card under the reply: the tool and its arguments, **result** to open what it returned, and
   the picture or clip it made. A render in it says **waiting for the card** until it runs.
4. Rendering a production's chapters, rerolling a shot, finishing, cancelling a run and deleting wait for you:
   **Allow** or **Decline**. Sending a new message instead declines it.
5. **Stop** cancels the reply and any render the chat queued; a run the chat was only following goes on.
6. **context** under the title is how full the model's memory of the chat is. Past the mark, the next reply first
   summarises the chat, keeping your newest two messages and what followed them as they were; **compact** asks for
   that now, for example before changing subject. Older messages stay on the page, dimmed above a divider that
   holds the summary, and the model can still see an old picture again by its path.

The model makes pictures and clips without asking, looks at each result and says what is wrong with it, and makes
one again only when told to. One message allows 20 rounds of tool calls. Web search goes through the search engines'
own pages; when they refuse, the chat gives a link to search in your browser, and you drop what you pick into it.

Over MCP: the same tools, called by your own agent; `search_images`, `fetch_image` and `search_web` are the chat's only.

# Use the VLM from other tools
Chat with the VLM (talk to the model), or reach difflab from another program.

- **Chat** tab: a conversation with the VLM, pictures included, that acts through difflab's tools; its reply waits
  its turn on the card.
- **OpenAI-compatible API** at `/v1/chat/completions` and `/v1/models`: streaming, tool calls and reasoning
  content. The API key is the server's token, when it has one.
- **MCP** at `/mcp`: these tools. Same token, as a bearer header.
