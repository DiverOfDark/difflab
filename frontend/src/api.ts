/** The server, as functions. Every call is one route in `difflab/api/app.py`. */
import { shallowRef } from 'vue'
import type { RunState } from './run'
import { apply, emptyRun } from './run'

export interface Job {
  mode: 'picture' | 'video' | 'upscale'
  idea: string
  prompt: string
  photos: string[]
  /** A picture starts from `photos[0]`; unset, every photo is a reference and the canvas is fresh. */
  start: boolean
  clip: string | null
  finish_megapixels: number | null
  grade_strength: number | null
  retime_fps: number | null
  /** "W:H". A created picture's: empty lets the writer pick one, or draws a prompt square. An edit's or a
   *  take's: the photo is brought to it by `fit`; empty keeps the photo's own. */
  aspect: string
  fit: 'crop' | 'extend'
  /** Each is a tool's whole effective settings, advanced fields the form does not draw included, so a
   *  machine's overrides reach a quick job; a field absent falls back to the code default. */
  video: VideoKnobs & Record<string, unknown>
  edit: {
    steps: number
    cfg: number
    seed: number
    megapixels: number
    engine?: string
    turbo?: boolean
    /** Qwen-Image 2.1's own, so switching engines changes neither. */
    steps_21?: number
    cfg_21?: number
    /** minimax-h3's clip, in seconds; the video card's duration is the video mode's. */
    seconds_h3?: number
    opening_h3?: string
    megapixels_h3?: number
  } & Record<string, unknown>
  /** The edit instruction's writer, and the H3 caption's: the video graph asked for different numbers. */
  vlm: VlmKnobs & Record<string, unknown>
  video_vlm: VlmKnobs & Record<string, unknown>
}

interface VideoKnobs {
  duration: number
  megapixels: number
  seed: number
  steps: number
  reference_size: 'match' | 'max'
}

interface VlmKnobs {
  max_new_tokens: number
  temperature: number
  top_p: number
  top_k: number
  seed: number
}

export interface Asset {
  path: string
  name: string
  url: string
  thumb?: string
  bytes: number
  width?: number
  height?: number
}

/** A run as the server reads its row against its newest event: `queued`, `live`, `paused`,
 *  `stopped` for one that went quiet without writing an end, or the state it ended in. */
export interface RunSummary {
  id: string
  title: string
  status: string
  state: string
  items: number
  error: string
  asked_by: string
  /** The production this run was built from, or null for a quick job. */
  production: number | null
  scope: string
  created: number
  started: number | null
  ended: number | null
  last: number | null
  jobs_done: number
  jobs_total: number
  /** Its newest picture, or the photo a run that made none was given; relative to output/ or absolute. */
  preview: string | null
}

export interface RunTask {
  name: string
  state: string
  done: number
  total: number | null
  unit: string
  detail: string
  seconds: number | null
}

/** One tool's turn inside a run, with the tasks it reported while it ran. */
export interface RunJob {
  id: number
  position: number
  tool: string
  label: string
  state: string
  inputs: Record<string, unknown>
  outputs: Record<string, unknown>
  settings: Record<string, unknown>
  progress: number | null
  started: number | null
  ended: number | null
  error: string
  tasks: RunTask[]
}

export interface RunArtifact {
  /** Relative to output/. */
  path: string
  kind: string
  seed: number | null
  job: number | null
}

export interface RunRecord extends RunSummary {
  /** What was asked for, as it was asked: the form goes on being edited, this does not. */
  spec: Job
  jobs: RunJob[]
  artifacts: RunArtifact[]
}

/** The newest edit that left shots stale, whichever screen made it: the shell's STALE banner reads it. */
export const staled = shallowRef<{ production: number; stale: Stale } | null>(null)

async function json<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, options)
  if (!res.ok) {
    const text = await res.text()
    let detail: unknown
    try {
      detail = JSON.parse(text)?.detail
    } catch {}
    throw new Error(detail == null ? text : typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  const found = await res.json()
  if (found?.stale?.shots?.length) staled.value = { production: found.id, stale: found.stale }
  return found as T
}

const send = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})
const post = (body: unknown) => send('POST', body)

export const checkJob = (job: Job) => json<{ problems: string[]; title: string; job: Job }>('/api/jobs/check', post(job))
export const listRuns = () => json<RunSummary[]>('/api/runs')

export const listUploads = () => json<Asset[]>('/api/uploads')
/** Queued, not started: the worker takes runs in the order they were asked for, and answers here at once. */
export const startRun = (job: Job) => json<{ run: string; title: string }>('/api/runs', post(job))
export const startRuns = (jobs: Job[]) => json<{ run: string; title: string }[]>('/api/runs/batch', post(jobs))
export const readRun = (runId: string) => json<RunRecord>(`/api/runs/${encodeURIComponent(runId)}`)

/** `pause`, `resume` and `cancel` are the run's; the rest go to the job in flight and act where it next
 *  looks - a chunk boundary or a sampling step - so nothing here is immediate. */
export const control = (runId: string, command: Record<string, unknown>) =>
  json<unknown>(`/api/runs/${encodeURIComponent(runId)}/commands`, post(command))

export async function upload(files: File[]): Promise<Asset[]> {
  const form = new FormData()
  for (const file of files) form.append('files', file)
  return json<Asset[]>('/api/uploads', { method: 'POST', body: form })
}

export interface GalleryItem {
  id: number
  /** Relative to output/, served under `/output/`. */
  path: string
  kind: 'image' | 'video'
  seed: number | null
  created: number
  bytes: number
  run: string | null
  title: string
  production: number | null
  /** A video's own still, when its job saved one. */
  poster: string | null
  /** What the run made on the way: its earlier files, an H3 edit's clip, the prompts it wrote. */
  steps: GalleryStep[]
}

export interface GalleryStep {
  path: string
  kind: 'image' | 'video' | 'text'
  label: string
  text?: string
}

export const listGallery = () => json<GalleryItem[]>('/api/gallery')
export const deleteResult = (id: number) => json<{ files: number; bytes: number }>(`/api/gallery/${id}`, { method: 'DELETE' })

export const thumbUrl = (path: string) => `/api/thumb?path=${encodeURIComponent(path)}`

/** By the path the run's own events name: relative to the run directory, or absolute (`/output/…`)
 *  for a chunk that lives under output/ rather than under the run. */
export const runFile = (runId: string, relative: string) =>
  relative.startsWith('/') ? relative : `/runs/${encodeURIComponent(runId)}/${relative}`

/**
 * The server replays the whole log on connect, so the state is rebuilt from scratch rather than
 * patched: a reconnect after a dropped stream cannot leave it half-right.
 */
export function watchRun(runId: string, onChange: (run: RunState) => void): () => void {
  let run = emptyRun(runId)
  const source = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`)
  source.onopen = () => {
    run = emptyRun(runId)
  }
  source.onmessage = (message) => {
    try {
      apply(run, JSON.parse(message.data))
      onChange(run)
    } catch {
      /* a line the page cannot read is not worth breaking the stream over */
    }
  }
  source.addEventListener('done', () => source.close())
  return () => source.close()
}

// --- productions ---------------------------------------------------------------------------------
export interface MadeThing {
  id: number
  state: string
  artifact: string | null
  present: boolean
  key: string | null
  job: number | null
  pinned: boolean
  review: string
  /** Who gave the last verdict: 'judge', or 'person' when someone approved or rejected it by hand. */
  reviewed_by: string
  reviewed_at: number | null
}

/** A member's look: the bible's (`id` null, its picture the member's base one) or an amended casting's. */
export interface LookRef {
  id: number | null
  member: string
  label: string
  /** What it has turned into and every amendment since, joined. */
  words: string
  /** What it has turned into; '' while it is still itself. */
  form: string
  /** The shot whose casting made it; null for the base. */
  shot: number | null
  image: 'done' | 'flag' | 'running' | 'pending' | 'failed'
  artifact: string | null
}

/** A member in a shot. `parent` null is the look it last had, 'base' the bible's, or a casting id;
 *  with `amend` set it is a look of its own, whose picture the made columns are. */
export interface CastingInfo extends MadeThing {
  member: string
  member_id: number
  kind: string
  position: number
  parent: string | null
  amend: string
  /** What this look turns the member into, its `amend` then the whole look; '' for a look that only adds. */
  form: string
  form_kind: string
  label: string
  /** "from text · ch N ¶P" when the planner wrote it, "by hand" when a person did. */
  origin: string
  size: number[]
  amended: boolean
  /** The look it starts from, the one it has, and the one it had last before this shot. */
  source: LookRef
  look: LookRef
  latest: LookRef
  is_revert: boolean
  /** Its source was decided by a cut back to the opening scene. */
  back: boolean
  /** Later shot ids that show this casting's look; only for an amended one. */
  carried_into: number[]
  /** 'base' and look ids whose pictures the clip is handed; null is the compiler's own pick. */
  pictured: string[] | null
}

export interface ShotInfo {
  id: number
  /** Its chapter's position, from 0. */
  chapter: number
  position: number
  action: string
  scene: string
  camera: string
  dialogue: string
  seconds: number
  chained: boolean
  /** Its plan was edited by hand: a re-plan leaves it as written. */
  pinned: boolean
  opening: 'keyframe' | 'chain' | 'refs'
  sound: string
  castings: CastingInfo[]
  /** Look ids the shot shows whose picture is not approved yet. */
  waiting: number[]
  keyframe: MadeThing & {
    source: string
    provided: string | null
    note: string
    landed: boolean
    /** A composed keyframe's inputs: the place it was made over, and each cast member's picture by id. */
    made_from: { base: string | null; pictures: Record<string, string>; note: string | null } | null
  }
  prompt: MadeThing & {
    text: string
    written_by: string
    pictures: string[]
    checks: { severity: string; code: string; message: string }[]
    marker: { kind: 'pinned' | 'changed' | 'rewritten' | 'written'; label: string; at: number | null } | null
  }
  clip: MadeThing & { seed: number; handoff: number | null; takes: { judged: number; rejected: number; retaken: number } }
  plan: Record<string, unknown>
}

export interface Trait {
  part: string
  phrase: string
  short: string
  name: string
}

/** Two photographs disagreeing (`shows` empty), or the story against a photograph: `says` is the story's
 *  words, `shows` the photograph's trait, `lines` how many sentences keeping the reference rewrites, `novel`
 *  the trait keeping the story takes. */
export interface Conflict {
  id: number
  says: string
  /** Which wording was kept; empty while it is open. */
  resolution: string
  shows: string
  lines: number
  novel: string
}

/** How many made things an edit moved to stale. */
export interface Stale {
  prompts: number
  keyframes: number
  clips: number
  /** Drawn places that will be drawn again at a new aspect. */
  references?: number
  /** 1 when turning plates on queued a run to draw the missing ones. */
  plating?: number
  /** The shots newly stale, and the chained shots after them that re-render because they open on a stale clip. */
  shots: StaleShot[]
  chained: StaleShot[]
  /** Measured GPU seconds to make them all again. */
  gpu: number
}

export interface StaleShot {
  id: number
  label: string
}

export interface MemberInfo {
  id: number
  slug: string
  kind: string
  noun: string
  possessive: string
  find: string
  identity: { traits?: Trait[]; lights?: string[]; photographed?: boolean }
  position: number
  /** Edited by hand: a re-plan's survey and tidy leave it as written. */
  pinned: boolean
  /** What the story calls it, comma-joined; the planner finds its mentions under these. */
  aliases: string
  /** Its look as a person wrote it; empty while its traits say it. */
  appearance: string
  /** What may differ from its pictures between shots: told to the judge and to the editor drawing a look. */
  allowed: string
  voice: string
  delivery: string
  /** Under output/, or empty. */
  reference_audio: string
  /** Its photographs are handed over on a flat backdrop, each reference's `plate` once drawn. */
  plate: boolean
  /** Garments it can change into: a look the plan names after one is drawn wearing that picture. */
  wardrobe: { name: string; path: string }[]
  /** Role `generated`: drawn by the editor from the member's words, with the seed it was drawn with. */
  references: (MadeThing & {
    role: string
    photo: string
    box: number[]
    size: number[]
    shows: string[]
    seed?: number | null
    plate: string | null
  })[]
  /** Drawings from its newest "draw 3" run, waiting to be picked as its reference. */
  variants?: { seed: number; frame: string | null; state: string }[]
  /** Pictures its prompts want that the last cut could not make from the photographs, and why. */
  gaps?: { role: string; why: string }[]
  conflicts: Conflict[]
  /** Its looks across the film, in film order. */
  looks?: (LookRef & { amend: string; origin: string; source: LookRef; carried_into: number[]; made: MadeThing })[]
  /** Each stretch of shot ids it is off screen between two appearances. */
  off_screen?: number[][]
}

export type StagePhase = 'done' | 'review' | 'running' | 'stale' | 'pending'

export interface ProductionSummary {
  id: number
  slug: string
  title: string
  story_kind: string
  created: number
  updated: number
  /** The newest thing that happened to it: an edit, or a run starting or ending. */
  when: number
  chapters: number
  shots: number
  seconds: number
  stages: Record<'ingest' | 'bible' | 'shots' | 'render' | 'assemble', StagePhase>
  /** The stage running, or the one after the furthest the work reached. */
  resume: 'ingest' | 'bible' | 'shots' | 'render' | 'assemble'
  /** Scope kinds of its runs queued or running: plan, chapter, references, assemble, film. */
  live: string[]
  paused: boolean
  clips_done: number
  clips_review: number
  bible_review: number
  shots_review: number
  to_draw: number
}

/** The card: what the job in flight is doing as a word, and the VRAM in GB, whoever holds it. */
export interface Gpu {
  state: 'idle' | 'planning' | 'drawing' | 'judging' | 'rendering' | 'assembling' | 'answering'
  tool: string
  run: string | null
  production: number | null
  title: string
  /** The job is waiting on weights it does not have yet: its bytes so far, of all it lacks. */
  download: { label: string; done: number; total: number; seconds: number } | null
  name?: string
  vram_used?: number
  vram_total?: number
}

export interface Production {
  id: number
  slug: string
  title: string
  story_kind: string
  story: string
  manifest: Record<string, unknown>
  created: number
  updated: number
  keep: boolean
  members: MemberInfo[]
  chapters: { id: number; position: number; title: string; shots: ShotInfo[] }[]
  runs: { id: string; title: string; state: string; scope: string; created: number }[]
  camera_moves: string[]
  trait_parts: string[]
  /** Members with no usable picture: none cut or given, and a drawn one absent, stale or failed. */
  missing_references: string[]
  /** Photographs the survey placed with no member: what the VLM saw, and the member it most resembles. */
  unmatched: { photo: string; about: string; likely: string }[]
  ideas: IdeaRound[]
  /** A short digest per stage of what it holds; it moves when the stage's content does. */
  hashes: Record<'ingest' | 'bible' | 'shots' | 'render' | 'assemble' | 'manifest', string>
  /** Per tool, measured on this machine: median seconds a job, and per second of clip for render_shot.
   *  `samples` 0 means no history yet, and the numbers are the code's fallback. */
  timings: Record<string, { seconds: number; per_second: number | null; samples: number }>
}

/** One round of the director rewriting the idea. `state` is its job's; `story` is empty until done. */
export interface IdeaRound {
  round: number
  source: string
  feedback: string
  revises: number | null
  story: string
  changes: string[]
  questions: string[]
  run: string
  state: string
  error: string
  seconds: number | null
  adopted: boolean
  created: number
}

/** A row of the references table: an upload, what it is, and the entity it shows. */
export type RefKind = 'character' | 'location' | 'prop' | 'style' | 'video'
export interface IngestRef {
  path: string
  kind: RefKind
  entity: string
  guess: string
}

export interface IngestStory {
  title: string
  story: string
  references: IngestRef[]
  /** The whole film, spread over the chapters by their words. */
  target_seconds: number
  /** The shortest shot the director plans; the longest is `video.duration`. */
  shortest: number
  /** Width over height of the canvas; null takes the first photo's own. */
  aspect: number | null
  /** How the first photo meets the canvas: cut to it, or kept whole with the sides drawn by the editor. */
  fit: 'crop' | 'extend'
  video: { duration: number; megapixels: number; steps: number }
  ingest: { heading_level: number; frames: number; min_sharpness: number }
  /** Queue every chapter's plan run, stopping before render. */
  plan: boolean
}

export interface RedoVideo {
  title: string
  source: string
  instruction: string
  photos: string[]
  clip_scale: number
  video: { duration: number; megapixels: number; steps: number }
}

export interface ReferenceFrame {
  path: string
  at: number
  sharpness: number
  kept: boolean
}

export interface IngestingStep {
  id: 'split' | 'describe' | 'read' | 'reconcile'
  name: string
  state: 'done' | 'running' | 'waiting' | 'pending' | 'failed'
  bar: number
  rows: { k: string; v: string }[]
  jobs: number
  skippable: boolean
}

/** A production's newest plan as the ingesting page shows it. */
export interface Ingesting {
  runs: { id: string; state: string; title: string }[]
  live: string[]
  steps: IngestingStep[]
  current: string
  progress: number
  bible_done: boolean
  plan: { jobs: number; settled: number }
  elapsed: number
  eta: number
  model: string
  vram: number | null
  found: { slug: string; kind: string; desc: string; src: string; chapter: number }[]
  log: { t: number; tag: string; msg: string; dur: string }[]
}

export type Timings = Production['timings']

export type Dot = 'done' | 'review' | 'stale' | 'pending'
type Image = LookRef['image']

/** A shot as the timeline draws it; `clip` and `still` are paths under output/ when the file is there. */
export interface TimelineShot {
  id: number
  chapter: number
  position: number
  scene: string
  seconds: number
  opening: ShotInfo['opening']
  /** Keyframe, prompt, clip. */
  pipe: [Dot, Dot, Dot]
  /** Looks it shows whose picture is not approved yet. */
  waiting: number
  clip: string | null
  still: string | null
  action: string
}

/** `a` and `b` index the timeline's shots, both ends in; a gap is a stretch off screen between two. */
export type TimelineRun =
  | { a: number; b: number; gap: true }
  | { a: number; b: number; gap: false; look: number | null; label: string; image: Image; revert: boolean; mark: boolean }

export interface TimelineTrack {
  member: string
  kind: string
  runs: TimelineRun[]
  /** Where a shot amends it. */
  marks: { at: number; look: number | null; label: string; image: Image; words: string }[]
}

export interface Timeline {
  shots: TimelineShot[]
  tracks: TimelineTrack[]
}

export const readTimeline = (id: number, chapter: number | null) =>
  json<Timeline>(`/api/productions/${id}/timeline${chapter === null ? '' : `?chapter=${chapter}`}`)
/** The run, its rows and the files it made that nothing else uses; refused (409) while it may still be working. */
export const deleteRun = (id: string) =>
  json<{ files: number; bytes: number; kept: number }>(`/api/runs/${encodeURIComponent(id)}`, { method: 'DELETE' })
export const listProductions = () => json<ProductionSummary[]>('/api/productions')
export const readProduction = (id: number) => json<Production>(`/api/productions/${id}`)
export const deleteProduction = (id: number) => json<unknown>(`/api/productions/${id}`, { method: 'DELETE' })
/** `title`, `story`, `opening` (the first frame in words, `{id}` per member; empty gives it back to the photo),
 *  `aspect` (width over height; null takes the first photo's), `fit` ('crop' or 'extend'), `keep` (never deleted by the cleanup). */
export const patchProduction = (id: number, body: Record<string, unknown>) =>
  json<Production & { stale: Stale }>(`/api/productions/${id}`, send('PATCH', body))
export const addMember = (id: number, body: Record<string, unknown>) =>
  json<Production>(`/api/productions/${id}/members`, post(body))
/** Pins the member; `{ pinned: false }` alone gives it back to the director. */
export const patchMember = (memberId: number, body: Record<string, unknown>) =>
  json<Production & { stale: Stale }>(`/api/members/${memberId}`, send('PATCH', body))
export const deleteMember = (memberId: number) =>
  json<Production & { stale: Stale }>(`/api/members/${memberId}`, { method: 'DELETE' })
export const detachRef = (refId: number) => json<Production & { stale: Stale }>(`/api/refs/${refId}`, { method: 'DELETE' })
/** An uploaded picture held as the member's `role` picture, fitted to the canvas like a crop. */
export const giveReference = (memberId: number, path: string, role: string) =>
  json<Production & { stale: Stale }>(`/api/members/${memberId}/given`, post({ path, role }))
export const resolveConflict = (conflictId: number, resolution: string) =>
  json<Production>(`/api/conflicts/${conflictId}/resolve`, post({ resolution }))
/** The story against a photograph settled: `reference` rewrites the story's sentences, `novel` takes its trait. */
export const keepConflict = (conflictId: number, keep: 'reference' | 'novel') =>
  json<Production>(`/api/conflicts/${conflictId}/keep`, post({ keep }))
/** An unmatched photograph held as member `to`'s picture - its garment when `garment` names one - or `to` 'ignore'. */
export const attachUnmatched = (id: number, photo: string, to: string, garment = '') =>
  json<Production & { stale: Stale }>(`/api/productions/${id}/unmatched`, post({ photo, to, garment }))
/** `path` null takes the garment out of the wardrobe. */
export const dress = (memberId: number, name: string, path: string | null) =>
  json<Production & { stale: Stale }>(`/api/members/${memberId}/wardrobe`, post({ name, path }))
/** The member read again from its photographs, as a run; it goes back to the director. */
export const describeMember = (memberId: number) => json<{ run: string }>(`/api/members/${memberId}/describe`, post({}))
export const ingestStory = (form: IngestStory) => json<Production & { runs: string[] }>('/api/ingest', post(form))
export const redoVideo = (form: RedoVideo) => json<Production>('/api/ingest/video', post(form))
export const referenceFrames = (path: string, frames: number, min_sharpness: number) =>
  json<{ seconds: number; frames: ReferenceFrame[] }>('/api/ingest/frames', post({ path, frames, min_sharpness }))
export const guessReferences = (paths: string[]) => json<{ run: string }>('/api/ingest/guess', post({ paths }))
export const readIngesting = (id: number) => json<Ingesting>(`/api/productions/${id}/ingesting`)
export const skipStep = (id: number, step: string) => json<{ skipped: number }>(`/api/productions/${id}/ingesting`, post({ skip: step }))
export const cancelIngest = (id: number) => json<{ cancelled: number }>(`/api/productions/${id}/ingesting`, post({ cancel: true }))
export const readTimings = () => json<Timings>('/api/timings')
/** `chapter` null is every chapter, a run each in chapter order; `run` is the first of `runs`. */
const chapterQuery = (chapter: number | null, more = '') => {
  const q = [chapter === null ? '' : `chapter=${chapter}`, more].filter(Boolean).join('&')
  return q ? `?${q}` : ''
}
/** `fresh` asks the director for new shots even when nothing they are planned from changed; `bible` builds the
 *  bible again from the story and the photographs, and plans no chapter. */
export const planProduction = (id: number, chapter: number | null, fresh = false, bible = false) =>
  json<{ run: string; runs: string[] }>(
    `/api/productions/${id}/plan${chapterQuery(chapter, [fresh ? 'fresh=true' : '', bible ? 'shots=false' : ''].filter(Boolean).join('&'))}`,
    post({}),
  )
export const approveBible = (id: number) => json<Production & { left: string[] }>(`/api/productions/${id}/bible/approve`, post({}))
/** Pictures given after ingest: they wait in the bible to be attached, and the next plan reads them. */
export const addPhotos = (id: number, paths: string[]) => json<Production>(`/api/productions/${id}/photos`, post({ paths }))
export const readGpu = () => json<Gpu>('/api/gpu')
export const renderProduction = (id: number, chapter: number | null) =>
  json<{ run: string; runs: string[] }>(`/api/productions/${id}/render${chapterQuery(chapter)}`, post({}))
/** Joined, levelled and subtitled from the clips there are; every chapter also queues the film. Nothing renders. */
export const assembleProduction = (id: number, chapter: number | null) =>
  json<{ run: string; runs: string[] }>(`/api/productions/${id}/assemble${chapterQuery(chapter)}`, post({}))
/** Enlarged to 4K and retimed to 60 fps from the clips there are, after they are reviewed; every chapter also queues the film. */
export const finishProduction = (id: number, chapter: number | null) =>
  json<{ run: string; runs: string[] }>(`/api/productions/${id}/finish${chapterQuery(chapter)}`, post({}))
export interface Mastered {
  video: string
  srt: string | null
  vtt: string | null
  seconds: number
  lines?: number
  job: number
  run: string
  hash: string
}
export interface ChapterMaster extends Mastered {
  lufs: number | null
  finished: boolean
  /** made from the chapter's clips as they are now */
  current: boolean
}
export interface Assembled {
  chapters: {
    id: number
    position: number
    marker: string
    shots: number
    boundaries: number
    chained: number
    lines: number
    native?: ChapterMaster
    finished?: ChapterMaster
  }[]
  film: (Mastered & { chapters: number }) | null
  busy: number
}
export const readAssembled = (id: number) => json<Assembled>(`/api/productions/${id}/assembled`)
export type MadeKind = 'keyframe' | 'prompt' | 'clip' | 'reference' | 'casting'
/** Approved as it is by a person: the judge no longer overrules it, and a run paused on the clip goes on. */
export const approveMade = (kind: MadeKind, id: number) => json<Production>(`/api/made/${kind}/${id}/approve`, post({}))
export const patchShot = (shotId: number, body: Record<string, unknown>) =>
  json<Production>(`/api/shots/${shotId}`, send('PATCH', body))
/** The shot and its made things; the shots after it close the gap. */
export const deleteShot = (shotId: number) => json<Production>(`/api/shots/${shotId}`, { method: 'DELETE' })
/** The shot's prose written again and its prompt compiled from it, as a run; a hand-written prompt is let go. */
export const rewriteShot = (shotId: number) => json<{ run: string; runs: string[] }>(`/api/shots/${shotId}/rewrite`, post({}))
/** The shot's prompt, keyframe and looks approved, each that is made; `left` says what was not. */
export const approveShot = (shotId: number) => json<Production & { left: string[] }>(`/api/shots/${shotId}/approve`, post({}))
export const addCasting = (shotId: number, body: { member: string; parent?: string | null; amend?: string; label?: string }) =>
  json<Production>(`/api/shots/${shotId}/castings`, post(body))
export const patchCasting = (
  castingId: number,
  body: { parent?: string | null; amend?: string; label?: string; pictured?: string[] | null; form?: string; form_kind?: string },
) =>
  json<Production>(`/api/castings/${castingId}`, send('PATCH', body))
export const deleteCasting = (castingId: number) => json<Production>(`/api/castings/${castingId}`, { method: 'DELETE' })
export const drawLook = (castingId: number) => json<{ run: string; runs: string[] }>(`/api/castings/${castingId}/draw`, post({}))
export const drawShotLooks = (shotId: number) => json<{ run: string; runs: string[] }>(`/api/shots/${shotId}/looks`, post({}))
/** An uploaded picture as the look's own instead of a drawn one, approved. */
export const uploadLook = (castingId: number, path: string) => json<Production>(`/api/castings/${castingId}/upload`, post({ path }))
/** Every look of the chapter, or of every one, with no picture yet, drawn as one run. */
export const drawProductionLooks = (id: number, chapter: number | null) =>
  json<{ run: string; runs: string[] }>(`/api/productions/${id}/looks${chapterQuery(chapter)}`, post({}))
/** Per chapter position, what would stop it rendering; nothing is queued. */
export const preflight = (id: number) => json<{ problems: Record<string, string[]> }>(`/api/productions/${id}/preflight`)

export type QueueState = 'done' | 'review' | 'running' | 'paused' | 'queued' | 'blocked' | 'failed' | 'stale'
export interface QueueRow {
  id: number
  /** `01.016`: chapter and shot from 1. */
  label: string
  chapter: number
  action: string
  seconds: number
  opening: 'keyframe' | 'chain' | 'refs'
  /** H3's mode: fl2va opens on a frame, ref2va on its references alone. */
  preset: string
  state: QueueState
  /** What the row says: "sampling 3/5", "blocked · look", "to render". */
  text: string
  clip: { id: number; state: string; seed: number; video: string | null }
  still: string | null
  /** For a clip in review: the judge's reason for the kept take, its score and how many takes it was the best of. */
  review: { why: string; score: number | null; takes: number; seeds: number[] } | null
}
export interface RenderQueue {
  rows: QueueRow[]
  total: number
  done: number
  flagged: number
  now: { label: string; doing: string } | null
  /** Seconds left of the queued clips; with nothing queued, of every clip not made. */
  eta: { seconds: number; clips: number; queued: boolean }
  judged: number
  rejected: number
  retaken: number
  status: 'running' | 'paused' | 'queued' | 'idle'
  runs: { id: string; state: string; title: string }[]
  auto_retry: boolean
  takes: number
  gate: number
}
export const readQueue = (id: number, chapter: number | null) => json<RenderQueue>(`/api/productions/${id}/queue${chapterQuery(chapter)}`)
/** `pause` every live render run of the production after its clip in flight, or let them go; `auto_retry` re-renders a clip
 *  below the gate on new seeds, keeping the best take, instead of pausing on it. */
export const steerQueue = (id: number, body: { pause?: boolean; auto_retry?: boolean }) =>
  json<{ told?: number; auto_retry?: boolean }>(`/api/productions/${id}/queue`, post(body))
export const rerollShot = (shotId: number, seed?: number) =>
  json<{ seed: number; run: string }>(`/api/shots/${shotId}/reroll`, post(seed === undefined ? {} : { seed }))
/** Every member with no usable picture, drawn from its words, as a run. */
export const drawReferences = (id: number) => json<{ run: string }>(`/api/productions/${id}/references`, post({}))
/** One member's picture; drawn again, a new seed unless one is given. `describe` has a place's words
 *  written from the story first. */
export const drawReference = (memberId: number, seed?: number, describe = false) =>
  json<{ run: string }>(`/api/members/${memberId}/reference`, post({ ...(seed === undefined ? {} : { seed }), ...(describe ? { describe } : {}) }))
export const drawVariants = (memberId: number) => json<{ run: string }>(`/api/members/${memberId}/variants`, post({}))
export const pickVariant = (memberId: number, seed: number) =>
  json<Production & { stale: Stale }>(`/api/members/${memberId}/variants/pick`, post({ seed }))
/** The shot's keyframe composed again with a note for the editor; only keyframes are made, not clips. */
export const redoKeyframe = (shotId: number, note: string) => json<{ run: string }>(`/api/shots/${shotId}/keyframe`, post({ note }))
export const renderKeyframes = (id: number, chapter: number | null) =>
  json<{ run: string; runs: string[] }>(`/api/productions/${id}/keyframes${chapterQuery(chapter)}`, post({}))

export const listIdeas = (id: number) => json<IdeaRound[]>(`/api/productions/${id}/ideas`)
/** `from_round` with `feedback` revises that round; without, a new rewrite of `idea` or of the rounds' own idea. */
export const improveIdea = (id: number, body: { feedback?: string; from_round?: number; idea?: string }) =>
  json<{ run: string }>(`/api/productions/${id}/ideas`, post(body))
export const adoptIdea = (id: number, round: number) =>
  json<Production & { stale: Stale }>(`/api/productions/${id}/ideas/${round}/adopt`, post({}))

// --- tools and settings ------------------------------------------------------------------------
export interface SettingField {
  name: string
  type: 'int' | 'float' | 'bool' | 'str'
  default: unknown
  choices: unknown[]
  label: string
  hint: string
  unit: string
  low: number | null
  high: number | null
  keyed: boolean
  advanced: boolean
  role: string
  multiple: boolean
}

export interface ToolInfo {
  name: string
  key_version: number
  tasks: string[]
  hint: string
  settings: SettingField[]
  machine?: Record<string, unknown>
  production?: Record<string, unknown>
  effective?: Record<string, unknown>
  notes?: string[]
}

export const listTools = (production?: number) =>
  json<ToolInfo[]>(production === undefined ? '/api/tools' : `/api/tools?production=${production}`)
export const putSettings = (tool: string, scopeKind: string, scopeId: string, overrides: Record<string, unknown>) =>
  json<{ overrides: Record<string, unknown>; effective: Record<string, unknown>; notes: string[] }>(
    `/api/settings/${encodeURIComponent(tool)}`,
    send('PUT', { scope_kind: scopeKind, scope_id: scopeId, overrides }),
  )
export const settingsImpact = (tool: string, scopeKind: string, scopeId: string, overrides: Record<string, unknown>) =>
  json<{ changed: string[]; stale: Record<string, number>; notes: string[] }>(
    '/api/settings/impact',
    post({ scope_kind: scopeKind, scope_id: scopeId, tool, overrides }),
  )
export const listModels = () => json<Record<string, string[]>>('/api/models')
export type ModelChoice = { value: string; name: string; size: number }
/** `value` when the link means one file; `choices` when a repo or a Civitai version holds several and the role takes one. */
export const resolveModel = (text: string, role: string) =>
  json<{ value?: string; choices?: ModelChoice[] }>('/api/models/resolve', post({ text, role }))
export const forgetModel = (role: string, value: string) =>
  json<Record<string, string[]>>('/api/models', send('DELETE', { role, value }))
export type Account = { source: string; account: string; tail: string }
export const listAccounts = () => json<Record<string, Account>>('/api/accounts')
export const saveAccount = (service: string, token: string) =>
  json<Record<string, Account>>(`/api/accounts/${service}`, send('PUT', { token }))
export const outputUrl = (path: string) => `/output/${path}`

// --- chats ---------------------------------------------------------------------------------------
export interface ChatSummary {
  id: number
  title: string
  created: number
  updated: number
}

/** An answer goes `queued` -> `writing` -> `done`, `stopped` or `failed`; `text` and `thinking` grow while it writes. */
export interface ChatMessage {
  id: number
  role: 'user' | 'assistant'
  text: string
  thinking: string
  /** Uploaded paths: pictures, videos (the VLM sees a frame a second), sounds (it cannot hear them). */
  media: string[]
  state: 'queued' | 'writing' | 'done' | 'stopped' | 'failed'
  error: string
  run: string | null
  created: number
}

export interface Chat extends ChatSummary {
  /** '' is `default_system`. */
  system: string
  default_system: string
  messages: ChatMessage[]
}

const chatUrl = (id: number, rest = '') => `/api/chats/${id}${rest}`
export const listChats = () => json<ChatSummary[]>('/api/chats')
export const startChat = () => json<Chat>('/api/chats', { method: 'POST' })
export const readChat = (id: number) => json<Chat>(chatUrl(id))
export const patchChat = (id: number, body: { title?: string; system?: string }) => json<Chat>(chatUrl(id), send('PATCH', body))
export const deleteChat = (id: number) => json<unknown>(chatUrl(id), { method: 'DELETE' })
export const sayInChat = (id: number, text: string, media: string[]) => json<Chat>(chatUrl(id, '/messages'), post({ text, media }))
export const chatAgain = (id: number) => json<Chat>(chatUrl(id, '/again'), { method: 'POST' })
export const chatRewind = (id: number, message: number) => json<Chat>(chatUrl(id, `/rewind/${message}`), { method: 'POST' })
export const chatStop = (id: number) => json<Chat>(chatUrl(id, '/stop'), { method: 'POST' })
