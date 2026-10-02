/** The server, as functions. Every call is one route in `difflab/web/api.py`. */
import type { RunState } from './run'
import { apply, emptyRun } from './run'

export interface Job {
  mode: 'video' | 'edit' | 'upscale'
  idea: string
  prompt: string
  photos: string[]
  clip: string | null
  finish_megapixels: number | null
  grade_strength: number | null
  retime_fps: number | null
  /** Each is a tool's whole effective settings, advanced fields the form does not draw included, so a
   *  machine's overrides reach a quick job; a field absent falls back to the code default. */
  video: VideoKnobs & Record<string, unknown>
  edit: { steps: number; cfg: number; seed: number; megapixels: number } & Record<string, unknown>
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

async function json<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, options)
  if (!res.ok) throw new Error(((await res.json().catch(() => ({}))) as any).detail ?? (await res.text()))
  return res.json() as Promise<T>
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
}

export interface ShotInfo {
  id: number
  position: number
  action: string
  scene: string
  camera: string
  dialogue: string
  seconds: number
  chained: boolean
  /** Its plan was edited by hand: a re-plan leaves it as written. */
  pinned: boolean
  keyframe: MadeThing & { source: string; provided: string | null; note: string; landed: boolean }
  prompt: MadeThing & { text: string; written_by: string; pictures: string[]; checks: { severity: string; code: string; message: string }[] }
  clip: MadeThing & { seed: number; handoff: number | null }
  plan: Record<string, unknown>
}

export interface Trait {
  part: string
  phrase: string
  short: string
  name: string
}

export interface Conflict {
  id: number
  says: string
  /** Which wording was kept; empty while it is open. */
  resolution: string
}

/** How many made things an edit moved to stale. */
export interface Stale {
  prompts: number
  keyframes: number
  clips: number
  /** Drawn places that will be drawn again at a new aspect. */
  references?: number
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
  /** Role `generated`: drawn by the editor from the member's words, with the seed it was drawn with. */
  references: (MadeThing & { role: string; photo: string; box: number[]; size: number[]; shows: string[]; seed?: number | null })[]
  conflicts: Conflict[]
}

export interface ProductionSummary {
  id: number
  slug: string
  title: string
  story_kind: string
  story: string
  manifest: Record<string, unknown>
  created: number
  updated: number
  chapters: number
  shots: number
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
  members: MemberInfo[]
  chapters: { id: number; position: number; title: string; shots: ShotInfo[] }[]
  runs: { id: string; title: string; state: string; scope: string; created: number }[]
  camera_moves: string[]
  trait_parts: string[]
  /** Members with no usable picture: none cut or given, and a drawn one absent, stale or failed. */
  missing_references: string[]
  ideas: IdeaRound[]
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

export interface Ingest {
  kind: 'idea' | 'text' | 'video'
  slug: string
  title: string
  idea: string
  text: string
  source: string
  instruction: string
  photos: string[]
  /** Seconds the director fills with as many shots as the story needs; per chapter for a text. */
  take_seconds: number
  /** Width over height of the canvas; null takes the first photo's own. */
  aspect: number | null
  clip_scale: number
  video: { duration: number; megapixels: number; steps: number }
  continuity: { tail_seconds: number; handoff_window: number; crossfade: number }
}

export const listProductions = () => json<ProductionSummary[]>('/api/productions')
export const readProduction = (id: number) => json<Production>(`/api/productions/${id}`)
export const deleteProduction = (id: number) => json<unknown>(`/api/productions/${id}`, { method: 'DELETE' })
/** `title`, `story`, `opening` (the first frame in words, `{id}` per member; empty gives it back to the photo),
 *  `aspect` (width over height; null takes the first photo's). */
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
export const resolveConflict = (conflictId: number, resolution: string) =>
  json<Production>(`/api/conflicts/${conflictId}/resolve`, post({ resolution }))
export const ingest = (form: Ingest) => {
  const common = { slug: form.slug, title: form.title, photos: form.photos, video: form.video, continuity: form.continuity, aspect: form.aspect }
  if (form.kind === 'idea') return json<Production>('/api/ingest/idea', post({ ...common, idea: form.idea, take_seconds: form.take_seconds }))
  if (form.kind === 'text') return json<Production>('/api/ingest/text', post({ ...common, text: form.text, take_seconds: form.take_seconds }))
  return json<Production>(
    '/api/ingest/video',
    post({ ...common, source: form.source, instruction: form.instruction, clip_scale: form.clip_scale }),
  )
}
/** `fresh` asks the director for new shots even when nothing they are planned from changed. */
export const planProduction = (id: number, chapter = 0, fresh = false) =>
  json<{ run: string }>(`/api/productions/${id}/plan?chapter=${chapter}${fresh ? '&fresh=true' : ''}`, post({}))
export const renderProduction = (id: number, chapter = 0) =>
  json<{ run: string }>(`/api/productions/${id}/render?chapter=${chapter}`, post({}))
export const addShot = (id: number, body: Record<string, unknown>) =>
  json<Production>(`/api/productions/${id}/shots`, post(body))
export const patchShot = (shotId: number, body: Record<string, unknown>) =>
  json<Production>(`/api/shots/${shotId}`, send('PATCH', body))
export const deleteShot = (shotId: number) => json<Production>(`/api/shots/${shotId}`, { method: 'DELETE' })
export const rerollShot = (shotId: number, seed?: number) =>
  json<{ seed: number; run: string }>(`/api/shots/${shotId}/reroll`, post(seed === undefined ? {} : { seed }))
/** Every member with no usable picture, drawn from its words, as a run. */
export const drawReferences = (id: number) => json<{ run: string }>(`/api/productions/${id}/references`, post({}))
/** One member's picture; drawn again, a new seed unless one is given. */
export const drawReference = (memberId: number, seed?: number) =>
  json<{ run: string }>(`/api/members/${memberId}/reference`, post(seed === undefined ? {} : { seed }))

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
export const rememberModel = (role: string, value: string) => json<Record<string, string[]>>('/api/models', post({ role, value }))
export const forgetModel = (role: string, value: string) =>
  json<Record<string, string[]>>('/api/models', send('DELETE', { role, value }))
export const outputUrl = (path: string) => `/output/${path}`
