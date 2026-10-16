// Written by `python -m difflab.core.protocol`. Do not edit: change the classes and rerun.

export interface RunStartEvent {
  kind: 'run.start'
  t: number
  title: string
  pid: number
}

export interface RunEndEvent {
  kind: 'run.end'
  t: number
  ok: boolean
  error: string
  seconds: number
}

export interface ErrorEvent {
  kind: 'error'
  t: number
  text: string
}

export interface StatsEvent {
  kind: 'stats'
  t: number
  rss: number
  vram_used?: number | null
  vram_total?: number | null
  ram_used?: number | null
  ram_total?: number | null
}

export interface LogEvent {
  kind: 'log'
  t: number
  text: string
}

export interface PanelEvent {
  kind: 'panel'
  t: number
  title: string
  rows: Record<string, string>
  color: string
}

export interface StageStartEvent {
  kind: 'stage.start'
  t: number
  label: string
  total?: number | null
  unit?: string | null
  steps?: number | null
  blocks?: number | null
}

export interface StageProgressEvent {
  kind: 'stage.progress'
  t: number
  label: string
  done: number
  elapsed: number
  total?: number | null
  unit?: string | null
}

export interface StageEndEvent {
  kind: 'stage.end'
  t: number
  label: string
  ok: boolean
  seconds: number
  detail?: string | null
  numbers?: Record<string, number> | null
}

export interface TallyEvent {
  kind: 'tally'
  t: number
  name: string
  stages: string[]
  index: number
  calls: number
  done: number
  seconds: number
}

export interface TallyEndEvent {
  kind: 'tally.end'
  t: number
  name: string
  seconds: number
}

export interface PlanEvent {
  kind: 'plan'
  t: number
  title: string
  items: string[]
}

export interface ItemBeginEvent {
  kind: 'item.begin'
  t: number
  index: number
}

export interface ItemEndEvent {
  kind: 'item.end'
  t: number
  index: number
  seconds: number
  activity: string
}

export interface ItemInfoEvent {
  kind: 'item.info'
  t: number
  index: number
  version: number
  prompt: string
  seed: number
  opening: string
  references: string[]
}

export interface ItemVideoEvent {
  kind: 'item.video'
  t: number
  index: number
  version: number
  video: string
  seed: number
}

export interface PreviewEvent {
  kind: 'preview'
  t: number
  item: number | null
  name: string
  step: number
  steps: number
  video: string
  poster: string
}

export interface PictureEvent {
  kind: 'picture'
  t: number
  label: string
  image: string
}

export interface ArtifactEvent {
  kind: 'artifact'
  t: number
  media: string
  name: string
  seed: number | null
  path: string
}

export interface FinalEvent {
  kind: 'final'
  t: number
  video: string
  frames?: number | null
  seconds?: number | null
}

export interface JobBeginEvent {
  kind: 'job.begin'
  t: number
  index: number
  total: number
  position: number
  mode: string
  label: string
  job: number | null
}

export interface JobEndEvent {
  kind: 'job.end'
  t: number
  index: number
  total: number
  ok: boolean
  status: string
  error: string
  made: string
  seconds: number
}

export interface FoundEvent {
  kind: 'found'
  t: number
  chapter: number
  members: Record<string, string>[]
}

export type Event =
  | RunStartEvent
  | RunEndEvent
  | ErrorEvent
  | StatsEvent
  | LogEvent
  | PanelEvent
  | StageStartEvent
  | StageProgressEvent
  | StageEndEvent
  | TallyEvent
  | TallyEndEvent
  | PlanEvent
  | ItemBeginEvent
  | ItemEndEvent
  | ItemInfoEvent
  | ItemVideoEvent
  | PreviewEvent
  | PictureEvent
  | ArtifactEvent
  | FinalEvent
  | JobBeginEvent
  | JobEndEvent
  | FoundEvent

export const KINDS = ['run.start', 'run.end', 'error', 'stats', 'log', 'panel', 'stage.start', 'stage.progress', 'stage.end', 'tally', 'tally.end', 'plan', 'item.begin', 'item.end', 'item.info', 'item.video', 'preview', 'picture', 'artifact', 'final', 'job.begin', 'job.end', 'found'] as const
