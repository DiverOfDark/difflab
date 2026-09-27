/** What the stage rail and the stage screens derive from one production, so they cannot disagree. */
import type { MadeThing, Production, ShotInfo } from './api'

export type StageId = 'ingest' | 'bible' | 'assets' | 'shots' | 'keyframes' | 'prompts' | 'render' | 'assemble'
export type Phase = 'done' | 'review' | 'running' | 'stale' | 'pending'

export const FINISHED = new Set(['made', 'approved', 'pinned'])
export const BUSY = new Set(['queued', 'running'])

/** The lifecycle of a made thing, folded to what one dot can say. */
export function phaseOf(state: string): Phase {
  if (FINISHED.has(state)) return 'done'
  if (BUSY.has(state)) return 'running'
  if (state === 'needs_review' || state === 'failed') return 'review'
  if (state === 'stale') return 'stale'
  return 'pending'
}

export function phaseOfAll(things: MadeThing[]): Phase {
  if (!things.length) return 'pending'
  const phases = things.map((one) => phaseOf(one.state))
  for (const p of ['running', 'review', 'stale', 'pending'] as Phase[]) if (phases.includes(p)) return p
  return 'done'
}

export const PHASE_LABEL: Record<Phase, string> = {
  done: 'DONE',
  review: 'REVIEW',
  running: 'RUNNING',
  stale: 'STALE',
  pending: 'PENDING',
}

export const pad = (n: number, width: number) => String(n).padStart(width, '0')
export const shotId = (chapter: number, shot: ShotInfo) => `${pad(chapter + 1, 2)}.${pad(shot.position + 1, 3)}`
export const castOf = (shot: ShotInfo) => (Array.isArray(shot.plan.cast) ? (shot.plan.cast as string[]) : [])
export const words = (text: string) => text.split(/\s+/).filter(Boolean).length

export function seconds(total: number): string {
  if (total < 60) return `${total.toFixed(1)} s`
  const m = Math.floor(total / 60)
  return `${m} min ${pad(Math.round(total - m * 60), 2)} s`
}

export function ago(t: number): string {
  const s = Date.now() / 1000 - t
  if (s < 90) return 'now'
  if (s < 3600) return `${Math.round(s / 60)} min`
  if (s < 86400) return `${Math.round(s / 3600)} h`
  return `${Math.round(s / 86400)} d`
}

export const runLive = (state: string) => state === 'live' || state === 'queued' || state === 'running' || state === 'paused'

/** Plan runs are scoped `plan:<chapter id>`, chapter runs `chapter:<chapter id>`. */
export function runsFor(production: Production, kind: 'plan' | 'chapter', chapterId: number) {
  return production.runs.filter((r) => r.scope === `${kind}:${chapterId}`)
}

/** The run drawing a production's references, while it is queued or running. */
export const drawingRun = (production: Production) =>
  production.runs.find((r) => r.scope === `references:${production.id}` && runLive(r.state))

export const GENERATED = 'generated'

export const ASPECTS: { label: string; value: number | null }[] = [
  { label: 'from the first photo', value: null },
  { label: '16:9', value: 16 / 9 },
  { label: '9:16', value: 9 / 16 },
  { label: '1:1', value: 1 },
  { label: '4:3', value: 4 / 3 },
  { label: '2.39:1', value: 2.39 },
]

/** Mirrors `pipeline.canvas_size`: the pixel budget at this aspect, each side rounded to 32. */
export function canvasFor(aspect: number, megapixels: number): [number, number] {
  const area = megapixels * 1024 * 1024
  const side = (v: number) => Math.max(32, Math.round(v / 32) * 32)
  return [side(Math.sqrt(area * aspect)), side(Math.sqrt(area / aspect))]
}
