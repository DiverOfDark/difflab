/** What the stage rail and the stage screens derive from one production, so they cannot disagree. */
import type { MadeThing, Production, ShotInfo } from './api'

export type StageId = 'ingest' | 'bible' | 'shots' | 'render' | 'assemble'
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
export const shotId = (shot: ShotInfo) => `${pad(shot.chapter + 1, 2)}.${pad(shot.position + 1, 3)}`
export const castOf = (shot: ShotInfo) => (Array.isArray(shot.plan.cast) ? (shot.plan.cast as string[]) : [])
export const words = (text: string) => text.split(/\s+/).filter(Boolean).length

/** `mm:ss`, and `mm:ss:ff` with frames at 24 fps. */
export const mmss = (t: number) =>
  t >= 3600 ? `${Math.floor(t / 3600)}:${pad(Math.floor(t / 60) % 60, 2)}:${pad(Math.floor(t % 60), 2)}` : `${pad(Math.floor(t / 60), 2)}:${pad(Math.floor(t % 60), 2)}`
export const timecode = (t: number) => `${mmss(t)}:${pad(Math.floor((t % 1) * 24), 2)}`

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
export function runsFor(production: Production, kind: 'plan' | 'chapter', chapterIds: number[]) {
  return production.runs.filter((r) => chapterIds.some((id) => r.scope === `${kind}:${id}`))
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

// `screenplay.config.DIRECTOR_CLIP_MIN` and `SHOT_SECONDS`: the shortest shot the director may plan, and
// the longest a shot may be set to.
export const CLIP_MIN = 6
export const SHOT_MAX = 15

/** Mirrors `screenplay.ingest.chapters_of`: a chapter per heading of `level` or above; text with no such
 *  heading over it, every 600 words. */
const HEADING = /^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/gm
export const WORDS_PER_CHAPTER = 600
export interface SplitChapter {
  title: string
  words: number
  headed: boolean
}
export function chaptersOf(text: string, level: number): SplitChapter[] {
  const found = [...text.matchAll(HEADING)].filter((m) => m[1].length <= level)
  const parts: [string, string][] = [['', found.length ? text.slice(0, found[0].index) : text]]
  found.forEach((m, i) => parts.push([m[2].trim(), text.slice(m.index! + m[0].length, found[i + 1]?.index ?? text.length)]))
  const out: SplitChapter[] = []
  for (const [title, body] of parts) {
    const n = words(body.replace(HEADING, '$2.'))
    if (title && n) out.push({ title, words: n, headed: true })
    else if (n) {
      const count = Math.max(1, Math.ceil(n / WORDS_PER_CHAPTER))
      const each = Math.ceil(n / count)
      for (let k = 0; k < count && k * each < n; k++) out.push({ title: `chapter ${out.length + 1}`, words: Math.min(each, n - k * each), headed: false })
    }
  }
  return out
}

/** "2 h", "90 min", "1 h 30 min", "1:30:00", "45 s" as seconds; a bare number is seconds. NaN when unreadable. */
export function parseLength(said: string): number {
  const text = said.trim().toLowerCase()
  if (/^\d+(:\d{1,2}){1,2}$/.test(text)) return text.split(':').reduce((t, part) => t * 60 + Number(part), 0)
  const units: Record<string, number> = { h: 3600, hr: 3600, hour: 3600, hours: 3600, m: 60, min: 60, mins: 60, minute: 60, minutes: 60, s: 1, sec: 1, secs: 1, second: 1, seconds: 1 }
  const parts = [...text.matchAll(/(\d+(?:\.\d+)?)\s*([a-z]*)/g)]
  if (!parts.length || parts.map((p) => p[0]).join('').replace(/\s/g, '') !== text.replace(/\s/g, '')) return NaN
  let total = 0
  for (const [, n, unit] of parts) {
    if (unit && !(unit in units)) return NaN
    total += Number(n) * (unit ? units[unit] : 1)
  }
  return total
}

/** "6–8 s" (or "6-8", "6 to 8 s", "8 s") as [shortest, longest] seconds; null when unreadable. */
export function parseRange(said: string): [number, number] | null {
  const found = said.trim().toLowerCase().match(/^(\d+(?:\.\d+)?)\s*(?:(?:–|-|to)\s*(\d+(?:\.\d+)?))?\s*s?$/)
  if (!found) return null
  return found[2] ? [Number(found[1]), Number(found[2])] : [CLIP_MIN, Number(found[1])]
}

/** Seconds as "2 h 04 min", "9 min 34 s", "45 s". */
export function length(total: number): string {
  if (total >= 3600) return `${Math.floor(total / 3600)} h ${pad(Math.round((total % 3600) / 60), 2)} min`
  if (total >= 60) return `${Math.floor(total / 60)} min ${pad(Math.round(total % 60), 2)} s`
  return `${Math.round(total)} s`
}
