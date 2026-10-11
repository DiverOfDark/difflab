/**
 * The server streams a run's events — a collapsed catch-up, then each row as it is written — and
 * replays from the start on every reconnect. So nothing here may accumulate outside the state
 * object: the state is rebuilt, never patched.
 */

import type { Event } from './events'
import { gb } from './production'

export type Status = 'live' | 'done' | 'failed' | 'stopped'
export type ItemState = 'pending' | 'running' | 'done' | 'failed'
/** `skipped` is a job whose artifacts were already there under their key, so nothing ran. */
export type StepState = ItemState | 'skipped'

export interface Step {
  index: number
  label: string
  mode: string
  /** The job row this was played from; `null` for a run handed one job rather than a project's list. */
  jobId: number | null
  state: StepState
  seconds: number
  error: string
  /** Relative to output/, as `job.end` names it. */
  made: string
}

export interface Item {
  index: number
  label: string
  state: ItemState
  activity: string
  seconds: number
  seed: number | null
  prompt: string
  /** Paths relative to the run directory, as the events name them. */
  opening: string
  references: string[]
  preview: string
  poster: string
  previewStep: [number, number] | null
  video: string
}

export interface Tally {
  name: string
  stages: string[]
  index: number
  calls: number
  done: number
  seconds: number
}

export interface Artifact {
  /** Relative to output/. */
  path: string
  media: 'image' | 'video'
  name: string
}

export interface RunState {
  id: string
  title: string
  status: Status
  /** One per job of the project's list, dense: `position` leaves a gap where a job was deleted. */
  steps: Step[]
  /** Which of them is in flight, -1 between two of them and after the last. */
  step: number
  /** What `job.*` says the list is, which is known from the first job rather than from the last. */
  stepsTotal: number
  items: Item[]
  elapsed: number
  error: string
  /** Relative to output/, not to the run. */
  finalVideo: string
  /** Newest last, and relative to output/ like finalVideo. */
  artifacts: Artifact[]
  /**
   * Set by `plan` only, never inferred from "are there items": a single clip emits a preview per
   * sampling step and the fold gives each one an item, so it would read as a chain.
   */
  planned: boolean
  startedAt: number
  stage: string
  /**
   * `null` when the stage counts nothing — a model loading, a VLM reporting words arriving rather
   * than a share of a known length. A take draws its bar from `fraction`, which is chunks done; one
   * clip has no chunks, so without this a `mode: video` render showed an empty bar for its length.
   */
  stageFraction: number | null
  tally: Tally | null
  vram: [number, number] | null
  ram: [number, number] | null
  log: string[]
  lastEvent: number
}

const LOG_LINES = 400
const STALE_AFTER = 15 // seconds without an event; the render's heartbeat is 2 s

export function emptyRun(id = ''): RunState {
  return {
    id,
    title: id,
    status: 'live',
    steps: [],
    step: -1,
    stepsTotal: 0,
    items: [],
    elapsed: 0,
    error: '',
    finalVideo: '',
    artifacts: [],
    planned: false,
    startedAt: 0,
    stage: '',
    stageFraction: null,
    tally: null,
    vram: null,
    ram: null,
    log: [],
    lastEvent: 0,
  }
}

function item(run: RunState, index: number): Item {
  while (run.items.length <= index) {
    run.items.push({
      index: run.items.length,
      label: `chunk ${run.items.length + 1}`,
      state: 'pending',
      activity: '',
      seconds: 0,
      seed: null,
      prompt: '',
      opening: '',
      references: [],
      preview: '',
      poster: '',
      previewStep: null,
      video: '',
    })
  }
  return run.items[index]
}

const STEP_STATES: StepState[] = ['pending', 'running', 'done', 'failed', 'skipped']

function step(run: RunState, index: number): Step {
  while (run.steps.length <= index) {
    run.steps.push({
      index: run.steps.length,
      label: '',
      mode: '',
      jobId: null,
      state: 'pending',
      seconds: 0,
      error: '',
      made: '',
    })
  }
  return run.steps[index]
}

/** Unknown kinds are ignored: the render says more than the page draws. */
/** The kinds are `core/protocol.py`'s, written into `events.ts`; a kind the page does not draw falls through. */
export function apply(run: RunState, event: Event): void {
  run.lastEvent = event.t ?? run.lastEvent
  if (run.startedAt && run.lastEvent) run.elapsed = run.lastEvent - run.startedAt

  switch (event.kind) {
    case 'run.start':
      if (run.title === run.id) run.title = event.title ?? run.id
      run.startedAt = event.t ?? 0
      break
    case 'job.begin': {
      const index = event.index
      run.stepsTotal = event.total ?? run.stepsTotal
      Object.assign(step(run, index), {
        state: 'running',
        label: event.label ?? '',
        mode: event.mode ?? '',
        jobId: typeof event.job === 'number' ? event.job : null,
      })
      run.step = index
      break
    }
    case 'job.end': {
      const index = event.index
      run.stepsTotal = event.total ?? run.stepsTotal
      const status = event.status as StepState
      Object.assign(step(run, index), {
        state: STEP_STATES.includes(status) ? status : event.ok === false ? 'failed' : 'done',
        seconds: event.seconds ?? 0,
        error: event.error ?? '',
        made: event.made ?? '',
      })
      if (run.step === index) run.step = -1
      break
    }
    case 'plan':
      run.title = event.title ?? run.title
      run.planned = true
      run.items = []
      ;(event.items ?? []).forEach((label: string, i: number) => (item(run, i).label = label))
      break
    case 'item.begin':
      Object.assign(item(run, event.index), { state: 'running', activity: 'preparing' })
      break
    case 'item.end':
      Object.assign(item(run, event.index), {
        state: 'done',
        seconds: event.seconds ?? 0,
        activity: event.activity ?? '',
      })
      break
    case 'item.info':
      Object.assign(item(run, event.index), {
        prompt: event.prompt ?? '',
        seed: event.seed ?? null,
        opening: event.opening ?? '',
        references: event.references ?? [],
      })
      break
    case 'item.video':
      Object.assign(item(run, event.index), { video: event.video ?? '', seed: event.seed ?? item(run, event.index).seed })
      break
    case 'preview':
      // A render with no chunks previews under `item: null`; the fold hangs those on a first item.
      Object.assign(item(run, event.item ?? 0), {
        preview: event.video ?? '',
        poster: event.poster ?? '',
        previewStep: [event.step ?? 0, event.steps ?? 0],
      })
      break
    case 'tally':
      run.tally = {
        name: event.name ?? '',
        stages: event.stages ?? [],
        index: event.index ?? -1,
        calls: event.calls ?? 0,
        done: event.done ?? 0,
        seconds: event.seconds ?? 0,
      }
      break
    case 'tally.end':
      run.tally = null
      break
    case 'stage.start':
      run.stage = event.label ?? ''
      run.stageFraction = null
      break
    case 'stage.progress': {
      // Two shapes, because two kinds of stage. One counts toward a known total - eight sampling
      // steps, forty references - and can fill a bar. The other counts up with no end in sight: a
      // VLM writes until it stops, and `1200 words` is the honest thing to say about it.
      const done = Number(event.done ?? 0)
      const total = Number(event.total ?? 0)
      const unit = String(event.unit ?? '')
      run.stage =
        unit === 'byte'
          ? `${event.label} · ${gb(done)} / ${gb(total)}`
          : total
            ? `${event.label} · ${done}/${total}`
            : `${event.label} · ${done}${unit ? ` ${unit}s` : ''}`
      run.stageFraction = total ? Math.min(1, done / total) : null
      break
    }
    case 'stage.end': {
      // The ✓ line a terminal used to print. Since `core/ui` stopped drawing one, this is the only
      // place a finished stage says what it produced and how long it took.
      const detail = String(event.detail ?? '').split('\n')[0]
      const seconds = event.seconds != null ? ` · ${event.seconds}s` : ''
      push(run, `${event.ok === false ? '✗' : '✓'} ${event.label ?? ''}${detail ? ` · ${detail}` : ''}${seconds}`)
      run.stage = ''
      run.stageFraction = null
      break
    }
    case 'artifact': {
      // An edit makes one picture and no chunks, so there is no item for the stage to draw. This is
      // what a run that produces files rather than a take has to show for itself.
      const path = String(event.path ?? '')
      if (path && !run.artifacts.some((a) => a.path === path)) {
        run.artifacts.push({ path, media: event.media === 'video' ? 'video' : 'image', name: String(event.name ?? '') })
      }
      break
    }
    case 'log':
      push(run, event.text ?? '')
      break
    case 'panel':
      // Until the take is planned there is no `plan` event to name the run, and "run.py" or a
      // timestamp tells nobody what is being written. The first panel does: `director · <slug>`.
      if ((run.title === run.id || run.title === 'run.py') && event.title) run.title = event.title
      push(run, `── ${event.title ?? ''}`)
      for (const [name, value] of Object.entries(event.rows ?? {})) push(run, `   ${name}: ${value}`)
      break
    case 'stats':
      if (event.vram_total) run.vram = [event.vram_used ?? 0, event.vram_total]
      if (event.ram_total) run.ram = [event.ram_used ?? 0, event.ram_total]
      break
    case 'final':
      run.finalVideo = event.video ?? ''
      break
    case 'error':
      run.error = event.text ?? ''
      break
    case 'run.end':
      run.status = event.ok ? 'done' : 'failed'
      run.error = run.error || (event.error ?? '')
      break
  }
}

function push(run: RunState, line: string): void {
  run.log.push(line)
  if (run.log.length > LOG_LINES) run.log.splice(0, run.log.length - LOG_LINES)
}

/**
 * A run whose log has stopped growing without saying `run.end` died. Said out loud rather than left
 * looking live, because a take that stopped an hour ago should not read as one still going.
 */
export function refreshStatus(run: RunState): void {
  if (run.status === 'live' && run.lastEvent && Date.now() / 1000 - run.lastEvent > STALE_AFTER) {
    run.status = 'stopped'
  }
}

/** A run of VLM jobs writes words, not files: a player, chunks and a finish would all be empty. */
export const madeNoFiles = (run: RunState) => !run.finalVideo && !run.artifacts.length && !run.items.length

export function currentIndex(run: RunState): number {
  return run.items.findIndex((it) => it.state === 'running')
}

export function pace(run: RunState): string {
  const done = run.items.filter((it) => it.state === 'done' && it.seconds).map((it) => it.seconds)
  if (!done.length) return ''
  return `${duration(done.reduce((a, b) => a + b, 0) / done.length)} a chunk`
}

export function timeLeft(run: RunState): string {
  if (run.status === 'failed') return 'stopped on an error'
  if (run.finalVideo) return 'delivered'
  // A run that is over is not "working", and a single clip never has an ETA to fall back on - it
  // has no plan to predict from - so without this every finished one read as still going.
  if (run.status !== 'live') return run.status === 'done' ? 'done' : run.status
  return 'working'
}

export function duration(seconds: number): string {
  if (!isFinite(seconds)) return '–'
  seconds = Math.max(0, Math.round(seconds))
  if (seconds < 60) return `${seconds}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${String(seconds % 60).padStart(2, '0')}s`
  return `${Math.floor(seconds / 3600)}h ${String(Math.floor((seconds % 3600) / 60)).padStart(2, '0')}m`
}
