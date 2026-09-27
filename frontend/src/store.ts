/**
 * One store: what the server lists, refreshed on a clock, and one run watched at a time. Everything
 * the page knows about a run it learns from that run's event stream, so a tab closed and reopened
 * mid-take picks up where the rows are.
 */
import { computed, reactive, shallowRef, watch } from 'vue'
import {
  listProductions,
  listRuns,
  listTools,
  readRun,
  watchRun,
  type Job,
  type ProductionSummary,
  type RunRecord,
  type RunSummary,
  type ToolInfo,
} from './api'
import { refreshStatus, type RunState } from './run'

export const state = reactive({
  runs: [] as RunSummary[],
  productions: [] as ProductionSummary[],
  tools: [] as ToolInfo[],
  /** The quick form's job, kept across screens so a run's "again" can fill it. */
  quick: null as Job | null,
})

export async function refresh(): Promise<void> {
  const [runs, productions] = await Promise.all([listRuns().catch(() => null), listProductions().catch(() => null)])
  if (runs) state.runs = runs
  if (productions) state.productions = productions
}

export async function refreshTools(production?: number): Promise<void> {
  const tools = await listTools(production).catch(() => null)
  if (tools) state.tools = tools
}

export const live = computed(() => state.runs.some((one) => one.status === 'live'))
export const queued = computed(() => state.runs.filter((one) => one.status === 'queued').length)

// --- the run being watched -----------------------------------------------------------------------
export const watching = shallowRef('')
export const run = shallowRef<RunState | null>(null)
export const record = shallowRef<RunRecord | null>(null)
let latest: RunState | null = null
let stop: (() => void) | null = null

export function unwatch(): void {
  stop?.()
  stop = null
  watching.value = ''
  run.value = null
  record.value = null
  latest = null
}

export function watch_(id: string): void {
  if (id === watching.value) return
  unwatch()
  watching.value = id
  stop = watchRun(id, (found) => {
    latest = found
    run.value = { ...found }
  })
  reread()
}

/** The spec is frozen; the jobs and artifacts are not until the run ends, so this runs again then. */
export function reread(): void {
  const id = watching.value
  if (!id) return
  readRun(id)
    .then((found) => {
      if (watching.value === id) record.value = found
    })
    .catch(() => undefined)
}

watch(
  () => run.value?.status,
  (now, before) => {
    if (before === 'live' && now && now !== 'live') reread()
  },
)

let timers: number[] = []
export function start(): void {
  refresh()
  timers = [
    window.setInterval(refresh, 10000),
    window.setInterval(() => {
      if (latest) {
        refreshStatus(latest)
        run.value = { ...latest }
      }
    }, 5000),
  ]
}

export function halt(): void {
  for (const timer of timers) clearInterval(timer)
  stop?.()
}
