<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import {
  addPhotos,
  approveBible,
  assembleProduction,
  finishProduction,
  drawProductionLooks,
  planProduction,
  preflight,
  readAssembled,
  readProduction,
  renderKeyframes,
  renderProduction,
  staled,
  steerQueue,
  upload,
  type Assembled,
  type Production,
  type ShotInfo,
} from '../api'
import AssembleStage from '../components/stages/AssembleStage.vue'
import BibleStage from '../components/stages/BibleStage.vue'
import IngestStage from '../components/stages/IngestStage.vue'
import RenderStage from '../components/stages/RenderStage.vue'
import ShotsStage from '../components/stages/ShotsStage.vue'
import {
  FINISHED,
  PHASE_COLOR,
  PHASE_LABEL,
  ago,
  count,
  length,
  pad,
  phaseOf,
  phaseOfAll,
  plural,
  runLive,
  runsFor,
  standing,
  type Phase,
  type StageId,
} from '../production'
import { refresh, state } from '../store'

const props = defineProps<{ id?: number }>()
const route = useRoute()
const router = useRouter()

const production = ref<Production | null>(null)
const missing = ref(false)
/** The chapter in view by position, or null for every chapter. */
const chapter = ref<number | null>(null)
const failure = ref('')
const busy = ref('')
/** The dry run's problems per chapter position, shown on the Render stage until closed. */
const checked = ref<Record<string, string[]> | null>(null)
const listOpen = ref(false)
const assembled = ref<Assembled | null>(null)
/** The chapter position the Assemble stage plays, -1 for the film. */
const previewing = ref<number | null>(null)

async function load() {
  if (props.id === undefined) {
    production.value = null
    missing.value = false
    return
  }
  const found = await readProduction(props.id).catch(() => null)
  missing.value = !found
  // An unchanged poll is not assigned: a new object redraws every stage, and a redraw writes each
  // `:value` field back over whatever is being typed into it.
  if (JSON.stringify(found) !== JSON.stringify(production.value)) production.value = found
  const joined = found ? await readAssembled(found.id).catch(() => null) : null
  if (JSON.stringify(joined) !== JSON.stringify(assembled.value)) assembled.value = joined
  if (found && chapter.value !== null && chapter.value >= found.chapters.length) chapter.value = null
}
watch(
  () => props.id,
  () => {
    production.value = null
    assembled.value = null
    previewing.value = null
    checked.value = null
    listOpen.value = false
    load()
  },
  { immediate: true },
)
const timer = window.setInterval(load, 8000)
onUnmounted(() => clearInterval(timer))

const fresh = computed(() => props.id === undefined)
const stage = computed<StageId>(() => {
  if (fresh.value) return 'ingest'
  const asked = DEFS.find((d) => d.id === route.query.stage)
  if (asked) return asked.id
  return stages.value.find((s) => s.phase !== 'done')?.id ?? 'assemble'
})
const focus = computed(() => (route.query.shot ? Number(route.query.shot) : null))
const focusMember = computed(() => (typeof route.query.member === 'string' ? route.query.member : null))
/** Shots a STALE banner's "show affected" asked the timeline to mark. */
const affected = computed(() => (typeof route.query.affected === 'string' ? route.query.affected.split(',').map(Number) : []))
/** The bible card open in the editor: a member id, or 'new' for one decided in words. */
const editing = computed(() => (stage.value === 'bible' && typeof route.query.edit === 'string' ? route.query.edit : null))
const editingMember = computed(() => production.value?.members.find((m) => String(m.id) === editing.value) ?? null)

function go(to: StageId, shot: number | null = null, member: string | null = null) {
  const keep = to === 'shots' && affected.value.length ? { affected: affected.value.join(',') } : {}
  router.replace({ query: { stage: to, ...(shot === null ? {} : { shot: String(shot) }), ...(member === null ? {} : { member }), ...keep } })
}
const edit = (id: number | 'new' | null) => router.replace({ query: { stage: 'bible', ...(id === null ? {} : { edit: String(id) }) } })

const inView = computed(() => {
  const all = production.value?.chapters ?? []
  return chapter.value === null ? all : all.filter((c) => c.position === chapter.value)
})
const shots = computed<ShotInfo[]>(() => inView.value.flatMap((c) => c.shots))
const whole = computed(() => chapter.value === null && (production.value?.chapters.length ?? 0) > 1)
const scope = computed(() => (whole.value ? 'all chapters' : 'chapter'))
const members = computed(() => production.value?.members ?? [])
const ids = computed(() => inView.value.map((c) => c.id))
/** The shots' own looks, the amended castings: each has a picture to draw and judge. */
const looks = computed(() => shots.value.flatMap((s) => s.castings.filter((c) => c.amended)))
const toDraw = computed(() => looks.value.filter((c) => ['missing', 'failed', 'stale'].includes(c.state)).length)
const composed = computed(() => shots.value.filter((s) => s.keyframe.source === 'composed'))
const toCompose = computed(() => composed.value.filter((s) => ['missing', 'failed', 'stale'].includes(s.keyframe.state)).length)
const planRun = computed(() => (production.value ? runsFor(production.value, 'plan', ids.value).find((r) => runLive(r.state)) : undefined))
const chapterRuns = computed(() => (production.value ? runsFor(production.value, 'chapter', ids.value) : []))
const renderRun = computed(() => chapterRuns.value.find((r) => runLive(r.state)))

/** Everything the page asks of the server goes through here, so a refusal lands on the page. */
async function act(name: string, what: () => Promise<unknown>) {
  failure.value = ''
  busy.value = name
  try {
    await what()
    await Promise.all([load(), refresh()])
  } catch (error) {
    failure.value = String(error)
  } finally {
    busy.value = ''
  }
}

const stalePrompts = computed(() => shots.value.filter((s) => s.prompt.state === 'stale' && !s.prompt.pinned))
const canPlan = computed(() => !!production.value?.story && production.value.story_kind !== 'video')
// A re-plan with nothing changed takes every answer from the director's cache and finishes in under a
// second, so without this the button looks dead.
const replanned = ref<{ run: string; before: string; started: number } | null>(null)
const notice = ref<{ text: string; unchanged: boolean; run: string | null } | null>(null)
const snapshot = () => JSON.stringify(shots.value.map((s) => [s.action, s.camera, s.dialogue, s.prompt.text]))
const plan = (fresh = false, bible = false) =>
  act(bible ? 'bible' : fresh ? 'fresh' : 'plan', async () => {
    notice.value = null
    const before = snapshot()
    // The last chapter's run ends last: the notice waits for it.
    const { runs } = await planProduction(props.id!, bible ? null : chapter.value, fresh, bible)
    replanned.value = { run: runs[runs.length - 1], before, started: Date.now() }
  })
const settle = window.setInterval(() => {
  if (replanned.value && !production.value?.runs.find((r) => r.id === replanned.value!.run && !runLive(r.state))) load()
}, 1500)
onUnmounted(() => clearInterval(settle))
watch(production, (p) => {
  const asked = replanned.value
  const ran = asked && p?.runs.find((r) => r.id === asked.run)
  if (!asked || !ran || runLive(ran.state)) return
  replanned.value = null
  const took = ((Date.now() - asked.started) / 1000).toFixed(1)
  if (ran.state !== 'done') {
    notice.value = { text: `The plan run ended ${ran.state}.`, unchanged: false, run: ran.id }
    return
  }
  const before = JSON.parse(asked.before) as string[][]
  const after = JSON.parse(snapshot()) as string[][]
  const moved = after.filter((row, i) => JSON.stringify(row) !== JSON.stringify(before[i])).length + Math.max(0, before.length - after.length)
  notice.value = moved
    ? { text: `Re-planned in ${took} s: ${moved} of ${after.length} shots changed.`, unchanged: false, run: ran.id }
    : {
        text: `Re-planned in ${took} s and nothing changed: the story, the cast and the settings are what the director already answered, so every answer came from its cache. Re-plan fresh asks it again for new shots; the cast stays as it is.`,
        unchanged: true,
        run: ran.id,
      }
})
const render = () => act('render', () => renderProduction(props.id!, chapter.value))
const renderPaused = computed(() => chapterRuns.value.some((r) => r.state === 'paused'))
const resume = () => act('render', () => steerQueue(props.id!, { pause: false }))
const check = () =>
  act('check', async () => {
    const { problems } = await preflight(props.id!)
    checked.value = chapter.value === null ? problems : { [chapter.value]: problems[chapter.value] ?? [] }
  })

interface StageDef {
  id: StageId
  name: string
  artifact: string
  desc: string
  divider?: string
}

const DEFS: StageDef[] = [
  {
    id: 'ingest',
    name: 'Ingest',
    artifact: 'Chapters and references',
    desc: 'The story split into chapters. Uploaded photos, videos and style images are matched to entities; anything unmatched waits in Bible.',
  },
  {
    id: 'bible',
    name: 'Bible',
    artifact: 'Standing cast, places, props, style',
    desc: 'The standing look of everyone and everything: what a shot starts from the first time an entity appears. Changes in the story (wet clothes, a rusted blade) live on the shot that makes them, not here.',
  },
  {
    id: 'shots',
    name: 'Shots',
    artifact: 'References, opening, prompt per shot',
    desc: 'The film as a timeline of 5–15 s shots. Each shot lists its own references: a bible entity as it stands, or its version from an earlier shot plus what is different now. Opening frame and H3 prompt are written from those.',
  },
  {
    id: 'render',
    name: 'Render',
    artifact: 'Clip + audio per shot',
    divider: 'RENDER · EXPENSIVE · RUN ONCE',
    desc: 'MiniMax H3 from each opening, with its own dialogue and sound from the prompt. ref2va where a shot opens on its references alone; chained shots open on the previous last frame. QA-gated, auto-retried, flagged.',
  },
  {
    id: 'assemble',
    name: 'Assemble',
    artifact: 'Chapter videos · full film',
    desc: 'Clips joined per scene with their own audio, loudness levelled, subtitles, crossfades, chapter markers.',
  },
]

interface StageRow extends StageDef {
  phase: Phase
  progress: string
  review: number
}

/** The bible's pictures a person has not approved yet, its open conflicts, and photographs placed with no one. */
const unapproved = computed(() => members.value.flatMap((m) => m.references).filter((r) => r.artifact && !(r.state === 'approved' && r.reviewed_by === 'person')).length)
const bibleReview = computed(
  () => unapproved.value + members.value.reduce((sum, m) => sum + m.conflicts.filter((c) => !c.resolution).length, 0) + (production.value?.unmatched?.length ?? 0),
)
const takes = computed(() =>
  shots.value.reduce((t, s) => ({ judged: t.judged + (s.clip.takes?.judged ?? 0), rejected: t.rejected + (s.clip.takes?.rejected ?? 0) }), { judged: 0, rejected: 0 }),
)

const stages = computed<StageRow[]>(() => {
  const p = production.value
  const list = shots.value
  const n = list.length
  const made = (pick: (s: ShotInfo) => { state: string }) => list.filter((s) => FINISHED.has(pick(s).state)).length
  const flagged = (pick: (s: ShotInfo) => { state: string }) => list.filter((s) => phaseOf(pick(s).state) === 'review').length
  const planning = !!planRun.value
  const rendering = !!renderRun.value
  const references = Array.isArray(p?.manifest.references) ? (p!.manifest.references as unknown[]).length : Array.isArray(p?.manifest.photos) ? (p!.manifest.photos as unknown[]).length : 0

  const row = (def: StageDef, phase: Phase, progress: string, review = 0): StageRow => ({
    ...def,
    phase: fresh.value ? (def.id === 'ingest' ? 'review' : 'pending') : phase,
    progress: fresh.value ? (def.id === 'ingest' ? 'waiting for a story' : '—') : progress,
    review: fresh.value ? 0 : review,
  })
  const planned = (phase: Phase) => (planning ? 'running' : phase)
  const rendered = (phase: Phase) => (rendering && phase !== 'done' ? 'running' : phase)

  return [
    row(DEFS[0], planning && !members.value.length ? 'running' : p?.chapters.length ? 'done' : 'pending', `${plural(p?.chapters.length ?? 0, 'chapter')} · ${plural(references, 'reference')}`),
    row(DEFS[1], planned(!members.value.length ? 'pending' : bibleReview.value ? 'review' : 'done'), bibleCounts.value, bibleReview.value),
    row(
      DEFS[2],
      planned(n ? phaseOfAll([...composed.value.map((s) => s.keyframe), ...list.map((s) => s.prompt), ...looks.value]) : 'pending'),
      `${plural(n, 'shot')} · ${length(list.reduce((s, x) => s + x.seconds, 0))}${toDraw.value ? ` · ${count(toDraw.value)} to draw` : ''}`,
      list.filter((s) => phaseOf(s.keyframe.state) === 'review' || s.prompt.checks.length).length + looks.value.filter((c) => phaseOf(c.state) === 'review').length,
    ),
    row(DEFS[3], rendered(phaseOfAll(list.map((s) => s.clip))), `${count(made((s) => s.clip))} / ${count(n)}${takes.value.rejected ? ` · ${count(takes.value.rejected)} rejected` : ''}`, flagged((s) => s.clip)),
    row(
      DEFS[4],
      assembling.value ? 'running' : mastered.value === inView.value.length && n ? 'done' : 'pending',
      `${count(mastered.value)} / ${plural(inView.value.length, 'chapter')}${assembled.value?.film ? ' · film' : ''}`,
    ),
  ]
})

const bibleCounts = computed(() => {
  const kinds = (k: string[]) => members.value.filter((m) => k.includes(m.kind)).length
  const said = (n: number, one: string) => (n ? plural(n, one) : '')
  return (
    [said(kinds(['person', 'animal']), 'character'), said(kinds(['environment']), 'location'), said(kinds(['object']), 'prop'), kinds(['style']) ? 'style' : '']
      .filter(Boolean)
      .join(' · ') || 'nobody yet'
  )
})
const cur = computed(() => {
  const found = stages.value.find((s) => s.id === stage.value) ?? stages.value[0]
  if (found.id === 'ingest' && fresh.value)
    return { ...found, desc: 'A story and, optionally, references. Everything downstream is planned from these; nothing touches the GPU until Render.' }
  if (found.id !== 'bible' || !editing.value) return found
  return {
    ...found,
    name: editingMember.value?.slug ?? 'A new member',
    desc: 'One card, edited once. Everything that reads it downstream is re-keyed.',
  }
})
const index = computed(() => stages.value.findIndex((s) => s.id === cur.value.id))
const crumb = computed(() => {
  if (editing.value) return `plan · bible · ${({ environment: 'location', object: 'prop', style: 'style' } as Record<string, string>)[editingMember.value?.kind ?? ''] ?? 'character'}`
  const where = index.value >= 3 ? 'render' : 'plan'
  return `${where} · stage ${index.value + 1} of ${stages.value.length}`
})
/** Chapters in view with a master made from their clips as they are now. */
const mastered = computed(
  () =>
    (assembled.value?.chapters ?? []).filter(
      (c) => (chapter.value === null || c.position === chapter.value) && (c.finished?.current || c.native?.current),
    ).length,
)
const assemblyRun = computed(() =>
  production.value?.runs.find((r) => [`assemble:${production.value!.id}`, `finish:${production.value!.id}`, `film:${production.value!.id}`].includes(r.scope) && runLive(r.state)),
)
const assembling = computed(() => !!assemblyRun.value || !!assembled.value?.busy)
const watchable = computed(() =>
  stage.value === 'assemble' ? (assemblyRun.value ?? renderRun.value) : stage.value === 'render' ? renderRun.value : planRun.value,
)
const target = computed(() => chapter.value ?? (production.value?.chapters.length === 1 ? 0 : null))
const assemble = () => act('assemble', () => assembleProduction(props.id!, chapter.value))
const finish = () => act('finish', () => finishProduction(props.id!, chapter.value))
const canPreview = computed(() => {
  const at = target.value
  const found = at === null ? null : assembled.value?.chapters.find((c) => c.position === at)
  return at === null ? !!assembled.value?.film : !!(found?.finished || found?.native)
})

// --- the budget ---------------------------------------------------------------------------------
const timings = computed(() => production.value?.timings)
/** Measured seconds to render and judge one shot of `seconds`. */
const clipCost = (seconds: number) => {
  const r = timings.value?.render_shot
  if (!r) return 0
  return (r.per_second ? r.per_second * seconds : r.seconds) + (timings.value?.judge?.seconds ?? 0)
}
const perClip = computed(() => {
  const n = shots.value.length
  const r = timings.value?.render_shot
  if (!r) return 0
  return Math.round(n && r.per_second ? (r.per_second * shots.value.reduce((s, x) => s + x.seconds, 0)) / n : r.seconds)
})
const card = computed(() => state.gpu?.name?.match(/\d{4}\w*/)?.[0] ?? state.gpu?.name ?? 'GPU')
const budget = computed(() => {
  const n = shots.value.length
  const clips = shots.value.map((s) => phaseOf(s.clip.state))
  const of = (p: Phase) => clips.filter((c) => c === p).length
  const share = (k: number) => `${n ? (k / n) * 100 : 0}%`
  const done = of('done')
  const left = shots.value.filter((s) => phaseOf(s.clip.state) !== 'done')
  const { judged, rejected } = takes.value
  return {
    done: share(done),
    flagged: share(of('review')),
    invalid: share(of('stale')),
    rendered: n ? `${count(done)} / ${count(n)}` : '—',
    remaining: n ? plural(n - done, 'clip') : '—',
    rejected: judged ? `${count(rejected)} · ${Math.round((rejected / judged) * 100)}%` : '—',
    wall: n ? (left.length ? length(left.reduce((t, s) => t + clipCost(s.seconds), 0)) : 'none') : '—',
  }
})

// --- the stage's own actions --------------------------------------------------------------------
interface Action {
  label: string
  run: () => void
  disabled?: boolean
  title?: string
}
const picker = ref<HTMLInputElement | null>(null)
const addReferences = (event: Event) => {
  const input = event.target as HTMLInputElement
  const files = [...(input.files ?? [])]
  input.value = ''
  if (!files.length) return
  act('photos', async () => {
    const stored = await upload(files)
    await addPhotos(props.id!, stored.map((a) => a.path))
    notice.value = { text: `${plural(stored.length, 'photograph')} wait in Bible to be attached; the next plan reads them too.`, unchanged: false, run: null }
  })
}
const reingest = () =>
  act('bible', async () => {
    await planProduction(props.id!, null, false, true)
    router.replace({ query: { stage: 'ingest', ingesting: '1' } })
  })
const approve = () =>
  act('approve', async () => {
    const { left } = await approveBible(props.id!)
    if (left.length) notice.value = { text: `Approved what was made. Not approved: ${left.join('; ')}.`, unchanged: false, run: null }
  })
const ing = (name: string, label: string) => (busy.value === name ? 'queueing…' : label)
const actions = computed<{ extra?: Action; secondary: Action; primary: Action } | null>(() => {
  const p = production.value
  if (!p || editing.value) return null
  const idle = !!busy.value
  const planning = !!planRun.value
  switch (stage.value) {
    case 'ingest':
      return {
        secondary: { label: ing('photos', 'Add references'), run: () => picker.value?.click(), disabled: idle, title: 'photographs to attach to members in Bible; the next plan reads them' },
        primary: { label: planning ? 'Ingesting…' : ing('bible', 'Re-ingest'), run: reingest, disabled: idle || !canPlan.value || planning, title: 'read the story and the references again: chapters, then the bible' },
      }
    case 'bible':
      return {
        secondary: { label: ing('bible', 'Rebuild from novel'), run: () => plan(false, true), disabled: idle || !canPlan.value || planning, title: 'the cast read again from the story and the photographs; no chapter is planned' },
        primary: {
          label: busy.value === 'approve' ? 'approving…' : unapproved.value ? 'Approve bible' : 'Bible approved',
          run: approve,
          disabled: idle || !unapproved.value,
          title: `every member's picture that is made, approved as it is: ${plural(unapproved.value, 'picture')} to approve`,
        },
      }
    case 'shots':
      return {
        secondary: {
          label: planning ? 'Planning…' : ing('plan', `Re-extract ${scope.value}`),
          run: () => plan(),
          disabled: idle || !canPlan.value || planning,
          title: 'the director plans the shots again; pinned shots and prompts stay',
        },
        primary: toDraw.value
          ? {
              label: ing('looks', `Generate ${plural(toDraw.value, 'pending image')}`),
              run: () => act('looks', () => drawProductionLooks(p.id, chapter.value)),
              disabled: idle,
              title: `every changed state with no picture yet, drawn from its source and judged, ~${Math.round((timings.value?.draw_look?.seconds ?? 0) + (timings.value?.judge?.seconds ?? 0))} s each`,
            }
          : {
              label: toCompose.value ? ing('keyframes', `Make ${plural(toCompose.value, 'keyframe')}`) : 'No image pending',
              run: () => act('keyframes', () => renderKeyframes(p.id, chapter.value)),
              disabled: idle || !toCompose.value,
              title: 'compose and judge every composed keyframe, without rendering clips',
            },
      }
    case 'render':
      return {
        secondary: { label: busy.value === 'check' ? 'checking…' : 'Dry run · CPU preflight', run: check, disabled: idle || !shots.value.length, title: 'what would stop each chapter rendering, checked on the CPU; nothing is queued' },
        primary: renderPaused.value
          ? { label: busy.value === 'render' ? 'resuming…' : 'Resume queue', run: resume, disabled: idle }
          : { label: renderRun.value ? 'Rendering…' : ing('render', `Render ${scope.value}`), run: render, disabled: idle || !shots.value.length || !!renderRun.value },
      }
    default:
      return {
        secondary: {
          label: target.value === null ? 'Preview film' : 'Preview chapter',
          run: () => (previewing.value = target.value ?? -1),
          disabled: !canPreview.value,
          title: target.value === null ? 'the full film, with its subtitles' : 'the chapter, with its subtitles',
        },
        extra: {
          label: ing('finish', target.value === null ? 'Finish film' : `Finish chapter ${pad(target.value + 1, 2)}`),
          run: finish,
          disabled: idle || !shots.value.length || assembling.value,
          title: 'enlarged to 4K and retimed to 60 fps from the clips there are: review the native render first',
        },
        primary: {
          label: assembling.value ? 'Assembling…' : ing('assemble', target.value === null ? 'Assemble film' : `Assemble chapter ${pad(target.value + 1, 2)}`),
          run: assemble,
          disabled: idle || !shots.value.length || assembling.value,
          title: 'joined, levelled and subtitled from the clips there are; nothing is rendered',
        },
      }
  }
})

const HASHED: Record<StageId, keyof Production['hashes']> = { ingest: 'ingest', bible: 'bible', shots: 'shots', render: 'render', assemble: 'assemble' }
const hash = computed(() => production.value?.hashes?.[HASHED[stage.value]] ?? '')

// --- the STALE banner ---------------------------------------------------------------------------
const said = (labels: string[]) => (labels.length > 1 ? `${labels[0]} → ${labels[labels.length - 1]}` : labels[0])
const banner = computed(() => {
  const heard = staled.value && staled.value.production === props.id ? staled.value.stale : null
  if (heard) {
    const kinds = (['keyframes', 'prompts', 'clips'] as const).filter((k) => heard[k]).map((k) => (heard.shots.length > 1 ? k : k.slice(0, -1)))
    const things = kinds.length ? kinds.join(kinds.length > 2 ? ', ' : ' and ').replace(/, (\w+)$/, ' and $1') : 'what was made'
    const one = heard.shots.length === 1
    const head = one ? `Shot ${heard.shots[0].label} changed · ${things} for that shot ${kinds.length === 1 && !things.endsWith('s') ? 'is' : 'are'} stale` : `${heard.shots.length} shots changed (${said(heard.shots.map((s) => s.label))}) · their ${things} are stale`
    const chain = heard.chained.length ? `; downstream chained shots (${said([heard.shots[heard.shots.length - 1].label, ...heard.chained.map((s) => s.label)])}) will re-render` : ''
    return { text: `${head}${chain}.${heard.gpu ? ` ~${length(heard.gpu)} of GPU.` : ''}`, ids: [...heard.shots, ...heard.chained].map((s) => s.id), replan: false }
  }
  if (stalePrompts.value.length && !planRun.value)
    return {
      text: `${count(stalePrompts.value.length)} of ${plural(shots.value.length, 'prompt')} were compiled before the story, the cast or a plan changed; a render refuses them until the ${scope.value} ${whole.value ? 'are' : 'is'} re-planned.`,
      ids: stalePrompts.value.map((s) => s.id),
      replan: true,
    }
  return null
})
function showAffected(ids: number[]) {
  staled.value = null
  router.replace({ query: { stage: 'shots', shot: String(ids[0]), affected: ids.join(',') } })
}

// --- the rail's head and the productions ---------------------------------------------------------
const meta = computed(() => {
  const p = production.value
  if (!p) return fresh.value ? 'nothing ingested yet' : ''
  const kind = { text: 'story', video: 'video' }[p.story_kind] ?? 'by hand'
  const aim = typeof p.manifest.target_seconds === 'number' ? (p.manifest.target_seconds as number) : 0
  return `${kind} · ${plural(p.chapters.length, 'chapter')} · ${aim ? `target ${length(aim)}` : length(p.chapters.reduce((t, c) => t + c.shots.reduce((u, s) => u + s.seconds, 0), 0))}`
})
const listed = computed(() =>
  state.productions.map((p) => ({ ...p, ...standing(p), when: ago(p.when) === 'now' ? 'now' : `${ago(p.when)} ago`, seg: Object.values(p.stages).map((ph) => PHASE_COLOR[ph]) })),
)
function open(p: { id: number; resume: StageId }) {
  listOpen.value = false
  router.push({ path: `/productions/${p.id}`, query: { stage: p.resume } })
}
function cancelFresh() {
  const last = state.productions[0]
  if (last) open(last)
}
watch(
  () => [production.value?.title, cur.value.name, production.value?.hashes?.manifest, fresh.value],
  () => {
    const p = production.value
    state.heading = fresh.value ? 'New production · no manifest yet' : p ? `${p.title} · ${cur.value.name} · manifest ${p.hashes?.manifest ?? '—'}` : ''
  },
  { immediate: true },
)
onUnmounted(() => (state.heading = ''))
</script>

<template>
  <div v-if="!missing" class="production">
    <aside class="rail" data-screen-label="Stage rail">
      <div class="who">
        <span class="row">
          <span class="eyebrow">production</span>
          <span class="links">
            <RouterLink v-if="production" class="link" :to="`/settings?production=${production.id}`">settings</RouterLink>
            <button v-if="fresh" class="link" :disabled="!state.productions.length" @click="cancelFresh">cancel</button>
            <button v-else class="link" @click="router.push('/productions')">new production</button>
          </span>
        </span>
        <button class="title" @click="listOpen = !listOpen">
          <span>{{ production?.title ?? (fresh ? 'New production' : '…') }}</span>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="7 15 12 20 17 15" />
            <polyline points="7 9 12 4 17 9" />
          </svg>
        </button>
        <span class="meta">{{ meta }}</span>
      </div>

      <div v-if="listOpen" class="list" data-screen-label="Productions">
        <span class="eyebrow">ALL PRODUCTIONS · resume where it stopped</span>
        <button v-for="p in listed" :key="p.id" :class="{ on: p.id === id }" @click="open(p)">
          <span class="line"><span class="name">{{ p.title }}</span><span class="when mono">{{ p.when }}</span></span>
          <span class="segs"><i v-for="(c, i) in p.seg" :key="i" :style="{ background: c }" /></span>
          <span class="line mono small"><span class="st" :style="{ color: p.color }">{{ p.state }}</span><span class="dim">{{ p.meta }}</span></span>
        </button>
        <button class="new mono" @click="router.push('/productions')">+ new production</button>
      </div>

      <label v-if="production" class="chapter">
        <span class="eyebrow">chapter</span>
        <select v-model="chapter">
          <option :value="null">all · {{ plural(production.chapters.reduce((n, c) => n + c.shots.length, 0), 'shot') }}</option>
          <option v-for="c in production.chapters" :key="c.id" :value="c.position">
            {{ pad(c.position + 1, 2) }}{{ c.title ? ` · ${c.title}` : '' }} · {{ plural(c.shots.length, 'shot') }}
          </option>
        </select>
      </label>

      <div class="stages">
        <template v-for="(s, i) in stages" :key="s.id">
          <div v-if="s.divider" class="divider">
            <span>{{ s.divider }}</span><i /><span v-if="perClip">~{{ perClip }} s / clip</span>
          </div>
          <button class="stage" :class="{ on: s.id === stage }" :disabled="fresh && s.id !== 'ingest'" @click="go(s.id)">
            <span class="track"><i class="dot" :class="s.phase" /><i v-if="i < stages.length - 1" class="wire" /></span>
            <span class="what">
              <span class="name">{{ s.name }}</span>
              <span class="artifact mono">{{ s.artifact }}</span>
              <span class="progress">{{ s.progress }}</span>
            </span>
            <span class="state">
              <span class="ph" :class="s.phase">{{ PHASE_LABEL[s.phase] }}</span>
              <span v-if="s.review" class="badge mono">{{ s.review }} review</span>
            </span>
          </button>
        </template>
      </div>

      <div class="budget">
        <span class="eyebrow">BUDGET · {{ card }}{{ perClip ? ` · ~${perClip} s / clip` : '' }}</span>
        <div class="bar3">
          <i :style="{ width: budget.done, background: 'var(--status-ready)' }" />
          <i :style="{ width: budget.flagged, background: 'var(--status-degraded)' }" />
          <i :style="{ width: budget.invalid, background: 'var(--status-error)' }" />
        </div>
        <div class="grid mono">
          <span class="dim">rendered</span><span>{{ budget.rendered }}</span>
          <span class="dim">remaining</span><span>{{ budget.remaining }}</span>
          <span class="dim">rejected</span><span>{{ budget.rejected }}</span>
          <span class="dim">est. wall</span><span>{{ budget.wall }}</span>
        </div>
      </div>
    </aside>

    <main class="work">
      <div class="heading">
        <div class="named">
          <span class="eyebrow">{{ crumb }}</span>
          <h3>{{ cur.name }}</h3>
          <span class="desc">{{ cur.desc }}</span>
        </div>
        <div v-if="actions" class="buttons">
          <span class="hashline mono"
            >{{ hash ? `version ${hash}` : 'unchanged work is skipped' }}<template v-if="watchable">
              · <RouterLink :to="`/runs/${watchable.id}`">watch run</RouterLink></template
            ></span
          >
          <button v-if="actions.extra" :disabled="actions.extra.disabled" :title="actions.extra.title" @click="actions.extra.run">{{ actions.extra.label }}</button>
          <button :disabled="actions.secondary.disabled" :title="actions.secondary.title" @click="actions.secondary.run">{{ actions.secondary.label }}</button>
          <button class="primary" :disabled="actions.primary.disabled" :title="actions.primary.title" @click="actions.primary.run">{{ actions.primary.label }}</button>
          <input ref="picker" type="file" accept="image/*" multiple hidden @change="addReferences" />
        </div>
      </div>

      <div v-if="banner" class="stale-banner">
        <span class="tag">STALE</span>
        <span class="text">{{ banner.text }}</span>
        <button v-if="banner.replan" class="link" :disabled="!!busy || !canPlan" @click="plan()">re-plan {{ scope }}</button>
        <button class="link" @click="showAffected(banner.ids)">show affected</button>
      </div>
      <p v-if="failure" class="band error">{{ failure }}</p>
      <p v-if="notice" class="band" :class="notice.unchanged ? 'hint' : 'delivered'">
        {{ notice.text }}
        <RouterLink v-if="notice.run" :to="`/runs/${notice.run}`">run</RouterLink>
        <button v-if="notice.unchanged" class="link" :disabled="!!busy || !canPlan" @click="plan(true)">Re-plan fresh</button>
        <button class="link" @click="notice = null">dismiss</button>
      </p>

      <IngestStage v-if="stage === 'ingest'" :production="production" :busy="busy" :act="act" />
      <template v-else-if="production">
        <BibleStage v-if="stage === 'bible'" :production="production" :shots="shots" :editing="editing" :busy="busy" :act="act" @edit="edit" @go="go" />
        <ShotsStage
          v-else-if="stage === 'shots'"
          :production="production"
          :chapter="chapter"
          :focus="focus"
          :member="focusMember"
          :affected="affected"
          :busy="busy"
          :act="act"
          @go="go"
          @unmark="router.replace({ query: { stage: 'shots', ...(focus === null ? {} : { shot: String(focus) }) } })"
        />
        <RenderStage
          v-else-if="stage === 'render'"
          :production="production"
          :chapter="chapter"
          :focus="focus"
          :busy="busy"
          :act="act"
          :checked="checked"
          @go="go"
          @uncheck="checked = null"
        />
        <AssembleStage
          v-else-if="stage === 'assemble'"
          :production="production"
          :shots="shots"
          :chapter="chapter"
          :runs="chapterRuns"
          :assembled="assembled"
          :preview="previewing"
          :act="act"
          @preview="(at) => (previewing = at)"
        />
      </template>
    </main>
  </div>
  <main v-else class="screen empty"><p class="note">No such production. <RouterLink to="/productions">Begin one.</RouterLink></p></main>
</template>

<style scoped>
.production {
  flex: 1;
  display: grid;
  grid-template-columns: minmax(240px, 272px) minmax(0, 1fr);
  min-height: 0;
}

.rail {
  border-right: 1px solid var(--border);
  background: var(--bg-surface);
  display: flex;
  flex-direction: column;
  position: sticky;
  top: 56px;
  height: calc(100vh - 56px);
  overflow: auto;
}

.who {
  padding: var(--space-2);
  border-bottom: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--space-1);
}

.links {
  display: flex;
  gap: var(--space-1);
  align-items: center;
}

.links .link {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-secondary);
  text-decoration: none;
  white-space: nowrap;
}

.links a.link:hover {
  color: var(--accent);
}

.title {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--space-1);
  background: none;
  border: 0;
  padding: 0;
  text-align: left;
  min-width: 0;
}

.title:hover {
  color: var(--accent);
}

.title > span {
  font-family: var(--font-heading);
  font-weight: var(--weight-semibold);
  font-size: var(--text-md);
  letter-spacing: -0.01em;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.title svg {
  flex-shrink: 0;
  color: var(--text-secondary);
}

.meta {
  font-size: var(--text-xs);
  font-weight: var(--weight-light);
  color: var(--text-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.list {
  border-bottom: 1px solid var(--border-strong);
  background: var(--bg-base);
  display: flex;
  flex-direction: column;
}

.list > .eyebrow {
  padding: var(--space-1) var(--space-2) 4px;
  font-size: 10px;
  text-transform: none;
  white-space: normal;
}

.list > button {
  display: flex;
  flex-direction: column;
  gap: 4px;
  text-align: left;
  border: 0;
  border-radius: 0;
  border-top: 1px solid var(--border);
  border-left: 2px solid transparent;
  padding: var(--space-1) var(--space-2) var(--space-1) 14px;
  font-weight: var(--weight-regular);
}

.list > button:hover {
  background: var(--bg-raised);
  color: inherit;
}

.list > button.on {
  background: var(--bg-raised);
  border-left-color: var(--accent);
}

.list .line {
  display: flex;
  justify-content: space-between;
  gap: var(--space-1);
  align-items: baseline;
  min-width: 0;
}

.list .name {
  font-size: var(--text-sm);
  font-weight: var(--weight-medium);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.list .when {
  font-size: 10px;
  color: var(--text-tertiary);
  white-space: nowrap;
}

.list .small {
  font-size: 10px;
}

.list .st {
  letter-spacing: 0.06em;
  white-space: nowrap;
}

.list .dim {
  color: var(--text-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.segs {
  display: flex;
  gap: 2px;
  height: 3px;
}

.segs i {
  flex: 1;
  border-radius: 1px;
}

.list > .new {
  flex-direction: row;
  padding: var(--space-1) var(--space-2);
  font-size: 11px;
  color: var(--text-secondary);
}

.chapter {
  padding: var(--space-1) var(--space-2);
  border-bottom: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.chapter select {
  background: var(--bg-raised);
  font-family: var(--font-mono);
  font-size: var(--text-xs);
}

.stages {
  padding: var(--space-2) var(--space-2) var(--space-1);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
}

.divider {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px var(--space-1);
  margin: var(--space-1) calc(-1 * var(--space-2)) 0;
  padding: var(--space-2) var(--space-2) var(--space-1);
  border-top: 1px solid var(--border-strong);
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--text-tertiary);
  white-space: nowrap;
}

.divider span:first-child {
  letter-spacing: 0.1em;
}

.divider i {
  flex: 1;
  min-width: 8px;
  height: 1px;
  background: var(--border);
}

.divider span:last-child {
  margin-left: auto;
}

.stage {
  display: grid;
  grid-template-columns: 16px minmax(0, 1fr) auto;
  gap: 0 var(--space-1);
  align-items: start;
  text-align: left;
  border: 0;
  border-radius: 0;
  border-left: 2px solid transparent;
  padding: var(--space-1) var(--space-2) var(--space-1) 14px;
  margin: 0 calc(-1 * var(--space-2));
  font-weight: var(--weight-regular);
}

.stage:hover:not(:disabled) {
  background: var(--bg-raised);
  color: inherit;
}

.stage:disabled {
  cursor: default;
  color: inherit;
}

.stage.on {
  background: var(--bg-raised);
  border-left-color: var(--accent);
}

.stage.on .name {
  color: var(--accent);
}

.track {
  display: flex;
  flex-direction: column;
  align-items: center;
  height: 100%;
  padding-top: 4px;
}

.track .dot {
  width: 10px;
  height: 10px;
}

.wire {
  width: 1px;
  flex: 1;
  background: var(--border);
  margin-top: 4px;
  min-height: 12px;
}

.what {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}

.what .name {
  font-size: var(--text-sm);
  font-weight: var(--weight-medium);
}

.what .artifact {
  font-size: 11px;
  color: var(--text-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.what .progress {
  font-size: 11px;
  font-weight: var(--weight-light);
  color: var(--text-secondary);
}

.state {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 4px;
  padding-top: 2px;
}

.badge {
  font-size: 10px;
  padding: 1px 6px;
  border: 1px solid var(--status-degraded);
  color: var(--status-degraded);
  border-radius: var(--radius-input);
  white-space: nowrap;
}

.budget {
  margin-top: auto;
  padding: var(--space-2);
  border-top: 1px solid var(--border-strong);
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.bar3 {
  height: 4px;
  background: var(--bg-raised);
  border-radius: 2px;
  overflow: hidden;
  display: flex;
}

.budget .grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 4px var(--space-2);
  font-size: 11px;
}

.budget .grid span:nth-child(even) {
  text-align: right;
}

.budget > .eyebrow {
  text-transform: none;
}

.budget .dim {
  color: var(--text-tertiary);
}

.work {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
  padding: var(--space-3);
  max-width: 1200px;
  width: 100%;
}

.heading {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: var(--space-2);
  flex-wrap: wrap;
}

.named {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}

.named h3 {
  font-size: var(--text-xl);
  margin: 0;
}

.desc {
  font-size: var(--text-sm);
  font-weight: var(--weight-light);
  color: var(--text-secondary);
  max-width: 64ch;
  text-wrap: pretty;
}

.buttons {
  display: flex;
  gap: var(--space-1);
  align-items: center;
  flex-wrap: wrap;
}

.hashline {
  font-size: var(--text-xs);
  color: var(--text-secondary);
  white-space: nowrap;
}

.hashline a {
  color: var(--text-secondary);
}

.hashline a:hover {
  color: var(--accent);
}

.stale-banner {
  border: 1px solid var(--status-degraded);
  border-radius: var(--radius-card);
  background: var(--bg-surface);
  padding: var(--space-1) var(--space-2);
  display: flex;
  align-items: center;
  gap: var(--space-2);
  flex-wrap: wrap;
}

.stale-banner .tag {
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.08em;
  color: var(--status-degraded);
}

.stale-banner .text {
  font-size: var(--text-xs);
  font-weight: var(--weight-light);
  flex: 1 1 320px;
}

.empty {
  display: flex;
  align-items: center;
  justify-content: center;
}

@media (max-width: 900px) {
  .production {
    grid-template-columns: 1fr;
  }

  .rail {
    position: static;
    height: auto;
  }
}
</style>
