<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import {
  assembleProduction,
  drawProductionLooks,
  planProduction,
  preflight,
  readAssembled,
  readProduction,
  renderKeyframes,
  renderProduction,
  steerQueue,
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
  PHASE_LABEL,
  ago,
  phaseOf,
  phaseOfAll,
  runLive,
  runsFor,
  seconds,
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
const listOpen = ref(props.id === undefined)
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
    listOpen.value = props.id === undefined
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
/** The bible card open in the editor: a member id, or 'new' for one decided in words. */
const editing = computed(() => (stage.value === 'bible' && typeof route.query.edit === 'string' ? route.query.edit : null))
const editingMember = computed(() => production.value?.members.find((m) => String(m.id) === editing.value) ?? null)

function go(to: StageId, shot: number | null = null, member: string | null = null) {
  router.replace({ query: { stage: to, ...(shot === null ? {} : { shot: String(shot) }), ...(member === null ? {} : { member }) } })
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
const composed = computed(() => shots.value.filter((s) => s.keyframe.source === 'composed').length)
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

const stalePrompts = computed(() => shots.value.filter((s) => s.prompt.state === 'stale' && !s.prompt.pinned).length)
const canPlan = computed(() => !!production.value?.story && production.value.story_kind !== 'video')
// A re-plan with nothing changed takes every answer from the director's cache and finishes in under a
// second, so without this the button looks dead.
const replanned = ref<{ run: string; before: string; started: number } | null>(null)
const notice = ref<{ text: string; unchanged: boolean; run: string } | null>(null)
const snapshot = () => JSON.stringify(shots.value.map((s) => [s.action, s.camera, s.dialogue, s.prompt.text]))
const plan = (fresh = false) =>
  act(fresh ? 'fresh' : 'plan', async () => {
    notice.value = null
    const before = snapshot()
    // The last chapter's run ends last: the notice waits for it.
    const { runs } = await planProduction(props.id!, chapter.value, fresh)
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
    desc: 'The story split into chapters. Uploaded photos, videos and style images are matched to entities; anything unmatched waits in Bible. Nothing touches the GPU until Render.',
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
    desc: 'MiniMax H3 from each keyframe, with its own dialogue and sound from the prompt. ref2va where a shot opens on its references alone; chained shots open on the previous last frame. QA-gated, auto-retried, flagged.',
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

const stages = computed<StageRow[]>(() => {
  const p = production.value
  const list = shots.value
  const n = list.length
  const made = (pick: (s: ShotInfo) => { state: string }) => list.filter((s) => FINISHED.has(pick(s).state)).length
  const flagged = (pick: (s: ShotInfo) => { state: string }) => list.filter((s) => phaseOf(pick(s).state) === 'review').length
  const planning = !!planRun.value
  const rendering = !!renderRun.value
  const conflicts = members.value.reduce((sum, m) => sum + m.conflicts.filter((c) => !c.resolution).length, 0) + (p?.unmatched?.length ?? 0)
  const photos = Array.isArray(p?.manifest.photos) ? (p!.manifest.photos as unknown[]).length : 0
  const references = Array.isArray(p?.manifest.references) ? (p!.manifest.references as unknown[]).length : 0

  const row = (def: StageDef, phase: Phase, progress: string, review = 0): StageRow => ({
    ...def,
    phase: fresh.value ? (def.id === 'ingest' ? 'review' : 'pending') : phase,
    progress: fresh.value ? (def.id === 'ingest' ? 'waiting for a story' : '—') : progress,
    review: fresh.value ? 0 : review,
  })
  const planned = (phase: Phase) => (planning ? 'running' : phase)
  const rendered = (phase: Phase) => (rendering && phase !== 'done' ? 'running' : phase)

  return [
    row(DEFS[0], planning && !members.value.length ? 'running' : 'done', `${p?.chapters.length ?? 0} chapters · ${references || photos} references`),
    row(
      DEFS[1],
      planned(!members.value.length ? 'pending' : conflicts ? 'review' : 'done'),
      `${bibleCounts.value}${conflicts ? ` · ${conflicts} to review` : ''}`,
      conflicts,
    ),
    row(
      DEFS[2],
      planned(n ? phaseOfAll([...list.flatMap((s) => [s.keyframe, s.prompt]), ...looks.value]) : 'pending'),
      `${n} shots · ${seconds(list.reduce((s, x) => s + x.seconds, 0))} · ${toDraw.value ? `${toDraw.value} to draw` : 'all drawn'}`,
      list.filter((s) => phaseOf(s.keyframe.state) === 'review' || s.prompt.checks.length).length + looks.value.filter((c) => phaseOf(c.state) === 'review').length,
    ),
    row(DEFS[3], rendered(phaseOfAll(list.map((s) => s.clip))), `${made((s) => s.clip)} / ${n} clips`, flagged((s) => s.clip)),
    row(
      DEFS[4],
      assembling.value || rendering ? 'running' : mastered.value === inView.value.length && n ? 'done' : 'pending',
      `${mastered.value} / ${inView.value.length} chapter${inView.value.length === 1 ? '' : 's'}${assembled.value?.film ? ' · film' : ''}`,
    ),
  ]
})

const bibleCounts = computed(() => {
  const count = (kinds: string[]) => members.value.filter((m) => kinds.includes(m.kind)).length
  const said = (n: number, one: string, many: string) => (n ? `${n} ${n === 1 ? one : many}` : '')
  const style = count(['style'])
  return [
    said(count(['person', 'animal']), 'character', 'characters'),
    said(count(['environment']), 'location', 'locations'),
    said(count(['object']), 'prop', 'props'),
    style ? 'style' : '',
  ]
    .filter(Boolean)
    .join(' · ') || 'nobody yet'
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
  if (editing.value) return `plan · bible · ${editingMember.value?.kind === 'environment' ? 'location' : editingMember.value?.kind === 'object' ? 'prop' : 'character'}`
  const where = index.value >= 3 ? 'render' : 'plan'
  return `${where} · stage ${index.value + 1} of ${stages.value.length}`
})
const rendersSide = computed(() => stage.value === 'render')
const planSide = computed(() => !['render', 'assemble'].includes(stage.value))
/** Chapters in view with a master made from their clips as they are now. */
const mastered = computed(
  () =>
    (assembled.value?.chapters ?? []).filter(
      (c) => (chapter.value === null || c.position === chapter.value) && (c.finished?.current || c.native?.current),
    ).length,
)
const assemblyRun = computed(() =>
  production.value?.runs.find((r) => [`assemble:${production.value!.id}`, `film:${production.value!.id}`].includes(r.scope) && runLive(r.state)),
)
const assembling = computed(() => !!assemblyRun.value || !!assembled.value?.busy)
const watchable = computed(() =>
  stage.value === 'assemble' ? (assemblyRun.value ?? renderRun.value) : rendersSide.value ? renderRun.value : planRun.value,
)
const target = computed(() => chapter.value ?? (production.value?.chapters.length === 1 ? 0 : null))
const assemble = () => act('assemble', () => assembleProduction(props.id!, chapter.value))
const canPreview = computed(() => {
  const at = target.value
  const found = at === null ? null : assembled.value?.chapters.find((c) => c.position === at)
  return at === null ? !!assembled.value?.film : !!(found?.finished || found?.native)
})

const budget = computed(() => {
  const n = shots.value.length || 1
  const clips = shots.value.map((s) => phaseOf(s.clip.state))
  const count = (p: Phase) => clips.filter((c) => c === p).length
  return {
    done: `${(count('done') / n) * 100}%`,
    review: `${(count('review') / n) * 100}%`,
    stale: `${(count('stale') / n) * 100}%`,
    rendered: count('done'),
    remaining: shots.value.length - count('done'),
    review_n: count('review'),
    film: seconds(shots.value.reduce((s, x) => s + x.seconds, 0)),
  }
})

const HASHED: Record<StageId, keyof Production['hashes']> = {
  ingest: 'ingest',
  bible: 'bible',
  shots: 'shots',
  render: 'render',
  assemble: 'assemble',
}
const hash = computed(() => production.value?.hashes?.[HASHED[stage.value]] ?? '')
const drawEach = computed(() => Math.round((production.value?.timings?.draw_look?.seconds ?? 0) + (production.value?.timings?.judge?.seconds ?? 0)))
const perClip = computed(() => {
  const t = production.value?.timings?.render_shot
  if (!t) return ''
  const n = shots.value.length
  const each = t.per_second && n ? (t.per_second * shots.value.reduce((s, x) => s + x.seconds, 0)) / n : t.seconds
  return `~${Math.round(each)} s/clip`
})

const meta = computed(() => {
  const p = production.value
  if (!p) return fresh.value ? 'a story and photographs, then plan' : ''
  const n = p.chapters.reduce((s, c) => s + c.shots.length, 0)
  return `${p.slug} · ${p.chapters.length} chapters · ${n} shots · ${ago(p.updated)} ago`
})

function pick(id: number) {
  listOpen.value = false
  router.push(`/productions/${id}`)
}
</script>

<template>
  <div v-if="!missing" class="production">
    <aside class="rail">
      <div class="who">
        <span class="row">
          <span class="eyebrow">production</span>
          <button class="link" @click="fresh ? (listOpen = !listOpen) : router.push('/')">{{ fresh ? 'all' : '+ new' }}</button>
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

      <div v-if="listOpen" class="list">
        <span class="eyebrow">all productions · resume where it stopped</span>
        <button v-for="p in state.productions" :key="p.id" :class="{ on: p.id === id }" @click="pick(p.id)">
          <span class="line"><span class="name">{{ p.title }}</span><span class="when mono">{{ ago(p.updated) }}</span></span>
          <span class="line mono small"
            ><span class="kind">{{ (p.story_kind || 'by hand').toUpperCase() }}</span
            ><span class="dim">{{ p.chapters }} ch · {{ p.shots }} shots</span></span
          >
        </button>
        <span v-if="!state.productions.length" class="note pad">None yet.</span>
        <button class="new mono" @click="router.push('/')">+ new production</button>
      </div>

      <label v-if="production && production.chapters.length > 1" class="chapter">
        <span class="eyebrow">chapter</span>
        <select v-model="chapter">
          <option :value="null">all · {{ production.chapters.reduce((n, c) => n + c.shots.length, 0) }} shots</option>
          <option v-for="c in production.chapters" :key="c.id" :value="c.position">
            {{ String(c.position + 1).padStart(2, '0') }}{{ c.title ? ` · ${c.title}` : '' }} · {{ c.shots.length }} shots
          </option>
        </select>
      </label>

      <div class="stages">
        <template v-for="(s, i) in stages" :key="s.id">
          <div v-if="s.divider" class="divider">
            <span>{{ s.divider }}</span><i /><span v-if="perClip">{{ perClip }}</span>
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

      <div v-if="production" class="budget">
        <span class="eyebrow">{{ scope }} · clips</span>
        <div class="bar3">
          <i :style="{ width: budget.done, background: 'var(--status-ready)' }" />
          <i :style="{ width: budget.review, background: 'var(--status-degraded)' }" />
          <i :style="{ width: budget.stale, background: 'var(--status-error)' }" />
        </div>
        <div class="grid mono">
          <span class="dim">rendered</span><span>{{ budget.rendered }}</span>
          <span class="dim">remaining</span><span>{{ budget.remaining }}</span>
          <span class="dim">needs review</span><span>{{ budget.review_n }}</span>
          <span class="dim">film</span><span>{{ budget.film }}</span>
        </div>
      </div>
    </aside>

    <main class="work">
      <div class="heading">
        <div class="named">
          <span class="eyebrow">{{ crumb }}<template v-if="hash"> · <span class="mono">{{ hash }}</span></template></span>
          <h3>{{ cur.name }}</h3>
          <span class="desc">{{ cur.desc }}</span>
        </div>
        <div v-if="production" class="buttons">
          <RouterLink v-if="watchable" class="button" :to="`/runs/${watchable.id}`">Watch run</RouterLink>
          <RouterLink v-else class="button" :to="`/settings?production=${production.id}`">Settings</RouterLink>
          <button v-if="stage === 'render'" :disabled="!!busy || !shots.length" title="what would stop each chapter rendering, checked on the CPU; nothing is queued" @click="check">
            {{ busy === 'check' ? 'checking…' : 'Dry run · CPU preflight' }}
          </button>
          <template v-if="stage === 'assemble'">
            <button :disabled="!canPreview" :title="target === null ? 'the full film, with its subtitles' : 'the chapter, with its subtitles'" @click="previewing = target ?? -1">
              {{ target === null ? 'Preview film' : 'Preview chapter' }}
            </button>
            <button
              class="primary"
              :disabled="!!busy || !shots.length || assembling"
              title="joined, levelled and subtitled from the clips there are; nothing is rendered"
              @click="assemble"
            >
              {{ busy === 'assemble' ? 'queueing…' : assembling ? 'Assembling…' : target === null ? 'Assemble film' : `Assemble chapter ${String(target + 1).padStart(2, '0')}` }}
            </button>
          </template>
          <button v-if="renderPaused && rendersSide" class="primary" :disabled="!!busy" @click="resume">
            {{ busy === 'render' ? 'resuming…' : 'Resume queue' }}
          </button>
          <button v-else-if="rendersSide" class="primary" :disabled="!!busy || !shots.length || !!renderRun" @click="render">
            {{ busy === 'render' ? 'queueing…' : renderRun ? 'Rendering…' : `Render ${scope}` }}
          </button>
          <button
            v-if="planSide && shots.length"
            :disabled="!!busy || !canPlan || !!planRun"
            title="Ask the director for new shots even though nothing they are planned from changed. The cast stays; pinned shots and prompts stay."
            @click="plan(true)"
          >
            {{ busy === 'fresh' ? 'queueing…' : 'Re-plan fresh' }}
          </button>
          <button v-if="planSide" :class="{ primary: stage !== 'shots' }" :disabled="!!busy || !canPlan || !!planRun" @click="plan()">
            {{ busy === 'plan' ? 'queueing…' : planRun ? 'Planning…' : `${shots.length ? 'Re-plan' : 'Plan'} ${scope}` }}
          </button>
          <button
            v-if="stage === 'shots' && composed"
            :disabled="!!busy"
            title="compose and judge every composed keyframe, without rendering clips"
            @click="act('keyframes', () => renderKeyframes(production!.id, chapter))"
          >
            {{ busy === 'keyframes' ? 'queueing…' : `Make ${composed} keyframe${composed === 1 ? '' : 's'}` }}
          </button>
          <button
            v-if="stage === 'shots'"
            class="primary"
            :disabled="!!busy || !toDraw"
            :title="`every changed state with no picture yet, drawn from its source and judged, ~${drawEach} s each`"
            @click="act('looks', () => drawProductionLooks(production!.id, chapter))"
          >
            {{ busy === 'looks' ? 'queueing…' : toDraw ? `Generate ${toDraw} pending image${toDraw === 1 ? '' : 's'}` : 'No image pending' }}
          </button>
        </div>
      </div>

      <p v-if="failure" class="band error">{{ failure }}</p>
      <p v-if="notice" class="band" :class="notice.unchanged ? 'hint' : 'delivered'">
        {{ notice.text }}
        <RouterLink :to="`/runs/${notice.run}`">run</RouterLink>
        <button v-if="notice.unchanged" class="link" :disabled="!!busy || !canPlan" @click="plan(true)">Re-plan fresh</button>
        <button class="link" @click="notice = null">dismiss</button>
      </p>
      <p v-if="stalePrompts && !planRun" class="band stale">
        {{ stalePrompts }} of {{ shots.length }} prompts were compiled before the story, the cast or a plan changed. A render refuses them
        until the {{ scope }} {{ whole ? 'are' : 'is' }} re-planned.
        <button class="link" :disabled="!!busy || !canPlan" @click="plan()">Re-plan {{ scope }}</button>
      </p>

      <IngestStage v-if="stage === 'ingest'" :production="production" :busy="busy" :act="act" />
      <template v-else-if="production">
        <BibleStage v-if="stage === 'bible'" :production="production" :shots="shots" :editing="editing" :busy="busy" :act="act" @edit="edit" @go="go" />
        <ShotsStage v-else-if="stage === 'shots'" :production="production" :chapter="chapter" :focus="focus" :member="focusMember" :busy="busy" :act="act" @go="go" />
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
  <main v-else class="screen empty"><p class="note">No such production. <RouterLink to="/">Begin one.</RouterLink></p></main>
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

.list .when,
.list .small {
  font-size: 10px;
  color: var(--text-tertiary);
  white-space: nowrap;
}

.list .kind {
  letter-spacing: 0.06em;
  color: var(--text-secondary);
}

.list > .new {
  font-size: 11px;
  color: var(--text-secondary);
}

.pad {
  padding: var(--space-1) var(--space-2);
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
  align-items: center;
  gap: var(--space-1);
  margin: var(--space-1) calc(-1 * var(--space-2)) 0;
  padding: var(--space-2) var(--space-2) var(--space-1);
  border-top: 1px solid var(--border-strong);
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.1em;
  color: var(--text-tertiary);
  white-space: nowrap;
}

.divider i {
  flex: 1;
  min-width: 8px;
  height: 1px;
  background: var(--border);
}

.divider span:first-child {
  overflow: hidden;
  text-overflow: ellipsis;
}

.divider span:last-child {
  flex-shrink: 0;
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

.button {
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-btn);
  padding: 7px 14px;
  color: var(--text-primary);
  text-decoration: none;
  font-size: var(--text-sm);
  font-weight: var(--weight-medium);
}

.button:hover {
  border-color: var(--accent);
  color: var(--accent);
}

.band.stale {
  border: 1px solid var(--status-error);
  color: var(--text-primary);
  display: flex;
  gap: var(--space-2);
  align-items: baseline;
  flex-wrap: wrap;
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
