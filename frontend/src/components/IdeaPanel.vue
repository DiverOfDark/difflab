<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue'
import { RouterLink } from 'vue-router'
import { adoptIdea, draftIdea, improveIdea, listIdeas, readRun, type IdeaRound, type Production } from '../api'
import { ago } from '../production'

/** A story not yet begun: what the director is given, before there is a production to hold its rounds. */
export interface Draft {
  story: string
  photos: string[]
  take_seconds: number
  seconds: number
}

const props = defineProps<{
  production?: Production | null
  draft?: Draft
  busy: string
  act: (name: string, what: () => Promise<unknown>) => Promise<void>
}>()
const emit = defineEmits<{ use: [story: string] }>()

const rounds = ref<IdeaRound[]>([])
const picked = ref<number | null>(null)
const feedback = ref('')
const view = ref<'story' | 'current' | 'previous'>('story')
const note = ref('')

watch(
  () => props.production?.ideas,
  (given) => {
    if (!props.production) return
    rounds.value = given ?? []
    if (picked.value === null || !rounds.value.some((r) => r.round === picked.value)) picked.value = rounds.value.at(-1)?.round ?? null
  },
  { immediate: true },
)

const story = computed(() => props.production?.story ?? props.draft?.story ?? '')
const LIVE = new Set(['queued', 'running', 'blocked'])
const live = computed(() => rounds.value.find((r) => LIVE.has(r.state)))
const shown = computed(() => rounds.value.find((r) => r.round === picked.value))
const revised = computed(() => rounds.value.find((r) => r.round === shown.value?.revises))
const inUse = (r: IdeaRound) => !!r.story && r.story.trim() === story.value.trim()
const source = computed(() => rounds.value.at(-1)?.source ?? story.value)
/** The story was written by hand since: neither the idea the rounds start from nor one of them. */
const handWritten = computed(() => rounds.value.length > 0 && story.value.trim() !== source.value.trim() && !rounds.value.some(inUse))
const ready = computed(() => !!story.value.trim() && (!!props.production || !!props.draft?.photos.length))

/** A draft round's state and answer, read off its run's one job. */
async function readDraft(round: IdeaRound): Promise<IdeaRound> {
  const job = (await readRun(round.run).catch(() => null))?.jobs[0]
  if (!job) return round
  const out = job.outputs as { story?: string; changes?: string[]; questions?: string[] }
  return {
    ...round,
    state: job.state,
    error: job.error,
    story: out.story ?? '',
    changes: out.changes ?? [],
    questions: out.questions ?? [],
    seconds: job.started && job.ended ? Math.round(job.ended - job.started) : null,
  }
}

// A round is one VLM job of about a minute; the production read polls every 8 s, which is too slow to watch.
const now = ref(Date.now())
const poll = window.setInterval(async () => {
  now.value = Date.now()
  if (!live.value) return
  const was = live.value.round
  const fresh = props.production
    ? await listIdeas(props.production.id).catch(() => null)
    : await Promise.all(rounds.value.map((r) => (LIVE.has(r.state) ? readDraft(r) : r)))
  if (!fresh) return
  rounds.value = fresh
  if (!fresh.some((r) => LIVE.has(r.state))) {
    picked.value = was
    view.value = 'story'
    if (props.production) await props.act('idea done', async () => {})
  }
}, 1500)
onUnmounted(() => clearInterval(poll))
const running = computed(() => (live.value ? Math.max(0, Math.round(now.value / 1000 - live.value.created)) : 0))

type Ask = { feedback?: string; from_round?: number; idea?: string }

async function askDraft(body: Ask) {
  const d = props.draft!
  const previous = rounds.value.find((r) => r.round === body.from_round)
  const round = (rounds.value.at(-1)?.round ?? 0) + 1
  const idea = body.idea ?? previous?.source ?? source.value
  const { run } = await draftIdea({ ...d, idea, round, feedback: body.feedback, previous: previous?.story })
  rounds.value.push({
    ...{ round, source: idea, feedback: previous ? (body.feedback ?? '') : '', revises: previous?.round ?? null, story: '' },
    ...{ changes: [], questions: [], run, state: 'queued', error: '', seconds: null, adopted: false, created: Date.now() / 1000 },
  })
}

const ask = (body: Ask) =>
  props.act('improve', async () => {
    note.value = ''
    if (props.production) {
      await improveIdea(props.production.id, body)
      rounds.value = await listIdeas(props.production.id)
    } else await askDraft(body)
    picked.value = rounds.value.at(-1)?.round ?? null
  })
const improve = () => ask({})
const again = () => ask({ feedback: feedback.value, from_round: picked.value! })
const rewriteCurrent = () => ask({ idea: story.value })
const adopt = () => {
  const p = props.production
  if (!p) {
    emit('use', shown.value!.story)
    note.value = `Round ${picked.value} is in the story box now.`
    return
  }
  return props.act('adopt', async () => {
    const { stale } = await adoptIdea(p.id, picked.value!)
    const shots = p.chapters.reduce((s, c) => s + c.shots.length, 0)
    note.value = shots
      ? `Round ${picked.value} is the story now. ${stale.prompts} prompts and ${stale.clips} clips went stale: re-plan the chapter to plan it from these words.`
      : `Round ${picked.value} is the story now. Plan the chapter to see it as shots.`
  })
}

watch(picked, () => {
  feedback.value = ''
  if (view.value === 'previous' && !revised.value) view.value = 'story'
})

// --- a word diff, so a revision shows what it moved -----------------------------------------------------
interface Piece {
  kind: 'same' | 'add' | 'del'
  text: string
}

function diff(before: string, after: string): Piece[] {
  const a = before.split(/(\s+)/)
  const b = after.split(/(\s+)/)
  const n = a.length
  const m = b.length
  const table = new Int32Array((n + 1) * (m + 1))
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      table[i * (m + 1) + j] = a[i] === b[j] ? table[(i + 1) * (m + 1) + j + 1] + 1 : Math.max(table[(i + 1) * (m + 1) + j], table[i * (m + 1) + j + 1])
  const out: Piece[] = []
  const push = (kind: Piece['kind'], text: string) => {
    const last = out.at(-1)
    if (last && last.kind === kind) last.text += text
    else out.push({ kind, text })
  }
  let i = 0
  let j = 0
  while (i < n && j < m) {
    if (a[i] === b[j]) {
      push('same', a[i++])
      j++
    } else if (table[(i + 1) * (m + 1) + j] >= table[i * (m + 1) + j + 1]) push('del', a[i++])
    else push('add', b[j++])
  }
  while (i < n) push('del', a[i++])
  while (j < m) push('add', b[j++])
  return out
}

const pieces = computed<Piece[]>(() => {
  const r = shown.value
  if (!r?.story) return []
  if (view.value === 'current') return diff(story.value, r.story)
  if (view.value === 'previous' && revised.value) return diff(revised.value.story, r.story)
  return [{ kind: 'same', text: r.story }]
})
</script>

<template>
  <section class="card idea">
    <div class="head">
      <h5>Improve with director</h5>
      <div v-if="rounds.length" class="chips">
        <button v-for="r in rounds" :key="r.round" class="chip" :class="{ on: r.round === picked }" :title="r.feedback || 'a rewrite of the idea'" @click="picked = r.round">
          <i class="dot" :class="LIVE.has(r.state) ? 'running' : r.state === 'done' || r.state === 'skipped' ? (inUse(r) ? 'done' : 'pending') : 'failed'" />
          R{{ r.round }}
        </button>
      </div>
    </div>
    <div class="body">
      <template v-if="!rounds.length">
        <p class="note">
          The director (the same VLM that plans) reads your idea and the photographs and rewrites it into a story the planner renders well:
          the place first, one beat per shot, every change and every exit spelled out, anything unusual described as the camera sees it, and a
          sound line. You read it, say what to change, ask again, and use the version you like. Nothing changes until you do.
        </p>
        <div class="buttons">
          <button class="primary" :disabled="!!busy || !ready" @click="improve">{{ busy === 'improve' ? 'queueing…' : 'Improve' }}</button>
          <span v-if="!ready" class="note">{{ story.trim() ? 'needs a photograph: the director writes the story around it' : 'needs a story' }}</span>
        </div>
      </template>

      <p v-if="live" class="band working mono">
        <i class="dot running" />
        <span>Round {{ live.round }} · {{ live.state }} · {{ running }} s</span>
        <RouterLink :to="`/runs/${live.run}`">watch run →</RouterLink>
      </p>

      <template v-if="shown && !LIVE.has(shown.state)">
        <div class="meta mono">
          <span>round {{ shown.round }}{{ shown.revises ? ` · revises R${shown.revises}` : ' · a rewrite of the idea' }}</span>
          <span v-if="shown.seconds !== null" class="dim">{{ shown.seconds }} s</span>
          <span class="dim">{{ ago(shown.created) }} ago</span>
          <span v-if="inUse(shown)" class="ph done">IN USE</span>
          <span v-else-if="shown.adopted" class="ph review">ADOPTED, EDITED SINCE</span>
        </div>
        <p v-if="shown.feedback" class="asked">“{{ shown.feedback }}”</p>

        <p v-if="!shown.story" class="band error">Round {{ shown.round }} ended {{ shown.state }}{{ shown.error ? `: ${shown.error}` : '' }}</p>
        <template v-else>
          <div class="seg">
            <button :class="{ on: view === 'story' }" @click="view = 'story'">proposed</button>
            <button :class="{ on: view === 'current' }" :disabled="inUse(shown)" @click="view = 'current'">vs current story</button>
            <button v-if="revised" :class="{ on: view === 'previous' }" @click="view = 'previous'">vs R{{ revised.round }}</button>
          </div>
          <p class="story" :class="{ diffed: view !== 'story' }">
            <template v-for="(p, i) in pieces" :key="i"
              ><span :class="p.kind">{{ p.text }}</span></template
            >
          </p>
          <div v-if="shown.changes.length" class="list">
            <span class="eyebrow">what changed</span>
            <ul>
              <li v-for="(c, i) in shown.changes" :key="i">{{ c }}</li>
            </ul>
          </div>
          <div v-if="shown.questions.length" class="list">
            <span class="eyebrow">open questions</span>
            <ul>
              <li v-for="(q, i) in shown.questions" :key="i">{{ q }}</li>
            </ul>
          </div>
        </template>

        <label class="field">
          <span class="eyebrow">feedback on round {{ shown.round }}</span>
          <textarea v-model="feedback" rows="3" class="prose" placeholder="make it a fireball, not a stream of flame" />
        </label>
        <div class="buttons">
          <button class="primary" :disabled="!!busy || !!live || !shown.story || inUse(shown)" @click="adopt">
            {{ busy === 'adopt' ? 'saving…' : inUse(shown) ? 'In use' : 'Use this' }}
          </button>
          <button :disabled="!!busy || !!live || !shown.story || !feedback.trim()" @click="again">Again with feedback</button>
          <button :disabled="!!busy || !!live" title="A new rewrite of the idea, not a revision of this round" @click="improve">New version</button>
          <button v-if="handWritten" class="link" :disabled="!!busy || !!live" @click="rewriteCurrent">rewrite the current story instead</button>
        </div>
      </template>

      <p v-if="note" class="band hint">{{ note }}</p>

      <details v-if="rounds.length" class="source">
        <summary class="eyebrow">the idea it starts from</summary>
        <p class="story">{{ source }}</p>
      </details>
    </div>
  </section>
</template>

<style scoped>
.head {
  align-items: center !important;
  flex-wrap: wrap;
}

.chips .chip {
  padding: 4px 10px;
  display: flex;
  align-items: center;
  gap: 6px;
}

.buttons {
  display: flex;
  gap: var(--space-1);
  align-items: center;
  flex-wrap: wrap;
}

.meta {
  display: flex;
  gap: var(--space-2);
  align-items: baseline;
  flex-wrap: wrap;
  font-size: 11px;
  color: var(--text-secondary);
}

.asked {
  margin: 0;
  font-size: var(--text-sm);
  font-weight: var(--weight-light);
  color: var(--text-secondary);
  border-left: 2px solid var(--border-strong);
  padding-left: var(--space-1);
}

.seg {
  align-self: flex-start;
}

.seg > button:disabled {
  opacity: 0.4;
}

.story {
  font-size: var(--text-sm);
  font-weight: var(--weight-light);
  line-height: 1.55;
  white-space: pre-wrap;
  margin: 0;
}

.story .add {
  background: var(--done-back);
  color: var(--done);
}

.story .del {
  background: var(--fail-back);
  color: var(--fail);
  text-decoration: line-through;
}

.list {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding-top: var(--space-2);
  border-top: 1px solid var(--border);
}

.list ul {
  margin: 0;
  padding-left: 18px;
  font-size: var(--text-sm);
  font-weight: var(--weight-light);
  line-height: 1.45;
}

.field {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.card textarea.prose {
  font-family: var(--font-body);
  font-weight: var(--weight-light);
  line-height: 1.5;
  padding: var(--space-1) var(--space-2);
}

.band.working {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  background: var(--bg-surface);
  border: 1px solid var(--border);
  color: var(--text-secondary);
  font-size: 11px;
}

.band.working a {
  margin-left: auto;
  color: var(--accent);
  text-decoration: none;
}

.source {
  padding-top: var(--space-2);
  border-top: 1px solid var(--border);
}

.source summary {
  cursor: pointer;
}

.source .story {
  margin-top: var(--space-1);
  color: var(--text-secondary);
}
</style>
