<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue'
import { approveMade, outputUrl, readQueue, redoKeyframe, rerollShot, steerQueue, type Production, type QueueRow, type RenderQueue } from '../../api'
import { length, type StageId } from '../../production'

const props = defineProps<{
  production: Production
  chapter: number | null
  focus: number | null
  busy: string
  act: (name: string, what: () => Promise<unknown>) => Promise<void>
  /** The dry run's answer, per chapter position; null until asked. */
  checked: Record<string, string[]> | null
}>()
const emit = defineEmits<{ go: [to: StageId, shot?: number | null]; uncheck: [] }>()

const queue = ref<RenderQueue | null>(null)
const failure = ref('')
let timer = 0
async function load() {
  window.clearTimeout(timer)
  try {
    queue.value = await readQueue(props.production.id, props.chapter)
    failure.value = ''
  } catch (error) {
    failure.value = String(error)
  }
  // A sampling step is ~40 s and the heartbeat 2 s: often enough to see the step move, not every heartbeat.
  timer = window.setTimeout(load, queue.value && queue.value.status !== 'idle' ? 2500 : 10000)
}
watch(() => [props.production, props.chapter], load, { immediate: true })
onUnmounted(() => window.clearTimeout(timer))

const q = computed(() => queue.value)
const pct = (n: number) => `${q.value && q.value.total ? (n / q.value.total) * 100 : 0}%`
const queueLine = computed(() => (q.value ? `${q.value.done} / ${q.value.total} clips · ${q.value.status}` : '…'))
const nowShot = computed(() => (q.value?.now ? `${q.value.now.label} · ${q.value.now.doing}` : '—'))
const eta = computed(() => {
  const e = q.value?.eta
  if (!e || !e.clips) return '—'
  return e.queued ? length(e.seconds) : `~${length(e.seconds)} · ${e.clips} not queued`
})
const rejectRate = computed(() => {
  const v = q.value
  if (!v || !v.judged) return '—'
  return `${Math.round((v.rejected / v.judged) * 100)}% · ${v.retaken} re-rolled`
})
const vram = computed(() => (q.value?.vram ? `${q.value.vram[0].toFixed(1)} / ${q.value.vram[1].toFixed(1)} GB` : '—'))
const live = computed(() => !!q.value && q.value.status !== 'idle')
const paused = computed(() => q.value?.status === 'paused')
const reviews = computed(() => q.value?.rows.filter((r) => r.review) ?? [])

const steer = (name: string, body: { pause?: boolean; auto_retry?: boolean }) =>
  props.act(name, async () => {
    await steerQueue(props.production.id, body)
    await load()
  })
const toggle = () => steer('pause', { pause: !paused.value })
const retry = (event: Event) => steer('retry', { auto_retry: (event.target as HTMLInputElement).checked })

const reroll = (r: QueueRow) => props.act(`reroll ${r.id}`, () => rerollShot(r.id))
const backToKeyframe = (r: QueueRow) => {
  if (r.opening !== 'keyframe') emit('go', 'shots', r.id)
  else props.act(`keyframe ${r.id}`, () => redoKeyframe(r.id, ''))
}
const score = (r: QueueRow) => (r.review?.score === null || r.review?.score === undefined ? '—' : r.review.score.toFixed(2))
const chapters = computed(() => Object.entries(props.checked ?? {}).sort(([a], [b]) => Number(a) - Number(b)))
const problems = computed(() => chapters.value.reduce((n, [, list]) => n + list.length, 0))
</script>

<template>
  <div class="render">
    <p v-if="failure" class="band error">{{ failure }}</p>
    <section v-if="checked" class="card check">
      <div class="head slim">
        <h5>Dry run · CPU preflight</h5>
        <span class="mono dim small">{{ problems ? `${problems} to fix · nothing queued` : `ready · ${chapters.length} chapter${chapters.length === 1 ? '' : 's'} · nothing queued` }}</span>
        <button class="link" @click="emit('uncheck')">close</button>
      </div>
      <template v-for="[at, list] in chapters" :key="at">
        <div v-if="list.length" class="chapter">
          <span class="mono dim small">ch {{ String(Number(at) + 1).padStart(2, '0') }}</span>
          <ul>
            <li v-for="(p, i) in list" :key="i">{{ p }}</li>
          </ul>
        </div>
      </template>
      <p v-if="!problems" class="note pad">Every chapter in view would queue: prompts compiled, keyframes and looks in place.</p>
    </section>

    <section class="card top">
      <div class="progress">
        <div class="line">
          <span class="title">Render queue · one pass</span>
          <span class="mono muted queueline">{{ queueLine }}</span>
        </div>
        <div class="bar3">
          <i :style="{ width: pct(q?.done ?? 0), background: 'var(--status-ready)' }" />
          <i :style="{ width: pct(q?.flagged ?? 0), background: 'var(--status-degraded)' }" />
        </div>
        <div class="facts mono">
          <span>NOW <b>{{ nowShot }}</b></span><span>ETA <b>{{ eta }}</b></span><span>REJECTION <b>{{ rejectRate }}</b></span><span>VRAM <b>{{ vram }}</b></span>
        </div>
      </div>
      <div class="buttons">
        <label class="retry" :title="`a clip below the gate (${q?.gate ?? '…'}) is rendered again on new seeds, ${q?.takes ?? 3} takes in all, and the best kept for review; off, the run pauses on it`">
          <input type="checkbox" :checked="q?.auto_retry ?? true" :disabled="!q || !!busy" @change="retry" />auto-retry below gate
        </label>
        <button v-if="live" class="ghost" :disabled="!!busy" @click="toggle">{{ paused ? 'Resume queue' : 'Pause after this clip' }}</button>
      </div>
    </section>

    <div class="pair">
      <section class="card">
        <div class="head slim"><h5>Queue</h5><span class="mono dim small">chapter order · chained shots adjacent</span></div>
        <div class="rows">
          <button v-for="r in q?.rows ?? []" :key="r.id" class="q" :class="{ on: r.id === focus }" @click="emit('go', 'render', r.id)">
            <i class="qdot" :class="r.state" />
            <span class="mono">{{ r.label }}</span>
            <span class="act" :title="r.action">{{ r.opening === 'chain' ? '│ ' : '' }}{{ r.action }}</span>
            <span class="mono preset">{{ r.preset }}</span>
            <span class="qstate" :class="r.state">{{ r.text.toUpperCase() }}</span>
          </button>
        </div>
      </section>

      <section class="card">
        <div class="head slim">
          <h5>Needs review</h5>
          <span class="mono dim small">{{ q?.auto_retry ? `survived ${q.takes} takes below gate` : 'below gate · auto-retry off' }}</span>
        </div>
        <p v-if="!reviews.length" class="note pad">Nothing to review.</p>
        <div v-for="r in reviews" :key="r.id" class="clip" :class="{ on: r.id === focus }">
          <video v-if="r.clip.video" class="thumb" :src="outputUrl(r.clip.video)" muted loop preload="metadata" @mouseenter="($event.target as HTMLVideoElement).play()" @mouseleave="($event.target as HTMLVideoElement).pause()" />
          <img v-else-if="r.still" class="thumb" :src="outputUrl(r.still)" alt="" />
          <div v-else class="thumb empty mono">{{ r.label }}</div>
          <div class="what">
            <span class="why">{{ r.review!.why }}</span>
            <span class="mono tiny dim">{{ r.label }} · best of {{ r.review!.takes }} · score {{ score(r) }} · {{ r.seconds.toFixed(1) }} s</span>
            <span class="acts mono">
              <button
                class="link"
                :disabled="!!busy"
                title="approve the clip as it is; a run paused on it goes on"
                @click="act(`accept ${r.id}`, () => approveMade('clip', r.clip.id))"
              >
                accept
              </button>
              <button class="link" :disabled="!!busy" title="a new seed, and the chapter queued again" @click="reroll(r)">re-roll</button>
              <button
                class="link"
                :disabled="!!busy"
                :title="r.opening === 'keyframe' ? 'compose the opening frame again, keyframes only' : 'this shot opens on no keyframe of its own: open it in Shots'"
                @click="backToKeyframe(r)"
              >
                back to keyframe
              </button>
              <button class="link" @click="emit('go', 'shots', r.id)">edit shot</button>
            </span>
          </div>
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.render {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.top {
  padding: var(--space-2) var(--space-3);
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2) var(--space-3);
  align-items: center;
}

.progress {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  min-width: 0;
  flex: 1 1 400px;
}

.line {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: var(--space-2);
  flex-wrap: wrap;
}

.title {
  font-family: var(--font-heading);
  font-weight: var(--weight-semibold);
  font-size: var(--text-md);
  white-space: nowrap;
}

.queueline {
  font-size: 12px;
  color: var(--text-secondary);
  white-space: nowrap;
}

.small {
  font-size: 11px;
}

.tiny {
  font-size: 10px;
}

.bar3 {
  height: 4px;
  background: var(--bg-raised);
  border-radius: 2px;
  overflow: hidden;
  display: flex;
}

.facts {
  display: flex;
  gap: var(--space-2) var(--space-3);
  flex-wrap: wrap;
  font-size: 11px;
  color: var(--text-tertiary);
  letter-spacing: 0.04em;
  white-space: nowrap;
}

.facts b {
  font-weight: var(--weight-regular);
  color: var(--text-primary);
}

.buttons {
  display: flex;
  gap: var(--space-1);
  white-space: nowrap;
  flex-wrap: wrap;
  align-items: center;
}

.retry {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  font-size: var(--text-xs);
  font-weight: var(--weight-light);
  color: var(--text-secondary);
  margin-right: var(--space-1);
}

.retry input {
  accent-color: var(--accent);
}

.ghost {
  background: transparent;
  border: 1px solid var(--border-strong);
  padding: 8px 16px;
  font-size: 13px;
  font-weight: var(--weight-medium);
}

.ghost:hover:not(:disabled) {
  border-color: var(--accent);
  color: var(--accent);
}

.pair {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr));
  gap: var(--space-3);
  align-items: start;
}

.rows {
  max-height: 70vh;
  overflow: auto;
}

.q {
  display: grid;
  grid-template-columns: 16px 56px minmax(0, 1fr) 64px 112px;
  gap: var(--space-2);
  align-items: center;
  width: 100%;
  text-align: left;
  border: 0;
  border-radius: 0;
  border-bottom: 1px solid var(--border);
  padding: var(--space-1) var(--space-2);
  font-size: var(--text-xs);
  font-weight: var(--weight-regular);
}

.q:hover,
.q.on {
  background: var(--bg-raised);
  color: inherit;
}

.qdot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  box-sizing: border-box;
  justify-self: center;
  border: 1.5px solid var(--border-strong);
}

.qdot.done,
.qdot.running,
.qdot.review,
.qdot.blocked,
.qdot.failed {
  border: 0;
}

.qdot.done {
  background: var(--status-ready);
}

.qdot.running {
  background: var(--status-building);
  animation: pulse 1.4s ease-in-out infinite;
}

.qdot.review {
  background: var(--status-degraded);
}

.qdot.blocked,
.qdot.failed {
  background: var(--status-error);
}

.act {
  font-weight: var(--weight-light);
  color: var(--text-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.preset {
  font-size: 11px;
  color: var(--text-tertiary);
  text-align: right;
}

.qstate {
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.06em;
  text-align: right;
  white-space: nowrap;
  color: var(--text-tertiary);
}

.qstate.done {
  color: var(--status-ready);
}

.qstate.running {
  color: var(--status-building);
}

.qstate.review {
  color: var(--status-degraded);
}

.qstate.blocked,
.qstate.failed {
  color: var(--status-error);
}

.pad {
  padding: var(--space-2);
}

.clip {
  display: grid;
  grid-template-columns: 96px minmax(0, 1fr);
  gap: var(--space-2);
  padding: var(--space-2);
  border-bottom: 1px solid var(--border);
}

.clip.on {
  background: var(--bg-raised);
}

.thumb {
  width: 96px;
  aspect-ratio: 16 / 9;
  object-fit: cover;
  background: var(--bg-raised);
  border: 1px solid var(--border);
  border-radius: 2px;
}

.empty {
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  color: var(--text-tertiary);
}

.what {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}

.why {
  font-size: var(--text-xs);
  line-height: 1.45;
}

.acts {
  display: flex;
  gap: var(--space-2);
  flex-wrap: wrap;
  margin-top: 4px;
  font-size: 11px;
}

.acts .link {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-secondary);
}

.acts .link:hover:not(:disabled) {
  color: var(--accent);
}

.check .head {
  gap: var(--space-2);
}

.check .head span {
  margin-left: auto;
}

.chapter {
  display: grid;
  grid-template-columns: 48px minmax(0, 1fr);
  gap: var(--space-2);
  padding: var(--space-1) var(--space-2);
  border-bottom: 1px solid var(--border);
}

.chapter ul {
  margin: 0;
  padding-left: var(--space-2);
  font-size: var(--text-xs);
  line-height: 1.5;
  color: var(--status-error);
}
</style>
