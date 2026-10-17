<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue'
import { cancelIngest, readIngesting, skipStep, type Ingesting, type Production } from '../../api'
import { mmss } from '../../production'

const props = defineProps<{
  production: Production
  busy: string
  act: (name: string, what: () => Promise<unknown>) => Promise<void>
}>()
const emit = defineEmits<{ done: [cancelled: boolean] }>()

const now = ref<Ingesting | null>(null)
const asked = ref(Date.now())
async function load() {
  now.value = await readIngesting(props.production.id).catch(() => now.value)
}
load()
const timer = window.setInterval(load, 1500)
onUnmounted(() => window.clearInterval(timer))

// The plan runs on after the bible is built, planning shots; the page moves on to the bible then.
watch(now, (state) => {
  if (!state || Date.now() - asked.value < 2500) return
  if (state.bible_done || !state.live.length) emit('done', !state.bible_done)
})

const STATE: Record<string, string> = { done: 'DONE', running: 'RUNNING', waiting: 'WAITING', pending: 'PENDING', failed: 'FAILED' }
const steps = computed(() =>
  (now.value?.steps ?? []).map((s) => ({
    ...s,
    label: STATE[s.state] ?? s.state.toUpperCase(),
    rows: s.rows.length > 6 ? [...s.rows.slice(0, 5), { k: `+ ${s.rows.length - 5} more`, v: '' }] : s.rows,
  })),
)
const current = computed(() => steps.value.find((s) => s.id === now.value?.current) ?? null)
const index = computed(() => Math.max(0, steps.value.findIndex((s) => s.id === now.value?.current)))
const WHAT: Record<string, string> = { split: 'splitting', pages: 'reading pages', describe: 'describing references', read: 'reading chapters', reconcile: 'reconciling', plan: 'writing the bible' }
const line = computed(() => (now.value ? `step ${Math.min(index.value + 1, 4)} of 4 · ${WHAT[now.value.current] ?? now.value.current}` : 'starting…'))
const clock = (t: number) => new Date(t * 1000).toTimeString().slice(0, 8)
const TAG: Record<string, string> = { DESCRIBE: 'var(--status-building)', READ: 'var(--status-building)', RECONCILE: 'var(--status-degraded)', PLAN: 'var(--accent)' }
const log = computed(() => [...(now.value?.log ?? [])].reverse())
const FILL: Record<string, string> = { CHARACTER: 'var(--bg-raised)', LOCATION: 'oklch(22% 0.02 260)', PROP: 'oklch(24% 0.02 160)', STYLE: 'oklch(26% 0.03 80)' }
const footer = computed(() => {
  const plan = now.value?.plan
  return plan?.jobs
    ? `Then plans every chapter: ${plan.settled} of ${plan.jobs} shot-planning jobs done so far, and stops before Render. The bible opens as soon as it is built.`
    : 'Builds the bible and stops there for review.'
})
const skipTitle = computed(() =>
  current.value?.skippable
    ? `End ${current.value.name} now: what it has not done is handed on as nothing, and the later steps run on what exists`
    : current.value
      ? `${current.value.name} cannot be skipped: every later step reads all of it`
      : '',
)

const skip = () => props.act('skip', () => skipStep(props.production.id, current.value!.id).then(load))
const cancelling = ref(false)
const cancel = () => {
  if (!confirm('Cancel the plan? What is done so far stays; the bible keeps what was found.')) return
  cancelling.value = true
  props.act('cancel', () => cancelIngest(props.production.id).then(load))
}
</script>

<template>
  <div class="ingesting">
    <section class="card head-card">
      <div class="top">
        <span class="title">{{ production.title }} · ingesting</span><span class="mono line">{{ line }}</span>
      </div>
      <div class="progress"><i :style="{ width: `${Math.round((now?.progress ?? 0) * 100)}%` }" /></div>
      <div class="facts mono">
        <span>ELAPSED <b>{{ mmss(now?.elapsed ?? 0) }}</b></span>
        <span>ETA <b>{{ now ? mmss(now.eta) : '—' }}</b></span>
        <span>MODEL <b>{{ now?.model ?? '—' }}{{ now?.vram ? ` · ${now.vram.toFixed(1)} GB` : '' }}</b></span>
        <span>GPU RENDER <b>none</b></span>
      </div>
    </section>

    <div class="steps">
      <article v-for="s in steps" :key="s.id" class="step" :class="s.state">
        <div class="step-head">
          <span class="name"><i class="dot" :class="s.state === 'waiting' ? 'pending' : s.state === 'failed' ? 'review' : s.state" />{{ s.name }}</span>
          <span class="mono state">{{ s.label }}</span>
        </div>
        <div class="step-body">
          <span class="desc">{{
            {
              split: 'Chapters on headings, or every ~600 words.',
              pages: 'Panels found in each page, each read for its balloons, cast and join; then cleaned of lettering.',
              describe: 'Qwen-VL writes what each upload shows; duplicates merge by face and hash.',
              read: 'Characters, locations, props extracted chapter by chapter.',
              reconcile: 'References override the story; conflicting prose found; voices assigned.',
            }[s.id]
          }}</span>
          <div class="rows">
            <div v-for="(r, i) in s.rows" :key="i" class="mono"><span :class="{ dim: r.v === '…' }">{{ r.k }}</span><span class="dim">{{ r.v }}</span></div>
          </div>
        </div>
        <div v-if="s.state === 'running' || s.state === 'waiting'" class="step-bar"><i :style="{ width: `${Math.round(s.bar * 100)}%` }" /></div>
      </article>
    </div>

    <div class="lower">
      <section class="card">
        <div class="head"><h5>Found so far</h5><span class="mono dim small">entities appear as chapters are read</span></div>
        <div class="found-list">
        <div v-for="f in now?.found ?? []" :key="f.slug" class="found">
          <span class="swatch" :style="{ background: FILL[f.kind] ?? 'transparent' }" />
          <span class="who"><span class="mono">{{ f.slug }}</span><span class="desc">{{ f.desc }}</span></span>
          <span class="mono kind">{{ f.kind }}</span>
          <span class="mono src" :class="{ given: f.src.startsWith('PROVIDED') }">{{ f.src }}</span>
        </div>
        <p v-if="!now?.found.length" class="note pad">Nothing yet: the first chapter is still being read.</p>
        </div>
      </section>
      <section class="card">
        <div class="head"><h5>Log</h5><span class="mono dim small">plan stages · no GPU render</span></div>
        <div class="log">
          <div v-for="(l, i) in log" :key="i" class="mono entry">
            <span class="dim">{{ clock(l.t) }}</span><span class="tag" :style="{ color: TAG[l.tag] ?? 'var(--text-tertiary)' }">{{ l.tag }}</span><span class="msg">{{ l.msg }}</span><span class="dim dur">{{ l.dur }}</span>
          </div>
        </div>
      </section>
    </div>

    <section class="foot">
      <span class="note">{{ cancelling ? 'Cancelling: the job in flight finishes first, then nothing more of this plan runs.' : footer }}</span>
      <div class="buttons">
        <button class="cancel" :disabled="!!busy || cancelling || !now?.live.length" @click="cancel">{{ cancelling ? 'Cancelling…' : 'Cancel' }}</button>
        <button :disabled="!!busy || cancelling || !current?.skippable" :title="skipTitle" @click="skip">
          {{ busy === 'skip' ? 'skipping…' : `Skip ${current ? current.name.toLowerCase() : 'step'}` }}
        </button>
      </div>
    </section>
  </div>
</template>

<style scoped>
.ingesting {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.head-card {
  padding: var(--space-2) var(--space-3);
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.top {
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
}

.line {
  font-size: var(--text-xs);
  color: var(--text-secondary);
  white-space: nowrap;
}

.progress,
.step-bar {
  background: var(--bg-raised);
  overflow: hidden;
}

.progress {
  height: 4px;
  border-radius: 2px;
}

.progress > i,
.step-bar > i {
  display: block;
  height: 100%;
  background: var(--accent);
  transition: width 300ms ease;
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

.steps {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr));
  gap: var(--space-2);
  align-items: start;
}

.step {
  border: 1px solid var(--border);
  border-radius: var(--radius-card);
  background: var(--bg-surface);
  display: flex;
  flex-direction: column;
}

.step.pending {
  opacity: 0.5;
}

.step.running {
  border-color: var(--status-building);
}

.step.failed {
  border-color: var(--status-error);
}

.step-head {
  padding: var(--space-1) var(--space-2);
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--space-1);
}

.step-head .name {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  font-size: var(--text-sm);
  font-weight: var(--weight-medium);
}

.step-head .state {
  font-size: 10px;
  letter-spacing: 0.08em;
  color: var(--text-tertiary);
}

.step.done .state {
  color: var(--status-ready);
}

.step.running .state {
  color: var(--status-building);
}

.step.running .dot {
  background: var(--status-building);
}

.step.failed .state {
  color: var(--status-error);
}

.step-body {
  padding: var(--space-2);
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  min-height: 104px;
}

.step-body .desc {
  font-size: var(--text-xs);
  font-weight: var(--weight-light);
  color: var(--text-secondary);
  line-height: 1.45;
}

.rows {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.rows > div {
  display: flex;
  justify-content: space-between;
  gap: var(--space-2);
  font-size: 11px;
}

.rows span:first-child {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.rows span:first-child {
  flex: 0 1 auto;
}

.rows span:last-child {
  flex: 0 1 auto;
  max-width: 60%;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  text-align: right;
}

.step-bar {
  height: 2px;
}

.step-bar > i {
  background: var(--status-building);
}

.lower {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr));
  gap: var(--space-3);
  align-items: start;
}

.head {
  padding: var(--space-1) var(--space-2) !important;
}

.head h5 {
  font-size: var(--text-base) !important;
}

.small {
  font-size: 11px;
}

.found-list {
  max-height: 480px;
  overflow: auto;
}

.found {
  display: grid;
  grid-template-columns: 24px minmax(0, 1fr) 72px 104px;
  gap: var(--space-2);
  align-items: center;
  padding: var(--space-1) var(--space-2);
  border-bottom: 1px solid var(--border);
  font-size: var(--text-xs);
}

.swatch {
  width: 24px;
  height: 24px;
  border: 1px solid var(--border);
  border-radius: 2px;
}

.who {
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.who .desc {
  font-size: 11px;
  font-weight: var(--weight-light);
  color: var(--text-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.kind,
.src {
  font-size: 10px;
  letter-spacing: 0.08em;
  color: var(--text-tertiary);
}

.src {
  text-align: right;
  white-space: nowrap;
  color: var(--text-secondary);
}

.src.given {
  color: var(--accent);
}

.pad {
  padding: var(--space-2);
}

.log {
  padding: var(--space-1) var(--space-2);
  max-height: 264px;
  overflow: auto;
  display: flex;
  flex-direction: column-reverse;
}

.entry {
  display: grid;
  grid-template-columns: 64px 88px minmax(0, 1fr) 48px;
  gap: var(--space-2);
  padding: 4px 0;
  font-size: var(--text-xs);
  line-height: 1.45;
}

.tag {
  letter-spacing: 0.06em;
  font-size: 10px;
  padding-top: 2px;
}

.msg {
  overflow-wrap: anywhere;
}

.dur {
  text-align: right;
}

.foot {
  border-top: 1px solid var(--border-strong);
  padding-top: var(--space-2);
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--space-2);
  flex-wrap: wrap;
}

.buttons {
  display: flex;
  gap: var(--space-1);
}

.cancel {
  border-color: transparent;
  color: var(--status-error);
}
</style>
