<script setup lang="ts">
import { computed } from 'vue'
import { RouterLink } from 'vue-router'
import { approveMade, control, deleteShot, outputUrl, rerollShot, type Production, type ShotInfo } from '../../api'
import { FINISHED, phaseOf, shotId, type StageId } from '../../production'
import { state } from '../../store'

const props = defineProps<{
  production: Production
  shots: ShotInfo[]
  focus: number | null
  run: Production['runs'][number] | undefined
  busy: string
  act: (name: string, what: () => Promise<unknown>) => Promise<void>
}>()
const emit = defineEmits<{ go: [to: StageId, shot?: number | null] }>()

const made = computed(() => props.shots.filter((s) => FINISHED.has(s.clip.state)).length)
const flagged = computed(() => props.shots.filter((s) => phaseOf(s.clip.state) === 'review').length)
const n = computed(() => props.shots.length || 1)
const summary = computed(() => (props.run ? state.runs.find((r) => r.id === props.run!.id) : undefined))
const now = computed(() => props.shots.find((s) => s.clip.state === 'running'))
const paused = computed(() => summary.value?.status === 'paused')

/** Flagged first: they are what a person is needed for. Then whatever is picked, then made. */
const clips = computed(() => {
  const rank = (s: ShotInfo) => (phaseOf(s.clip.state) === 'review' ? 0 : s.id === props.focus ? 1 : FINISHED.has(s.clip.state) ? 2 : 3)
  return props.shots.filter((s) => s.clip.artifact || phaseOf(s.clip.state) === 'review').sort((a, b) => rank(a) - rank(b))
})

const reroll = (s: ShotInfo) => {
  const typed = prompt(`Seed for shot ${shotId(s)} (blank: a random one)`, '')
  if (typed === null) return
  props.act(`reroll ${s.id}`, () => rerollShot(s.id, typed.trim() === '' ? undefined : Number(typed)))
}
const remove = (s: ShotInfo) => {
  if (confirm(`Delete shot ${shotId(s)}? Its clip stays on disk.`)) props.act(`remove ${s.id}`, () => deleteShot(s.id))
}
const toggle = () => props.act('pause', () => control(props.run!.id, { cmd: paused.value ? 'resume' : 'pause' }))
</script>

<template>
  <div class="render">
    <section class="card top">
      <div class="progress">
        <div class="line">
          <span class="title">Render queue · one pass</span>
          <span class="mono muted small">{{ made }} / {{ shots.length }} clips{{ flagged ? ` · ${flagged} flagged` : '' }}</span>
        </div>
        <div class="bar3">
          <i :style="{ width: `${(made / n) * 100}%`, background: 'var(--status-ready)' }" />
          <i :style="{ width: `${(flagged / n) * 100}%`, background: 'var(--status-degraded)' }" />
        </div>
        <div class="facts mono">
          <span>NOW <b>{{ now ? shotId(now) : run ? 'between jobs' : 'idle' }}</b></span>
          <span v-if="summary">JOBS <b>{{ summary.jobs_done }} / {{ summary.jobs_total }}</b></span>
          <span>RUN <b>{{ run ? `${run.id} · ${summary?.status ?? run.state}` : 'none' }}</b></span>
        </div>
      </div>
      <div v-if="run" class="buttons">
        <button :disabled="!!busy" @click="toggle">{{ paused ? 'Resume' : 'Pause' }}</button>
        <RouterLink class="button" :to="`/runs/${run.id}`">Watch run →</RouterLink>
      </div>
    </section>

    <div class="pair">
      <section class="card">
        <div class="head slim"><h5>Queue</h5><span class="mono dim small">chapter order · chained shots adjacent</span></div>
        <button v-for="s in shots" :key="s.id" class="q" :class="{ on: s.id === focus }" @click="emit('go', 'render', s.id)">
          <i class="dot" :class="phaseOf(s.clip.state)" />
          <span class="mono">{{ shotId(s) }}</span>
          <span class="act">{{ s.chained ? '│ ' : '' }}{{ s.action }}</span>
          <span class="mono dim small">{{ s.seconds }} s</span>
          <span class="ph" :class="phaseOf(s.clip.state)">{{ s.clip.state.toUpperCase() }}</span>
        </button>
      </section>

      <section class="card">
        <div class="head slim"><h5>Clips</h5><span class="mono dim small">flagged first · below the judge's gate after its retries</span></div>
        <p v-if="!clips.length" class="note pad">Nothing rendered yet.</p>
        <div v-for="s in clips" :key="s.id" class="clip" :class="{ on: s.id === focus }">
          <video v-if="s.clip.artifact && s.clip.present" :src="outputUrl(s.clip.artifact)" controls preload="metadata" />
          <div v-else class="frame empty mono">{{ shotId(s) }}</div>
          <div class="what">
            <span class="mono">{{ shotId(s) }} <span class="ph" :class="phaseOf(s.clip.state)">{{ s.clip.state }}</span></span>
            <span :class="s.clip.review ? 'warn' : 'why'">{{ s.clip.review || s.action }}</span>
            <span class="mono dim tiny">seed {{ s.clip.seed }}{{ s.clip.handoff !== null ? ` · hands off at frame ${s.clip.handoff}` : '' }}</span>
            <span class="acts">
              <button
                v-if="s.clip.state === 'needs_review'"
                class="link"
                :disabled="!!busy"
                title="Approve the clip as it is; a run paused on it goes on"
                @click="act(`accept ${s.id}`, () => approveMade('clip', s.clip.id))"
              >
                accept
              </button>
              <span v-else-if="s.clip.state === 'approved' && s.clip.reviewed_by === 'person'" class="mono dim tiny">accepted</span>
              <button class="link" :disabled="!!busy" @click="reroll(s)">re-roll</button>
              <button class="link" @click="emit('go', 'shots', s.id)">back to keyframe</button>
              <button class="link" @click="emit('go', 'shots', s.id)">edit shot</button>
              <button class="link del" :disabled="!!busy" @click="remove(s)">delete</button>
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
}

.facts b {
  font-weight: var(--weight-regular);
  color: var(--text-primary);
}

.buttons {
  display: flex;
  gap: var(--space-1);
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

.pair {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr));
  gap: var(--space-3);
  align-items: start;
}

.q {
  display: grid;
  grid-template-columns: 16px 64px minmax(0, 1fr) 64px 88px;
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

.q .dot {
  justify-self: center;
}

.act {
  font-weight: var(--weight-light);
  color: var(--text-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.q .ph {
  text-align: right;
}

.pad {
  padding: var(--space-2);
}

.clip {
  display: grid;
  grid-template-columns: 120px minmax(0, 1fr);
  gap: var(--space-2);
  padding: var(--space-2);
  border-bottom: 1px solid var(--border);
}

.clip.on {
  background: var(--bg-raised);
}

.clip video {
  width: 120px;
  max-height: 200px;
  background: var(--video);
  border: 1px solid var(--border);
  border-radius: 2px;
}

.empty {
  aspect-ratio: 3 / 4;
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
  font-size: var(--text-xs);
  line-height: 1.45;
}

.why {
  font-weight: var(--weight-light);
  color: var(--text-secondary);
}

.warn {
  color: var(--status-degraded);
}

.acts {
  display: flex;
  gap: var(--space-2);
  flex-wrap: wrap;
  margin-top: 4px;
}

.del:hover:not(:disabled) {
  color: var(--status-error);
}
</style>
