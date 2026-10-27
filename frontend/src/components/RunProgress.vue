<script setup lang="ts">
/** A run's progress in a line, for a card that only points at the run. */
import { computed, onBeforeUnmount, shallowRef, watch } from 'vue'
import { runFile, watchRun } from '../api'
import { duration, type RunState } from '../run'
import { state } from '../store'

const props = defineProps<{ id: string }>()

const run = shallowRef<RunState | null>(null)
let stop: (() => void) | null = null
let watched = ''

// A queued run's stream is a 404, which an EventSource never retries; it opens once the card has the run.
watch(
  () => [props.id, props.id === state.gpu?.run] as const,
  ([id, onCard]) => {
    if (id !== watched) {
      stop?.()
      stop = null
      run.value = null
    }
    if (!onCard || stop) return
    watched = id
    stop = watchRun(id, (found) => (run.value = { ...found }))
  },
  { immediate: true },
)
onBeforeUnmount(() => stop?.())

const percent = computed(() => {
  const r = run.value
  if (!r) return 0
  const share = r.planned ? r.items.filter((it) => it.state === 'done').length / Math.max(1, r.items.length) : (r.stageFraction ?? 0)
  return Math.round(100 * share)
})
const crawling = computed(() => !!run.value && !run.value.planned && run.value.stageFraction === null)

const where = computed(() => {
  const r = run.value
  if (!r) return ''
  const parts: string[] = []
  if (r.stepsTotal > 1) parts.push(`job ${(r.step >= 0 ? r.step : r.steps.length - 1) + 1}/${r.stepsTotal}`)
  if (r.planned) parts.push(`chunk ${Math.min(r.items.filter((it) => it.state === 'done').length + 1, r.items.length)}/${r.items.length}`)
  const t = r.tally
  if (t && t.index >= 0) parts.push(`${t.index + 1}/${t.stages.length} ${t.stages[t.index]}`)
  parts.push(duration(r.elapsed))
  return parts.join(' · ')
})

const poster = computed(() => {
  const r = run.value
  const it = r?.items.filter((one) => one.poster || one.opening).at(-1)
  if (r && it) return runFile(r.id, it.poster || it.opening)
  const made = r?.artifacts.filter((a) => a.media === 'image').at(-1)
  return made ? `/output/${made.path}` : ''
})
</script>

<template>
  <div v-if="run" class="run-progress">
    <img v-if="poster" :src="poster" alt="" />
    <div class="lines">
      <span class="mono stage">{{ run.stage || run.title }}</span>
      <div class="bar" :class="{ waiting: crawling }"><i :style="{ width: percent + '%' }" /></div>
      <span class="mono muted where">{{ where }}</span>
    </div>
  </div>
</template>

<style scoped>
.run-progress {
  display: flex;
  gap: 8px;
  align-items: center;
  font-size: var(--text-xs);
  max-width: 480px;
}

.run-progress img {
  height: 48px;
  border-radius: var(--radius-input);
}

.lines {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.stage,
.where {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.bar.waiting {
  background: linear-gradient(90deg, var(--surface) 0 40%, var(--accent-muted) 50%, var(--surface) 60% 100%);
  background-size: 250% 100%;
  animation: crawl 1.8s linear infinite;
}

.bar.waiting > i {
  display: none;
}

@keyframes crawl {
  from {
    background-position: 100% 0;
  }
  to {
    background-position: -150% 0;
  }
}
</style>
