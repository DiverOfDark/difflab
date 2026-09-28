<script setup lang="ts">
/** What a run that made no file has to show: each job's outputs, the last kind of job open first. */
import { computed, ref, watch } from 'vue'
import type { RunJob } from '../api'
import { duration } from '../run'
import OutputValue from './OutputValue.vue'

const props = defineProps<{ jobs: RunJob[] }>()

interface Group {
  tool: string
  title: string
  jobs: RunJob[]
  state: string
  seconds: number
}

/** "write shot 1…5" is one step to a reader, so consecutive jobs of one tool share a row. */
const groups = computed<Group[]>(() => {
  const out: Group[] = []
  for (const job of props.jobs) {
    const last = out[out.length - 1]
    if (last && last.tool === job.tool) last.jobs.push(job)
    else out.push({ tool: job.tool, title: '', jobs: [job], state: '', seconds: 0 })
  }
  for (const g of out) {
    g.title = g.jobs.length === 1 ? g.jobs[0].label || g.tool : `${g.tool.replace(/_/g, ' ')} × ${g.jobs.length}`
    const states = g.jobs.map((j) => j.state)
    g.state = states.includes('failed') ? 'failed' : states.includes('running') ? 'running' : states.every((s) => s === 'done' || s === 'skipped') ? 'done' : 'pending'
    g.seconds = g.jobs.reduce((sum, j) => sum + (j.started && j.ended ? j.ended - j.started : 0), 0)
  }
  return out
})

/** A failure is what someone opening a failed run came for; otherwise the last thing it made. */
const natural = () => {
  const failed = groups.value.findIndex((g) => g.state === 'failed')
  if (failed >= 0) return failed
  for (let i = groups.value.length - 1; i >= 0; i--) if (groups.value[i].state === 'done') return i
  return 0
}
const chosen = ref(natural())
watch(() => groups.value.length, () => (chosen.value = natural()))
const group = computed(() => groups.value[chosen.value] ?? null)

const empty = (job: RunJob) => !job.outputs || !Object.keys(job.outputs).length
</script>

<template>
  <div class="card results">
    <nav class="steps">
      <button v-for="(g, i) in groups" :key="i" class="step" :class="{ on: i === chosen }" @click="chosen = i">
        <i class="dot" :class="g.state" />
        <span class="name">{{ g.title }}</span>
        <span class="mono took">{{ g.seconds ? duration(g.seconds) : '' }}</span>
      </button>
    </nav>
    <section v-if="group" class="reader">
      <article v-for="job in group.jobs" :key="job.id" class="job">
        <header>
          <h5>{{ job.label || job.tool }}</h5>
          <span class="mono dim">{{ job.state }}{{ job.started && job.ended ? ` · ${duration(job.ended - job.started)}` : '' }}</span>
        </header>
        <div v-if="job.error" class="band error">{{ job.error }}</div>
        <p v-if="empty(job)" class="note">
          {{ job.state === 'done' || job.state === 'skipped' ? 'Returned nothing to show.' : `Not finished — ${job.state}.` }}
        </p>
        <OutputValue v-else :value="job.outputs" />
      </article>
    </section>
  </div>
</template>

<style scoped>
.results {
  display: grid;
  grid-template-columns: minmax(180px, 240px) minmax(0, 1fr);
  min-height: 480px;
}

.steps {
  display: flex;
  flex-direction: column;
  border-right: 1px solid var(--line);
  padding: var(--space-1) 0;
}

.step {
  display: grid;
  grid-template-columns: 12px minmax(0, 1fr) auto;
  gap: var(--space-1);
  align-items: center;
  padding: 6px var(--space-2);
  border: 0;
  border-left: 2px solid transparent;
  border-radius: 0;
  background: none;
  text-align: left;
  font-size: var(--text-xs);
  color: var(--muted);
}

.step.on {
  border-left-color: var(--accent);
  background: var(--raised);
  color: var(--text-primary);
}

.step .name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.took {
  font-size: 11px;
  color: var(--dim);
}

.reader {
  padding: var(--space-2);
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
  min-width: 0;
  max-height: 78vh;
  overflow-y: auto;
}

.job {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.job + .job {
  border-top: 1px solid var(--line);
  padding-top: var(--space-2);
}

.job header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: var(--space-2);
}

.job h5 {
  margin: 0;
  font-family: var(--font-heading);
  font-size: var(--text-base);
  font-weight: var(--weight-semibold);
}

.job header .mono {
  font-size: 11px;
}

@media (max-width: 720px) {
  .results {
    grid-template-columns: minmax(0, 1fr);
  }

  .steps {
    border-right: 0;
    border-bottom: 1px solid var(--line);
  }
}
</style>
