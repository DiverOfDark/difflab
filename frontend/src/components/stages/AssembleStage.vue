<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { RouterLink } from 'vue-router'
import { listTools, outputUrl, readRun, type Production, type ToolInfo } from '../../api'
import { ago, seconds } from '../../production'

const props = defineProps<{ production: Production; runs: Production['runs'] }>()

interface Joined {
  run: string
  state: string
  created: number
  video: string
  seconds: number | null
}
const joined = ref<Joined[]>([])
const known = new Map<string, Joined>()

// Only a finished run is cached: a live one has not written its assemble job's outputs yet.
watch(
  () => props.runs.map((r) => `${r.id}:${r.state}`).join(),
  async () => {
    const out: Joined[] = []
    for (const r of props.runs) {
      let found = known.get(r.id)
      if (!found) {
        const record = await readRun(r.id).catch(() => null)
        // The finished chapter once there is one; the native join until then.
        const job =
          record?.jobs.find((j) => j.tool === 'finish' && typeof j.outputs.video === 'string') ??
          record?.jobs.find((j) => j.tool === 'assemble')
        found = {
          run: r.id,
          state: r.state,
          created: r.created,
          video: typeof job?.outputs.video === 'string' ? job.outputs.video : '',
          seconds: typeof job?.outputs.seconds === 'number' ? job.outputs.seconds : null,
        }
        if (r.state === 'done' || r.state === 'failed') known.set(r.id, found)
      }
      out.push(found)
    }
    joined.value = out
  },
  { immediate: true },
)
const latest = computed(() => joined.value.find((j) => j.video))

const tool = ref<ToolInfo | null>(null)
listTools(props.production.id)
  .then((tools) => (tool.value = tools.find((t) => t.name === 'handoff') ?? null))
  .catch(() => undefined)
</script>

<template>
  <div class="assemble">
    <section class="card player">
      <div class="head slim">
        <h5>Chapter video</h5>
        <span class="mono dim small">{{ latest ? `${latest.video}${latest.seconds ? ` · ${seconds(latest.seconds)}` : ''}` : 'not assembled yet' }}</span>
      </div>
      <div class="body">
        <video v-if="latest" :key="latest.video" :src="outputUrl(latest.video)" controls preload="metadata" />
        <p v-else class="note">A chapter run ends with one assemble job; its video appears here.</p>
      </div>
    </section>

    <div class="pair">
      <section class="card">
        <div class="head slim"><h5>Chapter runs</h5><span class="mono dim small">newest first · a run with nothing new skips every job</span></div>
        <div v-for="j in joined" :key="j.run" class="row">
          <RouterLink class="mono" :to="`/runs/${j.run}`">{{ j.run }}</RouterLink>
          <span class="ph" :class="j.state === 'done' ? 'done' : j.state === 'failed' ? 'stale' : 'running'">{{ j.state.toUpperCase() }}</span>
          <a v-if="j.video" class="mono small" :href="outputUrl(j.video)" target="_blank">{{ j.video.split('/').at(-1) }}</a>
          <span v-else class="mono dim small">—</span>
          <span class="mono dim small">{{ ago(j.created) }}</span>
        </div>
        <p v-if="!joined.length" class="note pad">No chapter run yet.</p>
      </section>

      <section class="card">
        <div class="head slim">
          <h5>Knobs</h5>
          <RouterLink class="mono small" :to="`/settings?production=${production.id}`">change in settings →</RouterLink>
        </div>
        <div class="pad">
          <div v-for="f in tool?.settings.filter((x) => !x.advanced) ?? []" :key="f.name" class="knob">
            <div>
              <span class="name">{{ f.label || f.name }}</span>
              <span class="hint">{{ f.hint }}</span>
            </div>
            <span class="mono value">{{ tool?.effective?.[f.name] ?? f.default }}{{ f.unit ? ` ${f.unit}` : '' }}</span>
          </div>
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.assemble {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.head {
  align-items: center !important;
  flex-wrap: wrap;
}

.small {
  font-size: 11px;
}

.player video {
  width: 100%;
  max-height: 70vh;
  background: var(--video);
  border-radius: 2px;
}

.pair {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 400px), 1fr));
  gap: var(--space-3);
  align-items: start;
}

.row {
  display: grid;
  grid-template-columns: 150px 72px minmax(0, 1fr) 48px;
  gap: var(--space-2);
  align-items: center;
  padding: var(--space-1) var(--space-2);
  border-bottom: 1px solid var(--border);
  font-size: var(--text-xs);
}

.row a {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.pad {
  padding: var(--space-1) var(--space-2) var(--space-2);
}

.value {
  text-align: right;
  font-size: var(--text-xs);
}
</style>
