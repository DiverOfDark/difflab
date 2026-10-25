<script setup lang="ts">
import { computed, ref } from 'vue'
import { clearDisk, readDisk, type Disk, type DiskRow } from '../api'

const emit = defineEmits<{ failed: [message: string]; models: [] }>()

const disk = ref<Disk | null>(null)
const busy = ref('')
const armed = ref('')

const load = async () => (disk.value = await readDisk())
load().catch((error) => emit('failed', String(error)))

function size(bytes: number) {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`
  if (bytes >= 1e6) return `${Math.round(bytes / 1e6)} MB`
  return bytes ? `${Math.max(1, Math.round(bytes / 1e3))} KB` : '0'
}

const ABOUT: Record<DiskRow['id'], [title: string, note: (d: Disk, r: DiskRow) => string]> = {
  models: ['Models', () => 'Every downloaded model file. Delete them one by one under Downloads.'],
  results: ['Results', () => 'Finished pictures and videos, and uploaded files. Kept forever; delete them in the Gallery.'],
  made: ['Production shots', () => 'What productions show: keyframes, clips and the frames chained shots hand on. They go when a production sits idle for a week.'],
  steps: ['Step cache', (d) => `Renders and edits nothing shown points at, kept so the same job does not run twice. Cleaned to ${size(d.blob_budget)} automatically; clearing means a repeated job renders again.`],
  vlm: ['VLM prompt cache', (d) => `Prompt openings the VLM has already read, so it can skip them. Capped at ${size(d.vlm_budget)}; clearing makes the next director and judge calls read from scratch.`],
  logs: ['Logs and profiles', () => 'Old server logs and timing profiles. The current log stays.'],
  records: ['Database and run records', () => 'Productions, runs, chats and their backups. Never cleared from here.'],
  other: ['Other', (_, r) => (r.names?.length ? `Nothing difflab reads any more: ${r.names.join(', ')}. Delete by hand if unneeded.` : 'Nothing else.')],
}

const total = computed(() => (disk.value?.rows ?? []).reduce((n, r) => n + r.bytes, 0))
const pending = (id: string) => disk.value?.pending.includes(id) ?? false

async function clear(r: DiskRow) {
  if (armed.value !== r.id) {
    armed.value = r.id
    setTimeout(() => armed.value === r.id && (armed.value = ''), 4000)
    return
  }
  armed.value = ''
  busy.value = r.id
  try {
    disk.value = await clearDisk(r.id)
  } catch (error) {
    emit('failed', String(error))
  } finally {
    busy.value = ''
  }
}
</script>

<template>
  <section class="card disk">
    <div class="head">
      <h5>Disk</h5>
      <span class="note">{{ disk ? size(total) : '' }}</span>
      <button class="refresh" :disabled="!disk" @click="load().catch((e) => emit('failed', String(e)))">refresh</button>
    </div>
    <p v-if="!disk" class="note lead">measuring…</p>
    <template v-else>
      <div v-for="r in disk.rows" :key="r.id" class="line">
        <div class="what">
          <span class="name">{{ ABOUT[r.id][0] }}</span>
          <span class="sub">{{ ABOUT[r.id][1](disk, r) }}</span>
        </div>
        <span class="size">{{ size(r.bytes) }}</span>
        <button v-if="r.id === 'models'" @click="emit('models')">Downloads</button>
        <button
          v-else-if="r.id in { steps: 1, vlm: 1, logs: 1 }"
          class="clear"
          :class="{ armed: armed === r.id }"
          :disabled="!r.free || busy === r.id || pending(r.id)"
          :title="pending(r.id) ? 'waits for the job in flight' : `frees ${size(r.free)}`"
          @click="clear(r)"
          @mouseleave="armed === r.id && (armed = '')"
        >
          {{ busy === r.id ? 'clearing…' : pending(r.id) ? 'after this job' : armed === r.id ? `free ${size(r.free)}?` : 'clear' }}
        </button>
        <span v-else />
      </div>
    </template>
  </section>
</template>

<style scoped>
.lead {
  margin: 0;
  padding: var(--space-2);
  font-size: 11px;
}

.refresh {
  margin-left: auto;
}

.line {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 80px 120px;
  gap: var(--space-2);
  align-items: center;
  padding: 8px var(--space-2);
}

.line + .line {
  border-top: 1px dashed var(--line);
}

.what {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}

.name {
  font-size: var(--text-sm);
}

.sub {
  font-size: 11px;
  color: var(--muted);
}

.size {
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  text-align: right;
}

/* Full column width: arming changes the label, and a button that resized slid from under the cursor. */
.line > button {
  width: 100%;
  font-size: var(--text-xs);
  padding: 2px 8px;
  white-space: nowrap;
}

.clear:not(:disabled):hover {
  color: var(--status-error);
  border-color: var(--status-error);
}

.clear.armed {
  color: var(--bg-base);
  background: var(--status-error);
  border-color: var(--status-error);
}
</style>
