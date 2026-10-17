<script setup lang="ts">
import { computed, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { deleteRun, thumbUrl } from '../api'
import { refresh, state } from '../store'

const rows = computed(() => state.runs)
const when = (t: number | null) => (t ? new Date(t * 1000).toLocaleString() : '—')
const owner = (id: number | null) => state.productions.find((one) => one.id === id)
const working = (status: string) => ['live', 'queued', 'paused'].includes(status)
const deleting = ref('')

async function remove(id: string) {
  if (!confirm(`Delete run ${id}, its log and every file it made that nothing else uses?`)) return
  deleting.value = id
  try {
    await deleteRun(id)
    await refresh()
  } catch (problem) {
    alert(problem instanceof Error ? problem.message : String(problem))
  } finally {
    deleting.value = ''
  }
}
</script>

<template>
  <main class="screen stack">
    <section class="card">
      <div class="head">
        <h5>Runs</h5>
        <span class="note">newest first · a run is what was asked for, its jobs are what ran</span>
      </div>
      <table class="rows">
        <thead>
          <tr>
            <th></th>
            <th>run</th>
            <th>title</th>
            <th>status</th>
            <th>jobs</th>
            <th>asked by</th>
            <th>production</th>
            <th>started</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="one in rows" :key="one.id">
            <td class="preview">
              <RouterLink v-if="one.preview" :to="`/runs/${one.id}`">
                <img :src="thumbUrl(one.preview)" alt="" loading="lazy" @error="($event.target as HTMLImageElement).style.visibility = 'hidden'" />
              </RouterLink>
            </td>
            <td class="mono"><RouterLink :to="`/runs/${one.id}`">{{ one.id }}</RouterLink></td>
            <td>{{ one.title }}</td>
            <td><span class="pill" :class="one.status">{{ one.status }}</span></td>
            <td class="mono">{{ one.jobs_done }}/{{ one.jobs_total }}</td>
            <td>{{ one.asked_by }}</td>
            <td>
              <RouterLink v-if="one.production !== null && owner(one.production)" :to="`/productions/${one.production}`">
                {{ owner(one.production)?.title }}
              </RouterLink>
              <span v-else class="muted">—</span>
            </td>
            <td class="mono muted">{{ when(one.started) }}</td>
            <td>
              <button
                class="delete"
                :disabled="working(one.status) || deleting === one.id"
                :title="working(one.status) ? 'cancel the run first' : 'delete the run and the files it made'"
                @click="remove(one.id)"
              >
                {{ deleting === one.id ? 'deleting…' : 'delete' }}
              </button>
            </td>
          </tr>
          <tr v-if="!rows.length">
            <td colspan="9" class="note">Nothing has run yet.</td>
          </tr>
        </tbody>
      </table>
    </section>
  </main>
</template>

<style scoped>
.rows {
  width: 100%;
  border-collapse: collapse;
  font-size: var(--text-sm);
}

.rows th {
  text-align: left;
  font-weight: var(--weight-medium);
  color: var(--muted);
  font-size: var(--text-xs);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  padding: var(--space-1);
  border-bottom: 1px solid var(--line);
}

.rows td {
  padding: var(--space-1);
  border-bottom: 1px solid var(--line);
  vertical-align: middle;
}

.preview {
  width: 56px;
  padding-top: 2px;
}

.preview img {
  display: block;
  width: 48px;
  height: 48px;
  object-fit: cover;
  border-radius: var(--radius-input);
  background: var(--video);
}

.delete {
  font-size: var(--text-xs);
  padding: 2px 8px;
  border-color: var(--line);
  color: var(--muted);
}

.delete:not(:disabled):hover {
  color: var(--status-error);
  border-color: var(--status-error);
}

.pill {
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  padding: 2px 6px;
  border-radius: var(--radius-btn);
  border: 1px solid var(--line);
  color: var(--muted);
}

.pill.live {
  color: var(--accent);
  border-color: var(--accent);
}

.pill.done {
  color: var(--status-ready);
}

.pill.failed {
  color: var(--status-error);
}
</style>
