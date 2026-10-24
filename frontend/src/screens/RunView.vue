<script setup lang="ts">
/**
 * One run, as it happens or as it ended: the render screen while it is live, the delivery once it
 * is not, and a switch between them for a finished run someone wants to scrub through.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { deleteRun, type Job } from '../api'
import DeleteButton from '../components/DeleteButton.vue'
import DeliverScreen from '../components/DeliverScreen.vue'
import RenderScreen from '../components/RenderScreen.vue'
import { madeNoFiles } from '../run'
import { record, refresh, run, state, unwatch, watch_ } from '../store'

const props = defineProps<{ id: string }>()
const router = useRouter()
const held = ref<'render' | 'deliver' | null>(null)

const natural = computed(() => (run.value && run.value.status !== 'live' ? 'deliver' : 'render'))
const finished = computed(() => (run.value && madeNoFiles(run.value) && record.value?.jobs.length ? 'results' : 'deliver'))
const screen = computed(() => held.value ?? natural.value)

watch(
  () => props.id,
  (id) => {
    held.value = null
    watch_(id)
  },
  { immediate: true },
)

// A run this page saw finish moves on to its delivery; a click holds a screen until the next run.
watch(
  () => run.value?.status,
  (now, before) => {
    if (before === 'live' && now && now !== 'live') held.value = null
  },
)

onMounted(() => {
  if (!state.runs.length) return
})

const working = computed(() => !run.value || ['live', 'queued', 'paused'].includes(run.value.status))

async function deleted() {
  unwatch()
  await refresh()
  router.push('/runs')
}

/** From the run's own record, so the button means what it says: the job on screen, not the form. */
function again(spec: Job) {
  state.quick = JSON.parse(JSON.stringify(spec)) as Job
  router.push('/quick')
}
</script>

<template>
  <div class="switch">
    <button :class="{ on: screen === 'render' }" @click="held = 'render'">render</button>
    <button :class="{ on: screen === 'deliver' }" :disabled="!run || run.status === 'live'" @click="held = 'deliver'">
      {{ finished }}
    </button>
    <span class="mono muted">{{ id }}</span>
    <DeleteButton :remove="() => deleteRun(id)" what="the run and the files it made" why="cancel the run first" class="end" :disabled="working" @deleted="deleted" />
  </div>
  <RenderScreen v-if="screen === 'render' && run" :run="run" />
  <DeliverScreen v-else-if="screen === 'deliver' && run" :run="run" :record="record" @again="again" />
  <main v-else class="screen empty">
    <p class="note">Waiting for the worker — a cold start spends a minute loading before it writes anything.</p>
  </main>
</template>

<style scoped>
.switch {
  display: flex;
  gap: var(--space-1);
  align-items: center;
  padding: var(--space-1) var(--space-3) 0;
}

.switch button {
  font-size: var(--text-xs);
  padding: 2px 8px;
  border-color: var(--line);
  color: var(--muted);
}

.switch .end {
  margin-left: auto;
}

.switch button.on {
  color: var(--accent);
  border-color: var(--accent);
}

.empty {
  display: flex;
  align-items: center;
  justify-content: center;
}
</style>
