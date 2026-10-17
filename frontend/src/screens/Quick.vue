<script setup lang="ts">
/** The quick modes: a picture from words, a photo restyled, one clip, an upscale. A render queues and the page follows it. */
import { ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { checkJob, listTools, startRun, startRuns, type Job } from '../api'
import Compose from '../components/Compose.vue'
import { live, queued, refresh, state } from '../store'

/** The settings come from the server - the code defaults under this machine's overrides - so the form
 *  never carries a second copy of them. */
async function blank(): Promise<{ made: Job; models: (job: Job) => Job }> {
  const tools = await listTools()
  const effective = (name: string) => ({ ...(tools.find((t) => t.name === name)?.effective ?? {}) })
  const made: Job = {
    mode: 'video',
    idea: '',
    prompt: '',
    photos: [],
    clip: null,
    finish_megapixels: null,
    grade_strength: null,
    retime_fps: null,
    aspect: '',
    video: effective('render_chunk') as Job['video'],
    edit: effective('edit_image') as Job['edit'],
    vlm: effective('write_instruction') as Job['vlm'],
    video_vlm: effective('write_video_prompt') as Job['video_vlm'],
  }
  // A kept job brings its knobs along but never its models: those are what Settings says now, or a
  // model picked there after the form was filled is silently not the one that runs.
  const parts = { video: 'render_chunk', edit: 'edit_image', vlm: 'write_instruction', video_vlm: 'write_video_prompt' } as const
  const models = (job: Job): Job => {
    const out = { ...job }
    for (const [part, tool] of Object.entries(parts) as [keyof typeof parts, string][]) {
      const roles = tools.find((t) => t.name === tool)?.settings.filter((f) => f.role).map((f) => f.name) ?? []
      out[part] = { ...job[part], ...Object.fromEntries(roles.map((name) => [name, made[part][name as keyof Job[typeof part]]])) } as never
    }
    return out
  }
  return { made, models }
}

const router = useRouter()
const job = ref<Job | null>(null)
const problems = ref<string[]>([])
const starting = ref(false)
const failure = ref('')

// A run's "again" may predate a part of the job; the server's defaults fill what it lacks.
blank()
  .then(({ made, models }) => (job.value = models({ ...made, ...state.quick })))
  .catch((error) => (failure.value = `no settings from the server: ${error}`))

// Words alone have nothing to batch over, and the References card that turns batching off is not shown.
watch(
  () => job.value?.mode,
  (mode) => {
    if (mode === 'create') state.batching = false
  },
)

const expand = (base: Job, item: string): Job =>
  base.mode === 'upscale' ? { ...base, clip: item } : { ...base, photos: [item, ...base.photos] }

/** The server is what says whether a job is renderable - the same check a render runs. A batch is
 *  checked on its first item: the rest differ only in an uploaded file, which exists. */
let pending = 0
watch(
  [job, () => state.batch, () => state.batching],
  () => {
    if (!job.value) return
    const asked = ++pending
    state.quick = job.value
    if (state.batching && !state.batch.length) {
      problems.value = ['the batch is empty: drop what each run starts from']
      return
    }
    checkJob(state.batching ? expand(job.value, state.batch[0]!) : job.value)
      .then(({ problems: found }) => {
        if (asked === pending) problems.value = found
      })
      .catch(() => undefined)
  },
  { deep: true, immediate: true },
)

async function render() {
  if (!job.value) return
  failure.value = ''
  starting.value = true
  try {
    const base = job.value
    if (state.batching) {
      await startRuns(state.batch.map((item) => expand(base, item)))
      state.batch = []
      await refresh()
      router.push('/runs')
      return
    }
    const { run: id } = await startRun(base)
    await refresh()
    router.push(`/runs/${id}`)
  } catch (error) {
    failure.value = String(error)
  } finally {
    starting.value = false
  }
}
</script>

<template>
  <p v-if="failure" class="band error top">{{ failure }}</p>
  <Compose
    v-if="job"
    v-model="job"
    v-model:batch="state.batch"
    v-model:batching="state.batching"
    :problems="problems"
    :queued="queued"
    :busy="live"
    :starting="starting"
    @render="render"
  />
</template>

<style scoped>
.band.top {
  margin: var(--space-2) var(--space-3) 0;
}
</style>
