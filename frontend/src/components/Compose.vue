<script setup lang="ts">
/** A field a mode ignores is hidden rather than greyed: a disabled box still asks to be read. */
import { computed, ref, watch } from 'vue'
import type { Job } from '../api'
import PhotoPicker from './PhotoPicker.vue'

const job = defineModel<Job>({ required: true })
const batch = defineModel<string[]>('batch', { required: true })
const batching = defineModel<boolean>('batching', { required: true })
const props = defineProps<{
  problems: string[]
  /** Runs ahead of this one: the card renders one at a time and the worker keeps the order. */
  queued: number
  busy: boolean
  starting: boolean
}>()
const emit = defineEmits<{ render: [] }>()

const MODES = [
  { id: 'create', give: 'words alone', get: 'a new picture' },
  { id: 'video', give: 'a photo and words', get: 'one clip with audio' },
  { id: 'edit', give: 'a photo and words', get: 'a picture, restyled' },
  { id: 'upscale', give: 'a finished mp4', get: 'the same cut, larger' },
] as const

const USES: Record<string, string[]> = {
  create: ['words', 'edit', 'vlm'],
  video: ['words', 'photos', 'video', 'delivery', 'vlm'],
  edit: ['words', 'photos', 'edit', 'vlm'],
  upscale: ['clip', 'finish', 'delivery'],
}
const uses = computed(() => USES[job.value.mode] ?? [])
const shows = (part: string) => uses.value.includes(part)

// Its own state, not read off which box has text: an empty prompt has to be choosable, and the job may
// carry only one of the two, so switching moves the words across rather than copying them.
const chosen = ref<'idea' | 'prompt'>(job.value.prompt ? 'prompt' : 'idea')
watch(
  () => job.value.prompt,
  (prompt) => {
    if (prompt) chosen.value = 'prompt'
  },
)
const words = computed<'idea' | 'prompt'>({
  get: () => chosen.value,
  set: (kind) => {
    chosen.value = kind
    if (kind === 'prompt') {
      job.value.prompt ||= job.value.idea
      job.value.idea = ''
    } else {
      job.value.idea ||= job.value.prompt
      job.value.prompt = ''
    }
  },
})

const wordsNote = computed(() =>
  words.value === 'prompt'
    ? 'Used exactly as written. No VLM is loaded, which is about ninety seconds and 17 GB saved, and nothing rewrites what you typed.'
    : 'A rough sentence in your own words. The VLM turns it into the caption the model is handed, and it is loaded and evicted before any weights that draw.',
)

/**
 * Every row is named for the field it writes, because that is what the Python calls it: renaming
 * `megapixels` to "Quality" makes the two unsearchable from each other.
 */
interface Knob {
  key: string
  hint: string
  get: () => unknown
  set: (value: string) => void
  type?: 'number' | 'text'
  options?: [string, string][]
}

/** What the chosen engine will run: the Turbo's own 8, the base model's or the AIO's own field. */
const editSteps = (job: Job) => {
  if (job.mode !== 'create' && job.edit.engine !== 'qwen-image-2.1') return job.edit.steps
  return (job.edit.turbo ?? true) ? 8 : (job.edit.steps_21 ?? 40)
}

const num = (read: () => number, write: (n: number) => void) => ({
  get: read,
  set: (v: string) => write(Number(v)),
  type: 'number' as const,
})
/** A box that may be empty, where empty means "whatever the model does on its own". */
const opt = (read: () => number | null, write: (n: number | null) => void) => ({
  get: read,
  set: (v: string) => write(v.trim() === '' ? null : Number(v)),
  type: 'number' as const,
})

const groups = computed(() => {
  const j = job.value
  const out: { title: string; meta: string; rows: Knob[] }[] = []
  if (shows('video'))
    out.push({
      title: 'Video',
      meta: `${j.video.steps} steps · seed ${j.video.seed}`,
      rows: [
        {
          key: 'duration',
          hint: 'Seconds of clip. The frame count rounds up to the grid the VAE encodes, so 2 renders as 2.33.',
          ...num(
            () => j.video.duration,
            (n) => (j.video.duration = n),
          ),
        },
        {
          key: 'megapixels',
          hint: 'The canvas. 0.4 is the fast one, 0.98 is what the workflow was benchmarked at.',
          ...num(
            () => j.video.megapixels,
            (n) => (j.video.megapixels = n),
          ),
        },
        {
          key: 'steps',
          hint: '8 reproduces the graph, 4 the spec. More on a distilled model mostly buys nothing.',
          ...num(
            () => j.video.steps,
            (n) => (j.video.steps = n),
          ),
        },
        {
          key: 'seed',
          hint: 'The only thing that changes between two renders of the same words.',
          ...num(
            () => j.video.seed,
            (n) => (j.video.seed = n),
          ),
        },
        {
          key: 'reference_size',
          hint: 'max holds identity harder and is markedly slower.',
          get: () => j.video.reference_size,
          set: (v) => (j.video.reference_size = v as 'match' | 'max'),
          options: [
            ['match', "match · the clip's own size"],
            ['max', 'max · holds identity harder'],
          ],
        },
      ],
    })
  if (shows('edit')) {
    // A picture from words alone is Qwen-Image 2.1's: the AIO only edits.
    const creating = j.mode === 'create'
    const is21 = creating || j.edit.engine === 'qwen-image-2.1'
    const turbo = is21 && (j.edit.turbo ?? true)
    const engine: Knob = {
      key: 'engine',
      hint: 'Qwen-Image 2.1 draws with its own model and the LoRAs set on the Settings page. Each engine keeps its own steps and cfg.',
      get: () => j.edit.engine ?? 'qwen-image-edit',
      set: (v) => (j.edit.engine = v),
      options: [
        ['qwen-image-edit', 'qwen-image-edit · the AIO'],
        ['qwen-image-2.1', 'qwen-image-2.1 · experimental'],
      ],
    }
    const aspect: Knob = {
      key: 'aspect',
      hint: 'Width to height. Auto lets the writer pick one for the idea; a prompt is drawn square.',
      get: () => j.aspect ?? '',
      set: (v) => (j.aspect = v),
      options: [
        ['', 'auto'],
        ['1:1', '1:1 · square'],
        ['3:2', '3:2 · landscape'],
        ['2:3', '2:3 · portrait'],
        ['4:3', '4:3'],
        ['3:4', '3:4'],
        ['16:9', '16:9 · wide'],
        ['9:16', '9:16 · phone'],
      ],
    }
    out.push({
      title: is21 ? 'Qwen-Image 2.1 · experimental' : 'Qwen Image Edit',
      meta: `${editSteps(j)} steps · seed ${j.edit.seed}`,
      rows: [
        creating ? aspect : engine,
        ...(is21
          ? [
              {
                key: 'turbo',
                hint: "Qwen-Image 2.1 on the Turbo's own 8 steps; off for the base model, which takes the steps below.",
                get: () => String(j.edit.turbo ?? true),
                set: (v: string) => (j.edit.turbo = v === 'true'),
                options: [
                  ['true', 'on · 8 steps'],
                  ['false', 'off · base model'],
                ] as [string, string][],
              },
            ]
          : []),
        ...(turbo
          ? []
          : is21
            ? [
                {
                  key: 'steps',
                  hint: "The base model's steps. Qwen's own pipeline takes 40.",
                  ...num(
                    () => j.edit.steps_21 ?? 40,
                    (n) => (j.edit.steps_21 = n),
                  ),
                },
                {
                  key: 'cfg',
                  hint: 'Qwen-Image 2.1 is sampled without guidance: 1. Above it the negative prompt costs a second pass.',
                  ...num(
                    () => j.edit.cfg_21 ?? 1,
                    (n) => (j.edit.cfg_21 = n),
                  ),
                },
              ]
            : [
                {
                  key: 'steps',
                  hint: 'The all-in-one is distilled; eight is the whole of it.',
                  ...num(
                    () => j.edit.steps,
                    (n) => (j.edit.steps = n),
                  ),
                },
                {
                  key: 'cfg',
                  hint: 'Stays at 1. Above it the negative prompt starts costing a second pass.',
                  ...num(
                    () => j.edit.cfg,
                    (n) => (j.edit.cfg = n),
                  ),
                },
              ]),
        {
          key: 'megapixels',
          hint: creating ? 'The size it is drawn at.' : 'The delivered size. The input is resized to it before the model sees it.',
          ...num(
            () => j.edit.megapixels,
            (n) => (j.edit.megapixels = n),
          ),
        },
        {
          key: 'seed',
          hint: 'The only thing that changes between two runs of the same instruction.',
          ...num(
            () => j.edit.seed,
            (n) => (j.edit.seed = n),
          ),
        },
      ],
    })
  }
  if (shows('delivery'))
    out.push({
      title: 'Delivery',
      meta: 'after sampling',
      rows: [
        ...(shows('finish')
          ? [
              {
                key: 'finish_megapixels',
                hint: 'The size to deliver at. An ESRGAN draws the extra pixels over the finished frames.',
                ...opt(
                  () => j.finish_megapixels,
                  (n) => (j.finish_megapixels = n),
                ),
              },
            ]
          : []),
        {
          key: 'grade_strength',
          hint: 'Pull the finished video back toward the colour of its own first second. Half is usually right.',
          get: () => j.grade_strength,
          set: (v) => (j.grade_strength = v === '' ? null : Number(v)),
          options: [
            ['', 'as rendered'],
            ['0.5', '0.5 · hold'],
            ['0.75', '0.75 · hold hard'],
            ['1', '1 · match the opening'],
          ],
        },
        {
          key: 'retime_fps',
          hint: 'RIFE draws the frames between. Same seconds, same sound, the 24 fps file kept beside it.',
          get: () => j.retime_fps,
          set: (v) => (j.retime_fps = v === '' ? null : Number(v)),
          options: [
            ['', '24 · as rendered'],
            ['48', '48 · one frame between each'],
            ['60', '60 · smoothest'],
          ],
        },
      ],
    })
  // Each mode has its own writer: the edit instruction's, or the H3 caption's.
  const vlm = j.mode === 'video' ? j.video_vlm : j.vlm
  if (shows('vlm'))
    out.push({
      title: 'VLM',
      meta: `temp ${vlm.temperature}`,
      rows: [
        {
          key: 'max_new_tokens',
          hint: 'A cap, not a target: the writer is cut off here if it runs away.',
          ...num(
            () => vlm.max_new_tokens,
            (n) => (vlm.max_new_tokens = n),
          ),
        },
        {
          key: 'temperature',
          hint: 'Lower writes duller captions that repeat; higher invents things the photo does not show.',
          ...num(
            () => vlm.temperature,
            (n) => (vlm.temperature = n),
          ),
        },
        {
          key: 'top_p',
          hint: "Qwen's own recommendation is 0.8 with temperature 0.7.",
          ...num(
            () => vlm.top_p,
            (n) => (vlm.top_p = n),
          ),
        },
        {
          key: 'top_k',
          hint: 'Twenty. Rarely worth moving.',
          ...num(
            () => vlm.top_k,
            (n) => (vlm.top_k = n),
          ),
        },
        {
          key: 'seed',
          hint: '-1 draws a new one each time. A fixed one keeps the words when the clip is re-rolled.',
          ...num(
            () => vlm.seed,
            (n) => (vlm.seed = n),
          ),
        },
      ],
    })
  return out
})

const open = ref<Record<string, boolean>>({ Video: true })
const toggle = (title: string) => (open.value[title] = !open.value[title])

/** In the units the run reports back in, so the two can be read against each other. */
const estimate = computed(() => {
  const j = job.value
  const rows: [string, string][] = []
  if (shows('video')) rows.push(['LENGTH', `${j.video.duration.toFixed(1)}s`], ['CANVAS', `${j.video.megapixels} MP`])
  if (shows('edit')) rows.push(['SIZE', `${j.edit.megapixels} MP`], ['STEPS', String(editSteps(j))])
  if (shows('finish')) rows.push(['DELIVER', j.finish_megapixels ? `${j.finish_megapixels} MP` : '—'])
  return rows
})

const label = computed(() => {
  if (props.starting) return 'queueing…'
  const ahead = props.queued + (props.busy ? 1 : 0)
  if (batching.value) return ahead ? `Queue ${batch.value.length} behind ${ahead}` : `Queue ${batch.value.length}`
  return ahead ? `Queue behind ${ahead}` : 'Render'
})
</script>

<template>
  <main class="screen compose">
    <div class="stack">
      <section class="card">
        <div class="head">
          <h5>Mode</h5>
          <span class="label">{{ job.mode }}</span>
        </div>
        <div class="modes">
          <button
            v-for="m in MODES"
            :key="m.id"
            class="mode"
            :class="{ on: job.mode === m.id }"
            @click="job.mode = m.id"
          >
            <span class="id">{{ m.id }}</span>
            <span class="give">{{ m.give }}</span>
            <span class="get">→ {{ m.get }}</span>
          </button>
        </div>
      </section>

      <section v-if="shows('photos') || shows('clip')" class="card">
        <div class="head">
          <h5>References</h5>
          <div class="segmented">
            <button :class="{ on: !batching }" @click="batching = false">ONE</button>
            <button :class="{ on: batching }" @click="batching = true">BATCH</button>
          </div>
        </div>
        <div v-if="batching" class="body">
          <PhotoPicker
            v-if="shows('clip')"
            v-model:photos="batch"
            accept="video/*"
            each
            empty="Drop finished mp4s here. Each is enlarged in its own run, with the same settings."
          />
          <template v-else>
            <PhotoPicker
              v-model:photos="batch"
              accept="image/*"
              each
              empty="Drop photos here. Each opens a take of its own, with the same words and settings."
            />
            <span class="label">shared references · after each one, in this order</span>
            <PhotoPicker v-model:photos="job.photos" empty="Optional. Every take is handed these after its own photo." />
          </template>
        </div>
        <div v-else class="body">
          <label v-if="shows('clip')" class="stacked">
            <span class="label">clip</span>
            <input v-model="job.clip" placeholder="an mp4 under output/, or its full path" />
          </label>
          <PhotoPicker v-if="shows('photos')" v-model:photos="job.photos" />
        </div>
      </section>

      <section v-if="shows('words')" class="card">
        <div class="head">
          <h5>Words</h5>
          <div class="segmented">
            <button :class="{ on: words === 'idea' }" @click="words = 'idea'">IDEA</button>
            <button :class="{ on: words === 'prompt' }" @click="words = 'prompt'">PROMPT</button>
          </div>
        </div>
        <div class="body">
          <textarea
            v-if="words === 'idea'"
            v-model="job.idea"
            rows="5"
            placeholder="what should happen, in your own words"
          />
          <textarea v-else v-model="job.prompt" rows="6" placeholder="the caption, used exactly as written" />
          <p class="note">{{ wordsNote }}</p>
        </div>
      </section>
    </div>

    <aside class="aside">
      <section v-for="g in groups" :key="g.title" class="card">
        <button class="group" @click="toggle(g.title)">
          <span class="title">{{ g.title }}</span>
          <span class="label">{{ open[g.title] ? '—' : g.meta }}</span>
        </button>
        <div v-if="open[g.title]" class="knobs">
          <div v-for="k in g.rows" :key="k.key" class="knob">
            <div>
              <span class="name">{{ k.key }}</span>
              <span class="hint">{{ k.hint }}</span>
            </div>
            <select
              v-if="k.options"
              :value="k.get() ?? ''"
              @change="k.set((($event.target as HTMLSelectElement).value))"
            >
              <option v-for="[value, text] in k.options" :key="value" :value="value">{{ text }}</option>
            </select>
            <input
              v-else
              :type="k.type ?? 'text'"
              :value="k.get() ?? ''"
              placeholder="—"
              @change="k.set((($event.target as HTMLInputElement).value))"
            />
          </div>
        </div>
      </section>

      <section class="card commit">
        <div class="estimate">
          <div v-for="[k, v] in estimate" :key="k" class="row">
            <span class="label">{{ k }}</span>
            <span class="mono">{{ v }}</span>
          </div>
        </div>
        <p v-if="problems.length" class="band error">{{ problems.join('\n') }}</p>
        <div class="buttons">
          <button class="primary grow" :disabled="!!problems.length || starting" @click="emit('render')">
            {{ label }}
          </button>
        </div>
      </section>
    </aside>
  </main>
</template>

<style scoped>
.compose {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(280px, 320px);
  gap: var(--space-3);
  align-items: start;
}

.aside {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  position: sticky;
  top: 0;
}

.modes {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(152px, 1fr));
  gap: var(--space-1);
  padding: var(--space-2);
}

.mode {
  text-align: left;
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  padding: var(--space-2);
  min-height: 120px;
  min-width: 0;
  border-color: var(--line);
  background: var(--base);
}

.mode.on {
  border-color: var(--accent);
  background: var(--raised);
}

.mode .id {
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  letter-spacing: 0.06em;
  color: var(--muted);
}

.mode.on .id {
  color: var(--accent);
}

.mode .give {
  font-size: var(--text-sm);
  font-weight: var(--weight-light);
  color: var(--muted);
  line-height: 1.45;
}

.mode .get {
  font-size: var(--text-xs);
  color: var(--text);
  margin-top: auto;
}

.mode:hover:not(:disabled) {
  color: inherit;
}

/* Beside the label, not after the box: sharing that row it would be the first thing tabbed into. */

.segmented {
  display: flex;
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-btn);
  overflow: hidden;
}

.segmented > button {
  border: 0;
  border-radius: 0;
  padding: 6px var(--space-2);
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  letter-spacing: 0.06em;
  color: var(--muted);
}

.segmented > button.on {
  background: var(--accent);
  color: var(--accent-fg);
}

.group {
  width: 100%;
  background: none;
  border: 0;
  border-radius: 0;
  padding: var(--space-2);
  display: flex;
  justify-content: space-between;
  align-items: center;
  text-align: left;
}

.group .title {
  font-family: var(--font-heading);
  font-weight: var(--weight-semibold);
  font-size: var(--text-base);
}

.knobs {
  border-top: 1px solid var(--line);
  padding: var(--space-1) var(--space-2) var(--space-2);
  display: flex;
  flex-direction: column;
}

.commit {
  border-color: var(--border-strong);
  padding: var(--space-2);
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.estimate {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.estimate .row {
  display: flex;
  justify-content: space-between;
  font-size: var(--text-xs);
}

.buttons {
  display: flex;
  gap: var(--space-1);
}

.grow {
  flex: 1;
}

code {
  font-family: var(--font-mono);
  font-size: 0.92em;
}

@media (max-width: 960px) {
  .compose {
    grid-template-columns: minmax(0, 1fr);
  }
  .aside {
    position: static;
  }
}
</style>
