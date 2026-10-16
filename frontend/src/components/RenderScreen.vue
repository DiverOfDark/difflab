<script setup lang="ts">
/**
 * The stage falls back in order of how finished a chunk is: the decoded chunk, the taeh3 preview
 * clip from the last sampling step, that step's poster, then the frame the chunk opens on — the
 * only one of the four that exists in a chunk's first minutes.
 */
import { computed, ref, watch } from 'vue'
import { control, runFile } from '../api'
import { currentIndex, duration, pace, timeLeft, type Item, type RunState } from '../run'

const props = defineProps<{ run: RunState }>()

const following = ref(true)
const picked = ref(-1)
const showLog = ref(true)

const selected = computed(() => {
  if (following.value) return Math.max(currentIndex(props.run), 0)
  return Math.min(picked.value, props.run.items.length - 1)
})
const item = computed<Item | null>(() => props.run.items[selected.value] ?? null)
const live = computed(() => props.run.status === 'live')
/**
 * `planned`, NOT `items.length`: a single clip has one item too, invented by the fold to hang each
 * sampling step's preview on, and counting that left the bar at zero for an entire H3 render while
 * the card was visibly working. Where there are no chunks, the stage in flight is counted instead.
 */
const byChunks = computed(() => props.run.planned)

/**
 * "Something is happening with no end in sight" — true of a load or a decode, false of a run that
 * ended twenty minutes ago. Without the liveness test it crawled for ever after the last event.
 */
const crawling = computed(
  () => props.run.status === 'live' && !byChunks.value && props.run.stageFraction === null,
)

/** `run.stage` carries the live counters (`Sampling · 180/400`); the title never changes. */
const headline = computed(() => props.run.stage || props.run.title)
const percent = computed(() => {
  // A finished run is finished, whatever the last stage happened to be counting when it ended.
  if (props.run.status === 'done') return 100
  const done = props.run.items.filter((it) => it.state === 'done').length
  const share = byChunks.value ? done / Math.max(1, props.run.items.length) : (props.run.stageFraction ?? 0)
  return Math.round(100 * share)
})

/** Clicking takes the wheel: a chunk being re-watched must not be yanked away by the next one. */
function pick(index: number) {
  following.value = false
  picked.value = index
}

/** The newest thing a run wrote, for a run that has no chunks to show: an edit, a contact sheet. */
const artifact = computed(() => props.run.artifacts.at(-1) ?? null)

/** -1 follows the newest; the delivered artifact, once there is one, wins over every step before it. */
const pictured = ref(-1)
const short = (label: string) =>
  label
    .split(' · ')
    .filter((part) => /^(edit \d+ of \d+|amendment \d+|the photo)$/.test(part))
    .map((part) => part.replace(/ of /, '/').replace('amendment', 'fix'))
    .join(' · ')

const picture = computed(() => {
  const all = props.run.pictures
  if (pictured.value >= 0) return all[pictured.value] ?? null
  return artifact.value ? null : (all.at(-1) ?? null)
})

const delivered = computed(() =>
  props.run.finalVideo
    ? `/output/${props.run.finalVideo}`
    : artifact.value?.media === 'video'
      ? `/output/${artifact.value.path}`
      : '',
)

const movie = computed(() => {
  const it = item.value
  const relative = it ? it.video || it.preview : ''
  // A single clip's item carries only previews — its finished mp4 arrives as an artifact, not a
  // chunk — so without the delivered file winning, the stage stayed empty on exactly the run whose
  // result you most wanted to see.
  if (!relative || (props.run.status !== 'live' && delivered.value)) return delivered.value
  return runFile(props.run.id, relative)
})

const still = computed(() => {
  const it = item.value
  if (movie.value) return ''
  if (!it && picture.value) return runFile(props.run.id, picture.value.image)
  if (!it) return artifact.value?.media === 'image' ? `/output/${artifact.value.path}` : ''
  const relative = it.poster || it.opening
  return relative ? runFile(props.run.id, relative) : ''
})

const caption = computed(() => {
  // An unplanned run's one item is a peg for previews, so naming it "chunk 1" would be a fiction.
  if (!props.run.planned && picture.value) return picture.value.label
  if (!props.run.planned) return props.run.finalVideo || artifact.value?.path || ''
  const it = item.value
  if (!it) return ''
  const step = it.previewStep
  const seed = it.seed !== null ? ` · seed ${it.seed}` : ''
  if (it.video) return `chunk ${it.index + 1} · chunk${seed} · ${it.video}`
  if (it.preview) {
    const what = step ? `preview, step ${step[0]} of ${step[1]}` : 'preview'
    return `chunk ${it.index + 1} · ${what}${seed} · ${it.preview}`
  }
  if (it.poster && step) return `chunk ${it.index + 1} · taeh3 preview, step ${step[0]} of ${step[1]}${seed}`
  if (it.opening) return `chunk ${it.index + 1} · opening frame${seed} · ${it.opening}`
  return `chunk ${it.index + 1} · nothing rendered yet${seed}`
})

const gauges = computed(() => {
  const gb = (pair: [number, number] | null) => (pair ? `${pair[0].toFixed(1)}/${pair[1].toFixed(0)}` : '—')
  return [
    ['ELAPSED', duration(props.run.elapsed)],
    ['VRAM', gb(props.run.vram)],
    ['RAM', gb(props.run.ram)],
    ['RUN', props.run.id],
  ] as [string, string][]
})

/** `4/6 sampling` for a render; `4/8 opening 2/5` for the director, whose stages call per photograph. */
const where = computed(() => {
  const t = props.run.tally
  if (!t || t.index < 0) return ''
  const phase = `${t.index + 1}/${t.stages.length} ${t.stages[t.index]}`
  return t.calls > 0 ? `${phase} ${Math.min(t.done + 1, t.calls)}/${t.calls}` : phase
})

const pip = (i: number) => {
  const t = props.run.tally
  return !t ? '' : i < t.index ? 'done' : i === t.index ? 'running' : ''
}

/** Drawn only for a list: "job 1 of 1" is just the render, which the rest of this strip already is. */
const playing = computed(() => {
  const r = props.run
  if (r.stepsTotal < 2) return null
  const at = r.step >= 0 ? r.step : r.steps.length - 1
  const step = r.steps[at]
  if (!step) return null
  return { where: `job ${at + 1} of ${r.stepsTotal} · ${step.label || step.mode}` }
})

/**
 * The event stream carries no per-chunk fraction, so a running chunk is as far through as its last
 * preview said. Not guessed from elapsed time: the first minutes of a chunk are its slowest.
 */
function chunkPercent(it: Item): number {
  if (it.state === 'done') return 100
  if (it.state !== 'running') return 0
  const step = it.previewStep
  return step && step[1] ? Math.round((step[0] / step[1]) * 100) : 0
}

/** Two, not the design's four: a chunk's other two outputs are clips, and a clip as a 40px black
 *  rectangle says only that the layout wanted four of something. */
function previews(it: Item): { key: string; src: string; label: string }[] {
  const at = (relative: string) => runFile(props.run.id, relative)
  const out: { key: string; src: string; label: string }[] = []
  if (it.opening) out.push({ key: 'opening', src: at(it.opening), label: 'opens' })
  if (it.poster) out.push({ key: 'poster', src: at(it.poster), label: 'step' })
  return out
}

async function send(cmd: string, fields: Record<string, unknown> = {}) {
  try {
    await control(props.run.id, { cmd, ...fields })
  } catch (error) {
    alert(String(error))
  }
}

/** The one that loses work: the chunk in flight is dropped at its next step, and nothing after it runs. */
function cancel() {
  if (confirm('Cancel this run? The chunk being rendered is lost; what is finished stays on disk.')) send('cancel')
}

// A pinned tab is where a two-hour render is actually watched from.
watch(
  () => [props.run.title, props.run.status, percent.value] as const,
  () => {
    const run = props.run
    document.title =
      run.status === 'live'
        ? `${percent.value}% · ${timeLeft(run)} · ${run.title} — DiffLab`
        : `${run.title} ${run.status} — DiffLab`
  },
  { immediate: true },
)
</script>

<template>
  <main class="screen stack">
    <section class="card strip">
      <div class="progress">
        <div class="top">
          <span class="title">{{ headline }}</span>
          <span class="mono muted">{{ pace(run) }} · {{ timeLeft(run) }}</span>
        </div>
        <div class="bar" :class="{ waiting: crawling }">
          <i :style="{ width: percent + '%' }" />
        </div>
        <div v-if="playing" class="phases">
          <span class="dots">
            <i v-for="s in run.steps" :key="s.index" class="dot" :class="s.state" :title="s.label" />
          </span>
          <span class="mono where">{{ playing.where }}</span>
        </div>
        <!-- A render is six phases and the director is eight. -->
        <div v-if="run.tally" class="phases">
          <span class="dots">
            <i v-for="(name, i) in run.tally.stages" :key="name" class="dot" :class="pip(i)" :title="name" />
          </span>
          <span class="mono where">{{ where }}</span>
          <span class="label">{{ duration(run.tally.seconds) }}</span>
        </div>
        <div class="gauges">
          <span v-for="[k, v] in gauges" :key="k" class="label"
            >{{ k }} <span class="mono figure">{{ v }}</span></span
          >
        </div>
      </div>
      <div class="buttons">
        <button class="danger" :disabled="!live" @click="cancel">Cancel run</button>
      </div>
    </section>

    <div v-if="run.error" class="band error">{{ run.error.split('\n').at(-1) }}</div>
    <p v-if="!run.items.length && run.tally" class="note">
      {{ run.tally.name }} is writing the take — {{ where }}. The chunks appear once it has planned them.
    </p>

    <section v-if="movie || still" class="stage">
      <video v-if="movie" :key="movie" :src="movie" controls autoplay muted loop playsinline />
      <img v-else :src="still" alt="" />
      <p class="mono caption">{{ caption }}</p>
      <button v-if="!following" class="quiet" @click="following = true">follow the chunk being rendered</button>
      <button v-if="pictured >= 0" class="quiet" @click="pictured = -1">follow the newest picture</button>
    </section>

    <section v-if="run.pictures.length" class="pictures">
      <button
        v-for="(p, i) in run.pictures"
        :key="p.image"
        class="picture"
        :class="{ on: picture === p }"
        :title="p.label"
        @click="pictured = i"
      >
        <img :src="runFile(run.id, p.image)" alt="" loading="lazy" />
        <span>{{ short(p.label) }}</span>
      </button>
      <button v-if="artifact?.media === 'image'" class="picture" :class="{ on: !picture }" @click="pictured = -1">
        <img :src="`/output/${artifact.path}`" alt="" loading="lazy" />
        <span>delivered</span>
      </button>
    </section>

    <!-- A single clip's one item is a preview peg, not a chunk: drawn, it reads `chunk 1 · PENDING`
         beside a finished video — wrong in both halves. -->
    <section v-if="run.planned && run.items.length" class="chunks">
      <article
        v-for="it in run.items"
        :key="it.index"
        class="chunk"
        :class="{ on: it.index === selected, [it.state]: true }"
        @click="pick(it.index)"
      >
        <div class="chunk-head">
          <span class="mono"><i class="dot" :class="it.state" /> chunk {{ it.index + 1 }}</span>
          <span class="label">{{ it.state }}</span>
        </div>
        <div class="chunk-body">
          <div class="thumbs">
            <div v-for="p in previews(it)" :key="p.key" class="thumb">
              <img v-if="p.src" :src="p.src" alt="" />
              <span>{{ p.label }}</span>
            </div>
          </div>
          <div class="hair"><i :style="{ width: chunkPercent(it) + '%' }" /></div>
          <span class="summary">{{ it.activity || it.prompt || it.label }}</span>
          <div class="facts">
            <span>seed {{ it.seed ?? '—' }}</span>
            <span>{{ it.seconds ? duration(it.seconds) : '' }}</span>
          </div>
        </div>
      </article>
    </section>

    <section class="card">
      <div class="head">
        <h5>Log</h5>
        <span class="label" @click="showLog = !showLog">run {{ run.id }} · log</span>
      </div>
      <div v-if="showLog" class="log">
        <p v-for="(line, i) in run.log.slice(-120).reverse()" :key="i" class="mono">{{ line }}</p>
      </div>
    </section>
  </main>
</template>

<style scoped>
.strip {
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
  flex: 1 1 420px;
}

.progress .top {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: var(--space-2);
  flex-wrap: wrap;
}

.progress .title {
  font-family: var(--font-heading);
  font-weight: var(--weight-semibold);
  font-size: var(--text-md);
}

.gauges {
  display: flex;
  gap: var(--space-1) var(--space-3);
  flex-wrap: wrap;
}

.phases {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  flex-wrap: wrap;
}

.dots {
  display: flex;
  gap: 4px;
}

.phases .dot {
  width: 6px;
  height: 6px;
}

/* Nothing was rendered, and `pending` would read as "not yet". */
.phases .dot.skipped {
  background: var(--warn);
}

.where {
  font-size: var(--text-xs);
  color: var(--text);
}

.gauges .figure {
  color: var(--text);
  letter-spacing: 0;
  text-transform: none;
}

.buttons {
  display: flex;
  gap: var(--space-1);
  flex-wrap: wrap;
}

/* True black: a near-black surround makes a low-key frame look lifted. */
.stage {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  align-items: center;
  background: var(--video);
  border: 1px solid var(--line);
  border-radius: var(--radius-card);
  padding: var(--space-2);
}

.stage video,
.stage img {
  max-height: 52vh;
  max-width: 100%;
  border-radius: var(--radius-input);
}

.caption {
  font-size: var(--text-xs);
  color: var(--muted);
  margin: 0;
  text-align: center;
  overflow-wrap: anywhere;
}

.pictures {
  display: flex;
  gap: var(--space-1);
  overflow-x: auto;
  padding-bottom: 4px;
}

.picture {
  flex: 0 0 auto;
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 2px;
  border: 1px solid var(--line);
  border-radius: var(--radius-input);
  background: var(--video);
  cursor: pointer;
}

.picture.on {
  border-color: var(--accent);
}

.picture img {
  height: 112px;
  width: auto;
  border-radius: var(--radius-input);
}

.picture span {
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--muted);
  white-space: nowrap;
}

.chunks {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: var(--space-1);
}

.chunk {
  border: 1px solid var(--line);
  border-radius: var(--radius-card);
  background: var(--surface);
  display: flex;
  flex-direction: column;
  cursor: pointer;
  min-width: 0;
}

.chunk.pending {
  opacity: 0.55;
}

.chunk.on {
  border-color: var(--accent);
}

.chunk.failed {
  border-color: var(--fail);
}

.chunk-head {
  padding: var(--space-1) var(--space-2);
  border-bottom: 1px solid var(--line);
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: var(--text-xs);
  gap: var(--space-1);
}

.chunk-head .mono {
  display: flex;
  align-items: center;
  gap: var(--space-1);
}

.chunk-body {
  padding: var(--space-2);
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  flex: 1;
}

.thumbs {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 4px;
}

.thumb {
  aspect-ratio: 9 / 16;
  background: var(--video);
  border: 1px solid var(--line);
  border-radius: var(--radius-input);
  display: flex;
  align-items: flex-end;
  justify-content: center;
  padding-bottom: 4px;
  font-family: var(--font-mono);
  font-size: 9px;
  color: var(--dim);
  position: relative;
  overflow: hidden;
}

.thumb img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.thumb span {
  position: relative;
  text-shadow: 0 1px 2px #000;
}

/* A stage that counts nothing — a model loading, a VLM writing — has no share to draw. */
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

/* Two pixels: enough to read as a proportion, not enough to be mistaken for the run's own bar. */
.hair {
  height: 2px;
  background: var(--raised);
  border-radius: 1px;
  overflow: hidden;
}

.hair > i {
  display: block;
  height: 100%;
  background: var(--accent);
  transition: width 400ms ease;
}

.summary {
  font-size: var(--text-xs);
  font-weight: var(--weight-light);
  color: var(--muted);
  line-height: 1.45;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.facts {
  display: flex;
  justify-content: space-between;
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--dim);
  margin-top: auto;
  gap: var(--space-1);
}

.log {
  padding: var(--space-1) var(--space-2);
  max-height: 224px;
  overflow: auto;
}

.log p {
  margin: 0;
  padding: 2px 0;
  font-size: var(--text-xs);
  line-height: 1.45;
  color: var(--muted);
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
</style>
