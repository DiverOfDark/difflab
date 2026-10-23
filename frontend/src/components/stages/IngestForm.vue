<script setup lang="ts">
import { computed, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import {
  guessReferences,
  ingestStory,
  listTools,
  readRun,
  outputUrl,
  readTimings,
  redoVideo,
  referenceFrames,
  upload,
  type ReferenceFrame,
  type RefKind,
  type Timings,
} from '../../api'
import { ASPECTS, CLIP_MIN, FITS, SHOT_MAX, chaptersOf, length, pad, parseLength, parseRange, type Fit } from '../../production'

const props = defineProps<{
  busy: string
  act: (name: string, what: () => Promise<unknown>) => Promise<void>
}>()
const router = useRouter()

type StoryKind = 'text' | 'video'
const KINDS: { id: StoryKind; label: string }[] = [
  { id: 'text', label: 'STORY' },
  { id: 'video', label: 'REDO VIDEO' },
]
const storyKind = ref<StoryKind>('text')
const title = ref('')
const text = ref('')
const target = ref('')
const shot = ref('6–8 s')
const aspect = ref<number | null>(16 / 9)
const fit = ref<Fit>('crop')
const autoPlan = ref(true)
const knobs = reactive({ heading_level: 1, frames: 12, min_sharpness: 50 })
const problem = ref('')

onMounted(async () => {
  const tools = await listTools().catch(() => [])
  const own = tools.find((t) => t.name === 'ingest')
  for (const field of own?.settings ?? []) if (field.name in knobs) (knobs as Record<string, number>)[field.name] = Number(field.default)
  timings.value = await readTimings().catch(() => ({}))
})

// --- the story --------------------------------------------------------------------------------
const chapters = computed(() => (storyKind.value === 'text' ? chaptersOf(text.value, knobs.heading_level) : []))
const totalWords = computed(() => chapters.value.reduce((n, c) => n + c.words, 0))
const targetDefault = computed(() => (chapters.value.length > 1 ? '2 h' : '60 s'))
const targetSeconds = computed(() => parseLength(target.value || targetDefault.value))
const range = computed(() => parseRange(shot.value))
/** Mirrors `production.take_seconds`: the target spread by words, none under one shot. */
const chapterSeconds = computed(() =>
  chapters.value.map((c) => Math.max(CLIP_MIN, Math.round(((targetSeconds.value || 0) * c.words) / Math.max(1, totalWords.value) * 10) / 10)),
)
const meanShot = computed(() => (range.value ? (Math.max(CLIP_MIN, range.value[0]) + range.value[1]) / 2 : 7))
const shotsOf = (seconds: number) => Math.max(1, Math.round(seconds / meanShot.value))
const totalShots = computed(() => chapterSeconds.value.reduce((n, s) => n + shotsOf(s), 0))
const untitled = computed(() => chapters.value.map((c, i) => ({ ...c, n: i + 1 })).filter((c) => !c.headed))
const splitNote = computed(() => {
  if (!chapters.value.length) return 'Add a story to see how it splits.'
  if (chapters.value.length === 1 && chapters.value[0].headed && !chapters.value[0].title)
    return 'One chapter, planned whole. Once begun, the director can rewrite it into a story it renders well.'
  const loose = untitled.value.length
    ? ` ${untitled.value.length === chapters.value.length ? 'No heading' : `Chapter${untitled.value.length > 1 ? 's' : ''} ${untitled.value.map((c) => pad(c.n, 2)).join(', ')} ha${untitled.value.length > 1 ? 've' : 's'} no heading`}: split every 600 words.`
    : ''
  return `Each chapter becomes its own shot list and its own chapter video, so a 5-minute chapter can go end to end before the rest is touched.${loose}`
})

// --- the references ---------------------------------------------------------------------------
interface Row {
  path: string
  name: string
  thumb: string
  kind: RefKind
  entity: string
  /** What the VLM saw; for a video, the frames too. */
  about: string
  look: string
  guessed: boolean
  touched: boolean
  video: { seconds: number; frames: ReferenceFrame[] } | null
}
const rows = ref<Row[]>([])
const refInput = ref<HTMLInputElement | null>(null)
const dragging = ref(false)
const uploading = ref(false)
const VIDEO = /\.(mp4|mov|mkv|webm)$/i

async function addRefs(files: File[]) {
  if (!files.length) return
  uploading.value = true
  problem.value = ''
  try {
    const stored = await upload(files)
    const fresh: Row[] = []
    for (const asset of stored) {
      if (rows.value.some((r) => r.path === asset.path) || fresh.some((r) => r.path === asset.path)) continue
      const video = VIDEO.test(asset.path)
      fresh.push({
        path: asset.path,
        name: asset.name,
        thumb: asset.thumb ?? '',
        kind: video ? 'video' : 'character',
        entity: '',
        about: '',
        look: '',
        guessed: false,
        touched: false,
        video: null,
      })
    }
    rows.value.push(...fresh)
    await Promise.all(fresh.filter((r) => r.kind === 'video').map(extract))
    describe(fresh)
  } catch (error) {
    problem.value = String(error)
  } finally {
    uploading.value = false
  }
}

async function extract(row: Row) {
  const found = await referenceFrames(row.path, knobs.frames, knobs.min_sharpness)
  const live = rows.value.find((r) => r.path === row.path)
  if (!live) return
  live.video = found
  const first = found.frames.find((f) => f.kept) ?? found.frames[0]
  if (first) live.thumb = outputUrl(first.path)
}
let refresh = 0
watch(
  () => [knobs.frames, knobs.min_sharpness],
  () => {
    window.clearTimeout(refresh)
    refresh = window.setTimeout(() => rows.value.filter((r) => r.kind === 'video').forEach(extract), 500)
  },
)

/** The picture a row is described by: itself, or a video's sharpest kept frame. */
const pictureOf = (row: Row) => (row.video ? (row.video.frames.find((f) => f.kept) ?? row.video.frames[0])?.path : row.path)
/** An upload is named by its absolute path, a job by the one under output/. */
const under = (path: string) => path.replace(/^.*\/output\//, '')
const guessing = ref<string[]>([])
async function describe(fresh: Row[]) {
  const paths = fresh.map(pictureOf).filter((p): p is string => !!p)
  if (!paths.length) return
  const { run } = await guessReferences(paths)
  guessing.value.push(run)
}
const poll = window.setInterval(async () => {
  for (const run of [...guessing.value]) {
    const record = await readRun(run).catch(() => null)
    if (!record) continue
    for (const job of record.jobs) {
      const out = job.outputs as { path?: string; kind?: RefKind; about?: string; look?: string }
      if (!out.about) continue
      for (const row of rows.value.filter((r) => under(pictureOf(r) ?? '') === under(String(job.inputs.path)) && !r.guessed)) {
        row.about = out.about
        row.look = out.look ?? ''
        row.guessed = true
        if (!row.touched && row.kind !== 'video' && out.kind) row.kind = out.kind
      }
    }
    if (['done', 'failed', 'cancelled'].includes(record.state)) guessing.value = guessing.value.filter((r) => r !== run)
  }
}, 2000)
onUnmounted(() => window.clearInterval(poll))

const guessOf = (row: Row) => {
  const said = row.kind === 'style' ? row.look || row.about : row.about
  if (row.video) {
    const kept = row.video.frames.filter((f) => f.kept).length
    return `${row.video.seconds.toFixed(0)} s · ${row.video.frames.length} frames · ${kept} sharp${said ? ` · ${said}` : ''}`
  }
  return said || (guessing.value.length ? 'Qwen-VL describing…' : '—')
}
const framesTitle = (row: Row) => row.video?.frames.map((f) => `${f.at.toFixed(1)} s · sharpness ${f.sharpness}${f.kept ? '' : ' · dropped'}`).join('\n') ?? ''
const setKind = (row: Row, kind: RefKind) => {
  row.kind = kind
  row.touched = true
}
const remove = (row: Row) => (rows.value = rows.value.filter((r) => r !== row))

// --- the estimate -----------------------------------------------------------------------------
const timings = ref<Timings>({})
const each = (tool: string) => timings.value[tool]?.seconds ?? 0
const photos = computed(() => rows.value.filter((r) => r.kind !== 'style').reduce((n, r) => n + (r.video ? r.video.frames.filter((f) => f.kept).length : 1), 0))
const named = computed(() => new Set(rows.value.filter((r) => r.entity.trim()).map((r) => r.entity.trim().toLowerCase())))
const unnamed = computed(() => rows.value.filter((r) => !r.entity.trim() && r.kind !== 'video').length)
const planSeconds = computed(() => {
  const people = rows.value.filter((r) => r.kind === 'character' && !r.entity.trim()).length
  const pairs = (people * (people - 1)) / 2 + people * named.value.size
  const n = Math.max(1, chapters.value.length)
  let t = rows.value.length * each('guess_reference') + pairs * each('same_person') + n * each('survey')
  t += photos.value * (each('ground') + each('cards')) + each('cut_crops') + each('tidy')
  if (autoPlan.value) t += each('describe_opening') + each('arrange') + n * each('plan_shots') + totalShots.value * (each('write_shot') + each('compile_shot'))
  return t
})
const estimate = computed(() => {
  const n = chapters.value.length
  const kinds = (k: RefKind) => rows.value.filter((r) => r.kind === k).length
  return [
    { k: 'STORY', v: n ? `${totalWords.value} words · ${n} chapter${n === 1 ? '' : 's'}` : '—' },
    {
      k: 'REFERENCES',
      v: rows.value.length
        ? `${rows.value.length} · ${kinds('character')} character · ${kinds('video')} video${kinds('style') ? ` · ${kinds('style')} style` : ''}`
        : 'none yet · a photograph or a video is needed',
    },
    { k: 'ENTITIES', v: rows.value.length ? `≥ ${named.value.size + unnamed.value} from references · the rest from the story` : '—' },
    { k: 'FILM', v: n && targetSeconds.value ? `${length(targetSeconds.value)} · ~${totalShots.value} shots` : '—' },
    { k: 'PLAN TIME', v: n ? `~${length(planSeconds.value)} VLM · no GPU render` : '—' },
  ]
})

// --- a video to redo --------------------------------------------------------------------------
const redo = reactive({ source: '', name: '', instruction: '', clip_scale: 0.5 })
async function pickSource(files: File[]) {
  if (!files[0]) return
  uploading.value = true
  try {
    const [asset] = await upload([files[0]])
    redo.source = asset.path
    redo.name = asset.name
  } catch (error) {
    problem.value = String(error)
  } finally {
    uploading.value = false
  }
}

// --- begin ------------------------------------------------------------------------------------
const problems = computed(() => {
  const out: string[] = []
  if (!title.value.trim()) out.push('a title')
  if (storyKind.value === 'video') {
    if (!redo.source) out.push('the video to redo')
    if (!redo.instruction.trim()) out.push('what should be different about it')
    if (!range.value) out.push('a shot length like 6–8 s')
    return out
  }
  if (!chapters.value.length) out.push('a story')
  if (!(targetSeconds.value >= CLIP_MIN)) out.push(`a target length of at least ${CLIP_MIN} s, like 2 h or 90 min`)
  const r = range.value
  if (!r || r[1] < CLIP_MIN || r[1] > SHOT_MAX || r[0] > r[1]) out.push(`a shot length between ${CLIP_MIN} and ${SHOT_MAX} s, like 6–8 s`)
  if (!rows.value.some((row) => row.kind !== 'style')) out.push('a photograph or a video: chapter 1 opens on one')
  if (rows.value.some((row) => row.entity.trim() && !/[a-z]/i.test(row.entity))) out.push('entity names with letters')
  return out
})
const cta = computed(() => (storyKind.value === 'video' ? 'Begin' : autoPlan.value ? 'Ingest and plan' : 'Ingest · build bible'))
const footer = computed(() =>
  storyKind.value === 'video'
    ? 'Cut into shots of the longest length, each re-creating its stretch of the source.'
    : autoPlan.value
      ? 'Builds the bible, then plans every chapter, and stops before Render.'
      : 'Builds the bible and stops there for review.',
)

const begin = () =>
  props.act('begin', async () => {
    const r = range.value!
    const video = { duration: r[1], megapixels: 0.4, steps: 8 }
    if (storyKind.value === 'video') {
      const made = await redoVideo({ title: title.value, source: redo.source, instruction: redo.instruction, photos: [], clip_scale: redo.clip_scale, video })
      router.push(`/productions/${made.id}`)
      return
    }
    const made = await ingestStory({
      title: title.value,
      story: text.value,
      references: rows.value.map((row) => ({ path: row.path, kind: row.kind, entity: row.entity, guess: row.kind === 'style' ? row.look || row.about : row.about })),
      target_seconds: targetSeconds.value,
      shortest: Math.max(CLIP_MIN, r[0]),
      aspect: aspect.value,
      fit: fit.value,
      video,
      ingest: { ...knobs },
      plan: autoPlan.value,
    })
    router.push(`/productions/${made.id}?stage=ingest&ingesting=1`)
  })

const onDrop = (event: DragEvent, take: (files: File[]) => void) => {
  dragging.value = false
  take([...(event.dataTransfer?.files ?? [])])
}
const onPick = (event: Event, take: (files: File[]) => void) => {
  const input = event.target as HTMLInputElement
  take([...(input.files ?? [])])
  input.value = ''
}
</script>

<template>
  <div class="ingest">
    <section class="card">
      <div class="head">
        <h5>Story</h5>
        <div class="seg">
          <button v-for="k in KINDS" :key="k.id" :class="{ on: storyKind === k.id }" @click="storyKind = k.id">{{ k.label }}</button>
        </div>
      </div>
      <div class="body">
        <label class="field"><span class="eyebrow">TITLE</span><input v-model="title" class="plain" placeholder="Working title" /></label>

        <textarea
          v-if="storyKind === 'text'"
          v-model="text"
          rows="10"
          class="prose"
          placeholder="A few sentences of what happens, or the full text. Chapters split on headings, or every ~600 words if there are none."
        />
        <template v-else>
          <div
            class="drop"
            :class="{ over: dragging }"
            @click="refInput?.click()"
            @dragover.prevent="dragging = true"
            @dragleave="dragging = false"
            @drop.prevent="onDrop($event, pickSource)"
          >
            <span>{{ uploading ? 'uploading…' : redo.source ? `${redo.name} · replace` : 'Drop the video to redo' }}</span>
            <span class="mono dim">cut into shots of the longest length, each re-creating its stretch</span>
            <input ref="refInput" type="file" accept="video/*" hidden @change="onPick($event, pickSource)" />
          </div>
          <label class="field"><span class="eyebrow">WHAT CHANGES</span><input v-model="redo.instruction" class="plain" placeholder="make it look like a watercolour painting" /></label>
        </template>

        <div class="three ruled">
          <label v-if="storyKind !== 'video'" class="field"><span class="eyebrow">TARGET LENGTH</span><input v-model="target" :placeholder="targetDefault" /></label>
          <label class="field" :title="`the director plans shots inside this range; ${CLIP_MIN} s is the shortest it can hold`">
            <span class="eyebrow">{{ storyKind === 'video' ? 'LONGEST SHOT' : 'SHOT LENGTH' }}</span><input v-model="shot" placeholder="6–8 s" />
          </label>
          <label v-if="storyKind !== 'video'" class="field">
            <span class="eyebrow">ASPECT</span>
            <select v-model="aspect">
              <option v-for="a in ASPECTS" :key="a.label" :value="a.value">{{ a.label }}</option>
            </select>
          </label>
          <label v-if="storyKind !== 'video'" class="field" title="crop cuts the first photo to the canvas; extend keeps it whole and has the editor draw the sides">
            <span class="eyebrow">FIT</span>
            <select v-model="fit">
              <option v-for="f in FITS" :key="f.value" :value="f.value">{{ f.label }}</option>
            </select>
          </label>
          <label v-else class="field"><span class="eyebrow">CLIP SCALE</span><input v-model.number="redo.clip_scale" type="number" min="0.1" max="1" step="0.1" /></label>
        </div>

        <div v-if="storyKind === 'text'" class="split ruled">
          <span class="row mono"><span class="eyebrow">SPLIT PREVIEW</span><span class="muted">{{ chapters.length ? `${chapters.length} chapters · ~${totalShots} shots` : '—' }}</span></span>
          <div class="bars">
            <span
              v-for="(c, i) in chapters.slice(0, 60)"
              :key="i"
              :class="{ loose: !c.headed }"
              :title="`${c.title || 'the story'} · ${c.words} words · ${length(chapterSeconds[i])} · ~${shotsOf(chapterSeconds[i])} shots${c.headed ? '' : ' · no heading, split at 600 words'}`"
              :style="{ flex: c.words }"
              >{{ pad(i + 1, 2) }}</span
            >
          </div>
          <span class="note">{{ splitNote }}</span>
        </div>

        <div v-if="storyKind === 'text'" class="knobs ruled">
          <label class="field" title="a heading of this level or above starts a chapter">
            <span class="eyebrow">CHAPTER SPLIT</span>
            <select v-model.number="knobs.heading_level">
              <option v-for="n in 6" :key="n" :value="n">h{{ n }}{{ n > 1 ? ` · h1–h${n}` : '' }}</option>
            </select>
          </label>
          <label class="field" title="frames pulled from each reference video, evenly">
            <span class="eyebrow">FRAMES PER VIDEO</span><input v-model.number="knobs.frames" type="number" min="1" max="48" />
          </label>
          <label class="field" title="variance of the Laplacian at 512 px wide; frames blurrier than this are dropped, the sharpest is always kept">
            <span class="eyebrow">MIN SHARPNESS</span><input v-model.number="knobs.min_sharpness" type="number" min="0" max="5000" step="10" />
          </label>
        </div>
      </div>
    </section>

    <section v-if="storyKind === 'text'" class="card">
      <div class="head">
        <h5>References</h5>
        <span class="note">a photograph or a video at least · they override the story</span>
      </div>
      <div class="body">
        <div
          class="drop"
          :class="{ over: dragging }"
          @click="refInput?.click()"
          @dragover.prevent="dragging = true"
          @dragleave="dragging = false"
          @drop.prevent="onDrop($event, addRefs)"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></svg>
          <span>{{ uploading ? 'uploading…' : 'Drop photos, videos or style images' }}</span>
          <span class="mono dim">a face becomes a character · a place a location · a look the style · a video also a motion reference</span>
          <input ref="refInput" type="file" accept="image/*,video/*" multiple hidden @change="onPick($event, addRefs)" />
        </div>
        <div v-if="rows.length" class="refs">
          <div class="ref heading mono"><span /><span>FILE</span><span>KIND</span><span>ENTITY</span><span /></div>
          <div v-for="row in rows" :key="row.path" class="ref">
            <div class="thumb" :style="row.thumb ? { backgroundImage: `url(${row.thumb})` } : {}" :class="row.kind">
              <span v-if="row.video" class="badge mono">MP4</span>
            </div>
            <span class="what" :title="framesTitle(row)">
              <span class="mono name">{{ row.name }}</span>
              <span class="guess">{{ guessOf(row) }}</span>
            </span>
            <select :value="row.kind" class="mono" @change="setKind(row, ($event.target as HTMLSelectElement).value as RefKind)">
              <template v-if="!row.video">
                <option value="character">character</option>
                <option value="location">location</option>
                <option value="prop">prop</option>
                <option value="style">style</option>
              </template>
              <option v-else value="video">motion ref</option>
            </select>
            <input v-model="row.entity" class="mono" :placeholder="row.kind === 'style' ? 'style' : 'name or leave for the bible'" />
            <button class="x" title="remove" @click="remove(row)">×</button>
          </div>
        </div>
        <p class="note">
          Qwen-VL describes each upload and the description goes into the bible, so shots match what the picture shows, not what the text says. The
          same file uploaded twice is one reference; photos of one face are merged by a ruling. A video's sharp frames are photographs of its entity,
          and the clip is kept as its motion reference. The first location, or the first photograph, is where chapter 1 opens.
        </p>
      </div>
    </section>

    <section class="card go">
      <div class="estimate mono">
        <span v-for="e in estimate" :key="e.k"><span class="eyebrow">{{ e.k }}</span>{{ e.v }}</span>
      </div>
      <div class="buttons">
        <label v-if="storyKind !== 'video'" class="check"><input v-model="autoPlan" type="checkbox" />run all plan stages, stop before render</label>
        <button class="primary" :disabled="!!problems.length || !!busy || uploading" :title="problems.join('\n')" @click="begin">
          {{ busy === 'begin' ? 'beginning…' : cta }}
        </button>
      </div>
      <p class="note full">{{ problems.length ? `needs ${problems.join(' · ')}` : footer }}</p>
      <p v-if="problem" class="band error full">{{ problem }}</p>
    </section>
  </div>
</template>

<style scoped>
.ingest {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 440px), 1fr));
  gap: var(--space-3);
  align-items: start;
}

.head {
  align-items: center !important;
  flex-wrap: wrap;
}

.field {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  min-width: 0;
}

.eyebrow {
  font-size: 11px;
}

.ruled {
  padding-top: var(--space-2);
  border-top: 1px solid var(--border);
}

.three,
.knobs {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: var(--space-2);
}

.card textarea.prose,
.card input.plain {
  font-family: var(--font-body);
}

.card textarea.prose {
  font-weight: var(--weight-light);
  line-height: 1.5;
  padding: var(--space-1) var(--space-2);
}

.drop {
  border: 1px dashed var(--border-strong);
  border-radius: var(--radius-card);
  padding: var(--space-3);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--space-1);
  color: var(--text-secondary);
  font-size: var(--text-xs);
  font-weight: var(--weight-light);
  text-align: center;
  cursor: pointer;
}

.drop.over,
.drop:hover {
  border-color: var(--accent);
}

.drop .mono {
  font-size: 11px;
}

.split {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.row {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
}

.bars {
  display: flex;
  gap: 2px;
  height: 16px;
}

.bars span {
  background: var(--bg-raised);
  border: 1px solid var(--border);
  border-radius: 2px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--font-mono);
  font-size: 9px;
  color: var(--text-tertiary);
  overflow: hidden;
  min-width: 0;
}

.bars span.loose {
  border-style: dashed;
}

.refs {
  display: flex;
  flex-direction: column;
}

.ref {
  display: grid;
  grid-template-columns: 40px minmax(0, 1fr) 112px minmax(120px, 1fr) 24px;
  gap: var(--space-2);
  align-items: center;
  padding: var(--space-1) 0;
  border-bottom: 1px solid var(--border);
}

.ref.heading {
  padding: 0 0 var(--space-1);
  border-bottom-color: var(--border-strong);
  font-size: 10px;
  letter-spacing: 0.08em;
  color: var(--text-tertiary);
}

.thumb {
  width: 40px;
  height: 40px;
  background: var(--bg-raised) center / cover no-repeat;
  border: 1px solid var(--border);
  border-radius: 2px;
  display: flex;
  align-items: flex-end;
  justify-content: center;
}

.thumb.style {
  border-color: var(--accent-muted);
}

.badge {
  font-size: 9px;
  color: var(--text-primary);
  background: oklch(0% 0 0 / 0.6);
  padding: 0 3px;
  border-radius: 2px;
}

.what {
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.what > span {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.what .name {
  font-size: var(--text-xs);
}

.what .guess {
  font-size: 11px;
  font-weight: var(--weight-light);
  color: var(--text-secondary);
}

.ref select,
.ref input {
  font-size: 11px;
  min-width: 0;
}

.x {
  background: none;
  border: 0;
  padding: 0;
  color: var(--text-secondary);
  font-family: var(--font-mono);
  font-size: var(--text-xs);
}

.go {
  grid-column: 1 / -1;
  border-color: var(--border-strong);
  padding: var(--space-2) var(--space-3);
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--space-2) var(--space-3);
  flex-wrap: wrap;
}

.estimate {
  display: flex;
  gap: var(--space-3);
  flex-wrap: wrap;
  font-size: var(--text-xs);
}

.estimate > span {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.estimate .eyebrow {
  font-size: 10px;
}

.buttons {
  display: flex;
  gap: var(--space-1);
  align-items: center;
  flex-wrap: wrap;
}

.check {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  font-size: var(--text-xs);
  font-weight: var(--weight-light);
  color: var(--text-secondary);
  margin-right: var(--space-1);
}

.full {
  flex-basis: 100%;
}
</style>
