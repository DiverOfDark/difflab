<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { deleteProduction, ingest, outputUrl, patchProduction, patchShot, planProduction, type Ingest, type Production } from '../../api'
import { ASPECTS, canvasFor, seconds, words } from '../../production'
import { useDraft } from '../../draft'
import { state } from '../../store'
import IdeaPanel from '../IdeaPanel.vue'
import PhotoPicker from '../PhotoPicker.vue'

const props = defineProps<{
  production: Production | null
  busy: string
  act: (name: string, what: () => Promise<unknown>) => Promise<void>
}>()
const router = useRouter()

const form = ref<Ingest>({
  kind: 'idea',
  slug: '',
  title: '',
  idea: '',
  text: '',
  source: '',
  instruction: '',
  photos: [],
  take_seconds: 60,
  aspect: null,
  clip_scale: 0.5,
  video: { duration: 12, megapixels: 0.4, steps: 8 },
  continuity: { tail_seconds: 0.25, handoff_window: 6, crossfade: 0.25 },
})
const autoPlan = ref(true)
// `screenplay.config.DIRECTOR_CLIP_MIN`: the shortest shot the director may choose.
const CLIP_MIN = 6

const KINDS = [
  { id: 'idea', label: 'IDEA', placeholder: 'What happens, in your own words. The director plans one chapter from it.' },
  { id: 'text', label: 'TEXT', placeholder: '# A heading per chapter\n\nor plain text, split every 600 words' },
  { id: 'video', label: 'VIDEO', placeholder: '' },
] as const
const kind = computed(() => KINDS.find((k) => k.id === form.value.kind)!)

// Mirrors `screenplay.ingest.chapters_of`: a chapter per markdown heading, else 600-word stretches.
const HEADING = /^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$/gm
const WORDS_PER_CHAPTER = 600
const chapters = computed<{ title: string; words: number }[]>(() => {
  const text = form.value.text
  const found = [...text.matchAll(HEADING)]
  if (found.length)
    return found
      .map((m, i) => ({ title: m[1], words: words(text.slice(m.index! + m[0].length, found[i + 1]?.index ?? text.length)) }))
      .filter((c) => c.words)
  const n = words(text)
  const count = Math.max(1, Math.ceil(n / WORDS_PER_CHAPTER))
  const each = Math.ceil(n / count)
  return n ? Array.from({ length: count }, (_, i) => ({ title: `chapter ${i + 1}`, words: Math.min(each, n - i * each) })) : []
})

const problems = computed(() => {
  const f = form.value
  const out: string[] = []
  if (!/^[a-z0-9][a-z0-9_-]*$/.test(f.slug)) out.push('a slug: lower-case letters, digits, _ and -')
  if (state.productions.some((one) => one.slug === f.slug)) out.push(`${f.slug} exists`)
  if (f.kind === 'idea' && !f.idea.trim()) out.push('an idea, in your own words')
  if (f.kind === 'text' && !f.text.trim()) out.push('a text to make chapters of')
  if (f.kind === 'video' && !f.source.trim()) out.push('the video to redo: a path under output/')
  if (f.kind === 'video' && !f.instruction.trim()) out.push('what should be different about it')
  if (f.kind !== 'video' && !f.photos.length) out.push('at least one photograph: the first is where it opens')
  if (f.kind !== 'video' && f.take_seconds < CLIP_MIN) out.push(`at least ${CLIP_MIN} seconds: one shot`)
  return out
})

const estimate = computed(() => {
  const f = form.value
  const ch = f.kind === 'text' ? chapters.value.length : 1
  const fewest = ch
  const most = ch * Math.max(1, Math.floor(f.take_seconds / CLIP_MIN))
  return [
    { k: 'CHAPTERS', v: String(ch) },
    { k: 'SHOTS', v: f.kind === 'video' ? 'by the source' : fewest === most ? String(most) : `${fewest}–${most}, by the director` },
    { k: 'FILM', v: f.kind === 'video' ? 'as the source' : `up to ${seconds(ch * f.take_seconds)}` },
    { k: 'PHOTOS', v: String(f.photos.length) },
    { k: 'CANVAS', v: f.kind === 'video' || f.aspect === null ? 'as the first photo' : canvasFor(f.aspect, f.video.megapixels).join('×') },
  ]
})

const begin = () =>
  props.act('begin', async () => {
    const made = await ingest(form.value)
    const planning = autoPlan.value && form.value.kind !== 'video'
    if (planning) await planProduction(made.id, 0)
    // Begun without a plan, an idea opens where the director can improve it first.
    router.push(planning || form.value.kind !== 'idea' ? `/productions/${made.id}` : `/productions/${made.id}?stage=ingest`)
  })

const photos = computed(() => {
  const p = props.production
  return Array.isArray(p?.manifest.photos) ? (p!.manifest.photos as string[]) : []
})
const source = computed(() => (typeof props.production?.manifest.video === 'string' ? (props.production.manifest.video as string) : ''))
const name = (path: string) => path.split('/').at(-1)!.replace(/^[0-9a-f]{16}-/, '')

const { draft: title, reset: resetTitle } = useDraft(() => props.production?.title ?? '')
const { draft: story, reset: resetStory } = useDraft(() => props.production?.story ?? '')
const edited = computed(
  () => !!props.production && (title.value.trim() !== props.production.title || story.value.trim() !== props.production.story.trim()),
)
const saved = ref('')
const aspectNote = ref('')
const saveStory = () =>
  props.act('story', async () => {
    const p = props.production!
    const changes: Record<string, string> = {}
    if (title.value.trim() !== p.title) changes.title = title.value
    if (story.value.trim() !== p.story.trim()) changes.story = story.value
    const { stale } = await patchProduction(p.id, changes)
    saved.value = changes.story
      ? `Saved. ${stale.prompts} prompts and ${stale.clips} clips went stale: re-plan the chapter to plan it from the new words.`
      : 'Saved.'
  })

// --- the canvas's aspect ------------------------------------------------------------------------
const given = computed(() => (typeof props.production?.manifest.aspect === 'number' ? (props.production.manifest.aspect as number) : null))
const same = (a: number | null, b: number | null) => (a === null || b === null ? a === b : Math.abs(a - b) < 0.005)
const aspects = computed(() => {
  const g = given.value
  return ASPECTS.some((a) => same(a.value, g)) ? ASPECTS : [...ASPECTS, { label: `${g!.toFixed(2)}:1`, value: g }]
})
const { draft: aspect, reset: resetAspect } = useDraft(() => aspects.value.find((a) => same(a.value, given.value))?.value ?? null)
watch(
  () => props.production?.id,
  () => {
    resetTitle()
    resetStory()
    resetAspect()
    saved.value = ''
    aspectNote.value = ''
  },
)
const aspectChanged = computed(() => !!props.production && !same(aspect.value, given.value))
const canvasNow = computed(() => {
  const c = props.production?.manifest.canvas as { width?: number; height?: number } | undefined
  return c?.width && c.height ? { width: c.width, height: c.height } : null
})
const megapixels = computed(() => (canvasNow.value ? (canvasNow.value.width * canvasNow.value.height) / (1024 * 1024) : 0.4))
const photoSize = ref<{ width: number; height: number } | null>(null)
watch(
  () => photos.value[0],
  (first) => {
    photoSize.value = null
    if (!first) return
    const img = new Image()
    img.onload = () => (photoSize.value = { width: img.naturalWidth, height: img.naturalHeight })
    img.src = outputUrl(first)
  },
  { immediate: true },
)
const allShots = computed(() => props.production?.chapters.flatMap((c) => c.shots) ?? [])
const cost = computed(() => {
  const p = props.production
  if (!p) return ''
  const prompts = allShots.value.filter((s) => s.prompt.text).length
  const clips = allShots.value.filter((s) => s.clip.artifact).length
  const places = p.members.filter((m) => m.kind === 'environment').flatMap((m) => m.references).filter((r) => r.role === 'generated' && r.artifact).length
  const next = aspect.value === null ? 'the first photo’s shape' : canvasFor(aspect.value, megapixels.value).join('×')
  if (!prompts) return `The first plan cuts the canvas at ${next}.`
  const parts = ['re-plan the chapter']
  if (places) parts.push(`redraw ${places} drawn place${places > 1 ? 's' : ''}`)
  if (clips) parts.push(`re-render ${clips} clip${clips > 1 ? 's' : ''}`)
  return `Canvas about ${next}. It costs: ${parts.join(', ')}.`
})
/** At an aspect far from the first photo's, the opening cut keeps a thin strip of it. */
const strip = computed(() => {
  const photo = photoSize.value
  if (!photo) return null
  const own = photo.width / photo.height
  const target = aspect.value ?? own
  const width = own > target ? Math.round(photo.height * target) : photo.width
  const height = own > target ? photo.height : Math.round(photo.width / target)
  const kept = (width * height) / (photo.width * photo.height)
  return kept < 0.6 ? { width, height, percent: Math.round(kept * 100) } : null
})
const firstShot = computed(() => props.production?.chapters[0]?.shots[0])
const opensOnPhoto = computed(() => firstShot.value?.keyframe.source === 'provided' && firstShot.value.keyframe.note === 'the opening frame')
const saveAspect = () =>
  props.act('aspect', async () => {
    const { stale } = await patchProduction(props.production!.id, { aspect: aspect.value })
    aspectNote.value = stale.prompts
      ? `Saved. ${stale.prompts} prompts, ${stale.clips} clips and ${stale.references ?? 0} drawn places went stale: re-plan the chapter.`
      : 'Saved.'
  })
const composeFirst = () => props.act('compose', () => patchShot(firstShot.value!.id, { keyframe: 'composed' }))

const discard = () => {
  if (confirm('Delete this production, its cast and its shots? What was rendered stays on disk.'))
    props.act('discard', async () => {
      await deleteProduction(props.production!.id)
      router.push('/')
    })
}
</script>

<template>
  <div v-if="!production" class="ingest">
    <section class="card">
      <div class="head">
        <h5>Story</h5>
        <div class="seg">
          <button v-for="k in KINDS" :key="k.id" :class="{ on: form.kind === k.id }" @click="form.kind = k.id">{{ k.label }}</button>
        </div>
      </div>
      <div class="body">
        <div class="stacked">
          <label><span class="eyebrow">title</span><input v-model="form.title" placeholder="Working title" /></label>
          <label><span class="eyebrow">slug</span><input v-model="form.slug" placeholder="shisha_lounge" /></label>
        </div>
        <textarea v-if="form.kind === 'idea'" v-model="form.idea" rows="6" :placeholder="kind.placeholder" class="prose" />
        <textarea v-else-if="form.kind === 'text'" v-model="form.text" rows="12" :placeholder="kind.placeholder" class="prose" />
        <template v-else>
          <label class="field"><span class="eyebrow">video · under output/</span><input v-model="form.source" placeholder="videos/2026-09-26/….mp4" /></label>
          <label class="field"><span class="eyebrow">what changes</span><input v-model="form.instruction" placeholder="make it look like a watercolour painting" /></label>
        </template>
        <div class="stacked ruled">
          <label v-if="form.kind !== 'video'">
            <span class="eyebrow">length · s{{ form.kind === 'text' ? ' per chapter' : '' }}</span>
            <input v-model.number="form.take_seconds" type="number" :min="CLIP_MIN" max="600" />
          </label>
          <label><span class="eyebrow">longest shot · s</span><input v-model.number="form.video.duration" type="number" min="1" max="15" /></label>
          <label v-if="form.kind !== 'video'">
            <span class="eyebrow">aspect</span>
            <select v-model="form.aspect">
              <option v-for="a in ASPECTS" :key="a.label" :value="a.value">{{ a.label }}</option>
            </select>
          </label>
          <label><span class="eyebrow">megapixels</span><input v-model.number="form.video.megapixels" type="number" min="0.1" max="2" step="0.1" /></label>
          <label v-if="form.kind === 'video'">
            <span class="eyebrow">clip scale</span><input v-model.number="form.clip_scale" type="number" min="0.1" max="1" step="0.1" />
          </label>
        </div>
        <div v-if="form.kind === 'text'" class="split ruled">
          <span class="row mono"><span class="eyebrow">split preview</span><span class="muted">{{ chapters.length }} chapters · {{ words(form.text) }} words</span></span>
          <div class="bars">
            <span v-for="(c, i) in chapters" :key="i" :title="`${c.title} · ${c.words} words`" :style="{ flex: c.words }">{{ String(i + 1).padStart(2, '0') }}</span>
          </div>
          <span class="note">A chapter per heading; with none, every 600 words. Each is planned from its own text.</span>
        </div>
      </div>
    </section>

    <section class="card">
      <div class="head">
        <h5>References</h5>
        <span class="note">{{ form.kind === 'video' ? 'optional · they override the source' : 'the first photograph is where it opens' }}</span>
      </div>
      <div class="body">
        <PhotoPicker v-model:photos="form.photos" />
        <p class="note">
          Qwen-VL surveys every photograph once: who the story needs, and which photograph shows whom. What it sees goes into the
          bible, so shots match what the picture shows, not what the text says. The same file uploaded twice is one reference.
        </p>
      </div>
    </section>

    <section class="card go">
      <div class="estimate mono">
        <span v-for="e in estimate" :key="e.k"><span class="eyebrow">{{ e.k }}</span>{{ e.v }}</span>
      </div>
      <div class="buttons">
        <label v-if="form.kind !== 'video'" class="check">
          <input v-model="autoPlan" type="checkbox" />
          run the plan, stop before render
        </label>
        <span v-if="!autoPlan && form.kind === 'idea'" class="note">begin only: improve the idea with the director, then plan</span>
        <button class="primary" :disabled="!!problems.length || !!busy" :title="problems.join('\n')" @click="begin">
          {{ busy === 'begin' ? 'beginning…' : autoPlan && form.kind !== 'video' ? 'Begin and plan' : 'Begin' }}
        </button>
      </div>
      <p v-if="problems.length" class="note needs">needs {{ problems.join(' · ') }}</p>
    </section>
  </div>

  <div v-else class="ingest">
    <section class="card">
      <div class="head">
        <h5>Story</h5>
        <span class="eyebrow">{{ production.story_kind || 'shots by hand' }}</span>
      </div>
      <div class="body">
        <label class="field"><span class="eyebrow">title</span><input v-model="title" /></label>
        <textarea
          v-if="production.story_kind === 'idea' || production.story_kind === 'text'"
          v-model="story"
          :rows="production.story_kind === 'text' ? 12 : 10"
          class="prose"
          :placeholder="KINDS.find((k) => k.id === production!.story_kind)?.placeholder"
        />
        <p v-else class="story">{{ production.story || 'No story: the shots were added by hand.' }}</p>
        <div class="saving">
          <span class="note">{{ saved || (edited ? 'A changed story makes every prompt stale until the chapter is re-planned from it.' : '') }}</span>
          <button class="primary" :disabled="!edited || !!busy || !title.trim()" @click="saveStory">{{ busy === 'story' ? 'saving…' : 'Save' }}</button>
        </div>
        <div class="facts mono">
          <span v-for="c in production.chapters" :key="c.id">
            <span class="dim">{{ String(c.position + 1).padStart(2, '0') }}</span> {{ c.title || 'chapter' }} · {{ c.shots.length }} shots
          </span>
        </div>
      </div>
    </section>

    <IdeaPanel v-if="production.story_kind === 'idea'" :production="production" :busy="busy" :act="act" />

    <section class="card">
      <div class="head">
        <h5>References</h5>
        <span class="note">{{ photos.length }} photographs{{ source ? ' · a source video' : '' }}</span>
      </div>
      <div class="body">
        <video v-if="source" class="frame" :src="outputUrl(source)" controls muted preload="metadata" />
        <div class="photos">
          <a v-for="(p, i) in photos" :key="p" :href="outputUrl(p)" target="_blank">
            <img class="frame" :src="outputUrl(p)" alt="" />
            <span class="mono">{{ i === 0 ? 'opens · ' : '' }}{{ name(p) }}</span>
          </a>
        </div>
        <div v-if="production.story_kind === 'idea' || production.story_kind === 'text'" class="aspect ruled">
          <div class="saving">
            <label class="field">
              <span class="eyebrow">aspect · canvas {{ canvasNow ? `${canvasNow.width}×${canvasNow.height}` : 'not cut yet' }}</span>
              <select v-model="aspect">
                <option v-for="a in aspects" :key="a.label" :value="a.value">{{ a.label }}</option>
              </select>
            </label>
            <button class="primary" :disabled="!aspectChanged || !!busy" @click="saveAspect">{{ busy === 'aspect' ? 'saving…' : 'Change aspect' }}</button>
          </div>
          <span class="note">{{ aspectNote || (aspectChanged ? cost : 'The canvas every shot renders at, cut from the first photo at this shape.') }}</span>
          <p v-if="strip" class="band hint">
            At this aspect the opening is a {{ strip.width }}×{{ strip.height }} strip of the first photo, {{ strip.percent }}% of it.
            <template v-if="opensOnPhoto">
              Composed, shot 1's first frame is drawn by the editor at the canvas size instead.
              <button class="link" :disabled="!!busy" @click="composeFirst">compose shot 1</button>
            </template>
            <template v-else-if="!firstShot">After the plan, compose shot 1's keyframe to have it drawn at the canvas size.</template>
          </p>
        </div>
      </div>
    </section>

    <footer class="foot">
      <span class="note">Deleting drops the production, its cast and its shots. What was rendered stays on disk.</span>
      <div class="buttons">
        <button class="danger" :disabled="!!busy" @click="discard">Delete production</button>
      </div>
    </footer>
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

.stacked > label,
.field {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.ruled {
  padding-top: var(--space-2);
  border-top: 1px solid var(--border);
}

.card textarea.prose {
  font-family: var(--font-body);
  font-weight: var(--weight-light);
  line-height: 1.5;
  padding: var(--space-1) var(--space-2);
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

.go,
.foot {
  grid-column: 1 / -1;
}

.go {
  border-color: var(--border-strong);
  padding: var(--space-2) var(--space-3);
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--space-3);
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

.check input {
  width: auto;
  accent-color: var(--accent);
}

.needs {
  flex-basis: 100%;
}

.aspect {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.aspect .saving {
  align-items: flex-end;
}

.saving {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--space-2);
}

.story {
  font-size: var(--text-sm);
  font-weight: var(--weight-light);
  line-height: 1.5;
  white-space: pre-wrap;
  margin: 0;
}

.facts {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: var(--text-xs);
  padding-top: var(--space-2);
  border-top: 1px solid var(--border);
}

.photos {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(112px, 1fr));
  gap: var(--space-1);
}

.photos a {
  display: flex;
  flex-direction: column;
  gap: 4px;
  text-decoration: none;
  min-width: 0;
}

.photos img {
  aspect-ratio: 3 / 4;
}

.photos span {
  font-size: 10px;
  color: var(--text-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
</style>
