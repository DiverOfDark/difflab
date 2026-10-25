<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { deleteProduction, listTools, outputUrl, patchProduction, patchShot, putSettings, readIngesting, type Production } from '../../api'
import { ASPECTS, FITS, canvasFor, plural, runLive, type Fit } from '../../production'
import { useDraft } from '../../draft'
import IdeaPanel from '../IdeaPanel.vue'
import IngestForm from './IngestForm.vue'
import IngestingStage from './IngestingStage.vue'

const props = defineProps<{
  production: Production | null
  busy: string
  act: (name: string, what: () => Promise<unknown>) => Promise<void>
}>()
const router = useRouter()
const route = useRoute()


// --- while the plan builds the bible ------------------------------------------------------------
/** Asked for by the form that began it, or found: a plan whose bible is still being built. */
const building = ref(route.query.ingesting === '1')
async function look() {
  const p = props.production
  if (!p || building.value || !p.runs.some((r) => r.scope.startsWith('plan:') && runLive(r.state))) return
  const now = await readIngesting(p.id).catch(() => null)
  building.value = !!now && now.live.length > 0 && !now.bible_done
}
watch(() => props.production?.runs.map((r) => r.state).join(), look, { immediate: true })
function built(cancelled: boolean) {
  building.value = false
  router.replace({ query: cancelled ? { stage: 'ingest' } : { stage: 'bible' } })
}

// --- the knobs ----------------------------------------------------------------------------------
interface Knob {
  name: string
  label: string
  hint: string
  value: number
  low: number | null
  high: number | null
}
const knobs = ref<Knob[]>([])
const knobNote = ref('')
async function loadKnobs() {
  if (!props.production) return
  const tool = (await listTools(props.production.id).catch(() => [])).find((t) => t.name === 'ingest')
  knobs.value = (tool?.settings ?? []).map((f) => ({
    name: f.name,
    label: f.label,
    hint: f.hint,
    value: Number(tool!.effective?.[f.name] ?? f.default),
    low: f.low ?? null,
    high: f.high ?? null,
  }))
}
onMounted(loadKnobs)
watch(() => props.production?.id, loadKnobs)
const saveKnobs = () =>
  props.act('knobs', async () => {
    await putSettings('ingest', 'production', String(props.production!.id), Object.fromEntries(knobs.value.map((k) => [k.name, k.value])))
    knobNote.value = 'Saved. The chapter split applies when the story is next saved; frames, when references are next added.'
  })

/** The references as they were given: kind, entity, and for a video its frames. */
interface GivenRef {
  path: string
  kind: string
  entity: string
  guess: string
  frames?: { path: string; at: number; sharpness: number; kept: boolean }[]
}
const references = computed(() => (Array.isArray(props.production?.manifest.references) ? (props.production!.manifest.references as GivenRef[]) : []))

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
const givenFit = computed<Fit>(() => (props.production?.manifest.fit === 'extend' ? 'extend' : 'crop'))
const { draft: fit, reset: resetFit } = useDraft(() => givenFit.value)
watch(
  () => props.production?.id,
  () => {
    resetTitle()
    resetStory()
    resetAspect()
    resetFit()
    saved.value = ''
    aspectNote.value = ''
  },
)
const aspectChanged = computed(() => !!props.production && (!same(aspect.value, given.value) || fit.value !== givenFit.value))
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
  if (!photo || fit.value === 'extend') return null
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
    const changes: Record<string, unknown> = {}
    if (!same(aspect.value, given.value)) changes.aspect = aspect.value
    if (fit.value !== givenFit.value) changes.fit = fit.value
    const { stale } = await patchProduction(props.production!.id, changes)
    aspectNote.value = stale.prompts
      ? `Saved. ${stale.prompts} prompts, ${stale.clips} clips and ${stale.references ?? 0} drawn places went stale: re-plan the chapter.`
      : 'Saved.'
  })
const composeFirst = () => props.act('compose', () => patchShot(firstShot.value!.id, { keyframe: 'composed' }))

const toggleKeep = () => props.act('keep', () => patchProduction(props.production!.id, { keep: !props.production!.keep }))

const discard = () => {
  if (confirm('Delete this production, its cast and its shots? What was rendered stays on disk.'))
    props.act('discard', async () => {
      await deleteProduction(props.production!.id)
      router.push('/productions')
    })
}
</script>

<template>
  <IngestForm v-if="!production" :busy="busy" :act="act" />
  <IngestingStage v-else-if="building" :production="production" :busy="busy" :act="act" @done="built" />
  <div v-else class="ingest">
    <section class="card">
      <div class="head">
        <h5>Story</h5>
        <span class="eyebrow">{{ production.story_kind || 'shots by hand' }}</span>
      </div>
      <div class="body">
        <label class="field"><span class="eyebrow">title</span><input v-model="title" class="plain" /></label>
        <textarea
          v-if="production.story_kind === 'text'"
          v-model="story"
          :rows="production.chapters.length > 1 ? 12 : 10"
          class="prose"
          placeholder="What happens, in your own words. # A heading per chapter, or plain text split every 600 words."
        />
        <p v-else class="story">{{ production.story || 'No story: the shots were added by hand.' }}</p>
        <div class="saving">
          <span class="note">{{ saved || (edited ? 'A changed story makes every prompt stale until the chapter is re-planned from it.' : '') }}</span>
          <button class="primary" :disabled="!edited || !!busy || !title.trim()" @click="saveStory">{{ busy === 'story' ? 'saving…' : 'Save' }}</button>
        </div>

        <div v-if="production.story_kind === 'text'" class="aspect ruled">
          <div class="saving">
            <div class="three">
              <label class="field">
                <span class="eyebrow">aspect · canvas {{ canvasNow ? `${canvasNow.width}×${canvasNow.height}` : 'not cut yet' }}</span>
                <select v-model="aspect">
                  <option v-for="a in aspects" :key="a.label" :value="a.value">{{ a.label }}</option>
                </select>
              </label>
              <label class="field" title="crop cuts the first photo to the canvas; extend keeps it whole and has the editor draw the sides">
                <span class="eyebrow">fit</span>
                <select v-model="fit">
                  <option v-for="f in FITS" :key="f.value" :value="f.value">{{ f.label }}</option>
                </select>
              </label>
            </div>
            <button class="primary" :disabled="!aspectChanged || !!busy" @click="saveAspect">{{ busy === 'aspect' ? 'saving…' : 'Change canvas' }}</button>
          </div>
          <span class="note">{{ aspectNote || (aspectChanged ? cost : fit === 'extend' ? 'The canvas every shot renders at: the first photo whole, its sides drawn out to this shape by the editor.' : 'The canvas every shot renders at, cut from the first photo at this shape.') }}</span>
          <p v-if="strip" class="band hint">
            At this aspect the opening is a {{ strip.width }}×{{ strip.height }} strip of the first photo, {{ strip.percent }}% of it.
            <template v-if="opensOnPhoto">
              Composed, shot 1's first frame is drawn by the editor at the canvas size instead.
              <button class="link" :disabled="!!busy" @click="composeFirst">compose shot 1</button>
            </template>
            <template v-else-if="!firstShot">After the plan, compose shot 1's keyframe to have it drawn at the canvas size.</template>
          </p>
        </div>

        <div class="facts mono ruled">
          <span class="eyebrow">chapters</span>
          <span v-for="c in production.chapters" :key="c.id">
            <span class="dim">{{ String(c.position + 1).padStart(2, '0') }}</span> {{ c.title || 'chapter' }} · {{ c.shots.length }} shots
          </span>
        </div>

        <div v-if="knobs.length" class="knobbed ruled">
          <div class="knobs">
            <label v-for="k in knobs" :key="k.name" class="field" :title="k.hint">
              <span class="eyebrow">{{ k.label }}</span>
              <select v-if="k.name === 'heading_level'" v-model.number="k.value">
                <option v-for="n in 6" :key="n" :value="n">h{{ n }}{{ n > 1 ? ` · h1–h${n}` : '' }}</option>
              </select>
              <input v-else v-model.number="k.value" type="number" :min="k.low ?? undefined" :max="k.high ?? undefined" />
            </label>
          </div>
          <div class="saving">
            <span class="note">{{ knobNote || 'This production’s, over the machine’s.' }}</span>
            <button :disabled="!!busy" @click="saveKnobs">{{ busy === 'knobs' ? 'saving…' : 'Save knobs' }}</button>
          </div>
        </div>
      </div>
    </section>

    <div class="side">
      <section class="card">
        <div class="head">
          <h5>References</h5>
          <span class="note">{{ plural(photos.length, 'photograph') }}{{ source ? ' · a source video' : '' }}</span>
        </div>
        <div class="body">
          <video v-if="source" class="frame" :src="outputUrl(source)" controls muted preload="metadata" />
          <div v-if="references.length" class="given">
            <div v-for="r in references" :key="r.path" class="ref">
              <a :href="outputUrl(r.path)" target="_blank" class="thumb" :style="{ backgroundImage: `url(${outputUrl(r.frames?.find((f) => f.kept)?.path ?? r.path)})` }">
                <span v-if="r.frames" class="badge mono">MP4</span>
              </a>
              <span class="what">
                <span class="mono">{{ name(r.path) }}{{ photos[0] === r.path || r.frames?.some((f) => f.path === photos[0]) ? ' · opens' : '' }}</span>
                <span class="note">{{
                  r.frames
                    ? `${r.frames.length} frames · ${r.frames.filter((f) => f.kept).length} kept sharp · also a motion reference${r.guess ? ` · ${r.guess}` : ''}`
                    : r.guess || '—'
                }}</span>
              </span>
              <span class="mono dim">{{ r.kind === 'video' ? 'motion ref' : r.kind || 'unmatched' }}</span>
              <span class="mono">{{ r.entity || (r.kind === 'style' ? 'style' : '—') }}</span>
            </div>
          </div>
          <div v-else class="photos">
            <a v-for="(p, i) in photos" :key="p" :href="outputUrl(p)" target="_blank">
              <img class="frame" :src="outputUrl(p)" alt="" />
              <span class="mono">{{ i === 0 ? 'opens · ' : '' }}{{ name(p) }}</span>
            </a>
          </div>
        </div>
      </section>

      <IdeaPanel v-if="production.story_kind === 'text' && production.chapters.length === 1" :production="production" :busy="busy" :act="act" />
    </div>

    <footer class="foot">
      <span class="note">
        Deleting drops the production, its cast and its shots. What was rendered stays on disk.
        {{ production.keep ? 'Kept: the cleanup never deletes it.' : 'Past the 15 newest, a week untouched and the cleanup deletes it.' }}
      </span>
      <div class="buttons">
        <button :disabled="!!busy" @click="toggleKeep">{{ production.keep ? 'Stop keeping' : 'Keep' }}</button>
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

.field {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  min-width: 0;
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

.side {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
  min-width: 0;
}

.three,
.knobs {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: var(--space-2);
  flex: 1;
}

.knobbed {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.card input.plain {
  font-family: var(--font-body);
}

.foot {
  grid-column: 1 / -1;
}

.buttons {
  display: flex;
  gap: var(--space-1);
  align-items: center;
  flex-wrap: wrap;
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

.given {
  display: flex;
  flex-direction: column;
}

.given .ref {
  display: grid;
  grid-template-columns: 40px minmax(0, 1fr) 80px minmax(80px, 0.6fr);
  gap: var(--space-2);
  align-items: center;
  padding: var(--space-1) 0;
  border-bottom: 1px solid var(--border);
  font-size: var(--text-xs);
}

.given .thumb {
  width: 40px;
  height: 40px;
  background: var(--bg-raised) center / cover no-repeat;
  border: 1px solid var(--border);
  border-radius: 2px;
  display: flex;
  align-items: flex-end;
  justify-content: center;
  text-decoration: none;
}

.given .badge {
  font-size: 9px;
  color: var(--text-primary);
  background: oklch(0% 0 0 / 0.6);
  padding: 0 3px;
  border-radius: 2px;
}

.given .what {
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.given .what > span {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
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
