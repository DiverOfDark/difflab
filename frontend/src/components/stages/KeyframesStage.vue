<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { outputUrl, patchProduction, patchShot, redoKeyframe, renderKeyframes, type Production, type ShotInfo } from '../../api'
import { castOf, phaseOf, shotId, type Phase, type StageId } from '../../production'
import PhotoPicker from '../PhotoPicker.vue'

const props = defineProps<{
  production: Production
  chapter: number
  shots: ShotInfo[]
  focus: number | null
  busy: string
  act: (name: string, what: () => Promise<unknown>) => Promise<void>
}>()
const emit = defineEmits<{ go: [to: StageId, shot?: number | null] }>()

const aspect = computed(() => {
  const c = props.production.manifest.canvas as { width?: number; height?: number } | undefined
  return c?.width && c?.height ? `${c.width} / ${c.height}` : '16 / 9'
})
const dims = computed(() => {
  const c = props.production.manifest.canvas as { width?: number; height?: number } | undefined
  return c?.width && c?.height ? `${c.width}×${c.height}` : ''
})

const filter = ref<'all' | Phase>('all')
const filters = computed(() => {
  const of = (p: Phase) => props.shots.filter((s) => phaseOf(s.keyframe.state) === p).length
  return [
    { id: 'all' as const, label: 'all', count: props.shots.length },
    { id: 'review' as const, label: 'flagged', count: of('review') },
    { id: 'done' as const, label: 'made', count: of('done') },
    { id: 'pending' as const, label: 'not made', count: of('pending') + of('stale') },
  ]
})
const rows = computed(() =>
  props.shots.filter((s) => {
    const p = phaseOf(s.keyframe.state)
    return filter.value === 'all' || p === filter.value || (filter.value === 'pending' && p === 'stale')
  }),
)

const index = computed(() => props.shots.findIndex((s) => s.id === props.focus))
const shot = computed(() => props.shots[index.value] ?? null)
const step = (by: number) => {
  const n = props.shots.length
  emit('go', 'keyframes', props.shots[(index.value + by + n) % n].id)
}

const OPENING_NOTE = 'the opening frame'
const SOURCES = [
  { id: 'chained', label: 'CHAINED', why: "the previous shot's handoff frame" },
  { id: 'opening', label: 'OPENING', why: 'the first photograph, cut by the plan where it holds the most of the cast' },
  { id: 'composed', label: 'COMPOSED', why: 'the editor draws it from the words over a reference: of the place, or of who it names' },
  { id: 'provided', label: 'PROVIDED', why: 'a picture you give' },
]
const sourcesFor = (s: ShotInfo) => SOURCES.filter((src) => (s.position === 0 ? src.id !== 'chained' : src.id !== 'opening'))
const sourceOf = (s: ShotInfo) =>
  s.keyframe.source === 'provided' && s.keyframe.note === OPENING_NOTE ? 'opening' : s.keyframe.source
const setSource = (s: ShotInfo, source: string) => {
  if (source === 'provided' || source === sourceOf(s)) return
  props.act(`keyframe ${s.id}`, () =>
    patchShot(s.id, { keyframe: source === 'composed' ? 'composed' : source === 'opening' ? 'opening' : '' }),
  )
}

const written = computed(() => {
  const m = props.production.manifest
  return String(m.opening_words ?? m.opening_arrangement ?? '')
})
const byHand = computed(() => !!props.production.manifest.opening_words)
const words = ref('')
watch(written, (now) => (words.value = now), { immediate: true })
const saveWords = (text: string) => props.act('opening', () => patchProduction(props.production.id, { opening: text }))
const named = computed(() => props.production.members.map((m) => `{${m.slug}}`).join(' '))

const own = ref<string[]>([])
const bring = (s: ShotInfo, photos: string[]) => {
  own.value = photos
  const last = photos.at(-1)
  if (last) props.act(`keyframe ${s.id}`, () => patchShot(s.id, { keyframe: last })).then(() => (own.value = []))
}

/** The judge's verdict, parsed: its score, what it would fix, and its one-line summary. */
interface Verdict {
  score: number | null
  issues: string[]
  notes: string
}
function verdictOf(review: string): Verdict | null {
  if (!review) return null
  try {
    const v = JSON.parse(review)
    return { score: typeof v.score === 'number' ? v.score : null, issues: v.issues ?? [], notes: v.notes ?? '' }
  } catch {
    return { score: null, issues: [], notes: review }
  }
}
const verdict = computed(() => (shot.value ? verdictOf(shot.value.keyframe.review) : null))
/** What the frame was composed from, in the editor's own numbering: the place is Picture 1. */
const madeFrom = computed(() => {
  const m = shot.value?.keyframe.made_from
  if (!m) return []
  return [
    ...(m.base ? [{ label: 'Picture 1 · the place', path: m.base }] : []),
    ...Object.entries(m.pictures).map(([slug, path], i) => ({ label: `Picture ${i + 2} · ${slug}`, path })),
  ]
})

const JUDGED = 'Fix:'
const note = ref('')
watch(
  () => shot.value?.id,
  () => (note.value = (shot.value?.keyframe.note ?? '').split(JUDGED)[0].trim()),
  { immediate: true },
)
const takeIssues = () => {
  const issues = verdict.value?.issues ?? []
  note.value = [note.value.trim(), ...issues].filter(Boolean).join('; ')
}
const redo = (s: ShotInfo) => props.act(`keyframe ${s.id}`, () => redoKeyframe(s.id, note.value))
const makeAll = () => props.act('keyframes', () => renderKeyframes(props.production.id, props.chapter))
const composedCount = computed(() => props.shots.filter((s) => s.keyframe.source === 'composed').length)

const landable = (s: ShotInfo) => s.position > 0 && s.keyframe.source !== 'chained'
const landed = (s: ShotInfo) => landable(s) && s.keyframe.landed
const setLanded = (s: ShotInfo, on: boolean) => props.act(`keyframe ${s.id}`, () => patchShot(s.id, { landed: on }))

const impact = computed(() => {
  const s = shot.value
  if (!s) return ''
  const next = props.shots[index.value + 1]
  const tail = next?.chained ? ' and the chained shot after it' : ''
  const before = landed(s) ? `, the clip before it, which ends on it,` : ''
  return `Changing this frame makes the clip behind it stale${before}${tail}; nothing else moves. A frame costs seconds, the clip ~minutes.`
})
</script>

<template>
  <div v-if="shot" class="editor">
    <div class="nav">
      <span class="left">
        <button class="link big" @click="emit('go', 'keyframes')">← keyframes</button>
        <span class="steps"><button @click="step(-1)">‹</button><button @click="step(1)">›</button></span>
      </span>
      <span class="mono muted">{{ shotId(chapter, shot) }} · {{ shot.camera || 'camera free' }} · {{ shot.seconds }} s</span>
    </div>

    <div class="split">
      <div class="col">
        <section class="card">
          <div class="head slim">
            <div class="seg">
              <button
                v-for="src in sourcesFor(shot)"
                :key="src.id"
                :class="{ on: sourceOf(shot) === src.id }"
                :title="src.why"
                :disabled="!!busy"
                @click="setSource(shot, src.id)"
              >
                {{ src.label }}
              </button>
            </div>
            <span class="mono dim small">{{ shot.keyframe.state }}{{ shot.keyframe.pinned ? ' · pinned' : '' }}</span>
          </div>
          <div class="body">
            <div class="stage" :style="{ aspectRatio: aspect }">
              <img v-if="shot.keyframe.artifact && shot.keyframe.present" :src="outputUrl(shot.keyframe.artifact)" alt="" />
              <span v-else class="mono dim">{{ shot.keyframe.source === 'chained' ? 'made when the shot before renders' : shot.keyframe.state }}</span>
              <span v-if="dims" class="dims mono">{{ dims }}</span>
              <span class="score mono" :class="phaseOf(shot.keyframe.state)">{{ shot.keyframe.state.toUpperCase() }}</span>
            </div>
          </div>
        </section>

        <section class="card">
          <div class="head slim">
            <h5>QA · the judge against the shot and the cast's photos</h5>
            <span v-if="verdict?.score != null" class="mono small" :class="{ warn: verdict.score < 0.5 }">{{ verdict.score.toFixed(2) }}</span>
          </div>
          <div class="check">
            <i class="dot" :class="phaseOf(shot.keyframe.state)" />
            <span class="what">
              <span>{{ verdict ? verdict.notes || 'Verdict' : 'Not judged yet' }}</span>
              <ul v-if="verdict?.issues.length" class="issues">
                <li v-for="i in verdict.issues" :key="i">{{ i }}</li>
              </ul>
              <span v-else class="why">{{ verdict ? 'Nothing to fix.' : 'Composed frames are judged as they are made.' }}</span>
            </span>
          </div>
        </section>

        <section v-if="shot.keyframe.source === 'composed'" class="card">
          <div class="head slim">
            <h5>Redo this frame</h5>
            <span class="note">a new seed, and your note to the editor</span>
          </div>
          <div class="body">
            <textarea v-model="note" rows="3" class="prose" placeholder="the robe is the dark green one from his photo; nobody else in the courtyard" />
            <div class="row">
              <button class="link" :disabled="!verdict?.issues.length" @click="takeIssues">add the judge's issues</button>
              <button class="primary" :disabled="!!busy" @click="redo(shot)">
                {{ busy === `keyframe ${shot.id}` ? 'queueing…' : 'Redo the frame' }}
              </button>
            </div>
            <span class="note">Only the keyframes are made, not the clips; render the chapter when they are right.</span>
          </div>
        </section>

        <section v-if="madeFrom.length" class="card">
          <div class="head slim">
            <h5>Made from</h5>
            <span class="note">what the editor was handed</span>
          </div>
          <div class="refs">
            <a v-for="m in madeFrom" :key="m.label" class="ref" :href="outputUrl(m.path)" target="_blank">
              <img class="frame chip-img" :src="outputUrl(m.path)" alt="" />
              <span class="what"><span class="mono">{{ m.label }}</span><span class="why">{{ m.path.split('/').at(-1) }}</span></span>
            </a>
          </div>
        </section>
      </div>

      <aside class="col">
        <section v-if="shot.position === 0" class="card">
          <div class="head slim">
            <h5>The first frame, in words</h5>
            <span class="note">{{ byHand ? 'by hand · a re-plan keeps it' : 'read off the first photo' }}</span>
          </div>
          <div class="body">
            <textarea v-model="words" rows="5" class="prose" />
            <span class="note">
              Every clip after it follows from this sentence, and a composed frame is drawn from it. Name each member as
              <span class="mono">{id}</span>: {{ named }}. Saving makes every prompt stale until a re-plan.
            </span>
            <div class="row">
              <button class="link" :disabled="!!busy || !byHand" @click="saveWords('')">back to the photo's words</button>
              <button class="primary" :disabled="!!busy || words.trim() === written.trim() || !words.trim()" @click="saveWords(words)">
                {{ busy === 'opening' ? 'saving…' : 'Save' }}
              </button>
            </div>
          </div>
        </section>

        <section v-if="landable(shot)" class="card">
          <div class="head slim">
            <h5>The join</h5>
            <span class="mono dim small">{{ shotId(chapter, shots[index - 1]) }} → {{ shotId(chapter, shot) }}</span>
          </div>
          <div class="body">
            <label class="land">
              <input type="checkbox" :checked="shot.keyframe.landed" :disabled="!!busy" @change="setLanded(shot, ($event.target as HTMLInputElement).checked)" />
              the shot before ends on this frame
            </label>
            <span class="note">
              {{
                shot.keyframe.landed
                  ? 'Its last frame is pinned to this picture and its motion carries into this shot, so the join does not jump. A hard landing can freeze or morph its last half second.'
                  : 'The shot before ends wherever the model takes it, and this one opens fresh on this frame: a cut.'
              }}
            </span>
          </div>
        </section>

        <section class="card">
          <div class="head slim">
            <h5>Composition</h5>
            <span class="note">from the shot</span>
          </div>
          <div class="body">
            <div class="field">
              <span class="row"><span class="eyebrow">action · from shot list</span><button class="link" @click="emit('go', 'shots')">edit in shot list →</button></span>
              <span class="quote">{{ shot.action || '—' }}</span>
            </div>
            <div v-if="shot.keyframe.note" class="field">
              <span class="eyebrow">frame note</span>
              <span class="quote">{{ shot.keyframe.note }}</span>
            </div>
            <div class="facts mono">
              <span class="dim">cast</span><span>{{ castOf(shot).join(', ') || '—' }}</span>
              <span class="dim">camera</span><span>{{ shot.camera || '—' }}</span>
              <span class="dim">source</span><span>{{ shot.keyframe.source }}</span>
            </div>
          </div>
        </section>

        <section class="card">
          <div class="head slim">
            <h5>Pictures the clip is handed</h5>
            <span class="mono dim small">{{ shot.prompt.pictures.length }}</span>
          </div>
          <div class="refs">
            <div v-if="shot.chained && shot.position > 0" class="ref">
              <span class="frame chip-img chain" />
              <span class="what"><span class="mono">│ previous shot · handoff</span><span class="why">chained · this frame is the opening, not composed</span></span>
              <span class="mono dim small">locked</span>
            </div>
            <a v-for="(p, i) in shot.prompt.pictures" :key="p" class="ref" :href="outputUrl(p)" target="_blank">
              <img class="frame chip-img" :src="outputUrl(p)" alt="" />
              <span class="what"><span class="mono">Picture {{ i + 1 }}</span><span class="why">{{ p.split('/').at(-1) }}</span></span>
            </a>
          </div>
        </section>

        <section class="card">
          <div class="head slim"><h5>Or bring your own</h5></div>
          <div class="body">
            <PhotoPicker :photos="own" @update:photos="bring(shot, $event)" />
            <span class="note">Used as this keyframe as it is; the shot's source becomes provided.</span>
          </div>
        </section>
      </aside>
    </div>

    <footer class="foot">
      <span class="note">{{ impact }}</span>
      <div class="buttons">
        <button @click="emit('go', 'prompts', shot.id)">Prompt →</button>
        <button @click="emit('go', 'render', shot.id)">Clip →</button>
      </div>
    </footer>
  </div>

  <div v-else class="grid-view">
    <div class="bar-row">
      <div class="chips">
        <button v-for="f in filters" :key="f.id" class="chip" :class="{ on: filter === f.id }" @click="filter = f.id">
          {{ f.label }} <span class="n">{{ f.count }}</span>
        </button>
      </div>
      <span class="acts">
        <span class="note">chained frames are made by the chapter run</span>
        <button class="primary" :disabled="!!busy || !composedCount" title="compose and judge every composed keyframe, without rendering clips" @click="makeAll">
          {{ busy === 'keyframes' ? 'queueing…' : `Make ${composedCount} keyframe${composedCount === 1 ? '' : 's'}` }}
        </button>
      </span>
    </div>
    <p v-if="!rows.length" class="band hint">Nothing here.</p>
    <div class="grid">
      <article v-for="s in rows" :key="s.id" class="card kf" :class="phaseOf(s.keyframe.state)">
        <button class="pic" :style="{ aspectRatio: aspect }" @click="emit('go', 'keyframes', s.id)">
          <img v-if="s.keyframe.artifact && s.keyframe.present" :src="outputUrl(s.keyframe.artifact)" alt="" />
          <span v-else class="mono dim">{{ s.keyframe.source }}</span>
          <span class="id mono">{{ shotId(chapter, s) }}</span>
          <span class="score mono" :class="phaseOf(s.keyframe.state)">{{ s.keyframe.state }}</span>
        </button>
        <div class="text">
          <span class="action">{{ s.action || '—' }}</span>
          <span v-if="verdictOf(s.keyframe.review)" class="review">
            {{ verdictOf(s.keyframe.review)!.score?.toFixed(2) ?? '' }} · {{ verdictOf(s.keyframe.review)!.issues.join('; ') || verdictOf(s.keyframe.review)!.notes }}
          </span>
          <span class="meta mono"><span>{{ s.keyframe.source }}{{ landed(s) ? ' · landed' : '' }}{{ s.keyframe.pinned ? ' · pinned' : '' }}</span><span>{{ s.seconds }} s</span></span>
        </div>
        <div class="acts mono">
          <button class="link" @click="emit('go', 'prompts', s.id)">prompt</button>
          <button class="link push" @click="emit('go', 'keyframes', s.id)">edit frame</button>
        </div>
      </article>
    </div>
  </div>
</template>

<style scoped>
.editor,
.grid-view {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.editor {
  gap: var(--space-3);
}

.nav,
.bar-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--space-2);
  flex-wrap: wrap;
}

.nav .left {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

.big {
  font-size: var(--text-xs);
}

.steps {
  display: flex;
  gap: var(--space-1);
}

.steps button {
  padding: 2px 8px;
  font-family: var(--font-mono);
  color: var(--text-secondary);
}

.nav > .mono {
  font-size: var(--text-xs);
}

.split {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 440px), 1fr));
  gap: var(--space-3);
  align-items: start;
}

.col {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  min-width: 0;
}

.head {
  align-items: center !important;
  flex-wrap: wrap;
}

.small {
  font-size: 11px;
}

.stage {
  background: oklch(22% 0.02 260);
  border: 1px solid var(--border-strong);
  border-radius: 2px;
  position: relative;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  max-height: 70vh;
  margin: 0 auto;
  width: 100%;
}

.stage img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: var(--video);
}

.dims,
.score {
  position: absolute;
  bottom: 8px;
  font-size: 10px;
}

.dims {
  left: 8px;
  color: var(--text-tertiary);
}

.score {
  right: 8px;
  padding: 1px 6px;
  border-radius: 2px;
  background: var(--bg-base);
  color: var(--text-secondary);
  border: 1px solid var(--border);
}

.score.review {
  background: var(--status-degraded);
  color: var(--accent-fg);
  border-color: var(--status-degraded);
}

.score.stale {
  background: var(--status-error);
  color: var(--accent-fg);
  border-color: var(--status-error);
}

.check,
.ref {
  display: grid;
  grid-template-columns: 16px minmax(0, 1fr) auto;
  gap: var(--space-2);
  align-items: start;
  padding: var(--space-1) var(--space-2);
  border-bottom: 1px solid var(--border);
  font-size: var(--text-xs);
}

.land {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  font-size: var(--text-xs);
}

.land input {
  width: auto;
  accent-color: var(--accent);
}

.check .dot {
  margin-top: 5px;
  justify-self: center;
}

.what {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}

.why {
  font-weight: var(--weight-light);
  color: var(--text-secondary);
  line-height: 1.45;
  overflow-wrap: anywhere;
}

.field {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.row {
  display: flex;
  justify-content: space-between;
}

.card textarea.prose {
  font-family: var(--font-body);
  font-weight: var(--weight-light);
  line-height: 1.5;
  padding: var(--space-1) var(--space-2);
}

.row .primary {
  margin-left: auto;
}

.quote {
  font-size: var(--text-sm);
  line-height: 1.5;
  font-weight: var(--weight-light);
  color: var(--text-secondary);
  padding: var(--space-1) var(--space-2);
  background: var(--bg-raised);
  border-radius: 2px;
  border: 1px solid var(--border);
}

.facts {
  display: grid;
  grid-template-columns: 72px 1fr;
  gap: 4px var(--space-2);
  font-size: 11px;
}

.refs {
  padding: 0 var(--space-2) var(--space-2);
  display: flex;
  flex-direction: column;
}

.ref {
  grid-template-columns: 32px minmax(0, 1fr) auto;
  align-items: center;
  padding: var(--space-1) 0;
  text-decoration: none;
  color: inherit;
}

.chip-img {
  width: 32px;
  height: 32px;
}

.chain {
  border-color: var(--accent);
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: var(--space-1);
}

.kf {
  display: flex;
  flex-direction: column;
}

.kf.review {
  border-color: var(--status-degraded);
}

.kf.stale {
  border-color: var(--status-error);
}

.kf.pending {
  opacity: 0.7;
}

.pic {
  position: relative;
  border: 0;
  border-bottom: 1px solid var(--border);
  border-radius: 0;
  padding: 0;
  background: oklch(22% 0.02 260);
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  font-size: 10px;
}

.pic img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.pic .id,
.pic .score {
  position: absolute;
  top: 4px;
  bottom: auto;
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 2px;
  background: var(--bg-base);
  border: 1px solid var(--border);
  color: var(--text-primary);
}

.pic .id {
  left: 4px;
}

.pic .score {
  right: 4px;
  color: var(--text-secondary);
}

.text {
  padding: var(--space-1);
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 1;
}

.action {
  font-size: 11px;
  font-weight: var(--weight-light);
  line-height: 1.4;
  color: var(--text-secondary);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.review {
  font-size: 11px;
  line-height: 1.4;
  color: var(--status-degraded);
}

.meta {
  margin-top: auto;
  display: flex;
  justify-content: space-between;
  font-size: 10px;
  color: var(--text-tertiary);
}

.acts {
  padding: 4px var(--space-1);
  border-top: 1px solid var(--border);
  display: flex;
  gap: 12px;
}

.acts .link {
  font-size: 10px;
}

.push {
  margin-left: auto;
}

.issues {
  margin: 4px 0 0;
  padding-left: 16px;
  font-size: var(--text-xs);
  color: var(--text-secondary);
}

.warn {
  color: var(--status-degraded);
}

.acts {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}
</style>
