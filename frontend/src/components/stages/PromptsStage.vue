<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useDraft } from '../../draft'
import { outputUrl, patchShot, type ShotInfo } from '../../api'
import { FINISHED, phaseOf, shotId, words, type StageId } from '../../production'

const props = defineProps<{
  chapter: number
  shots: ShotInfo[]
  focus: number | null
  busy: string
  act: (name: string, what: () => Promise<unknown>) => Promise<void>
}>()
const emit = defineEmits<{ go: [to: StageId, shot?: number | null] }>()

type Filter = 'all' | 'findings' | 'pinned' | 'missing'
const filter = ref<Filter>('all')
const matches = (s: ShotInfo, f: Filter) =>
  f === 'all' ||
  (f === 'findings' && s.prompt.checks.length > 0) ||
  (f === 'pinned' && s.prompt.pinned) ||
  (f === 'missing' && !FINISHED.has(s.prompt.state))
const filters = computed(() =>
  (['all', 'findings', 'pinned', 'missing'] as Filter[]).map((id) => ({ id, count: props.shots.filter((s) => matches(s, id)).length })),
)
const rows = computed(() => props.shots.filter((s) => matches(s, filter.value)))

const index = computed(() => Math.max(0, props.shots.findIndex((s) => s.id === props.focus)))
const shot = computed(() => props.shots[index.value] ?? null)
const { draft, reset } = useDraft(() => shot.value?.prompt.text ?? '')
watch(() => shot.value?.id, reset)
const edited = computed(() => !!shot.value && draft.value !== shot.value.prompt.text)
const step = (by: number) => {
  const n = props.shots.length
  emit('go', 'prompts', props.shots[(index.value + by + n) % n].id)
}

const label = (s: ShotInfo) => {
  if (s.prompt.pinned) return ['PINNED', 'done']
  if (s.prompt.checks.length) return ['FINDINGS', 'review']
  if (FINISHED.has(s.prompt.state)) return ['WRITTEN', 'done']
  return [s.prompt.state.toUpperCase(), phaseOf(s.prompt.state)]
}

/** The compiler writes `section:` lines; a count per section shows where the words went. */
const sections = computed(() => {
  const out: { k: string; n: number }[] = []
  for (const line of draft.value.split('\n')) {
    const head = /^([a-z_]+):\s*$/.exec(line.trim())
    if (head) out.push({ k: head[1], n: 0 })
    else if (out.length) out[out.length - 1].n += words(line)
  }
  return out
})

const inputs = computed(() => {
  const plan = shot.value?.plan ?? {}
  const text = (v: unknown) => (Array.isArray(v) ? v.join(', ') : typeof v === 'string' ? v : v == null ? '' : JSON.stringify(v))
  return [
    ['PURPOSE', plan.purpose],
    ['CAST', plan.cast],
    ['CAMERA', [plan.camera_move, plan.camera_toward].filter(Boolean).join(' → ')],
    ['DIALOGUE', plan.dialogue],
    ['SOUNDS', plan.sounds],
    ['CLOSING', plan.closing],
  ]
    .map(([k, v]) => ({ k: k as string, v: text(v) }))
    .filter((r) => r.v)
})

const save = () => props.act(`prompt ${shot.value!.id}`, () => patchShot(shot.value!.id, { prompt: draft.value }))
</script>

<template>
  <p v-if="!shots.length" class="band hint">No shots yet: the plan run writes a prompt for each.</p>
  <div v-else class="prompts">
    <section class="card list">
      <div class="chips pad">
        <button v-for="f in filters" :key="f.id" class="chip small" :class="{ on: filter === f.id }" @click="filter = f.id">
          {{ f.id }} <span class="n">{{ f.count }}</span>
        </button>
      </div>
      <div class="rows">
        <button v-for="s in rows" :key="s.id" class="row" :class="{ on: s.id === shot?.id }" @click="emit('go', 'prompts', s.id)">
          <img v-if="s.keyframe.artifact && s.keyframe.present" class="frame thumb" :src="outputUrl(s.keyframe.artifact)" alt="" />
          <span v-else class="frame thumb" />
          <span class="what"><span class="mono">{{ shotId(chapter, s) }}</span><span class="sum">{{ s.action }}</span></span>
          <span class="ph" :class="label(s)[1]">{{ label(s)[0] }}</span>
        </button>
      </div>
    </section>

    <div v-if="shot" class="main">
      <section class="card">
        <div class="head slim">
          <span class="mono small">{{ shotId(chapter, shot) }} <span class="dim">· {{ shot.camera || 'camera free' }} · {{ shot.seconds }} s</span></span>
          <span class="mono dim small">{{ words(draft) }} words · by {{ shot.prompt.written_by || '—' }}</span>
        </div>
        <div class="body">
          <span class="row-between">
            <span class="eyebrow">h3 prompt · used verbatim</span>
            <span class="note" :class="{ warn: edited }">{{ edited ? 'edited · saving pins it' : shot.prompt.pinned ? 'pinned · a re-plan leaves it alone' : 'as compiled' }}</span>
          </span>
          <textarea v-model="draft" class="text" rows="16" spellcheck="false" />
          <div class="segs">
            <span v-for="sg in sections" :key="sg.k" class="mono">{{ sg.k }} <span class="dim">{{ sg.n }}</span></span>
          </div>
          <div class="opening">
            <img v-if="shot.keyframe.artifact && shot.keyframe.present" class="frame" :src="outputUrl(shot.keyframe.artifact)" alt="" />
            <span v-else class="frame" />
            <span class="what">
              <span class="eyebrow">keyframe · opening</span>
              <span class="note">The prompt says what happens from this frame; it must not re-describe what the frame already shows.</span>
            </span>
          </div>
        </div>
      </section>

      <div class="pair">
        <section class="card">
          <div class="head slim"><h5>Written from</h5><span class="mono dim small">the plan · change these, not the prose</span></div>
          <div v-for="i in inputs" :key="i.k" class="kv">
            <span class="eyebrow">{{ i.k }}</span>
            <span class="v">{{ i.v }}</span>
          </div>
          <div v-if="!inputs.length" class="kv"><span class="note">Written by hand: no plan behind it.</span></div>
        </section>
        <section class="card" :class="{ flagged: shot.prompt.checks.length }">
          <div class="head slim">
            <h5>Checks</h5>
            <span class="mono small" :class="shot.prompt.checks.length ? 'warn' : 'dim'">
              {{ shot.prompt.checks.length ? `${shot.prompt.checks.length} findings` : 'all passed' }}
            </span>
          </div>
          <div v-for="c in shot.prompt.checks" :key="c.code + c.message" class="check">
            <i class="dot" :class="c.severity === 'error' ? 'stale' : 'review'" />
            <span class="what"><span>{{ c.code }}</span><span class="why">{{ c.message }}</span></span>
            <span class="mono small" :class="c.severity === 'error' ? 'fail' : 'warn'">{{ c.severity }}</span>
          </div>
          <div class="pictures">
            <span class="eyebrow">pictures handed · {{ shot.prompt.pictures.length }}</span>
            <div class="thumbs">
              <a v-for="(p, i) in shot.prompt.pictures" :key="p" :href="outputUrl(p)" target="_blank" :title="`Picture ${i + 1}`">
                <img class="frame" :src="outputUrl(p)" alt="" />
              </a>
            </div>
          </div>
        </section>
      </div>

      <footer class="foot">
        <span class="note">Saving pins this prompt: the clip goes stale and re-renders on the next chapter run; a re-plan leaves the prompt alone.</span>
        <div class="buttons">
          <button @click="step(-1)">‹</button>
          <button @click="step(1)">›</button>
          <button class="primary" :disabled="!!busy || !edited" @click="save">Save and pin</button>
        </div>
      </footer>
    </div>
  </div>
</template>

<style scoped>
.prompts {
  display: grid;
  grid-template-columns: minmax(240px, 300px) minmax(0, 1fr);
  gap: var(--space-3);
  align-items: start;
}

.list {
  display: flex;
  flex-direction: column;
  max-height: calc(100vh - 220px);
  position: sticky;
  top: 80px;
}

.pad {
  padding: var(--space-1) var(--space-2);
  border-bottom: 1px solid var(--border);
}

.chip.small {
  padding: 4px 8px;
  font-size: 11px;
}

.rows {
  overflow: auto;
  display: flex;
  flex-direction: column;
}

.row {
  display: grid;
  grid-template-columns: 40px minmax(0, 1fr) auto;
  gap: var(--space-1);
  align-items: center;
  text-align: left;
  border: 0;
  border-radius: 0;
  border-bottom: 1px solid var(--border);
  border-left: 2px solid transparent;
  padding: var(--space-1) var(--space-2) var(--space-1) 14px;
  font-weight: var(--weight-regular);
}

.row:hover {
  background: var(--bg-raised);
  color: inherit;
}

.row.on {
  background: var(--bg-raised);
  border-left-color: var(--accent);
}

.thumb {
  width: 40px;
  aspect-ratio: 1;
}

.what {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}

.what .mono {
  font-size: var(--text-xs);
}

.sum {
  font-size: 11px;
  font-weight: var(--weight-light);
  color: var(--text-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.main {
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

.row-between {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: var(--space-2);
  flex-wrap: wrap;
}

.card textarea.text {
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  line-height: 1.6;
  padding: var(--space-1) var(--space-2);
}

.segs {
  display: flex;
  gap: var(--space-1);
  flex-wrap: wrap;
}

.segs span {
  font-size: 10px;
  letter-spacing: 0.06em;
  padding: 2px 6px;
  border-radius: 2px;
  border: 1px solid var(--border);
  color: var(--text-secondary);
}

.opening {
  display: grid;
  grid-template-columns: 120px minmax(0, 1fr);
  gap: var(--space-2);
  align-items: start;
}

.opening .frame {
  aspect-ratio: 3 / 4;
}

.pair {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr));
  gap: var(--space-2);
  align-items: start;
}

.flagged {
  border-color: var(--status-degraded);
}

.kv {
  display: grid;
  grid-template-columns: 88px minmax(0, 1fr);
  gap: var(--space-2);
  padding: var(--space-1) var(--space-2);
  border-bottom: 1px solid var(--border);
  font-size: var(--text-xs);
}

.kv .eyebrow {
  font-size: 10px;
  padding-top: 2px;
}

.v {
  line-height: 1.45;
  font-weight: var(--weight-light);
  color: var(--text-secondary);
}

.check {
  display: grid;
  grid-template-columns: 16px minmax(0, 1fr) auto;
  gap: var(--space-2);
  padding: var(--space-1) var(--space-2);
  border-bottom: 1px solid var(--border);
  font-size: var(--text-xs);
}

.check .dot {
  margin-top: 5px;
  justify-self: center;
}

.why {
  font-weight: var(--weight-light);
  color: var(--text-secondary);
  line-height: 1.45;
}

.pictures {
  padding: var(--space-1) var(--space-2) var(--space-2);
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.pictures .eyebrow {
  font-size: 10px;
}

.thumbs {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(56px, 1fr));
  gap: 4px;
}

.thumbs img {
  aspect-ratio: 1;
}

.warn {
  color: var(--status-degraded);
}

.fail {
  color: var(--status-error);
}

@media (max-width: 900px) {
  .prompts {
    grid-template-columns: 1fr;
  }

  .list {
    position: static;
    max-height: 320px;
  }
}
</style>
