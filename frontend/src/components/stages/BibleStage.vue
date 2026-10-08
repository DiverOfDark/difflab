<script setup lang="ts">
import { computed, reactive, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { attachUnmatched, describeMember, drawReference, drawReferences, outputUrl, type MemberInfo, type Production, type ShotInfo } from '../../api'
import { KIND_LABEL, TABS, appearance, changesIn, drawnRef, meta, pictures, shotLabel, shotsOf, source, states } from '../../bible'
import { drawingRun, type StageId } from '../../production'
import { state } from '../../store'
import BibleEditor from './BibleEditor.vue'

const props = defineProps<{
  production: Production
  shots: ShotInfo[]
  /** The member id open in the editor, 'new' for one decided in words, or null for the grid. */
  editing: string | null
  busy: string
  act: (name: string, what: () => Promise<unknown>) => Promise<void>
}>()
const emit = defineEmits<{ edit: [id: number | 'new' | null]; go: [to: StageId, shot?: number | null, member?: string | null] }>()

const tab = ref('all')
const tabs = computed(() =>
  TABS.map((t) => ({ ...t, count: props.production.members.filter((m) => !t.kinds.length || t.kinds.includes(m.kind)).length })),
)
const shown = computed(() => {
  const kinds = TABS.find((t) => t.id === tab.value)?.kinds ?? []
  return props.production.members.filter((m) => !kinds.length || kinds.includes(m.kind))
})
const editingMember = computed(() => props.production.members.find((m) => String(m.id) === props.editing) ?? null)

const novel = (m: MemberInfo) => m.conflicts.filter((c) => c.shows && !c.resolution)
const disagree = (m: MemberInfo) => m.conflicts.filter((c) => !c.shows && !c.resolution)
const rowsOf = (m: MemberInfo) => states(props.production, m)

// --- drawing references ---------------------------------------------------------------------------
const drawing = computed(() => drawingRun(props.production))
const drawingNow = computed(() => (drawing.value ? state.runs.find((r) => r.id === drawing.value!.id) : undefined))
const missingRefs = computed(() => props.production.missing_references ?? [])
const inDrawing = (m: MemberInfo) => {
  const r = drawnRef(m)
  return !!drawing.value && !!r && (r.state === 'queued' || r.state === 'running')
}
/** A photographed member is read again from its photographs; one drawn from words is drawn anew. */
const action = (m: MemberInfo) => (source(m).provided ? 'describe again' : drawnRef(m)?.artifact ? 'regenerate' : 'draw from words')
const run = (m: MemberInfo) =>
  source(m).provided
    ? props.act(`describe ${m.id}`, () => describeMember(m.id))
    : props.act(`draw ${m.id}`, () => drawReference(m.id))
const drawMissing = () => props.act('draw references', () => drawReferences(props.production.id))

// --- unmatched photographs --------------------------------------------------------------------------
const picks = reactive<Record<string, string>>({})
const pickOf = (u: Production['unmatched'][number]) => picks[u.photo] ?? (u.likely || '')
const name = (path: string) => path.split('/').at(-1)!.replace(/^[0-9a-f]{16}-/, '')
const attach = (u: Production['unmatched'][number]) =>
  props.act(`attach ${u.photo}`, () => attachUnmatched(props.production.id, u.photo, pickOf(u)))
</script>

<template>
  <BibleEditor
    v-if="editing"
    :production="production"
    :member="editingMember"
    :busy="busy"
    :act="act"
    @close="emit('edit', null)"
    @go="(to, shot, member) => emit('go', to, shot, member)"
  />

  <div v-else class="bible">
    <div class="bar-row">
      <div class="chips">
        <button v-for="t in tabs" :key="t.id" class="chip" :class="{ on: tab === t.id }" @click="tab = t.id">
          {{ t.id }} <span class="n">{{ t.count }}</span>
        </button>
      </div>
      <span class="acts">
        <button class="link" :disabled="!production.members.length" @click="emit('edit', 'new')">+ add a member in words</button>
        <button
          :disabled="!!busy || !!drawing || !missingRefs.length"
          :title="missingRefs.length ? `no usable picture: ${missingRefs.join(', ')}` : 'every member has a picture'"
          @click="drawMissing"
        >
          {{
            busy === 'draw references'
              ? 'queueing…'
              : drawing
                ? 'Drawing references…'
                : missingRefs.length
                  ? `Generate ${missingRefs.length} missing reference${missingRefs.length === 1 ? '' : 's'}`
                  : 'No reference missing'
          }}
        </button>
      </span>
    </div>

    <p v-if="drawing" class="band drawing mono">
      <i class="dot running" />
      <span>Drawing references · {{ drawingNow ? `${drawingNow.jobs_done} / ${drawingNow.jobs_total} jobs · ${drawingNow.status}` : drawing.state }}</span>
      <RouterLink :to="`/runs/${drawing.id}`">watch run →</RouterLink>
    </p>
    <p v-if="!production.members.length" class="band hint">No cast yet. Plan the chapter: the survey names who the story needs from the photographs.</p>

    <div class="cards">
      <article v-for="m in shown" :key="m.id" class="card" :class="{ conflicted: novel(m).length || disagree(m).length }">
        <div class="top">
          <span class="named"><span class="mono slug">{{ m.slug }}</span><span class="eyebrow">{{ KIND_LABEL[m.kind] ?? m.kind }}</span></span>
          <span class="source mono" :class="{ provided: source(m).provided }">{{ source(m).label.toUpperCase() }}</span>
        </div>
        <div class="inside">
          <div class="pics">
            <span v-if="pictures(m)[0]" class="leadpic">
              <img class="frame lead" :src="outputUrl(pictures(m)[0].artifact!)" :alt="pictures(m)[0].role" />
              <span class="tag mono" :class="{ gen: pictures(m)[0].role === 'generated' }">{{ pictures(m)[0].role === 'generated' ? 'GEN' : 'PHOTO' }}</span>
            </span>
            <div v-else class="frame lead empty mono">{{ inDrawing(m) ? 'drawing…' : 'no reference' }}</div>
            <div class="thumbs">
              <img v-for="r in pictures(m).slice(1, 4)" :key="r.id" class="frame" :src="outputUrl(r.artifact!)" :alt="r.role" :title="r.role" />
            </div>
          </div>
          <div class="words">
            <p class="desc">{{ appearance(m) || m.find || m.noun }}</p>
            <div class="meta">
              <span class="eyebrow small">{{ meta(production, m).k }}</span><span>{{ meta(production, m).v }}</span>
            </div>
            <div v-for="c in novel(m)" :key="c.id" class="conflict">
              <span class="eyebrow small warn">REFERENCE OVERRIDES NOVEL</span>
              <span>Novel: “{{ c.says }}”. Photo shows {{ c.shows }}; {{ c.lines }} line{{ c.lines === 1 ? '' : 's' }} to rewrite.</span>
            </div>
            <div v-for="c in disagree(m)" :key="c.id" class="conflict">
              <span class="eyebrow small warn">THE PHOTOGRAPHS DISAGREE</span>
              <span>{{ c.says }}</span>
            </div>
            <div v-for="g in m.gaps ?? []" :key="g.role" class="conflict">
              <span class="eyebrow small warn">NO {{ g.role.toUpperCase() }} PICTURE</span>
              <span>{{ g.why }}</span>
            </div>
            <div v-if="rowsOf(m).length" class="changes">
              <span class="eyebrow small">{{ changesIn(rowsOf(m)) }}</span>
              <span class="list">
                <span class="state mono base">base</span>
                <span
                  v-for="r in rowsOf(m)"
                  :key="r.id ?? 'base'"
                  class="state mono"
                  :class="{ flag: r.image === 'flag' || r.image === 'failed', pending: r.image === 'pending' || r.image === 'running' }"
                  :title="r.amend"
                  >{{ r.shot ? shotLabel(r.shot).split('.')[1] : '—' }} · {{ r.label }}</span
                >
              </span>
            </div>
          </div>
        </div>
        <div class="bottom mono">
          <span class="dim">in {{ shotsOf(production, m.slug).length }} shots</span>
          <span class="acts">
            <button v-if="m.kind !== 'style'" class="link" :disabled="!!busy || (!source(m).provided && !!drawing)" @click="run(m)">
              {{ inDrawing(m) ? 'drawing…' : action(m) }}
            </button>
            <button class="link" @click="emit('edit', m.id)">edit</button>
          </span>
        </div>
      </article>
    </div>

    <section v-if="production.unmatched?.length" class="card unmatched">
      <div class="head slim"><h5>Unmatched references</h5><span class="note">uploads that matched no entity · which entity is this?</span></div>
      <div v-for="u in production.unmatched" :key="u.photo" class="row">
        <img class="frame" :src="outputUrl(u.photo)" alt="" />
        <span class="what">
          <span class="mono file">{{ name(u.photo) }}</span>
          <span class="guess">Qwen-VL: {{ u.about || 'no description' }}{{ u.likely ? ` · likely ${u.likely}` : ' · no matching member in the bible' }}</span>
        </span>
        <select :value="pickOf(u)" class="mono" @change="picks[u.photo] = ($event.target as HTMLSelectElement).value">
          <option value="" disabled>which entity…</option>
          <option v-for="m in production.members" :key="m.id" :value="m.slug">{{ m.slug }}</option>
          <option value="ignore">ignore</option>
        </select>
        <button :disabled="!!busy || !pickOf(u)" @click="attach(u)">{{ busy === `attach ${u.photo}` ? 'attaching…' : 'Attach' }}</button>
      </div>
    </section>
  </div>
</template>

<style scoped>
.bible {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.bar-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--space-2);
  flex-wrap: wrap;
}

.acts {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

.band.drawing {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  background: var(--bg-surface);
  border: 1px solid var(--border);
  color: var(--text-secondary);
  font-size: 11px;
}

.band.drawing a {
  margin-left: auto;
  color: var(--accent);
  text-decoration: none;
}

.cards {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 340px), 1fr));
  gap: var(--space-2);
}

.cards > .card {
  display: flex;
  flex-direction: column;
}

.card.conflicted {
  border-color: var(--status-degraded);
}

.top,
.bottom {
  padding: var(--space-1) var(--space-2);
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--space-1);
}

.top {
  border-bottom: 1px solid var(--border);
}

.bottom {
  border-top: 1px solid var(--border);
  font-size: 11px;
  margin-top: auto;
}

.bottom .acts {
  gap: var(--space-2);
}

.named {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  min-width: 0;
}

.slug {
  font-size: 13px;
}

.named .eyebrow,
.eyebrow.small {
  font-size: 10px;
}

.source {
  font-size: 10px;
  letter-spacing: 0.06em;
  padding: 2px 6px;
  border-radius: 2px;
  border: 1px solid var(--border-strong);
  color: var(--text-secondary);
  white-space: nowrap;
}

.source.provided {
  border-color: var(--accent);
  color: var(--accent);
}

.inside {
  padding: var(--space-2);
  display: grid;
  grid-template-columns: 96px minmax(0, 1fr);
  gap: var(--space-2);
}

.pics {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.leadpic {
  position: relative;
  display: block;
}

.lead {
  aspect-ratio: 3 / 4;
}

.leadpic .lead {
  display: block;
  width: 100%;
}

.tag {
  position: absolute;
  top: 4px;
  left: 4px;
  font-size: 9px;
  letter-spacing: 0.06em;
  padding: 1px 4px;
  border-radius: 2px;
  background: var(--accent);
  color: var(--accent-fg);
}

.tag.gen {
  background: var(--bg-base);
  color: var(--text-secondary);
}

.empty {
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  color: var(--text-tertiary);
  text-align: center;
  padding: 4px;
}

.thumbs {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 4px;
}

.thumbs img {
  aspect-ratio: 1;
}

.words {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  min-width: 0;
}

.desc {
  margin: 0;
  font-size: var(--text-xs);
  line-height: 1.5;
  font-weight: var(--weight-light);
  color: var(--text-secondary);
}

.meta,
.changes {
  display: flex;
  flex-direction: column;
  gap: 2px;
  font-size: var(--text-xs);
  line-height: 1.45;
}

.changes {
  gap: 4px;
}

.conflict {
  border: 1px solid var(--status-degraded);
  border-radius: 2px;
  padding: var(--space-1);
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: var(--text-xs);
  font-weight: var(--weight-light);
  line-height: 1.45;
}

.warn {
  color: var(--status-degraded);
}

.list {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}

.state {
  font-size: 10px;
  padding: 1px 6px;
  border: 1px solid var(--border-strong);
  border-radius: 2px;
  white-space: nowrap;
}

.state.base {
  color: var(--text-secondary);
}

.state.flag {
  border-color: var(--status-degraded);
  color: var(--status-degraded);
}

.state.pending {
  color: var(--text-tertiary);
}

.unmatched .head {
  flex-wrap: wrap;
  align-items: center;
}

.unmatched .row {
  padding: var(--space-1) var(--space-2);
  border-bottom: 1px solid var(--border);
  display: grid;
  grid-template-columns: 40px minmax(0, 1fr) minmax(140px, 200px) auto;
  gap: var(--space-2);
  align-items: center;
}

.unmatched .row:last-child {
  border-bottom: 0;
}

.unmatched img {
  width: 40px;
  height: 40px;
}

.what {
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.file {
  font-size: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.guess {
  font-size: 11px;
  font-weight: var(--weight-light);
  color: var(--text-secondary);
}

.unmatched select {
  font-size: 12px;
  min-width: 0;
}

.unmatched button {
  font-size: 12px;
  padding: 6px 16px;
}
</style>
