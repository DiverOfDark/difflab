<script setup lang="ts">
import { computed, nextTick, ref } from 'vue'
import { deleteShot, patchShot, type Production, type ShotInfo } from '../../api'
import { useRowDrafts } from '../../draft'
import { castOf, phaseOf, shotId, type Phase, type StageId } from '../../production'

const props = defineProps<{
  production: Production
  chapter: number
  shots: ShotInfo[]
  busy: string
  act: (name: string, what: () => Promise<unknown>) => Promise<void>
}>()
const emit = defineEmits<{ go: [to: StageId, shot?: number | null] }>()

const phases = (s: ShotInfo) => [s.keyframe, s.prompt, s.clip].map((m) => phaseOf(m.state))
const filter = ref<'all' | Phase>('all')
const filters = computed(() => {
  const of = (p: Phase) => props.shots.filter((s) => phases(s).includes(p)).length
  return [
    { id: 'all' as const, label: 'all', count: props.shots.length },
    { id: 'review' as const, label: 'needs review', count: of('review') },
    { id: 'stale' as const, label: 'stale', count: of('stale') },
    { id: 'pending' as const, label: 'not made', count: of('pending') },
  ]
})
const rows = computed(() => props.shots.filter((s) => filter.value === 'all' || phases(s).includes(filter.value)))

const PIPE = [
  { l: 'K', what: 'keyframe', stage: 'keyframes' },
  { l: 'P', what: 'prompt', stage: 'prompts' },
  { l: 'V', what: 'clip with audio', stage: 'render' },
] as const

const typed = useRowDrafts()
// A save that failed keeps what was typed; one that landed hands the field back to the server.
async function save(key: string, server: () => string, what: () => Promise<unknown>) {
  await props.act(key, what)
  await nextTick()
  if (typed.typed[key] === server()) typed.drop(key)
}
const saveAction = (s: ShotInfo, value: string) => {
  if (value.trim() === s.action) return typed.drop(`action ${s.id}`)
  save(`action ${s.id}`, () => props.shots.find((x) => x.id === s.id)?.action ?? '', () => patchShot(s.id, { action: value.trim() }))
}
const saveSeconds = (s: ShotInfo, value: string) => {
  const n = Number(value)
  if (!n || n === s.seconds) return typed.drop(`seconds ${s.id}`)
  save(`seconds ${s.id}`, () => String(props.shots.find((x) => x.id === s.id)?.seconds ?? ''), () => patchShot(s.id, { seconds: n }))
}
const LIVING = new Set(['person', 'animal'])
const kindOf = (slug: string) => props.production.members.find((m) => m.slug === slug)?.kind ?? ''
const planned = (s: ShotInfo) => Array.isArray(s.plan.beats) && (s.plan.beats as unknown[]).length > 0
const toggleCast = (s: ShotInfo, slug: string) => {
  const cast = castOf(s)
  const next = cast.includes(slug) ? cast.filter((c) => c !== slug) : [...cast, slug]
  props.act(`cast ${s.id}`, () => patchShot(s.id, { cast: next }))
}
const setCamera = (s: ShotInfo, field: 'camera_move' | 'camera_toward', value: string) =>
  save(`${field} ${s.id}`, () => String(props.shots.find((x) => x.id === s.id)?.plan[field] ?? ''), () =>
    patchShot(s.id, { [field]: value }),
  )

interface Line {
  who: string
  line: string
}
const lines = (s: ShotInfo) => (Array.isArray(s.plan.dialogue) ? (s.plan.dialogue as Line[]) : [])
const speaking = ref<number | null>(null)
const draftLines = ref('')
const speak = (s: ShotInfo) => {
  speaking.value = s.id
  draftLines.value = lines(s)
    .map((l) => `${l.who}: ${l.line}`)
    .join('\n')
}
/** One line per row, "who: words", as the dialogue column shows it. */
const saveLines = (s: ShotInfo) => {
  const given = draftLines.value
    .split('\n')
    .map((row) => row.match(/^\s*([a-z_]+)\s*:\s*(.+)$/))
    .filter((m): m is RegExpMatchArray => !!m)
  const kept = new Map(lines(s).map((l) => [`${l.who}: ${l.line}`, l]))
  const dialogue = given.map(([, who, line]) => kept.get(`${who}: ${line.trim()}`) ?? { who, line: line.trim() })
  speaking.value = null
  props.act(`dialogue ${s.id}`, () => patchShot(s.id, { dialogue }))
}
const saveClosing = (s: ShotInfo, value: string) => {
  if (value.trim() === String(s.plan.closing ?? '').trim()) return typed.drop(`closing ${s.id}`)
  save(`closing ${s.id}`, () => String(props.shots.find((x) => x.id === s.id)?.plan.closing ?? ''), () =>
    patchShot(s.id, { closing: value.trim() }),
  )
}
const unpin = (s: ShotInfo) => props.act(`unpin ${s.id}`, () => patchShot(s.id, { pinned: false }))

const remove = (s: ShotInfo) => {
  if (confirm(`Delete shot ${shotId(props.chapter, s)}? Its clip stays on disk.`)) props.act(`remove ${s.id}`, () => deleteShot(s.id))
}
</script>

<template>
  <div class="shots">
    <div class="bar-row">
      <div class="chips">
        <button v-for="f in filters" :key="f.id" class="chip" :class="{ on: filter === f.id }" @click="filter = f.id">
          {{ f.label }} <span class="n">{{ f.count }}</span>
        </button>
      </div>
      <span class="note">Edit the spec, not the output. A changed cast, camera or line recompiles the prompt at once and pins the shot.</span>
    </div>

    <div class="card table">
      <div class="tr th mono">
        <span>SHOT</span><span></span><span>ACTION</span><span>CAST</span><span>CAMERA</span><span>DIALOGUE</span><span class="r">SEC</span><span class="r">PIPELINE</span>
      </div>
      <div v-if="!rows.length" class="tr empty note">No shots. Plan the chapter, or add shots by hand through the API.</div>
      <div v-for="s in rows" :key="s.id" class="tr">
        <span class="id">
          <button class="link" @click="emit('go', 'keyframes', s.id)">{{ shotId(chapter, s) }}</button>
          <span class="mono dim">{{ s.scene }}</span>
          <button v-if="s.pinned" class="link hand mono" title="edited by hand: a re-plan leaves its plan alone · click to give it back" @click="unpin(s)">BY HAND</button>
        </span>
        <span class="chain mono" :title="s.chained ? 'opens on the previous shot\'s handoff frame' : 'opens on its own keyframe'">{{ s.chained ? '│' : '' }}</span>
        <span class="spec">
          <textarea
            class="action"
            rows="2"
            :value="typed.value(`action ${s.id}`, s.action)"
            :disabled="!!busy"
            @input="typed.input(`action ${s.id}`, $event)"
            @change="saveAction(s, ($event.target as HTMLTextAreaElement).value)"
          />
          <label v-if="planned(s)" class="closing">
            <span class="eyebrow">closes on · the next shot opens here</span>
            <textarea
              class="action"
              rows="3"
              :value="typed.value(`closing ${s.id}`, String(s.plan.closing ?? ''))"
              :disabled="!!busy"
              @input="typed.input(`closing ${s.id}`, $event)"
              @change="saveClosing(s, ($event.target as HTMLTextAreaElement).value)"
            />
          </label>
        </span>
        <span class="cast">
          <template v-if="planned(s)">
            <button
              v-for="m in production.members"
              :key="m.slug"
              class="mono"
              :class="{ on: castOf(s).includes(m.slug) }"
              :disabled="!!busy || m.kind === 'environment'"
              :title="castOf(s).includes(m.slug) ? `take ${m.slug} out of the shot` : `put ${m.slug} in the shot`"
              @click="toggleCast(s, m.slug)"
            >
              {{ m.slug }}
            </button>
          </template>
          <span v-for="c in castOf(s)" v-else :key="c" class="mono on">{{ c }}</span>
        </span>
        <span v-if="planned(s)" class="camera">
          <select
            class="mono"
            :value="typed.value(`camera_move ${s.id}`, String(s.plan.camera_move ?? ''))"
            :disabled="!!busy"
            @change="typed.input(`camera_move ${s.id}`, $event), setCamera(s, 'camera_move', ($event.target as HTMLSelectElement).value)"
          >
            <option v-for="c in production.camera_moves" :key="c" :value="c">{{ c }}</option>
          </select>
          <select
            class="mono"
            :value="typed.value(`camera_toward ${s.id}`, String(s.plan.camera_toward ?? ''))"
            :disabled="!!busy"
            @change="typed.input(`camera_toward ${s.id}`, $event), setCamera(s, 'camera_toward', ($event.target as HTMLSelectElement).value)"
          >
            <option v-for="c in castOf(s)" :key="c" :value="c">→ {{ c }}</option>
          </select>
        </span>
        <span v-else class="mono muted small">{{ s.camera }}</span>
        <span v-if="speaking === s.id" class="speak">
          <textarea v-model="draftLines" rows="3" class="lines mono" :placeholder="castOf(s).filter((c) => LIVING.has(kindOf(c))).map((c) => `${c}: …`).join('\n')" />
          <span class="row"><button class="link" @click="speaking = null">cancel</button><button class="link" :disabled="!!busy" @click="saveLines(s)">save</button></span>
        </span>
        <button v-else-if="planned(s)" class="dialogue link-like" :class="{ none: !s.dialogue }" :disabled="!!busy" title="edit the lines: one per row, who: words" @click="speak(s)">
          {{ s.dialogue || 'none · add a line' }}
        </button>
        <span v-else class="dialogue" :class="{ none: !s.dialogue }">{{ s.dialogue || 'none' }}</span>
        <input
          class="sec mono"
          :value="typed.value(`seconds ${s.id}`, String(s.seconds))"
          :disabled="!!busy"
          @input="typed.input(`seconds ${s.id}`, $event)"
          @change="saveSeconds(s, ($event.target as HTMLInputElement).value)" />
        <span class="pipe">
          <button
            v-for="(p, i) in PIPE"
            :key="p.l"
            :class="phases(s)[i]"
            :title="`${p.what} · ${[s.keyframe, s.prompt, s.clip][i].state}`"
            @click="emit('go', p.stage, s.id)"
          >
            {{ p.l }}
          </button>
          <button class="link del" title="delete the shot" :disabled="!!busy" @click="remove(s)">×</button>
        </span>
      </div>
    </div>

    <div class="legend mono">
      <span>PIPELINE · K keyframe · P prompt · V clip with audio</span>
      <span>■ made · ▢ not made · <span class="warn">■</span> needs review · <span class="fail">■</span> stale</span>
      <span>│ chained from the previous shot's handoff frame</span>
    </div>
  </div>
</template>

<style scoped>
.shots {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.bar-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--space-2);
  flex-wrap: wrap;
}

.table {
  overflow: auto;
}

.tr {
  display: grid;
  grid-template-columns: 64px 12px minmax(200px, 1.5fr) minmax(140px, 1fr) 124px minmax(160px, 1.2fr) 56px 104px;
  gap: 0 var(--space-2);
  padding: var(--space-1) var(--space-2);
  border-bottom: 1px solid var(--border);
  align-items: start;
  font-size: var(--text-xs);
  min-width: 980px;
}

.tr:hover:not(.th) {
  background: var(--bg-raised);
}

.th {
  border-bottom-color: var(--border-strong);
  font-size: 10px;
  letter-spacing: 0.08em;
  color: var(--text-tertiary);
}

.empty {
  display: block;
}

.r {
  text-align: right;
}

.id {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}

.id .link {
  font-size: var(--text-xs);
  color: var(--text-primary);
  text-align: left;
}

.id .dim {
  font-size: 10px;
  overflow: hidden;
  text-overflow: ellipsis;
}

.chain {
  color: var(--accent);
}

.card textarea.action {
  background: transparent;
  border: 1px solid transparent;
  padding: 0 4px;
  margin: 0 -4px;
  resize: none;
  font-family: var(--font-body);
  font-size: var(--text-xs);
  font-weight: var(--weight-light);
  line-height: 1.45;
}

.spec {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}

.closing {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.closing .eyebrow {
  font-size: 9px;
}

.closing textarea.action {
  color: var(--text-secondary);
}

.card textarea.action:hover {
  border-color: var(--border);
}

.cast {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}

.cast > * {
  font-size: 10px;
  padding: 1px 6px;
  border: 1px dashed var(--border);
  border-radius: 2px;
  white-space: nowrap;
  background: transparent;
  color: var(--text-tertiary);
  font-weight: var(--weight-regular);
}

.cast > .on {
  border: 1px solid var(--border-strong);
  color: var(--text-primary);
}

.cast > button:disabled {
  cursor: default;
}

.camera {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.camera select {
  background: transparent;
  border-color: transparent;
  padding: 0 2px;
  font-size: 11px;
  color: var(--text-secondary);
}

.camera select:hover {
  border-color: var(--border);
}

.hand {
  font-size: 9px;
  letter-spacing: 0.06em;
  color: var(--accent) !important;
  text-align: left;
}

.speak {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.card textarea.lines {
  font-size: 11px;
  padding: 4px;
}

.row {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
}

button.link-like {
  text-align: left;
  background: transparent;
  border: 1px solid transparent;
  padding: 0 4px;
  margin: 0 -4px;
  font-size: var(--text-xs);
  border-radius: 2px;
}

button.link-like:hover:not(:disabled) {
  border-color: var(--border);
  color: var(--text-primary);
}

.small {
  font-size: 11px;
}

.dialogue {
  font-weight: var(--weight-light);
  line-height: 1.45;
  color: var(--text-secondary);
}

.dialogue.none {
  font-style: italic;
  color: var(--text-tertiary);
}

.card input.sec {
  background: transparent;
  border-color: transparent;
  padding: 0 4px;
  font-size: 11px;
  text-align: right;
}

.card input.sec:hover {
  border-color: var(--border);
}

.pipe {
  display: flex;
  justify-content: flex-end;
  gap: 4px;
  padding-top: 2px;
}

.pipe > button:not(.link) {
  width: 16px;
  height: 16px;
  padding: 0;
  border-radius: 2px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-family: var(--font-mono);
  font-size: 8px;
  border: 1px solid var(--border-strong);
  color: var(--text-tertiary);
  background: transparent;
}

.pipe > .done {
  background: var(--text-secondary) !important;
  border-color: var(--text-secondary) !important;
  color: var(--bg-base) !important;
}

.pipe > .review {
  background: var(--status-degraded) !important;
  border-color: var(--status-degraded) !important;
  color: var(--accent-fg) !important;
}

.pipe > .stale {
  background: var(--status-error) !important;
  border-color: var(--status-error) !important;
  color: var(--accent-fg) !important;
}

.pipe > .running {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  animation: pulse 1.6s ease-in-out infinite;
}

.del {
  margin-left: 4px;
  font-size: var(--text-xs);
}

.legend {
  display: flex;
  gap: var(--space-3);
  font-size: 11px;
  color: var(--text-tertiary);
  flex-wrap: wrap;
}

.warn {
  color: var(--status-degraded);
}

.fail {
  color: var(--status-error);
}
</style>
