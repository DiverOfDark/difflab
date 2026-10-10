<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import {
  addCasting,
  approveMade,
  approveShot,
  deleteCasting,
  drawLook,
  drawShotLooks,
  outputUrl,
  patchCasting,
  patchShot,
  redoKeyframe,
  rewriteShot,
  upload,
  uploadLook,
  type CastingInfo,
  type LookRef,
  type MemberInfo,
  type Production,
  type ShotInfo,
} from '../../api'
import { useRowDrafts } from '../../draft'
import { shotId } from '../../production'

export type Tab = 'refs' | 'shot' | 'prompt' | 'changes'

const props = defineProps<{
  production: Production
  shot: ShotInfo
  /** Every shot of the production in film order: a look's source is one from before this shot. */
  film: ShotInfo[]
  /** Shot ids the chapter filter keeps. */
  inView: Set<number>
  focus: string | null
  tab: Tab
  busy: string
  act: (name: string, what: () => Promise<unknown>) => Promise<void>
}>()
const emit = defineEmits<{ tab: [t: Tab]; select: [shot: number, member: string | null]; next: []; said: [text: string] }>()

const KIND: Record<string, string> = { person: 'CHARACTER', animal: 'CHARACTER', environment: 'LOCATION', object: 'PROP', style: 'STYLE' }
const kindOf = (kind: string) => KIND[kind] ?? kind.toUpperCase()
const byId = computed(() => new Map(props.film.map((s, i) => [s.id, { shot: s, at: i }])))
const many = computed(() => props.production.chapters.length > 1)
/** A shot as the inspector names it: its number in the chapter, or chapter.number across several. */
const named = (id: number | null) => {
  const s = id === null ? undefined : byId.value.get(id)?.shot
  return s ? (many.value ? shotId(s) : String(s.position + 1).padStart(3, '0')) : '—'
}
const t = (tool: string) => props.production.timings?.[tool]?.seconds ?? 0
const drawSeconds = computed(() => Math.round(t('draw_look') + t('judge')))
const rewriteSeconds = computed(() => Math.max(1, Math.round(t('write_shot') + t('compile_shot'))))
const composeSeconds = computed(() => Math.round(t('compose_keyframe') + t('judge')))

function verdict(review: string): { score: number | null; issues: string[] } | null {
  if (!review) return null
  try {
    const v = JSON.parse(review)
    return { score: typeof v.score === 'number' ? v.score : null, issues: v.issues ?? [] }
  } catch {
    return null
  }
}
const score = (review: string) => {
  const s = verdict(review)?.score
  return s == null ? '' : ` ${s.toFixed(2)}`
}
/** `state` is the look's made state: drawn and not judged yet reads `made`, which the image calls pending. */
function statusOf(look: LookRef, review: string, state: string): string {
  if (look.image === 'done') return review ? `QA${score(review)}` : 'APPROVED'
  if (look.image === 'flag') return `REVIEW${score(review)}`
  if (look.image === 'running') return 'GENERATING'
  if (look.image === 'failed') return 'FAILED'
  return state === 'made' ? 'JUDGING' : 'TO GENERATE'
}
const STATUS_COLOR: Record<LookRef['image'], string> = {
  done: 'var(--text-secondary)',
  flag: 'var(--status-degraded)',
  running: 'var(--status-building)',
  failed: 'var(--status-error)',
  pending: 'var(--text-tertiary)',
}

// --- references ---------------------------------------------------------------------------------
const members = computed(() => new Map(props.production.members.map((m) => [m.slug, m])))
const typed = useRowDrafts()
// Clicking an action blurs the field just typed in, so its save and the action leave together; a job takes
// the words the server has when it is queued, and a look drew the old ones whenever it arrived first.
const saving = new Map<string, Promise<unknown>>()
const saved = () => Promise.all(saving.values())
async function save(key: string, what: () => Promise<unknown>) {
  const done = props.act(key, what)
  saving.set(key, done)
  await done
  saving.delete(key)
  await nextTick()
  typed.drop(key)
}

const refs = computed(() =>
  props.shot.castings.map((c) => {
    const member = members.value.get(c.member)
    const here = byId.value.get(props.shot.id)?.at ?? 0
    const prior = (member?.looks ?? []).filter((l) => l.shot !== null && (byId.value.get(l.shot)?.at ?? Infinity) < here)
    const sources = [{ id: null as number | null, label: 'bible · base' }, ...prior.map((l) => ({ id: l.id, label: `${named(l.shot)} · ${l.label}` }))].map(
      (o) => ({ ...o, on: o.id === c.source.id, latest: o.id === c.latest.id }),
    )
    const look = c.amended ? c.look : null
    const carried = c.carried_into
    const carriedText = carried.length
      ? `Carried into ${carried.length > 3 ? `${carried.length} shots, to ${named(carried[carried.length - 1])}` : carried.map(named).join(', ')}.`
      : 'Not carried further.'
    const issues = verdict(c.review)?.issues ?? []
    let effect: string
    if (look) effect = `${look.image === 'flag' && issues.length ? `${issues.join('; ')}. ` : ''}Qwen-Image-Edit draws “${look.label}” from “${c.source.label}” plus this amendment. ${carriedText}`
    else if (c.back) effect = `Returns to “${c.source.label}” with the cut back to the opening scene.`
    else if (c.is_revert) effect = `Reverts to “${c.source.label}”; the latest was “${c.latest.label}”.`
    else effect = `As it stands: ${c.source.shot === null ? 'the bible base' : `“${c.source.label}” from ${named(c.source.shot)}`}.`
    return { c, kind: kindOf(c.kind), sources, look, effect, flagged: look?.image === 'flag', member }
  }),
)
const pickSource = (c: CastingInfo, id: number | null) => {
  if (id === c.source.id) return
  const parent = id === c.latest.id ? null : id === null ? 'base' : String(id)
  props.act(`casting ${c.id}`, () => patchCasting(c.id, { parent }))
}
const setAmend = (c: CastingInfo, value: string) => {
  if (value.trim() === c.amend) return typed.drop(`amend ${c.id}`)
  save(`amend ${c.id}`, () => patchCasting(c.id, { amend: value }))
}
const setLabel = (c: CastingInfo, value: string) => {
  if (value.trim() === c.label) return typed.drop(`label ${c.id}`)
  save(`label ${c.id}`, () => patchCasting(c.id, { label: value }))
}
const draw = async (c: CastingInfo) => {
  await saved()
  return props.act(`draw ${c.id}`, () => drawLook(c.id))
}
const give = (c: CastingInfo, event: Event) => {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  input.value = ''
  if (file)
    props.act(`upload ${c.id}`, async () => {
      const [asset] = await upload([file])
      await uploadLook(c.id, asset.path)
    })
}
const approveLook = (c: CastingInfo) => props.act(`approve ${c.id}`, () => approveMade('casting', c.id))
const remove = (c: CastingInfo) => props.act(`casting ${c.id}`, () => deleteCasting(c.id))
const addable = computed(() => {
  const here = new Set(props.shot.castings.map((c) => c.member))
  const place = props.shot.castings.some((c) => c.kind === 'environment')
  return props.production.members.filter((m) => !here.has(m.slug) && !(place && m.kind === 'environment'))
})
const add = (m: MemberInfo) => props.act(`add ${m.slug}`, () => addCasting(props.shot.id, { member: m.slug }))

// --- shot ---------------------------------------------------------------------------------------
const field = (key: string, server: string) => typed.value(`${key} ${props.shot.id}`, server)
const input = (key: string, event: Event) => typed.input(`${key} ${props.shot.id}`, event)
const planned = computed(() => Array.isArray(props.shot.plan.beats) && (props.shot.plan.beats as unknown[]).length > 0)
const cast = computed(() => (Array.isArray(props.shot.plan.cast) ? (props.shot.plan.cast as string[]) : []))
function setShot(key: string, value: string, server: string, body: () => Record<string, unknown>) {
  if (value.trim() === server.trim()) return typed.drop(`${key} ${props.shot.id}`)
  save(`${key} ${props.shot.id}`, () => patchShot(props.shot.id, body()))
}
/** "who: words / who: words", as the shot's dialogue reads; a line kept as it was keeps its timing and voice. */
const setDialogue = (value: string) => {
  const kept = new Map(((props.shot.plan.dialogue as { who: string; line: string }[] | undefined) ?? []).map((l) => [`${l.who}: ${l.line}`, l]))
  const dialogue = value
    .split(' / ')
    .map((row) => row.match(/^\s*([a-z_]+)\s*:\s*(.+)$/))
    .filter((m): m is RegExpMatchArray => !!m)
    .map(([, who, line]) => kept.get(`${who}: ${line.trim()}`) ?? { who, line: line.trim() })
  setShot('dialogue', value, props.shot.dialogue, () => ({ dialogue }))
}
const setSeconds = (value: string) => {
  const n = Number(value)
  if (!n) return typed.drop(`seconds ${props.shot.id}`)
  setShot('seconds', String(n), String(props.shot.seconds), () => ({ seconds: n }))
}
const camera = (key: 'camera_move' | 'camera_toward') => String(props.shot.plan[key] ?? '')
const setCamera = (key: 'camera_move' | 'camera_toward', value: string) => setShot(key, value, camera(key), () => ({ [key]: value }))

const OPENINGS = [
  { id: 'keyframe', label: 'KEYFRAME' },
  { id: 'chain', label: 'PREVIOUS LAST FRAME' },
  { id: 'refs', label: 'REFERENCES ONLY' },
] as const
const before = computed(() => {
  const at = byId.value.get(props.shot.id)?.at ?? 0
  const prev = props.film[at - 1]
  return prev && prev.chapter === props.shot.chapter ? prev : null
})
const setOpening = (id: ShotInfo['opening']) => {
  if (id !== props.shot.opening) props.act(`opening ${props.shot.id}`, () => patchShot(props.shot.id, { opening: id }))
}
const waiting = computed(() => props.shot.waiting.length)
const keyframeVerdict = computed(() => verdict(props.shot.keyframe.review))
const openingNote = computed(() => {
  const s = props.shot
  if (s.opening === 'chain')
    return `The sharpest of the last frames of ${named(before.value?.id ?? null)}, within the handoff window. References still steer identity; a change made here has to become visible from that frame on.`
  if (s.opening === 'refs') return 'H3 ref2va starts from the references alone. Composition is left to the model; use for inserts and establishing shots.'
  if (s.keyframe.source === 'provided') return 'A picture given by hand, used as it is.'
  if (waiting.value) return 'Qwen-Image composes the first frame from the references as they stand here. It waits until every changed state has an approved image.'
  const qa = keyframeVerdict.value?.score
  return `Composed by Qwen-Image from the references as they stand here${qa == null ? '' : ` · QA ${qa.toFixed(2)}`}${s.keyframe.reviewed_by === 'person' ? ` · ${s.keyframe.state === 'approved' ? 'approved' : 'rejected'} by you` : ''}.`
})
const note = ref('')
watch(
  () => props.shot.id,
  () => (note.value = (props.shot.keyframe.note ?? '').split('Fix:')[0].trim()),
  { immediate: true },
)
const takeIssues = () => (note.value = [note.value.trim(), ...(keyframeVerdict.value?.issues ?? [])].filter(Boolean).join('; '))
const redo = async () => {
  await saved()
  return props.act(`keyframe ${props.shot.id}`, () => redoKeyframe(props.shot.id, note.value))
}
const approveFrame = () => props.act(`approve ${props.shot.id}`, () => approveMade('keyframe', props.shot.keyframe.id))
const ownFrame = (event: Event) => {
  const el = event.target as HTMLInputElement
  const file = el.files?.[0]
  el.value = ''
  if (file)
    props.act(`keyframe ${props.shot.id}`, async () => {
      const [asset] = await upload([file])
      await patchShot(props.shot.id, { keyframe: asset.path })
    })
}
const setLanded = (on: boolean) => props.act(`keyframe ${props.shot.id}`, () => patchShot(props.shot.id, { landed: on }))

// --- prompt -------------------------------------------------------------------------------------
const promptKey = computed(() => `prompt ${props.shot.id}`)
const setPrompt = (value: string) => {
  if (value === props.shot.prompt.text) return typed.drop(promptKey.value)
  save(promptKey.value, () => patchShot(props.shot.id, { prompt: value }))
}
const MARKER_COLOR: Record<string, string> = {
  changed: 'var(--status-degraded)',
  pinned: 'var(--accent)',
  rewritten: 'var(--text-tertiary)',
  written: 'var(--text-tertiary)',
}
const rewrite = async () => {
  await saved()
  return props.act(`rewrite ${props.shot.id}`, () => rewriteShot(props.shot.id))
}

// --- all changes --------------------------------------------------------------------------------
const changes = computed(() =>
  props.production.members
    .flatMap((m) => (m.looks ?? []).map((l) => ({ m, l })))
    .filter(({ l }) => l.shot !== null && props.inView.has(l.shot))
    .sort((a, b) => (byId.value.get(a.l.shot!)?.at ?? 0) - (byId.value.get(b.l.shot!)?.at ?? 0))
    .map(({ m, l }) => ({
      key: String(l.id),
      shot: l.shot!,
      member: m.slug,
      from: l.source.label,
      to: l.label,
      amend: l.amend,
      status: statusOf(l, l.made.review, l.made.state),
      color: l.image === 'done' ? 'var(--text-secondary)' : STATUS_COLOR[l.image],
      on: l.shot === props.shot.id,
    })),
)

const tabs = computed(() => [
  { id: 'refs' as const, label: 'REFERENCES', n: String(props.shot.castings.length) },
  { id: 'shot' as const, label: 'SHOT', n: '' },
  { id: 'prompt' as const, label: 'PROMPT', n: '' },
  { id: 'changes' as const, label: 'ALL CHANGES', n: String(changes.value.length) },
])

// --- footer -------------------------------------------------------------------------------------
const own = computed(() => props.shot.castings.filter((c) => c.amended))
const toDraw = computed(() => own.value.filter((c) => c.look.image === 'pending' || c.look.image === 'failed').length)
const flagged = computed(() => own.value.filter((c) => c.look.image === 'flag').length)
const carriedInto = computed(() => new Set(own.value.flatMap((c) => c.carried_into)).size)
const impact = computed(() => {
  const parts = []
  if (toDraw.value) parts.push(`${toDraw.value} state image${toDraw.value > 1 ? 's' : ''} to draw, ~${drawSeconds.value} s each.`)
  if (flagged.value) parts.push(`${flagged.value} below the QA gate.`)
  parts.push(
    carriedInto.value
      ? `Changes here carry into ${carriedInto.value} later shot${carriedInto.value > 1 ? 's' : ''}; their openings and clips follow.`
      : 'Nothing changed here carries past this shot.',
  )
  return parts.join(' ')
})
const drawAll = () => props.act(`looks ${props.shot.id}`, () => drawShotLooks(props.shot.id))
const approve = () =>
  props.act(`approve shot ${props.shot.id}`, async () => {
    const { left } = await approveShot(props.shot.id)
    if (left.length) emit('said', `Shot ${named(props.shot.id)}: ${left.join('; ')}`)
    else emit('next')
  })
</script>

<template>
  <section class="inspector">
    <div class="topbar">
      <div class="seg">
        <button v-for="x in tabs" :key="x.id" :class="{ on: tab === x.id }" @click="emit('tab', x.id)">
          {{ x.label }} <span class="n">{{ x.n }}</span>
        </button>
      </div>
      <span class="meta">{{ +shot.seconds.toFixed(2) }} s · {{ shot.camera || 'camera free' }}</span>
    </div>

    <div class="body">
      <template v-if="tab === 'refs'">
        <div v-for="r in refs" :key="r.c.id" class="ref" :class="{ focus: focus === r.c.member }">
          <div class="ref-top">
            <span class="who">
              <span class="e">{{ r.c.member }}</span><span class="kind">{{ r.kind }}</span>
              <span v-if="r.c.origin" class="origin">{{ r.c.origin }}</span>
            </span>
            <button v-if="r.c.kind !== 'environment'" class="link" :disabled="!!busy" @click="remove(r.c)">remove</button>
          </div>
          <div class="ref-main">
            <div class="thumbs">
              <div class="thumb">
                <img v-if="r.c.source.artifact" class="pic" :src="outputUrl(r.c.source.artifact)" alt="" />
                <div v-else class="pic" />
                <span class="where">{{ r.c.source.shot === null ? 'bible' : `from ${named(r.c.source.shot)}` }}</span>
                <span class="lbl">{{ r.c.source.label }}</span>
              </div>
              <span class="arrow">→</span>
              <div class="thumb" :class="{ unchanged: !r.look }">
                <img v-if="r.look?.artifact && r.look.image !== 'pending'" class="pic child" :class="r.look.image" :src="outputUrl(r.look.artifact)" alt="" />
                <div v-else class="pic child" :class="r.look ? r.look.image : 'none'" />
                <span class="where status-line" :style="{ color: r.look ? STATUS_COLOR[r.look.image] : undefined }">{{ r.look ? statusOf(r.look, r.c.review, r.c.state) : '' }}</span>
                <span class="lbl">{{ r.look ? r.look.label : 'unchanged' }}</span>
              </div>
            </div>
            <div class="controls">
              <div class="sources">
                <button v-for="o in r.sources" :key="String(o.id)" :class="{ on: o.on }" :disabled="!!busy" @click="pickSource(r.c, o.id)">
                  {{ o.label }} <span v-if="o.latest" class="n">latest</span>
                </button>
              </div>
              <input
                class="amend"
                :value="typed.value(`amend ${r.c.id}`, r.c.amend)"
                placeholder="what is different in this shot · empty keeps it as it stands"
                :disabled="!!busy && busy !== `amend ${r.c.id}`"
                @input="typed.input(`amend ${r.c.id}`, $event)"
                @change="setAmend(r.c, ($event.target as HTMLInputElement).value)"
              />
              <div v-if="r.look" class="row">
                <input
                  class="state"
                  :value="typed.value(`label ${r.c.id}`, r.c.label)"
                  placeholder="state name"
                  @input="typed.input(`label ${r.c.id}`, $event)"
                  @change="setLabel(r.c, ($event.target as HTMLInputElement).value)"
                />
                <button class="link on" :disabled="!!busy || r.look.image === 'running'" @click="draw(r.c)">
                  {{ r.look.image === 'running' ? 'generating…' : `${r.look.image === 'done' || r.look.image === 'flag' ? 'regenerate' : 'generate'} · ~${drawSeconds} s` }}
                </button>
                <label class="link" :class="{ off: !!busy }">upload instead<input type="file" accept="image/*" hidden @change="give(r.c, $event)" /></label>
                <button v-if="r.flagged" class="link" :disabled="!!busy" @click="approveLook(r.c)">approve as it is</button>
              </div>
              <span class="effect" :class="{ warn: r.flagged }">{{ r.effect }}</span>
            </div>
          </div>
        </div>
        <div class="add">
          <span class="eyebrow">ADD FROM BIBLE</span>
          <button v-for="m in addable" :key="m.slug" :disabled="!!busy" @click="add(m)">+ {{ m.slug }}</button>
        </div>
      </template>

      <div v-else-if="tab === 'shot'" class="form">
        <label class="f">
          <span class="eyebrow">ACTION</span>
          <textarea
            rows="3"
            :value="field('action', shot.action)"
            @input="input('action', $event)"
            @change="setShot('action', ($event.target as HTMLTextAreaElement).value, shot.action, () => ({ action: ($event.target as HTMLTextAreaElement).value.trim() }))"
          />
        </label>
        <label class="f">
          <span class="eyebrow">DIALOGUE</span>
          <input
            :value="field('dialogue', shot.dialogue)"
            placeholder="none"
            :disabled="!planned"
            title="who: words, a line each, joined by ' / '"
            @input="input('dialogue', $event)"
            @change="setDialogue(($event.target as HTMLInputElement).value)"
          />
        </label>
        <div class="pair">
          <label class="f">
            <span class="eyebrow">CAMERA</span>
            <span v-if="planned" class="camera">
              <select class="mono" :value="field('camera_move', camera('camera_move'))" @change="input('camera_move', $event), setCamera('camera_move', ($event.target as HTMLSelectElement).value)">
                <option v-for="c in production.camera_moves" :key="c" :value="c">{{ c }}</option>
              </select>
              <select class="mono" :value="field('camera_toward', camera('camera_toward'))" @change="input('camera_toward', $event), setCamera('camera_toward', ($event.target as HTMLSelectElement).value)">
                <option v-for="c in cast" :key="c" :value="c">→ {{ c }}</option>
              </select>
            </span>
            <input v-else class="mono" :value="shot.camera" disabled />
          </label>
          <label class="f">
            <span class="eyebrow">SECONDS · 5–15</span>
            <input class="mono num" :value="field('seconds', String(+shot.seconds.toFixed(2)))" @input="input('seconds', $event)" @change="setSeconds(($event.target as HTMLInputElement).value)" />
          </label>
        </div>
        <label class="f">
          <span class="eyebrow">SOUND</span>
          <input
            :value="field('sound', shot.sound)"
            :disabled="!planned"
            @input="input('sound', $event)"
            @change="setShot('sound', ($event.target as HTMLInputElement).value, shot.sound, () => ({ sound: ($event.target as HTMLInputElement).value }))"
          />
        </label>
        <div class="f">
          <span class="eyebrow">OPENING</span>
          <div class="seg small">
            <button
              v-for="o in OPENINGS"
              :key="o.id"
              :class="{ on: shot.opening === o.id }"
              :disabled="!!busy || (o.id === 'chain' && !before)"
              @click="setOpening(o.id)"
            >
              {{ o.label }}
            </button>
          </div>
          <span class="effect" :class="{ warn: shot.opening === 'keyframe' && waiting }">{{ openingNote }}</span>
          <template v-if="shot.opening === 'keyframe'">
            <textarea v-model="note" rows="2" class="note-in" placeholder="a note for the editor when the frame is redone: the robe is the dark green one from his photo" />
            <div class="row">
              <button class="link on" :disabled="!!busy || !!waiting" @click="redo">{{ shot.keyframe.artifact ? 'redo the frame' : 'compose the frame' }} · ~{{ composeSeconds }} s</button>
              <button v-if="keyframeVerdict?.issues.length" class="link" @click="takeIssues">add the judge's issues</button>
              <button
                v-if="['made', 'needs_review'].includes(shot.keyframe.state) && shot.keyframe.artifact && shot.keyframe.reviewed_by !== 'person'"
                class="link"
                :disabled="!!busy"
                @click="approveFrame"
              >
                approve as it is
              </button>
              <label class="link" :class="{ off: !!busy }">use a picture of my own<input type="file" accept="image/*" hidden @change="ownFrame" /></label>
            </div>
            <label v-if="before" class="land">
              <input type="checkbox" :checked="shot.keyframe.landed" :disabled="!!busy" @change="setLanded(($event.target as HTMLInputElement).checked)" />
              {{ named(before.id) }} ends on this frame
            </label>
          </template>
        </div>
      </div>

      <div v-else-if="tab === 'prompt'" class="form tight">
        <span class="between">
          <span class="eyebrow">H3 PROMPT</span>
          <span v-if="shot.prompt.marker" class="marker" :style="{ color: MARKER_COLOR[shot.prompt.marker.kind] }">{{ shot.prompt.marker.label }}</span>
        </span>
        <textarea
          class="prompt"
          rows="10"
          spellcheck="false"
          :value="typed.value(promptKey, shot.prompt.text)"
          @input="typed.input(promptKey, $event)"
          @change="setPrompt(($event.target as HTMLTextAreaElement).value)"
        />
        <span v-for="c in shot.prompt.checks" :key="c.code + c.message" class="effect" :class="c.severity === 'error' ? 'fail' : 'warn'">{{ c.code }} · {{ c.message }}</span>
        <div class="between wrap">
          <span class="fine">Written from the shot and the states in References. Names never reach H3. An edit pins it.</span>
          <button class="link" :disabled="!!busy || !planned" @click="rewrite">rewrite · ~{{ rewriteSeconds }} s</button>
        </div>
      </div>

      <template v-else>
        <p v-if="!changes.length" class="empty">No shot changes anyone's look yet. Type what is different in a shot's References.</p>
        <button v-for="ch in changes" :key="ch.key" class="change" :class="{ on: ch.on }" @click="emit('select', ch.shot, ch.member)">
          <span class="mono">{{ named(ch.shot) }}</span>
          <span class="mono small">{{ ch.member }}</span>
          <span class="what">
            <span class="mono small dim2">{{ ch.from }} → <span class="to">{{ ch.to }}</span></span>
            <span class="amend-text">{{ ch.amend }}</span>
          </span>
          <span class="status" :style="{ color: ch.color }">{{ ch.status }}</span>
        </button>
      </template>
    </div>

    <div class="foot-bar">
      <span class="impact">{{ impact }}</span>
      <div class="buttons">
        <button v-if="toDraw" class="secondary" :disabled="!!busy" @click="drawAll">Draw {{ toDraw }} image{{ toDraw > 1 ? 's' : '' }}</button>
        <button class="primary" :disabled="!!busy" @click="approve">Approve · next</button>
      </div>
    </div>
  </section>
</template>

<style scoped>
.inspector {
  border: 1px solid var(--border);
  border-radius: var(--radius-card);
  background: var(--bg-surface);
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.topbar {
  padding: var(--space-1) var(--space-2);
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--space-2);
  flex-wrap: wrap;
}

.seg button {
  background: transparent;
  color: var(--text-secondary);
  border: 0;
  border-radius: 0;
  padding: 4px 12px;
  font-size: 10px;
  font-family: var(--font-mono);
  font-weight: var(--weight-regular);
  letter-spacing: 0.06em;
  white-space: nowrap;
}

.seg.small button {
  padding: 4px 8px;
}

.seg button.on {
  background: var(--bg-raised);
  color: var(--accent);
}

.seg button:disabled {
  color: var(--border-strong);
}

.seg .n {
  color: var(--text-tertiary);
}

.meta {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-tertiary);
  white-space: nowrap;
}

.body {
  max-height: 520px;
  overflow: auto;
}

.ref {
  padding: var(--space-2);
  border-bottom: 1px solid var(--border);
  border-left: 2px solid transparent;
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.ref.focus {
  border-left-color: var(--accent);
}

.ref-top,
.between {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--space-2);
}

.between.wrap {
  flex-wrap: wrap;
}

.who {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  flex-wrap: wrap;
}

.who .e {
  font-family: var(--font-mono);
  font-size: var(--text-sm);
}

.who .kind {
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.08em;
  color: var(--text-tertiary);
}

.origin {
  font-family: var(--font-mono);
  font-size: 10px;
  padding: 1px 6px;
  border: 1px solid var(--border);
  border-radius: var(--radius-input);
  color: var(--text-secondary);
  white-space: nowrap;
}

.ref-main {
  display: flex;
  gap: var(--space-2);
  flex-wrap: wrap;
  align-items: flex-start;
}

.thumbs {
  display: flex;
  align-items: flex-start;
  gap: var(--space-1);
  flex-shrink: 0;
}

.thumb {
  display: flex;
  flex-direction: column;
  gap: 4px;
  width: 56px;
}

.thumb.unchanged {
  opacity: 0.4;
}

.pic {
  aspect-ratio: 3 / 4;
  width: 56px;
  background: var(--bg-raised);
  border: 1px solid var(--border);
  border-radius: var(--radius-input);
  object-fit: cover;
  display: block;
}

.pic.child.done,
.pic.child.flag {
  background: oklch(22% 0.02 260);
  border-color: var(--accent);
}

.pic.child.flag {
  border-color: var(--status-degraded);
}

.pic.child.running {
  background: var(--bg-base);
  border-color: var(--status-building);
  animation: pulse 1.4s ease-in-out infinite;
}

.pic.child.pending {
  background: var(--bg-base);
  border: 1px dashed var(--accent);
}

.pic.child.failed {
  background: var(--bg-base);
  border-color: var(--status-error);
}

.where,
.lbl {
  font-family: var(--font-mono);
  font-size: 10px;
  line-height: 1.4;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.where {
  color: var(--text-tertiary);
}

.where.status-line {
  overflow: visible;
}

.arrow {
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  color: var(--text-tertiary);
  padding-top: 32px;
}

.controls {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  flex: 1 1 220px;
  min-width: 0;
}

.sources {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}

.sources button {
  background: transparent;
  border: 1px solid var(--border);
  border-radius: var(--radius-input);
  padding: 2px 8px;
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: var(--weight-regular);
  color: var(--text-secondary);
  white-space: nowrap;
}

.sources button.on {
  background: var(--bg-raised);
  border-color: var(--accent);
  color: var(--accent);
}

.sources .n {
  color: var(--text-tertiary);
}

input,
textarea,
select {
  background: var(--bg-raised);
  border: 1px solid var(--border);
  border-radius: var(--radius-input);
  padding: 6px 8px;
  font-size: var(--text-sm);
  font-weight: var(--weight-light);
  min-width: 0;
}

input.state {
  width: 144px;
  padding: 4px 8px;
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  font-weight: var(--weight-regular);
}

.row {
  display: flex;
  gap: var(--space-1) var(--space-2);
  align-items: center;
  flex-wrap: wrap;
}

.link.on {
  color: var(--accent);
}

label.link {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-secondary);
  white-space: nowrap;
  cursor: pointer;
}

label.link:hover {
  color: var(--accent);
}

label.link.off {
  pointer-events: none;
  opacity: 0.5;
}

.effect {
  font-size: var(--text-xs);
  font-weight: var(--weight-light);
  line-height: 1.45;
  color: var(--text-secondary);
}

.effect.warn {
  color: var(--status-degraded);
}

.effect.fail {
  color: var(--status-error);
}

.add {
  padding: var(--space-1) var(--space-2);
  display: flex;
  gap: var(--space-1);
  align-items: center;
  flex-wrap: wrap;
}

.add button {
  background: transparent;
  border: 1px dashed var(--border-strong);
  border-radius: var(--radius-input);
  padding: 2px 8px;
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: var(--weight-regular);
  color: var(--text-secondary);
  white-space: nowrap;
}

.form {
  padding: var(--space-2);
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.form.tight {
  gap: var(--space-1);
}

.f {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.f textarea {
  padding: 8px 16px;
  line-height: 1.5;
}

.pair {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: var(--space-2);
}

.mono,
.camera select {
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  font-weight: var(--weight-regular);
}

.camera {
  display: flex;
  gap: 4px;
}

.num {
  text-align: right;
}

.seg.small {
  align-self: flex-start;
}

.note-in {
  font-size: var(--text-xs);
}

.land {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-secondary);
}

.marker {
  font-family: var(--font-mono);
  font-size: 11px;
  white-space: nowrap;
}

textarea.prompt {
  width: 100%;
  padding: 8px 16px;
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  font-weight: var(--weight-regular);
  line-height: 1.6;
}

.fine {
  font-size: 11px;
  font-weight: var(--weight-light);
  color: var(--text-secondary);
}

.change {
  display: grid;
  grid-template-columns: 56px 88px minmax(0, 1fr) auto;
  gap: var(--space-2);
  padding: var(--space-1) var(--space-2);
  border: 0;
  border-radius: 0;
  border-bottom: 1px solid var(--border);
  width: 100%;
  text-align: left;
  font-size: var(--text-xs);
  font-weight: var(--weight-regular);
  align-items: start;
}

.change:hover:not(:disabled),
.change.on {
  background: var(--bg-raised);
  color: inherit;
}

.change .small {
  font-size: 11px;
  overflow: hidden;
  text-overflow: ellipsis;
}

.change .what {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}

.dim2 {
  color: var(--text-secondary);
}

.to {
  color: var(--text-primary);
}

.amend-text {
  font-weight: var(--weight-light);
  color: var(--text-secondary);
  line-height: 1.45;
}

.status {
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.06em;
  white-space: nowrap;
}

.empty {
  padding: var(--space-2);
  margin: 0;
  font-size: var(--text-xs);
  font-weight: var(--weight-light);
  color: var(--text-secondary);
}

.foot-bar {
  padding: var(--space-1) var(--space-2);
  border-top: 1px solid var(--border-strong);
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--space-2);
  flex-wrap: wrap;
  margin-top: auto;
}

.impact {
  font-size: var(--text-xs);
  font-weight: var(--weight-light);
  color: var(--text-secondary);
  line-height: 1.45;
  flex: 1 1 240px;
}

.buttons {
  display: flex;
  gap: var(--space-1);
}

.buttons button {
  padding: 6px 12px;
  font-size: var(--text-xs);
  font-weight: var(--weight-medium);
  white-space: nowrap;
}
</style>
