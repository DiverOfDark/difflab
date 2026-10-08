<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import {
  addMember,
  deleteMember,
  describeMember,
  detachRef,
  drawVariants,
  giveReference,
  keepConflict,
  outputUrl,
  patchMember,
  pickVariant,
  resolveConflict,
  upload,
  type Conflict,
  type MemberInfo,
  type Production,
  type Stale,
} from '../../api'
import { done, filmOf, hours, impact, linesOf, photographs, shotLabel, shotsOf, states } from '../../bible'
import { GENERATED, drawingRun, pad, type StageId } from '../../production'

const props = defineProps<{
  production: Production
  /** null: a new member, decided in words. */
  member: MemberInfo | null
  busy: string
  act: (name: string, what: () => Promise<unknown>) => Promise<void>
}>()
const emit = defineEmits<{ close: []; go: [to: StageId, shot: number | null, member: string | null] }>()

interface Trait {
  part: string
  phrase: string
  short: string
  name: string
}
interface Draft {
  slug: string
  kind: string
  noun: string
  possessive: string
  aliases: string
  appearance: string
  traits: Trait[]
  allowed: string
  lights: string
  voice: string
  delivery: string
  reference_audio: string
}

const KINDS = [
  { value: 'person', label: 'character · person' },
  { value: 'animal', label: 'character · animal' },
  { value: 'environment', label: 'location' },
  { value: 'object', label: 'prop' },
]
const PART_FOR: Record<string, string> = { person: 'accessory', animal: 'markings', object: 'component', environment: 'landmark' }
const IDENTITY: (keyof Draft)[] = ['kind', 'noun', 'possessive', 'appearance', 'traits', 'allowed', 'lights']
const VOICE: (keyof Draft)[] = ['voice', 'delivery', 'reference_audio']

const written = (m: MemberInfo) => (m.identity.traits ?? []).map((t) => t.phrase).join(', ')
function draftOf(m: MemberInfo | null): Draft {
  if (!m) {
    return { slug: '', kind: 'object', noun: '', possessive: 'its', aliases: '', appearance: '', traits: [], allowed: '', lights: '', voice: '', delivery: '', reference_audio: '' }
  }
  return {
    slug: m.slug,
    kind: m.kind,
    noun: m.noun,
    possessive: m.possessive,
    aliases: m.aliases,
    appearance: m.appearance || written(m),
    traits: (m.identity.traits ?? []).map((t) => ({ part: t.part, phrase: t.phrase, short: t.short, name: t.name })),
    allowed: m.allowed,
    lights: (m.identity.lights ?? []).join(', '),
    voice: m.voice,
    delivery: m.delivery,
    reference_audio: m.reference_audio,
  }
}
const draft = ref<Draft>(draftOf(props.member))
const saved = ref<Draft>(draftOf(props.member))
// A poll that brings the member back changed (a plate drawn, a describe finished) resets the form only
// while nothing is being typed into it.
watch(
  () => props.member,
  (m) => {
    const fresh = draftOf(m)
    if (JSON.stringify(draft.value) === JSON.stringify(saved.value)) draft.value = fresh
    saved.value = fresh
  },
)
const changed = (k: keyof Draft) => JSON.stringify(draft.value[k]) !== JSON.stringify(saved.value[k])
const identityDirty = computed(() => IDENTITY.some(changed))
const voiceDirty = computed(() => VOICE.some(changed))
const dirty = computed(() => identityDirty.value || voiceDirty.value || changed('aliases') || (!props.member && !!draft.value.slug))

const slug = computed(() => props.member?.slug ?? (draft.value.slug || 'new'))
const living = computed(() => draft.value.kind === 'person' || draft.value.kind === 'animal')
const film = computed(() => filmOf(props.production))
const cast = computed(() => (props.member ? shotsOf(props.production, props.member.slug) : []))

// --- identity --------------------------------------------------------------------------------------
const photoCount = computed(() => (props.member ? new Set(photographs(props.member).map((r) => r.photo)).size : 0))
const authored = computed(() => {
  const m = props.member
  if (!m) return 'decided in words'
  const by = m.identity.photographed === false ? 'written by the director from the story' : `written by Qwen-VL from ${photoCount.value} photo${photoCount.value === 1 ? '' : 's'}`
  return `${by}${m.appearance || m.pinned ? ' · edited' : ''}`
})
const trait = ref('')
const part = ref(PART_FOR[draft.value.kind] ?? 'accessory')
function addTrait() {
  const words = trait.value.split(/\s+/).filter(Boolean).join(' ')
  if (!words) return
  draft.value.traits.push({ part: part.value, phrase: words, short: words, name: words.split(' ').at(-1)! })
  trait.value = ''
}

// --- states ---------------------------------------------------------------------------------------
const rows = computed(() => (props.member ? states(props.production, props.member) : []))
const base = computed(() => props.member?.references.find((r) => r.artifact && r.present && r.role !== 'face') ?? props.member?.references.find((r) => r.artifact && r.present))
const open = (shot: number) => emit('go', 'shots', shot, props.member?.slug ?? null)

// --- voice ----------------------------------------------------------------------------------------
const spoken = computed(() => (props.member ? film.value.filter((s) => linesOf(s, props.member!.slug) > 0) : []))
const heard = computed(() => spoken.value.find((s) => done(s.clip) && s.clip.present && s.clip.artifact))
const hearing = ref(false)
const audioName = (path: string) => path.split('/').at(-1)!.replace(/^[0-9a-f]{16}-/, '')
const audioPick = ref('')
async function audioChosen(event: Event) {
  const select = event.target as HTMLSelectElement
  if (select.value === 'none') draft.value.reference_audio = ''
  else if (select.value === 'upload') audioInput.value?.click()
  select.value = draft.value.reference_audio ? 'current' : 'none'
}
const audioInput = ref<HTMLInputElement | null>(null)
async function audioUploaded(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  input.value = ''
  if (!file) return
  await props.act('audio', async () => {
    const [asset] = await upload([file])
    draft.value.reference_audio = asset.path
  })
}
const audioUrl = computed(() => {
  const path = draft.value.reference_audio
  if (!path) return ''
  return path.startsWith('/') ? `/uploads/${path.split('/').at(-1)}` : outputUrl(path)
})

// --- conflicts --------------------------------------------------------------------------------------
const conflicts = computed(() => props.member?.conflicts ?? [])
const textual = computed(() => conflicts.value.filter((c) => c.shows))
const photos = computed(() => conflicts.value.filter((c) => !c.shows))
const openCount = computed(() => conflicts.value.filter((c) => !c.resolution).length)
/** "fur colour: dark-brown in photo 2 (primary) vs black in photo 1": the two wordings to keep one of. */
function choices(c: Conflict): { label: string; resolution: string }[] {
  const found = c.says.match(/^(.+?):\s*(.+?) in photo (\d+)(?:\s*\([^)]*\))?(?:,| vs)\s*(.+?) in photo (\d+)/)
  if (!found) return [{ label: 'resolved', resolution: 'resolved by hand' }]
  const [, what, first, a, second, b] = found
  return [
    { label: `keep photo ${a} · ${first}`, resolution: `${what}: ${first}, as photo ${a} shows` },
    { label: `keep photo ${b} · ${second}`, resolution: `${what}: ${second}, as photo ${b} shows` },
  ]
}
const settle = (c: Conflict, resolution: string) => props.act(`conflict ${c.id}`, () => resolveConflict(c.id, resolution))
const keep = (c: Conflict, kept: 'reference' | 'novel') => props.act(`conflict ${c.id}`, () => keepConflict(c.id, kept))

// --- references -------------------------------------------------------------------------------------
type Ref = MemberInfo['references'][number]
const refs = computed(() => (props.member?.references ?? []).filter((r) => r.artifact))
const provided = computed(() => refs.value.filter((r) => r.role !== GENERATED).length)
const generated = computed(() => refs.value.filter((r) => r.role === GENERATED).length)
const shownAs = (r: Ref) => (props.member?.plate && r.plate ? r.plate : r.artifact!)
const refName = (r: Ref) => (r.role === GENERATED ? `drawn · seed ${r.seed ?? '—'}` : `${r.role} · ${audioName(r.photo || r.artifact!)}`)
const drawing = computed(() => drawingRun(props.production))
const GIVEN_ROLE: Record<string, string> = { environment: 'environment', person: 'body', animal: 'view', object: 'object' }
const dragging = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)
function give(files: FileList | null | undefined) {
  const file = files?.[0]
  const m = props.member
  if (!file || !m) return
  props.act(`give ${m.id}`, async () => {
    const [asset] = await upload([file])
    told((await giveReference(m.id, asset.path, GIVEN_ROLE[m.kind] ?? 'body')).stale)
  })
}
const plating = computed(() => props.production.runs.find((r) => r.scope === `plates:${props.production.id}` && ['queued', 'running', 'live'].includes(r.state)))
const plates = computed(() => refs.value.filter((r) => r.role !== GENERATED))
const plated = computed(() => plates.value.filter((r) => r.plate).length)
const togglePlate = (on: boolean) => props.member && props.act(`plate ${props.member.id}`, async () => told((await patchMember(props.member!.id, { plate: on })).stale))
const describe = () => props.member && props.act(`describe ${props.member.id}`, () => describeMember(props.member!.id))
const three = () => props.member && props.act(`draw ${props.member.id}`, () => drawVariants(props.member!.id))
const pick = (seed: number) => props.member && props.act(`draw ${props.member.id}`, async () => told((await pickVariant(props.member!.id, seed)).stale))
const detach = (r: Ref) => props.act(`ref ${r.id}`, async () => told((await detachRef(r.id)).stale))

// --- appears in -------------------------------------------------------------------------------------
const appears = computed(() => {
  const m = props.member
  if (!m) return []
  const chapters = [...new Set(cast.value.map((s) => s.chapter + 1))].sort((a, b) => a - b)
  const span = !chapters.length ? '—' : chapters.length > 2 && chapters.at(-1)! - chapters[0] === chapters.length - 1 ? `${pad(chapters[0], 2)}–${pad(chapters.at(-1)!, 2)}` : chapters.map((c) => pad(c, 2)).join(', ')
  return [
    { k: 'chapters', v: span },
    { k: 'shots', v: String(cast.value.length) },
    { k: 'dialogue lines', v: String(cast.value.reduce((n, s) => n + linesOf(s, m.slug), 0)) },
    { k: 'keyframes done', v: String(cast.value.filter((s) => s.keyframe.source === 'composed' && done(s.keyframe)).length) },
    { k: 'clips done', v: String(cast.value.filter((s) => done(s.clip)).length) },
  ]
})

// --- saving -----------------------------------------------------------------------------------------
const cost = computed(() => (props.member ? impact(props.production, props.member.slug, !identityDirty.value) : { keyframes: 0, clips: 0, seconds: 0 }))
const marks = computed(() => (identityDirty.value || voiceDirty.value ? cost.value.keyframes + cost.value.clips : 0))
const pronoun = computed(() => ({ his: 'his', her: 'her', their: 'their' })[draft.value.possessive] ?? 'its')
const impactLine = computed(() => {
  const who = slug.value
  if (!props.member) return problems.value.length ? `Needs ${problems.value.join(' · ')}.` : `${who} is added pinned: a re-plan casts it where the story needs it.`
  if (!identityDirty.value && !voiceDirty.value)
    return changed('aliases')
      ? 'Aliases reach the next plan: re-plan to find its mentions by them. Nothing goes stale.'
      : `No changes yet. Identity fields invalidate keyframes and clips; voice fields only clips with ${pronoun.value} lines.`
  const { keyframes, clips, seconds } = cost.value
  const what = identityDirty.value ? 'Appearance changed' : 'Voice changed'
  const stale = keyframes + clips ? `${keyframes} keyframe${keyframes === 1 ? '' : 's'} and ${clips} clip${clips === 1 ? '' : 's'} with ${who} become stale · ~${hours(seconds)} of GPU to redo` : identityDirty.value ? `nothing with ${who} is made yet, so nothing goes stale` : `no rendered clip has ${who} speak, so nothing goes stale`
  const tail = identityDirty.value ? ` Voice changes re-render only the clips where ${who} speaks.` : ''
  return `${what} · ${stale}.${tail}`
})
const problems = computed(() => {
  const d = draft.value
  const out: string[] = []
  if (!props.member && !/^[a-z][a-z_]*$/.test(d.slug)) out.push('a name: one lower-case word, or words joined by _')
  if (!props.member && props.production.members.some((m) => m.slug === d.slug)) out.push(`${d.slug} exists`)
  if (d.kind === 'style') return out
  if (!d.noun.trim()) out.push('a noun')
  if (!d.traits.length) out.push('at least one trait that must not drift')
  if (d.kind === 'environment' && !d.lights.trim()) out.push('a light for the place')
  return out
})
const note = ref('')
function told(stale: Stale) {
  const parts = Object.entries(stale)
    .filter(([k, n]) => n && k !== 'plating')
    .map(([k, n]) => `${n} ${k}`)
  note.value = [stale.plating ? 'Plates are being drawn.' : '', parts.length ? `${parts.join(', ')} went stale.` : ''].filter(Boolean).join(' ')
}
function save() {
  const d = draft.value
  const lights = d.lights.split(',').map((x) => x.trim()).filter(Boolean)
  const m = props.member
  if (!m) {
    props.act('member new', async () => {
      await addMember(props.production.id, { slug: d.slug, kind: d.kind, noun: d.noun, possessive: d.possessive, traits: d.traits, lights })
      emit('close')
    })
    return
  }
  const body: Record<string, unknown> = {}
  if (changed('aliases')) body.aliases = d.aliases
  if (identityDirty.value) {
    Object.assign(body, { kind: d.kind, noun: d.noun, possessive: d.possessive, traits: d.traits, allowed: d.allowed, lights })
    body.appearance = d.appearance.trim() === written(m) ? '' : d.appearance
  }
  if (voiceDirty.value) Object.assign(body, { voice: d.voice, delivery: d.delivery, reference_audio: d.reference_audio })
  props.act(`member ${m.id}`, async () => {
    told((await patchMember(m.id, body)).stale)
    saved.value = JSON.parse(JSON.stringify(draft.value))
  })
}
const discard = () => {
  draft.value = JSON.parse(JSON.stringify(saved.value))
  emit('close')
}
const remove = () => {
  const m = props.member
  if (m && confirm(`Delete ${m.slug}? A re-plan will not bring it back.`))
    props.act(`member ${m.id}`, async () => {
      await deleteMember(m.id)
      emit('close')
    })
}
</script>

<template>
  <div class="editor">
    <div class="nav">
      <button class="link back" @click="discard">← bible</button>
      <span class="mono where">
        {{ slug }} · in {{ cast.length }} shots
        <button v-if="member && member.kind !== 'environment'" class="link" :disabled="!!busy" @click="remove">delete</button>
      </span>
    </div>

    <p v-if="note" class="band hint">{{ note }}</p>

    <div class="split">
      <div class="col">
        <section class="card">
          <div class="head"><h5>Identity</h5><span class="note">what the shot writer and the QA gate check against</span></div>
          <div class="body">
            <div class="three">
              <label>
                <span class="eyebrow">NAME</span>
                <input v-if="!member" v-model="draft.slug" class="mono" placeholder="the_keeper" />
                <input v-else :value="member.slug" class="mono" readonly title="the plan names it by this" />
              </label>
              <label>
                <span class="eyebrow">KIND</span>
                <select v-model="draft.kind" class="mono" @change="part = PART_FOR[draft.kind] ?? part">
                  <option v-for="k in KINDS" :key="k.value" :value="k.value">{{ k.label }}</option>
                </select>
              </label>
              <label>
                <span class="eyebrow">ALIASES IN TEXT</span>
                <input v-model="draft.aliases" class="plain" placeholder="Tomas, the keeper, the old man" />
              </label>
            </div>
            <div class="three">
              <label>
                <span class="eyebrow">NOUN</span>
                <input v-model="draft.noun" class="plain" placeholder="lighthouse keeper" />
              </label>
              <label>
                <span class="eyebrow">POSSESSIVE</span>
                <select v-model="draft.possessive" class="mono">
                  <option v-for="p in ['his', 'her', 'their', 'its']" :key="p" :value="p">{{ p }}</option>
                </select>
              </label>
            </div>
            <label class="field">
              <span class="split-label"><span class="eyebrow">APPEARANCE</span><span class="eyebrow quiet">{{ authored }}</span></span>
              <textarea v-model="draft.appearance" rows="4" class="prose" />
            </label>
            <div class="field">
              <span class="split-label"><span class="eyebrow">MUST NOT DRIFT</span><span class="note">traits the QA gate rejects a keyframe or clip for</span></span>
              <div class="traits">
                <span v-for="(t, i) in draft.traits" :key="i" class="trait mono" :title="`${t.part} · ${t.phrase}`">
                  {{ t.short || t.phrase }}<button class="x" title="remove" @click="draft.traits.splice(i, 1)">×</button>
                </span>
                <span class="adder">
                  <select v-model="part" class="mono" title="which part of it">
                    <option v-for="p in production.trait_parts" :key="p" :value="p">{{ p }}</option>
                  </select>
                  <input v-model="trait" class="mono" placeholder="+ trait" @keydown.enter.prevent="addTrait" @blur="addTrait" />
                </span>
              </div>
            </div>
            <label class="field">
              <span class="eyebrow">ALLOWED TO CHANGE</span>
              <input v-model="draft.allowed" class="plain" placeholder="expression, wet or dry hair, coat open or buttoned" />
            </label>
            <label v-if="draft.kind === 'environment'" class="field">
              <span class="eyebrow">LIGHTS</span>
              <input v-model="draft.lights" class="plain" placeholder="soft overcast daylight from above, a warm lamp on the left" />
            </label>
          </div>
        </section>

        <section v-if="member" class="card">
          <div class="head"><h5>States across the film</h5><span class="note">this card is the base · each change lives on the shot that makes it</span></div>
          <div class="staterow">
            <img v-if="base" class="frame thumb on" :src="outputUrl(base.artifact!)" alt="" />
            <span v-else class="frame thumb on" />
            <span class="what"><span class="mono">base</span><span class="sub">as described above · first appearance</span></span>
            <span class="eyebrow small">BIBLE</span>
          </div>
          <div v-for="r in rows" :key="r.id ?? 'base'" class="staterow">
            <img v-if="r.artifact" class="frame thumb" :class="r.image" :src="outputUrl(r.artifact)" alt="" />
            <span v-else class="frame thumb" :class="r.image" />
            <span class="what"
              ><span class="mono">{{ r.label }} <span class="dim">· from {{ r.parent }}</span></span><span class="sub">{{ r.amend }}</span></span
            >
            <button v-if="r.shot" class="link" @click="open(r.shot.id)">{{ shotLabel(r.shot) }} →</button>
          </div>
          <p v-if="!rows.length" class="note pad">No shot changes {{ member.slug }}: it looks as the base all film long.</p>
        </section>

        <section v-if="member && living" class="card">
          <div class="head"><h5>Voice</h5><span class="note">written into every H3 prompt this character speaks in</span></div>
          <div class="body">
            <div class="four">
              <label><span class="eyebrow">VOICE</span><input v-model="draft.voice" class="plain" placeholder="a low weathered male baritone" /></label>
              <label><span class="eyebrow">DELIVERY</span><input v-model="draft.delivery" class="plain" placeholder="slow, pauses mid-sentence" /></label>
              <label>
                <span class="eyebrow">REFERENCE AUDIO</span>
                <select v-model="audioPick" class="mono" @change="audioChosen">
                  <option value="" disabled hidden>{{ draft.reference_audio ? audioName(draft.reference_audio) : 'none' }}</option>
                  <option value="none">none</option>
                  <option value="upload">upload…</option>
                </select>
                <input ref="audioInput" type="file" accept="audio/*" hidden @change="audioUploaded" />
              </label>
              <button class="hear" :disabled="!heard" :title="heard ? '' : spoken.length ? 'no clip where it speaks is rendered yet' : 'it speaks in no shot'" @click="hearing = !hearing">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="6 4 20 12 6 20 6 4" /></svg>
                {{ heard ? `Hear in clip ${shotLabel(heard)}` : spoken.length ? 'No clip rendered yet' : 'Speaks in no shot' }}
              </button>
            </div>
            <audio v-if="audioUrl" class="player" :src="audioUrl" controls preload="none" />
            <video v-if="hearing && heard" class="frame player" :src="outputUrl(heard.clip.artifact!)" controls autoplay />
            <span class="note">The reference audio is kept and played here; H3 is not handed it (see the voice line in each prompt).</span>
          </div>
        </section>

        <section v-if="conflicts.length" class="card" :class="{ warned: openCount }">
          <div class="head">
            <h5>Conflicts with the text</h5>
            <span class="mono small" :class="{ warn: openCount }">{{ textual.length ? 'REFERENCE OVERRIDES NOVEL' : 'THE PHOTOGRAPHS DISAGREE' }} · {{ openCount }}</span>
          </div>
          <div v-for="c in textual" :key="c.id" class="conflict" :class="{ settled: c.resolution }">
            <span class="col3"><span class="eyebrow small">NOVEL SAYS</span><span class="struck">“{{ c.says }}”</span></span>
            <span class="col3"><span class="eyebrow small">REFERENCE SHOWS</span><span>{{ c.shows }}</span></span>
            <span class="col3">
              <span class="eyebrow small">RESOLUTION</span>
              <span v-if="c.resolution" class="mono small">{{ c.resolution }}</span>
              <span v-else class="keeps">
                <button class="keep on" :disabled="!!busy || !c.lines" :title="c.lines ? 'the story’s sentences rewritten to what the photograph shows; every shot goes stale' : 'the story says it in no sentence that can be rewritten'" @click="keep(c, 'reference')">keep reference · {{ c.lines }} line{{ c.lines === 1 ? '' : 's' }}</button>
                <button class="keep" :disabled="!!busy" :title="`the card takes the story’s ${c.novel || 'version'}`" @click="keep(c, 'novel')">keep novel</button>
              </span>
            </span>
          </div>
          <div v-for="c in photos" :key="c.id" class="conflict" :class="{ settled: c.resolution }">
            <span class="col3 wide"><span class="eyebrow small">THE PHOTOGRAPHS DISAGREE</span><span>{{ c.says }}</span></span>
            <span class="col3">
              <span class="eyebrow small">RESOLUTION</span>
              <span v-if="c.resolution" class="mono small">kept · {{ c.resolution }} <button class="link" @click="settle(c, '')">reopen</button></span>
              <span v-else class="keeps">
                <button v-for="o in choices(c)" :key="o.label" class="keep" :disabled="!!busy" @click="settle(c, o.resolution)">{{ o.label }}</button>
              </span>
            </span>
          </div>
        </section>
      </div>

      <aside v-if="member" class="col aside">
        <section class="card">
          <div class="head"><h5 class="small">References</h5><span class="mono small dim">{{ provided }} provided · {{ generated }} generated</span></div>
          <div class="body">
            <div class="refs">
              <div v-for="r in refs" :key="r.id" class="ref">
                <a class="pic" :href="outputUrl(shownAs(r))" target="_blank">
                  <img class="frame" :class="{ lead: r === refs[0] }" :src="outputUrl(shownAs(r))" :alt="r.role" />
                  <span class="tag mono" :class="{ on: r.role !== 'generated' && r === refs[0] }">{{ r.role === GENERATED ? 'GEN' : member.plate && r.plate ? 'PLATE' : 'PHOTO' }}</span>
                  <button class="x" title="it does not show this member: let go of it, for good" @click.prevent="detach(r)">×</button>
                </a>
                <span class="mono name">{{ refName(r) }}</span>
              </div>
            </div>
            <div
              class="drop"
              :class="{ over: dragging }"
              @click="fileInput?.click()"
              @dragover.prevent="dragging = true"
              @dragleave="dragging = false"
              @drop.prevent="((dragging = false), give($event.dataTransfer?.files))"
            >
              <span>{{ busy === `give ${member.id}` ? 'fitting it…' : 'Drop another photo' }}</span>
              <span class="mono hint">frontal, sharp, plain background works best</span>
              <input ref="fileInput" type="file" accept="image/*" hidden @change="give(($event.target as HTMLInputElement).files)" />
            </div>
            <div v-for="g in member.gaps ?? []" :key="g.role" class="gap note">no {{ g.role }} picture: {{ g.why }}</div>
            <div v-if="member.kind !== 'environment'" class="plate">
              <span class="eyebrow">PLATE</span>
              <label>
                <input type="checkbox" :checked="member.plate" :disabled="!!busy || !plates.length" @change="togglePlate(($event.target as HTMLInputElement).checked)" />
                <span>Replace each photo's background with a flat backdrop, so the model copies the {{ member.noun }} and not the room it was photographed in.</span>
              </label>
              <span v-if="member.plate" class="mono small dim">{{ plating ? 'plating…' : `${plated} of ${plates.length} plated` }}</span>
            </div>
            <div class="links">
              <button class="link" :disabled="!!busy" title="its card and conflicts read again from the photographs, and given back to the director" @click="describe">describe again from photos</button>
              <button class="link" :disabled="!!busy || !!drawing" title="three drawings on new seeds, from its words; pick one as its reference" @click="three">generate 3 views</button>
            </div>
            <div v-if="member.variants?.length" class="variants">
              <span v-for="v in member.variants" :key="v.seed" class="variant">
                <img v-if="v.frame" class="frame" :src="outputUrl(v.frame)" alt="" />
                <span v-else class="frame mono wait">{{ v.state }}</span>
                <button class="link" :disabled="!!busy || !v.frame || member.references.some((r) => r.role === GENERATED && r.seed === v.seed)" @click="pick(v.seed)">
                  {{ member.references.some((r) => r.role === GENERATED && r.seed === v.seed) ? 'in use' : 'use this' }}
                </button>
              </span>
            </div>
          </div>
        </section>
        <section class="card">
          <div class="head"><h5 class="small">Appears in</h5></div>
          <div class="appears">
            <div v-for="a in appears" :key="a.k" class="fact mono"><span>{{ a.k }}</span><span class="dim">{{ a.v }}</span></div>
          </div>
        </section>
      </aside>
    </div>

    <footer class="foot">
      <span class="note">{{ impactLine }}</span>
      <div class="buttons">
        <button @click="discard">Discard</button>
        <button class="primary" :disabled="!!busy || !dirty || !!problems.length" @click="save">
          {{ busy.startsWith('member') ? 'saving…' : marks ? `Save · mark ${marks} stale` : 'Save' }}
        </button>
      </div>
    </footer>
  </div>
</template>

<style scoped>
.editor {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.nav {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--space-2);
  flex-wrap: wrap;
}

.back,
.where {
  font-size: 12px;
}

.where {
  color: var(--text-secondary);
  display: flex;
  gap: var(--space-2);
  align-items: center;
}

.split {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr));
  gap: var(--space-3);
  align-items: start;
}

@media (min-width: 1100px) {
  .split {
    grid-template-columns: minmax(0, 1.6fr) minmax(300px, 1fr);
  }
}

.col {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
  min-width: 0;
}

.aside {
  gap: var(--space-2);
}

.card > .head {
  flex-wrap: wrap;
}

.card > .head > h5.small {
  font-size: var(--text-sm);
}

.three,
.four {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: var(--space-2);
}

.four {
  align-items: end;
}

.three > label,
.four > label,
.field {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  min-width: 0;
}

.card input.plain,
.card textarea.prose {
  font-family: var(--font-body);
  font-weight: var(--weight-light);
  font-size: 13px;
}

.card textarea.prose {
  line-height: 1.5;
  padding: var(--space-1) var(--space-2);
}

.card input[readonly] {
  color: var(--text-secondary);
}

.split-label {
  display: flex;
  justify-content: space-between;
  gap: var(--space-1);
  flex-wrap: wrap;
}

.eyebrow.quiet {
  text-transform: none;
  letter-spacing: 0;
}

.eyebrow.small,
.small {
  font-size: 10px;
}

.traits {
  display: flex;
  gap: var(--space-1);
  flex-wrap: wrap;
  align-items: center;
}

.trait {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  font-size: 11px;
  padding: 4px 8px;
  border: 1px solid var(--border-strong);
  border-radius: 2px;
}

.x {
  background: none;
  border: 0;
  padding: 0;
  color: var(--text-tertiary);
  font-family: var(--font-mono);
  font-size: 11px;
}

.adder {
  display: flex;
  gap: 4px;
}

.adder select {
  width: auto;
  font-size: 11px;
  padding: 4px;
}

.adder input {
  background: transparent;
  border: 1px dashed var(--border-strong);
  font-size: 11px;
  padding: 4px 8px;
  width: 140px;
}

.staterow {
  display: grid;
  grid-template-columns: 40px minmax(0, 1fr) auto;
  gap: var(--space-2);
  align-items: center;
  padding: var(--space-1) var(--space-2);
  border-bottom: 1px solid var(--border);
}

.staterow:last-child {
  border-bottom: 0;
}

.thumb {
  width: 40px;
  aspect-ratio: 3 / 4;
}

.thumb.on,
.thumb.done {
  border-color: var(--accent);
}

.thumb.flag,
.thumb.failed {
  border-color: var(--status-degraded);
}

.thumb.pending,
.thumb.running {
  border-color: var(--border-strong);
}

.what {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
  font-size: 12px;
}

.sub {
  font-weight: var(--weight-light);
  color: var(--text-secondary);
  line-height: 1.45;
}

.pad {
  padding: var(--space-1) var(--space-2);
}

.hear {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-1);
  white-space: nowrap;
}

.player {
  width: 100%;
}

video.player {
  max-height: 320px;
  object-fit: contain;
}

.card.warned {
  border-color: var(--status-degraded);
}

.warn {
  color: var(--status-degraded);
}

.conflict {
  padding: var(--space-2);
  border-bottom: 1px solid var(--border);
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: var(--space-2);
  align-items: start;
}

.conflict:last-child {
  border-bottom: 0;
}

.conflict.settled {
  opacity: 0.6;
}

.col3 {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
  line-height: 1.45;
}

.col3.wide {
  grid-column: span 2;
}

.struck {
  font-weight: var(--weight-light);
  text-decoration: line-through;
  color: var(--text-secondary);
}

.keeps {
  display: flex;
  gap: var(--space-1);
  flex-wrap: wrap;
}

.keep {
  padding: 4px 8px;
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: var(--weight-regular);
  white-space: nowrap;
}

.keep.on {
  background: var(--bg-raised);
  border-color: var(--accent);
  color: var(--accent);
}

.refs {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--space-1);
}

.ref {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}

.pic {
  position: relative;
  display: block;
}

.pic img {
  aspect-ratio: 3 / 4;
}

.pic img.lead {
  border-color: var(--accent);
}

.tag {
  position: absolute;
  top: 4px;
  left: 4px;
  font-size: 9px;
  letter-spacing: 0.06em;
  padding: 1px 4px;
  border-radius: 2px;
  background: var(--bg-base);
  color: var(--text-secondary);
}

.tag.on {
  background: var(--accent);
  color: var(--accent-fg);
}

.pic .x {
  position: absolute;
  top: 2px;
  right: 4px;
  opacity: 0;
}

.pic:hover .x {
  opacity: 1;
}

.name {
  font-size: 10px;
  color: var(--text-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.drop {
  border: 1px dashed var(--border-strong);
  border-radius: var(--radius-btn);
  padding: var(--space-2);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  color: var(--text-secondary);
  font-size: 12px;
  font-weight: var(--weight-light);
  text-align: center;
  cursor: pointer;
}

.drop.over {
  border-color: var(--accent);
  color: var(--accent);
}

.drop .hint {
  font-size: 10px;
  color: var(--text-tertiary);
}

.plate {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  padding-top: var(--space-2);
  border-top: 1px solid var(--border);
}

.plate label {
  display: flex;
  gap: var(--space-1);
  align-items: flex-start;
  font-size: 12px;
  font-weight: var(--weight-light);
  color: var(--text-secondary);
  line-height: 1.4;
  cursor: pointer;
}

.plate input {
  margin: 2px 0 0;
}

.links {
  display: flex;
  gap: var(--space-2);
  flex-wrap: wrap;
}

.variants {
  display: flex;
  gap: var(--space-1);
  overflow-x: auto;
}

.variant {
  display: flex;
  flex-direction: column;
  gap: 4px;
  align-items: flex-start;
}

.variant .frame {
  height: 120px;
  width: auto;
  aspect-ratio: 3 / 4;
}

.wait {
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  color: var(--text-tertiary);
}

.appears {
  padding: var(--space-1) var(--space-2) var(--space-2);
  display: flex;
  flex-direction: column;
}

.fact {
  display: flex;
  justify-content: space-between;
  gap: var(--space-2);
  padding: var(--space-1) 0;
  border-bottom: 1px solid var(--border);
  font-size: 12px;
}

.dim {
  color: var(--text-tertiary);
}
</style>
