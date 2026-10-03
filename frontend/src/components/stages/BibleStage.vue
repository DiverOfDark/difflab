<script setup lang="ts">
import { computed, ref } from 'vue'
import { RouterLink } from 'vue-router'
import {
  addMember,
  deleteMember,
  detachRef,
  drawReference,
  drawReferences,
  giveReference,
  outputUrl,
  patchMember,
  resolveConflict,
  upload,
  type Conflict,
  type MemberInfo,
  type Production,
  type ShotInfo,
} from '../../api'
import { GENERATED, castOf, drawingRun, type StageId } from '../../production'
import { state } from '../../store'

const props = defineProps<{
  production: Production
  shots: ShotInfo[]
  busy: string
  act: (name: string, what: () => Promise<unknown>) => Promise<void>
}>()
defineEmits<{ go: [to: StageId, shot?: number | null] }>()

const tab = ref('all')
const tabs = computed(() => {
  const kinds = [...new Set(props.production.members.map((m) => m.kind))]
  return [
    { id: 'all', count: props.production.members.length },
    ...kinds.map((k) => ({ id: k, count: props.production.members.filter((m) => m.kind === k).length })),
  ]
})
const shown = computed(() => props.production.members.filter((m) => tab.value === 'all' || m.kind === tab.value))

const inShots = (slug: string) => props.shots.filter((s) => castOf(s).includes(slug)).length
const traits = (m: MemberInfo) => m.identity.traits ?? []
const lead = (m: MemberInfo) => m.references.find((r) => r.artifact && r.present)
const rest = (m: MemberInfo) => m.references.filter((r) => r.artifact && r.present && r !== lead(m)).slice(0, 3)
const open = (m: MemberInfo) => m.conflicts.filter((c) => !c.resolution)

// --- references drawn from words ---------------------------------------------------------------------
const drawing = computed(() => drawingRun(props.production))
const drawingNow = computed(() => (drawing.value ? state.runs.find((r) => r.id === drawing.value!.id) : undefined))
const missingRefs = computed(() => props.production.missing_references ?? [])
const drawn = (m: MemberInfo) => m.references.find((r) => r.role === GENERATED)
/** A member with a picture cut from a photograph, or given, has no need of a drawn one. */
const photographic = (m: MemberInfo) => m.references.some((r) => r.role !== GENERATED && r.artifact && r.present)
const inDrawing = (m: MemberInfo) => {
  const r = drawn(m)
  return !!drawing.value && !!r && (r.state === 'queued' || r.state === 'running')
}
const drawMissing = () => props.act('draw references', () => drawReferences(props.production.id))
const drawOne = (m: MemberInfo) => props.act(`draw ${m.id}`, () => drawReference(m.id))
/** A gap filled by the user's own picture; a face is not drawn, since a drawn picture is a whole body. */
const give = (m: MemberInfo, role: string, event: Event) => {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  input.value = ''
  if (file)
    props.act(`give ${m.id}`, async () => {
      const [asset] = await upload([file])
      said((await giveReference(m.id, asset.path, role)).stale)
    })
}
const photoName = (path: string) => {
  if (!path) return 'drawn from its words'

  const photos = Array.isArray(props.production.manifest.photos) ? (props.production.manifest.photos as string[]) : []
  const at = photos.indexOf(path)
  return at >= 0 ? `photo ${at + 1}` : path.split('/').at(-1)!
}

/** "fur colour: dark-brown in photo 2 (primary) vs black in photo 1", or "…in photo 1, … in photo 2":
 *  the two wordings to keep one of. */
function choices(c: Conflict): { label: string; resolution: string }[] {
  const found = c.says.match(/^(.+?):\s*(.+?) in photo (\d+)(?:\s*\([^)]*\))?(?:,| vs)\s*(.+?) in photo (\d+)/)
  if (!found) return [{ label: 'resolved', resolution: 'resolved by hand' }]
  const [, what, first, a, second, b] = found
  return [
    { label: `keep photo ${a} · ${first}`, resolution: `${what}: ${first}, as photo ${a} shows` },
    { label: `keep photo ${b} · ${second}`, resolution: `${what}: ${second}, as photo ${b} shows` },
  ]
}
const resolve = (c: Conflict, resolution: string) => props.act(`conflict ${c.id}`, () => resolveConflict(c.id, resolution))

// --- the editor ------------------------------------------------------------------------------------
interface Draft {
  id: number | null
  slug: string
  noun: string
  kind: string
  possessive: string
  find: string
  photographed: boolean
  traits: { part: string; phrase: string; short: string }[]
  lights: string[]
}

const draft = ref<Draft | null>(null)
const editing = computed(() => (draft.value?.id ? props.production.members.find((m) => m.id === draft.value!.id) ?? null : null))
const KINDS = ['person', 'animal', 'object', 'environment']
const place = computed(() => props.production.members.find((m) => m.kind === 'environment'))

function edit(m: MemberInfo) {
  draft.value = {
    id: m.id,
    slug: m.slug,
    noun: m.noun,
    kind: m.kind,
    possessive: m.possessive,
    find: m.find,
    photographed: m.identity.photographed ?? true,
    traits: traits(m).map((t) => ({ part: t.part, phrase: t.phrase, short: t.short })),
    lights: [...(m.identity.lights ?? [])],
  }
}
function fresh() {
  draft.value = { id: null, slug: '', noun: '', kind: 'object', possessive: 'its', find: '', photographed: false, traits: [], lights: [] }
  addTrait()
}
const addTrait = () => draft.value!.traits.push({ part: props.production.trait_parts[0] ?? 'coat', phrase: '', short: '' })

const problems = computed(() => {
  const d = draft.value
  if (!d) return []
  const out: string[] = []
  if (!d.id && !/^[a-z][a-z_]*$/.test(d.slug)) out.push('an id: one lower-case word, or words joined by _')
  if (!d.id && props.production.members.some((m) => m.slug === d.slug)) out.push(`${d.slug} exists`)
  if (!d.noun.trim()) out.push('a noun')
  if (!d.traits.some((t) => t.phrase.trim())) out.push('at least one trait')
  if (d.kind === 'environment' && !d.lights.some((l) => l.trim())) out.push('a light for the place')
  return out
})

const body = () => {
  const d = draft.value!
  return {
    noun: d.noun,
    kind: d.kind,
    possessive: d.possessive,
    find: d.find,
    traits: d.traits.filter((t) => t.phrase.trim()).map((t) => ({ part: t.part, phrase: t.phrase, short: t.short || t.phrase })),
    lights: d.lights.filter((l) => l.trim()),
  }
}
const note = ref('')
const said = (stale: { prompts: number; keyframes: number; clips: number }) => {
  const parts = Object.entries(stale)
    .filter(([, n]) => n)
    .map(([k, n]) => `${n} ${k}`)
  note.value = parts.length ? `${parts.join(', ')} went stale: re-plan the chapter.` : 'Saved. Re-plan to write it into the shots.'
}
const save = () => {
  const d = draft.value!
  props.act(`member ${d.id ?? 'new'}`, async () => {
    if (d.id) said((await patchMember(d.id, { ...body(), photographed: d.photographed })).stale)
    else {
      await addMember(props.production.id, { slug: d.slug, ...body() })
      note.value = `${d.slug} added. Re-plan to cast it where the story needs it.`
    }
    draft.value = null
  })
}
const unpin = (m: MemberInfo) => props.act(`member ${m.id}`, () => patchMember(m.id, { pinned: false }))
const remove = (m: MemberInfo) => {
  if (confirm(`Delete ${m.slug}? A re-plan will not bring it back.`))
    props.act(`member ${m.id}`, async () => {
      said((await deleteMember(m.id)).stale)
      draft.value = null
    })
}
const detach = (refId: number) => props.act(`ref ${refId}`, async () => said((await detachRef(refId)).stale))
</script>

<template>
  <div v-if="draft" class="editor">
    <div class="nav">
      <button class="link big" @click="draft = null">← bible</button>
      <span class="mono muted">{{ draft.id ? `${draft.slug} · ${editing?.pinned ? 'edited by hand' : 'as the director wrote it'}` : 'a new member, decided in words' }}</span>
    </div>

    <div class="split">
      <div class="col">
        <section class="card">
          <div class="head slim"><h5>Identity</h5><span class="note">the id is how the plan names it</span></div>
          <div class="body">
            <div class="stacked">
              <label v-if="!draft.id"><span class="eyebrow">id</span><input v-model="draft.slug" placeholder="fried_pigeon" class="mono" /></label>
              <label><span class="eyebrow">noun · what it is</span><input v-model="draft.noun" placeholder="cat" /></label>
            </div>
            <div class="field">
              <span class="eyebrow">kind</span>
              <div class="seg">
                <button
                  v-for="k in KINDS"
                  :key="k"
                  :class="{ on: draft.kind === k }"
                  :disabled="k === 'environment' ? !!place && place.id !== draft.id : !!place && place.id === draft.id"
                  @click="draft.kind = k"
                >
                  {{ k.toUpperCase() }}
                </button>
              </div>
            </div>
            <div class="field">
              <span class="eyebrow">possessive</span>
              <div class="seg">
                <button v-for="p in ['his', 'her', 'its']" :key="p" :class="{ on: draft.possessive === p }" @click="draft.possessive = p">{{ p }}</button>
              </div>
            </div>
            <label class="field">
              <span class="eyebrow">find · how to know it, or what the story asks of it</span>
              <textarea v-model="draft.find" rows="2" class="prose" />
            </label>
            <div class="field">
              <span class="eyebrow">look</span>
              <div class="seg">
                <button :class="{ on: draft.photographed }" :disabled="!draft.id" title="a re-plan cuts its references from the photographs the survey finds it in" @click="draft.photographed = true">PHOTOGRAPHED</button>
                <button :class="{ on: !draft.photographed }" @click="draft.photographed = false">DECIDED</button>
              </div>
              <span class="note">
                {{
                  draft.photographed
                    ? 'Its references are cut from the photographs that show it.'
                    : 'No photograph shows it: every crop of it is let go of on save, and a clip has the words below, and the picture drawn from them once there is one.'
                }}
              </span>
            </div>
          </div>
        </section>

        <section class="card">
          <div class="head slim"><h5>Must not drift</h5><span class="note">said in every prompt it is in</span></div>
          <div class="body">
            <div v-for="(t, i) in draft.traits" :key="i" class="trait-row">
              <select v-model="t.part" class="mono">
                <option v-for="p in production.trait_parts" :key="p" :value="p">{{ p }}</option>
              </select>
              <input v-model="t.phrase" placeholder="the whole phrase: dark grey cracked asphalt with faded white lines" />
              <input v-model="t.short" placeholder="short: cracked grey asphalt" />
              <button class="link" title="remove" @click="draft.traits.splice(i, 1)">×</button>
            </div>
            <button class="link add" @click="addTrait">+ trait</button>
          </div>
        </section>

        <section v-if="draft.kind === 'environment'" class="card">
          <div class="head slim"><h5>Lights</h5><span class="note">each visible light, named</span></div>
          <div class="body">
            <div v-for="(_, i) in draft.lights" :key="i" class="light-row">
              <input v-model="draft.lights[i]" placeholder="soft overcast daylight from above" />
              <button class="link" title="remove" @click="draft.lights.splice(i, 1)">×</button>
            </div>
            <button class="link add" @click="draft.lights.push('')">+ light</button>
          </div>
        </section>
      </div>

      <aside class="col">
        <section v-if="editing" class="card">
          <div class="head slim"><h5>References</h5><span class="mono dim small">{{ editing.references.length }}</span></div>
          <div class="refs">
            <p v-if="!editing.references.length" class="note pad">None: {{ editing.identity.photographed === false ? 'decided in words.' : 'plan the chapter to cut them.' }}</p>
            <div v-for="r in editing.references" :key="r.id" class="ref">
              <a :href="r.artifact ? outputUrl(r.artifact) : undefined" target="_blank">
                <img v-if="r.artifact && r.present" class="frame" :src="outputUrl(r.artifact)" :alt="r.role" />
              </a>
              <span v-if="r.role === 'generated'" class="what"
                ><span class="mono">generated · {{ r.state }}</span><span class="why">drawn from its words · seed {{ r.seed }}</span></span
              >
              <span v-else class="what"><span class="mono">{{ r.role }} · {{ photoName(r.photo) }}</span><span class="why">shows {{ r.shows.join(', ') }}</span></span>
              <button class="link" :disabled="!!busy" title="it does not show this member: let go of it, for good" @click="detach(r.id)">detach</button>
            </div>
          </div>
        </section>

        <section v-if="editing?.conflicts.length" class="card">
          <div class="head slim"><h5>Conflicts</h5><span class="note">the photographs disagree</span></div>
          <div class="body">
            <div v-for="c in editing.conflicts" :key="c.id" class="conflict" :class="{ settled: c.resolution }">
              <span>{{ c.says }}</span>
              <span v-if="c.resolution" class="row mono small"><span>kept · {{ c.resolution }}</span><button class="link" @click="resolve(c, '')">reopen</button></span>
              <span v-else class="keeps">
                <button v-for="o in choices(c)" :key="o.label" class="chip" :disabled="!!busy" @click="resolve(c, o.resolution)">{{ o.label }}</button>
              </span>
            </div>
            <span class="note">Keeping a wording records it. Where the traits still say the other one, edit them here.</span>
          </div>
        </section>
      </aside>
    </div>

    <footer class="foot">
      <span class="note">
        Saving pins it: a re-plan leaves it as written.
        <template v-if="draft.id"> {{ inShots(draft.slug) }} shots cast it; their prompts, composed keyframes and clips go stale until you re-plan.</template>
        <template v-if="problems.length"> Needs {{ problems.join(' · ') }}.</template>
      </span>
      <div class="buttons">
        <button v-if="editing && editing.kind !== 'environment'" class="danger" :disabled="!!busy" @click="remove(editing)">Delete</button>
        <button v-if="editing?.pinned" :disabled="!!busy" title="a re-plan describes it again from the photographs" @click="unpin(editing)">Give back to the director</button>
        <button @click="draft = null">Cancel</button>
        <button class="primary" :disabled="!!busy || !!problems.length" @click="save">{{ busy.startsWith('member') ? 'saving…' : 'Save' }}</button>
      </div>
    </footer>
  </div>

  <div v-else class="bible">
    <div class="bar-row">
      <div class="chips">
        <button v-for="t in tabs" :key="t.id" class="chip" :class="{ on: tab === t.id }" @click="tab = t.id">
          {{ t.id }} <span class="n">{{ t.count }}</span>
        </button>
      </div>
      <span class="acts">
        <button class="link" :disabled="!production.members.length" @click="fresh">+ add a member in words</button>
        <button
          class="primary"
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

    <p v-if="note" class="band hint">{{ note }}</p>
    <p v-if="!production.members.length" class="band hint">No cast yet. Plan the chapter: the survey names who the story needs from the photographs.</p>

    <div class="cards">
      <article v-for="m in shown" :key="m.id" class="card" :class="{ conflicted: open(m).length }">
        <div class="top">
          <span class="named"><span class="mono">{{ m.slug }}</span><span class="eyebrow">{{ m.kind }}</span></span>
          <span class="flags">
            <span v-if="m.pinned" class="source mono hand" title="edited by hand: a re-plan leaves it as written">PINNED</span>
            <span class="source mono" :class="{ photo: m.identity.photographed }">{{ m.identity.photographed ? 'PHOTOGRAPHED' : 'DECIDED' }}</span>
          </span>
        </div>
        <div class="inside">
          <div class="pics">
            <span v-if="lead(m)" class="leadpic">
              <img class="frame lead" :src="outputUrl(lead(m)!.artifact!)" :alt="lead(m)!.role" />
              <span v-if="lead(m)!.role === 'generated'" class="gen mono" :class="{ stale: lead(m)!.state === 'stale' }" title="drawn by the editor from the words">GEN</span>
            </span>
            <div v-else class="frame lead empty mono">{{ inDrawing(m) ? 'drawing…' : 'no reference' }}</div>
            <div class="thumbs">
              <img v-for="r in rest(m)" :key="r.id" class="frame" :src="outputUrl(r.artifact!)" :alt="r.role" :title="r.role" />
            </div>
          </div>
          <div class="words">
            <p class="find">{{ m.find || m.noun }}</p>
            <div v-if="traits(m).length" class="traits">
              <span class="eyebrow">must not drift</span>
              <span class="list">
                <span v-for="t in traits(m)" :key="t.part + t.phrase" class="trait mono" :title="t.phrase">{{ t.short || t.phrase }}</span>
              </span>
            </div>
            <div v-if="m.identity.lights?.length" class="traits">
              <span class="eyebrow">lights</span>
              <span class="plain">{{ m.identity.lights.join(', ') }}</span>
            </div>
            <div v-for="g in m.gaps ?? []" :key="g.role" class="conflict gap">
              <span class="eyebrow warn">no {{ g.role }} picture</span>
              <span>{{ g.why }}</span>
              <span class="keeps">
                <button
                  v-if="g.role !== 'face'"
                  class="chip"
                  :disabled="!!busy || !!drawing"
                  title="the editor draws it from its words: the right clothes, but not the person in the photographs"
                  @click="drawOne(m)"
                >
                  draw from words
                </button>
                <label class="chip" :class="{ off: !!busy }" title="a photo of it, fitted to the canvas like a crop">
                  upload one<input type="file" accept="image/*" hidden :disabled="!!busy" @change="give(m, g.role, $event)" />
                </label>
              </span>
            </div>
            <div v-for="c in open(m)" :key="c.id" class="conflict">
              <span class="eyebrow warn">the photographs disagree</span>
              <span>{{ c.says }}</span>
              <span class="keeps">
                <button v-for="o in choices(c)" :key="o.label" class="chip" :disabled="!!busy" @click="resolve(c, o.resolution)">{{ o.label }}</button>
              </span>
            </div>
          </div>
        </div>
        <div class="bottom mono">
          <span class="dim">in {{ inShots(m.slug) }} shots · {{ m.noun }}, {{ m.possessive }}</span>
          <span class="acts">
            <button
              v-if="!photographic(m)"
              class="link"
              :disabled="!!busy || !!drawing"
              :title="drawn(m)?.artifact ? 'draw it again with a new seed; the new picture replaces this one' : 'draw it from its words with the editor'"
              @click="drawOne(m)"
            >
              {{ inDrawing(m) ? 'drawing…' : drawn(m)?.artifact ? 'regenerate' : 'generate reference' }}
            </button>
            <button class="link" @click="$emit('go', 'assets')">references</button>
            <button class="link" @click="edit(m)">edit →</button>
          </span>
        </div>
      </article>
    </div>
  </div>
</template>

<style scoped>
.bible,
.editor {
  display: flex;
  flex-direction: column;
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

.nav > .mono {
  font-size: var(--text-xs);
}

.big {
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

.field,
.stacked > label {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.card textarea.prose {
  font-family: var(--font-body);
  font-weight: var(--weight-light);
  line-height: 1.5;
  padding: var(--space-1) var(--space-2);
}

.trait-row {
  display: grid;
  grid-template-columns: 104px minmax(0, 2fr) minmax(0, 1fr) 16px;
  gap: var(--space-1);
  align-items: center;
}

.light-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 16px;
  gap: var(--space-1);
  align-items: center;
}

.trait-row select {
  background: var(--bg-raised);
  font-size: var(--text-xs);
}

.add {
  align-self: flex-start;
  font-size: var(--text-xs);
}

.refs {
  padding: 0 var(--space-2) var(--space-2);
  display: flex;
  flex-direction: column;
}

.ref {
  display: grid;
  grid-template-columns: 40px minmax(0, 1fr) auto;
  gap: var(--space-2);
  align-items: center;
  padding: var(--space-1) 0;
  border-bottom: 1px solid var(--border);
  font-size: var(--text-xs);
}

.ref img {
  width: 40px;
  height: 40px;
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
}

.pad {
  padding: var(--space-1) 0;
}

.small {
  font-size: 11px;
}

.row {
  display: flex;
  justify-content: space-between;
  gap: var(--space-1);
}

.keeps {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}

.keeps .chip {
  font-size: 10px;
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

.acts {
  display: flex;
  align-items: center;
  gap: 12px;
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

.leadpic {
  position: relative;
  display: block;
}

.leadpic .lead {
  display: block;
  width: 100%;
}

.gen {
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

.gen.stale {
  background: var(--status-error);
}

.named {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  font-size: var(--text-sm);
}

.named .eyebrow {
  font-size: 10px;
}

.flags {
  display: flex;
  gap: 4px;
}

.source {
  font-size: 10px;
  letter-spacing: 0.06em;
  padding: 2px 6px;
  border-radius: 2px;
  border: 1px solid var(--border-strong);
  color: var(--text-secondary);
}

.source.photo,
.source.hand {
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

.lead {
  aspect-ratio: 3 / 4;
}

.empty {
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  color: var(--text-tertiary);
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

.find {
  margin: 0;
  font-size: var(--text-xs);
  line-height: 1.5;
  font-weight: var(--weight-light);
  color: var(--text-secondary);
}

.traits {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.traits .eyebrow,
.conflict .eyebrow {
  font-size: 10px;
}

.list {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}

.trait {
  font-size: 10px;
  padding: 1px 6px;
  border: 1px solid var(--border-strong);
  border-radius: 2px;
  white-space: nowrap;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
}

.plain {
  font-size: var(--text-xs);
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

.gap label.chip {
  cursor: pointer;
  border: 1px solid var(--border);
  border-radius: var(--radius-btn);
}

.gap label.chip.off {
  pointer-events: none;
  opacity: 0.5;
}

.conflict.settled {
  border-color: var(--border);
  color: var(--text-secondary);
}

.warn {
  color: var(--status-degraded);
}
</style>
