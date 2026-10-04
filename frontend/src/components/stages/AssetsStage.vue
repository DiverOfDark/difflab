<script setup lang="ts">
import { computed, ref } from 'vue'
import { detachRef, drawReference, drawVariants, giveReference, outputUrl, pickVariant, upload, type MemberInfo, type Production } from '../../api'
import { GENERATED, PHASE_LABEL, drawingRun, phaseOf, phaseOfAll, type Phase } from '../../production'

const props = defineProps<{
  production: Production
  busy: string
  act: (name: string, what: () => Promise<unknown>) => Promise<void>
}>()
const detach = (refId: number) => props.act(`ref ${refId}`, () => detachRef(refId))
const drawing = computed(() => drawingRun(props.production))
const photographic = (m: MemberInfo) => m.references.some((r) => r.role !== GENERATED && r.artifact && r.present)
const drawn = (m: MemberInfo) => m.references.find((r) => r.role === GENERATED)
const drawOne = (m: MemberInfo) => props.act(`draw ${m.id}`, () => drawReference(m.id))
const describeAndDraw = (m: MemberInfo) => props.act(`draw ${m.id}`, () => drawReference(m.id, undefined, true))
const drawThree = (m: MemberInfo) => props.act(`draw ${m.id}`, () => drawVariants(m.id))
const pick = (m: MemberInfo, seed: number) => props.act(`draw ${m.id}`, () => pickVariant(m.id, seed))
/** The role an uploaded picture stands in for: a place's whole view, anyone else's body. */
const GIVEN_ROLE: Record<string, string> = { environment: 'environment', person: 'body', animal: 'view', object: 'object' }
const give = (m: MemberInfo, event: Event) => {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  input.value = ''
  if (file)
    props.act(`give ${m.id}`, async () => {
      const [asset] = await upload([file])
      await giveReference(m.id, asset.path, GIVEN_ROLE[m.kind] ?? 'body')
    })
}
/** Every keyframe that casts the member: its pictures beside what was made from them. */
const keyframesOf = (m: MemberInfo) =>
  props.production.chapters
    .flatMap((c) => c.shots)
    .filter((s) => ((s.plan.cast as string[] | undefined) ?? []).includes(m.slug) && s.keyframe.artifact && s.keyframe.present)

const filter = ref<'all' | Phase>('all')
const filters = computed(() => {
  const members = props.production.members
  const of = (p: Phase) => members.filter((m) => m.references.some((r) => phaseOf(r.state) === p)).length
  return [
    { id: 'all' as const, label: 'all', count: members.length },
    { id: 'review' as const, label: 'needs review', count: of('review') },
    { id: 'stale' as const, label: 'stale', count: of('stale') },
    { id: 'pending' as const, label: 'missing', count: of('pending') },
  ]
})
const rows = computed(() =>
  props.production.members.filter((m) => filter.value === 'all' || m.references.some((r) => phaseOf(r.state) === filter.value)),
)
const phase = (m: MemberInfo) => phaseOfAll(m.references)
/** The judge's verdict as a line: its score against the gate, then what it saw. */
function verdict(review: string): string {
  try {
    const v = JSON.parse(review)
    const said = [...(v.issues ?? []), v.notes].filter(Boolean).join('; ')
    return `judge ${Number(v.score).toFixed(2)} · gate ${v.gate}${said ? ` · ${said}` : ''}`
  } catch {
    return review
  }
}
const name = (path: string) => path.split('/').at(-1)!.replace(/^[0-9a-f]{16}-/, '')
</script>

<template>
  <div class="assets">
    <div class="bar-row">
      <div class="chips">
        <button v-for="f in filters" :key="f.id" class="chip" :class="{ on: filter === f.id }" @click="filter = f.id">
          {{ f.label }} <span class="n">{{ f.count }}</span>
        </button>
      </div>
      <span class="note">cut by the plan run's cut_crops · a re-plan re-cuts only what moved · GEN: drawn by the editor, which a re-plan leaves alone</span>
    </div>

    <p v-if="!rows.length" class="band hint">Nothing here.</p>

    <article v-for="m in rows" :key="m.id" class="card" :class="phase(m)">
      <div class="top">
        <span class="named"><i class="dot" :class="phase(m)" /><span class="mono">{{ m.slug }}</span><span class="eyebrow">{{ m.kind }}</span></span>
        <span class="mono small">
          <button
            v-if="!photographic(m)"
            class="link"
            :disabled="!!busy || !!drawing"
            :title="drawn(m)?.artifact ? 'draw it again with a new seed; the new picture replaces this one' : 'draw it from its words with the editor'"
            @click="drawOne(m)"
          >
            {{ drawn(m)?.artifact ? 'regenerate' : 'generate reference' }}
          </button>
          <button v-if="m.kind === 'environment'" class="link" :disabled="!!busy || !!drawing" title="the 27B writes its look from the story's words, then the editor draws it: over its own words, even edited ones" @click="describeAndDraw(m)">describe from the story &amp; draw</button>
          <button v-if="!photographic(m)" class="link" :disabled="!!busy || !!drawing" title="three drawings on new seeds; pick one as its reference" @click="drawThree(m)">draw 3 to choose</button>
          <label class="link upload" :class="{ off: !!busy }" title="a picture of it of your own, fitted to the canvas like a crop">upload one<input type="file" accept="image/*" hidden @change="give(m, $event)" /></label>
          <span class="dim">{{ m.references.length }} references</span><span class="ph" :class="phase(m)">{{ PHASE_LABEL[phase(m)] }}</span>
        </span>
      </div>
      <div class="inside">
        <div class="tiles">
          <a v-for="r in m.references" :key="r.id" class="tile" :href="r.artifact ? outputUrl(r.artifact) : undefined" target="_blank">
            <span class="pic">
              <img v-if="r.artifact && r.present" class="frame" :src="outputUrl(r.artifact)" :alt="r.role" />
              <span v-else class="frame placeholder mono">{{ r.state }}</span>
              <span class="tag mono" :class="{ on: r.role === 'face', gen: r.role === 'generated' }">{{ r.role === 'generated' ? 'GEN' : r.role.toUpperCase() }}</span>
              <span v-if="r.pinned" class="pin mono">PINNED</span>
            </span>
            <span class="under mono"><span>{{ r.size.join('×') }}</span><span class="ph" :class="phaseOf(r.state)">{{ r.state }}</span></span>
          </a>
        </div>
        <div class="side">
          <span class="eyebrow">shows · from which photograph</span>
          <div v-for="r in m.references" :key="r.id" class="fact">
            <span class="k">{{ r.role === 'generated' ? `generated · ${m.slug}` : `${r.role} · ${r.shows.join(', ')}` }}</span>
            <span class="v">
              <span v-if="r.role === 'generated'" class="mono">drawn from its words · seed {{ r.seed }}</span>
              <a v-else class="mono" :href="outputUrl(r.photo)" target="_blank" :title="r.photo">{{ name(r.photo) }}</a>
              <button class="link" :disabled="!!busy" title="it does not show this member: let go of it, for good" @click="detach(r.id)">detach</button>
            </span>
          </div>
          <div v-for="r in m.references.filter((x) => x.review)" :key="`review-${r.id}`" class="fact" :class="{ warn: phaseOf(r.state) === 'review' }">
            <span class="k">{{ verdict(r.review) }}</span>
          </div>
          <img v-if="m.references[0]?.photo" class="source frame" :src="outputUrl(m.references[0].photo)" alt="" />
        </div>
      </div>
      <div v-if="m.variants?.length" class="strip">
        <span class="eyebrow">drawings to choose from · the reference is the one you use</span>
        <div class="row-pics">
          <span v-for="v in m.variants" :key="v.seed" class="choice">
            <img v-if="v.frame" class="frame" :src="outputUrl(v.frame)" alt="" />
            <span v-else class="frame placeholder mono">{{ v.state }}</span>
            <button class="link" :disabled="!!busy || !v.frame || drawn(m)?.seed === v.seed" @click="pick(m, v.seed)">
              {{ drawn(m)?.seed === v.seed ? 'in use' : 'use this' }}
            </button>
          </span>
        </div>
      </div>
      <div v-if="keyframesOf(m).length" class="strip">
        <span class="eyebrow">in keyframes · compare with its pictures above</span>
        <div class="row-pics">
          <a v-for="s in keyframesOf(m)" :key="s.id" class="choice" :href="outputUrl(s.keyframe.artifact!)" target="_blank">
            <img class="frame" :src="outputUrl(s.keyframe.artifact!)" alt="" />
            <span class="mono small">shot {{ s.position + 1 }}</span>
          </a>
        </div>
      </div>
    </article>
  </div>
</template>

<style scoped>
.assets {
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

.card.review {
  border-color: var(--status-degraded);
}

.card.stale {
  border-color: var(--status-error);
}

.top {
  padding: var(--space-1) var(--space-2);
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--space-2);
  flex-wrap: wrap;
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

.small {
  display: flex;
  gap: var(--space-2);
  font-size: 11px;
}

.inside {
  padding: var(--space-2);
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(200px, 280px);
  gap: var(--space-3);
  align-items: start;
}

.tiles {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(112px, 1fr));
  gap: var(--space-1);
}

.tile {
  display: flex;
  flex-direction: column;
  gap: 4px;
  text-decoration: none;
  color: inherit;
}

.pic {
  position: relative;
}

.pic .frame {
  aspect-ratio: 3 / 4;
}

.placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  color: var(--text-tertiary);
}

.tag,
.pin {
  position: absolute;
  top: 4px;
  font-size: 9px;
  letter-spacing: 0.06em;
  padding: 1px 4px;
  border-radius: 2px;
  background: var(--bg-base);
  color: var(--text-secondary);
  border: 1px solid var(--border);
}

.tag {
  left: 4px;
}

.tag.gen {
  border-color: var(--accent);
  color: var(--accent);
}

.tag.on {
  background: var(--accent);
  color: var(--accent-fg);
  border-color: var(--accent);
}

.pin {
  right: 4px;
}

.under {
  display: flex;
  justify-content: space-between;
  font-size: 10px;
  color: var(--text-tertiary);
}

.side {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  min-width: 0;
}

.side .eyebrow {
  font-size: 10px;
}

.fact {
  display: flex;
  justify-content: space-between;
  gap: var(--space-2);
  font-size: var(--text-xs);
  line-height: 1.45;
  padding: 4px 0;
  border-bottom: 1px solid var(--border);
}

.fact .k {
  font-weight: var(--weight-light);
}

.fact .v {
  display: flex;
  gap: var(--space-1);
  font-size: 11px;
  color: var(--text-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 50%;
  text-decoration: none;
}

.fact.warn {
  color: var(--status-degraded);
}

.source {
  max-height: 160px;
  object-fit: contain;
}

@media (max-width: 700px) {
  .inside {
    grid-template-columns: 1fr;
  }
}

.strip {
  padding: var(--space-1) var(--space-2) var(--space-2);
  border-top: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.row-pics {
  display: flex;
  gap: var(--space-1);
  overflow-x: auto;
}

.choice {
  display: flex;
  flex-direction: column;
  gap: 4px;
  align-items: flex-start;
  text-decoration: none;
  color: inherit;
}

.choice .frame {
  height: 140px;
  width: auto;
}

.upload {
  cursor: pointer;
}

.upload.off {
  pointer-events: none;
  opacity: 0.5;
}
</style>
