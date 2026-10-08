<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import type { Dot, TimelineShot, TimelineTrack } from '../../api'
import { mmss, pad } from '../../production'

export type Zoom = 'fit' | 4 | 8 | 16

const props = defineProps<{
  shots: TimelineShot[]
  tracks: TimelineTrack[]
  starts: number[]
  total: number
  open: number
  focus: string | null
  /** Read on every frame of playback; passed as a getter so a new time is no new prop to redraw for. */
  clock: { t: number }
  summary: string
  zoom: Zoom
}>()
const emit = defineEmits<{ seek: [t: number, member: string | null]; zoom: [z: Zoom] }>()

const ZOOMS: { z: Zoom; label: string }[] = [
  { z: 'fit', label: 'FIT' },
  { z: 4, label: '2×' },
  { z: 8, label: '4×' },
  { z: 16, label: '8×' },
]
const KIND: Record<string, string> = { person: 'CHARACTER', animal: 'CHARACTER', environment: 'LOCATION', object: 'PROP', style: 'STYLE' }
// Blocks, runs and ticks this far outside the scrolled view are still drawn, so a fast scroll shows no gap.
const OVERDRAW = 400

const scroller = ref<HTMLElement | null>(null)
const head = ref<HTMLElement | null>(null)
const line = ref<HTMLElement | null>(null)
const width = ref(0)
const scrollLeft = ref(0)
let observer: ResizeObserver | undefined
let frame = 0
onMounted(() => {
  observer = new ResizeObserver(() => (width.value = scroller.value?.clientWidth ?? 0))
  if (scroller.value) observer.observe(scroller.value)
  width.value = scroller.value?.clientWidth ?? 0
  place()
})
onUnmounted(() => {
  observer?.disconnect()
  cancelAnimationFrame(frame)
})
const onScroll = () => {
  cancelAnimationFrame(frame)
  frame = requestAnimationFrame(() => (scrollLeft.value = scroller.value?.scrollLeft ?? 0))
}

const fit = computed(() => (props.total ? Math.max(width.value - 1, 1) / props.total : 1))
// 2×, 4× and 8× are that many fits, but never under 4, 8 and 16 px a second: a long film's fit is a
// fraction of a pixel a second, and 8× of it would still not show a shot's number.
const z = computed(() => (props.zoom === 'fit' ? fit.value : Math.max(props.zoom, (fit.value * props.zoom) / 2)))
const px = (n: number) => `${n}px`
const end = (i: number) => props.starts[i] + props.shots[i].seconds

/** The first shot ending after `t`. */
function at(t: number): number {
  let lo = 0
  let hi = props.shots.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (end(mid) <= t) lo = mid + 1
    else hi = mid
  }
  return lo
}
const view = computed(() => {
  const t0 = Math.max(0, (scrollLeft.value - OVERDRAW) / z.value)
  const t1 = (scrollLeft.value + width.value + OVERDRAW) / z.value
  return { t0, t1, i0: props.shots.length ? at(t0) : 0, i1: props.shots.length ? at(t1) : -1 }
})

const STEPS = [5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600]
const ticks = computed(() => {
  const step = STEPS.find((s) => s * z.value >= 60) ?? STEPS[STEPS.length - 1]
  const out: { left: string; label: string }[] = []
  for (let t = Math.floor(view.value.t0 / step) * step; t <= Math.min(props.total, view.value.t1); t += step) out.push({ left: px(t * z.value), label: mmss(t) })
  return out
})

const scenes = computed(() => {
  const out: { name: string; a: number; b: number }[] = []
  props.shots.forEach((s, i) => {
    const last = out[out.length - 1]
    if (last && last.name === s.scene) last.b = i
    else out.push({ name: s.scene, a: i, b: i })
  })
  return out
})
const sceneBand = computed(() =>
  scenes.value
    .filter((s) => s.b >= view.value.i0 && s.a <= view.value.i1)
    .map((s) => ({ name: s.name, left: px(props.starts[s.a] * z.value), width: px((end(s.b) - props.starts[s.a]) * z.value) })),
)

const PIPE: Record<Dot, string> = {
  done: 'var(--text-secondary)',
  review: 'var(--status-degraded)',
  stale: 'var(--status-error)',
  pending: 'var(--border)',
}
const blocks = computed(() => {
  const out = []
  for (let i = view.value.i0; i <= view.value.i1; i++) {
    const s = props.shots[i]
    const w = s.seconds * z.value - 2
    const on = i === props.open
    out.push({
      key: s.id,
      title: `${pad(s.chapter + 1, 2)}.${pad(s.position + 1, 3)} · ${s.seconds.toFixed(1)} s · ${s.action}`,
      label: w >= 40 ? `${pad(s.position + 1, 3)}${s.opening === 'chain' ? ' │' : ''}` : '',
      style: {
        left: px(props.starts[i] * z.value + 1),
        width: px(Math.max(w, 1)),
        background: on ? 'var(--bg-raised)' : s.pipe[2] === 'done' ? 'oklch(24% 0.02 260)' : 'var(--bg-base)',
        borderColor: on ? 'var(--accent)' : s.waiting ? 'var(--status-degraded)' : 'var(--border-strong)',
        padding: w >= 12 ? '4px' : '4px 0',
      },
      color: on ? 'var(--accent)' : 'var(--text-secondary)',
      // At FIT a long film's blocks are a pixel wide; drawn bare, 1,084 shots took the timeline from 10,704 nodes to 5,284.
      pipe: w >= 6 ? s.pipe.map((p) => PIPE[p]) : [],
    })
  }
  return out
})

const MARK: Record<string, { background: string; border: string; animation?: string }> = {
  done: { background: 'var(--accent)', border: '0' },
  flag: { background: 'var(--status-degraded)', border: '0' },
  running: { background: 'var(--status-building)', border: '0', animation: 'pulse 1.4s ease-in-out infinite' },
  pending: { background: 'var(--bg-surface)', border: '1.5px dashed var(--accent)' },
  failed: { background: 'var(--bg-surface)', border: '1.5px solid var(--status-error)' },
}
const rows = computed(() =>
  props.tracks.map((t) => {
    const { i0, i1 } = view.value
    const runs = t.runs
      .filter((r) => r.b >= i0 && r.a <= i1)
      .map((r) => {
        const left = props.starts[r.a] * z.value
        const w = end(r.b) * z.value - left - 2
        if (r.gap) return { key: `g${r.a}`, gap: true, title: `${t.member} off screen · look kept`, label: '', style: { left: px(left), width: px(w) } }
        const flagged = r.image === 'flag'
        return {
          key: `r${r.a}`,
          gap: false,
          title: `${t.member} · ${r.label}`,
          label: w >= 36 ? `${r.revert ? '↩ ' : ''}${r.label}` : '',
          style: {
            left: px(left + 1),
            width: px(Math.max(w, 1)),
            borderColor: flagged ? 'var(--status-degraded)' : 'var(--border-strong)',
            paddingLeft: r.mark ? '16px' : '4px',
            color: r.image === 'done' ? 'var(--text-secondary)' : 'var(--text-tertiary)',
          },
        }
      })
    const marks = t.marks
      .filter((m) => m.at >= i0 && m.at <= i1)
      .map((m) => ({ key: m.at, title: `${t.member} · ${m.label} · ${m.words}`, style: { left: px(props.starts[m.at] * z.value + 3), ...MARK[m.image] } }))
    return { member: t.member, meta: `${KIND[t.kind] ?? t.kind.toUpperCase()}${t.marks.length ? ` · ${t.marks.length} change${t.marks.length > 1 ? 's' : ''}` : ''}`, runs, marks }
  }),
)

const dense = computed(() => props.shots.length * 2 > width.value)
const overview = computed(() =>
  props.shots.map((s, i) => ({
    key: s.id,
    title: `${pad(s.chapter + 1, 2)}.${pad(s.position + 1, 3)} · ${s.seconds.toFixed(1)} s`,
    style: {
      flex: `${s.seconds} 1 0`,
      background: s.pipe[2] === 'done' ? 'var(--text-primary)' : s.waiting ? 'var(--status-degraded)' : s.pipe[0] === 'done' ? 'var(--text-secondary)' : 'var(--bg-raised)',
      outline: `1px solid ${i === props.open ? 'var(--accent)' : 'transparent'}`,
    },
  })),
)

function seek(event: MouseEvent) {
  const rect = (event.currentTarget as HTMLElement).getBoundingClientRect()
  const t = Math.max(0, Math.min(props.total - 0.01, (event.clientX - rect.left) / z.value))
  const row = (event.target as HTMLElement).closest?.('[data-e]')
  emit('seek', t, row ? row.getAttribute('data-e') : null)
}
const pick = (i: number) => emit('seek', props.starts[i], null)

/** The playhead moves on every frame of playback: it is set on the element, not drawn by the template. */
function place() {
  const left = px(props.clock.t * z.value)
  if (head.value) head.value.style.left = left
  if (line.value) line.value.style.left = left
  const el = scroller.value
  const at = props.clock.t * z.value
  if (el && (at < el.scrollLeft + 16 || at > el.scrollLeft + el.clientWidth - 48)) el.scrollLeft = Math.max(0, at - el.clientWidth / 3)
}
watch(() => [props.clock.t, z.value], place, { flush: 'post' })
</script>

<template>
  <section class="timeline" data-screen-label="Timeline">
    <div class="top">
      <span class="named"><h5>Timeline</h5><span class="summary">{{ summary }}</span></span>
      <div class="zoom">
        <span class="eyebrow">ZOOM</span>
        <div class="seg">
          <button v-for="o in ZOOMS" :key="o.label" :class="{ on: zoom === o.z }" @click="emit('zoom', o.z)">{{ o.label }}</button>
        </div>
      </div>
    </div>
    <div class="overview">
      <div class="strip" :class="{ dense }">
        <span v-for="(o, i) in overview" :key="o.key" :title="o.title" :style="o.style" @click="pick(i)" />
      </div>
      <div class="ends">
        <span>00:00</span>
        <span class="keys"><span>▮ clip rendered</span><span class="kf">▮ keyframe ready</span><span class="wait">▮ waiting on a state image</span></span>
        <span>{{ mmss(total) }}</span>
      </div>
    </div>
    <div class="grid">
      <div class="labels">
        <span class="lab">TIME</span>
        <span class="lab">SCENE</span>
        <span class="lab shots">SHOTS</span>
        <span v-for="t in rows" :key="t.member" class="track-label" :class="{ on: focus === t.member }">
          <span class="e">{{ t.member }}</span><span class="meta">{{ t.meta }}</span>
        </span>
      </div>
      <div ref="scroller" class="scroller" @scroll.passive="onScroll">
        <div class="inner" :style="{ width: px(total * z) }" @click="seek">
          <div class="ruler">
            <span v-for="t in ticks" :key="t.left" class="tick" :style="{ left: t.left }">{{ t.label }}</span>
          </div>
          <div class="scenes">
            <span v-for="s in sceneBand" :key="s.left" class="scene" :style="{ left: s.left, width: s.width }">{{ s.name }}</span>
          </div>
          <div class="blocks">
            <span v-for="b in blocks" :key="b.key" class="block" :title="b.title" :style="b.style">
              <template v-if="b.pipe.length">
                <span class="n" :style="{ color: b.color }">{{ b.label }}</span>
                <span class="pipe"><span v-for="(c, j) in b.pipe" :key="j" :style="{ background: c }" /></span>
              </template>
            </span>
          </div>
          <div v-for="t in rows" :key="t.member" class="track" :class="{ on: focus === t.member }" :data-e="t.member">
            <span v-for="r in t.runs" :key="r.key" :class="r.gap ? 'gap' : 'run'" :data-e="t.member" :title="r.title" :style="r.style">{{ r.label }}</span>
            <span v-for="m in t.marks" :key="m.key" class="mark" :data-e="t.member" :title="m.title" :style="m.style" />
          </div>
          <span ref="line" class="playline" />
          <span ref="head" class="playhead" />
        </div>
      </div>
    </div>
    <div class="legend">
      <span><i class="k-run" />in shot · state label</span>
      <span><i class="k-mark" />changes here</span>
      <span><i class="k-todo" />image to generate</span>
      <span><i class="k-flag" />below QA gate</span>
      <span><i class="k-gap" />off screen · state kept</span>
      <span><b>│</b>opens on previous last frame</span>
      <span class="hint">click anywhere to seek · a track row also focuses that entity</span>
    </div>
  </section>
</template>

<style scoped>
.timeline {
  border: 1px solid var(--border);
  border-radius: var(--radius-card);
  background: var(--bg-surface);
  min-width: 0;
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
  align-items: baseline;
  gap: var(--space-2);
  flex-wrap: wrap;
}

.named h5 {
  font-family: var(--font-heading);
  font-weight: var(--weight-semibold);
  font-size: var(--text-base);
  margin: 0;
}

.summary {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-tertiary);
  white-space: nowrap;
}

.zoom {
  display: flex;
  align-items: center;
  gap: var(--space-1);
}

.eyebrow {
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.08em;
  color: var(--text-tertiary);
}

.seg {
  display: flex;
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-btn);
  overflow: hidden;
}

.seg button {
  background: transparent;
  color: var(--text-secondary);
  border: 0;
  border-radius: 0;
  padding: 4px 8px;
  font-size: 10px;
  font-family: var(--font-mono);
  font-weight: var(--weight-regular);
  letter-spacing: 0.06em;
  white-space: nowrap;
}

.seg button.on {
  background: var(--bg-raised);
  color: var(--accent);
}

.overview {
  padding: var(--space-1) var(--space-2);
  border-bottom: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.strip {
  display: flex;
  gap: 1px;
  height: 8px;
}

.strip span {
  cursor: pointer;
  min-width: 1px;
  outline-offset: 1px;
}

.strip.dense {
  gap: 0;
}

.strip.dense span {
  min-width: 0;
}

.ends {
  display: flex;
  justify-content: space-between;
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--text-tertiary);
}

.keys {
  display: flex;
  gap: var(--space-2);
}

.keys .kf {
  color: var(--text-secondary);
}

.keys .wait {
  color: var(--status-degraded);
}

.grid {
  display: grid;
  grid-template-columns: 144px minmax(0, 1fr);
}

.labels {
  border-right: 1px solid var(--border-strong);
  display: flex;
  flex-direction: column;
}

.lab {
  height: 24px;
  padding: 0 var(--space-2);
  display: flex;
  align-items: center;
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.08em;
  color: var(--text-tertiary);
  border-bottom: 1px solid var(--border);
}

.lab.shots {
  height: 56px;
  border-bottom-color: var(--border-strong);
}

.track-label {
  height: 32px;
  padding: 0 var(--space-2);
  display: flex;
  flex-direction: column;
  justify-content: center;
  border-bottom: 1px solid var(--border);
  min-width: 0;
}

.track-label.on,
.track.on {
  background: var(--bg-raised);
}

.track-label .e {
  font-family: var(--font-mono);
  font-size: 11px;
  line-height: 1.3;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.track-label .meta {
  font-family: var(--font-mono);
  font-size: 9px;
  line-height: 1.3;
  letter-spacing: 0.06em;
  color: var(--text-tertiary);
  white-space: nowrap;
}

.scroller {
  overflow-x: auto;
  overflow-y: hidden;
  min-width: 0;
}

.inner {
  position: relative;
  cursor: crosshair;
}

.ruler,
.scenes {
  position: relative;
  height: 24px;
  border-bottom: 1px solid var(--border);
}

.tick {
  position: absolute;
  top: 0;
  bottom: 0;
  border-left: 1px solid var(--border);
  padding-left: 4px;
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--text-tertiary);
  display: flex;
  align-items: center;
  white-space: nowrap;
}

.scene {
  position: absolute;
  top: 4px;
  bottom: 4px;
  border-left: 2px solid var(--border-strong);
  padding-left: 4px;
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--text-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  display: flex;
  align-items: center;
}

.blocks {
  position: relative;
  height: 56px;
  border-bottom: 1px solid var(--border-strong);
}

.block {
  position: absolute;
  top: 4px;
  bottom: 4px;
  border: 1px solid;
  border-radius: 2px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

.block .n {
  font-family: var(--font-mono);
  font-size: 10px;
  line-height: 1.4;
  white-space: nowrap;
  overflow: hidden;
}

.pipe {
  display: flex;
  gap: 1px;
}

.pipe span {
  flex: 1;
  height: 3px;
  border-radius: 1px;
}

.track {
  position: relative;
  height: 32px;
  border-bottom: 1px solid var(--border);
}

.run {
  position: absolute;
  top: 8px;
  height: 16px;
  background: var(--bg-raised);
  border: 1px solid;
  border-radius: 2px;
  padding-right: 4px;
  font-family: var(--font-mono);
  font-size: 10px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 14px;
}

.gap {
  position: absolute;
  top: 15px;
  height: 0;
  border-top: 1px dashed var(--border-strong);
}

.mark {
  position: absolute;
  top: 11px;
  width: 10px;
  height: 10px;
  border-radius: 2px;
}

.playline {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 1px;
  background: var(--accent);
  pointer-events: none;
}

.playhead {
  position: absolute;
  top: 0;
  width: 9px;
  height: 9px;
  margin-left: -4px;
  background: var(--accent);
  pointer-events: none;
}

.legend {
  padding: var(--space-1) var(--space-2);
  border-top: 1px solid var(--border);
  display: flex;
  gap: var(--space-1) var(--space-3);
  flex-wrap: wrap;
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-tertiary);
  align-items: center;
}

.legend > span {
  display: flex;
  align-items: center;
  gap: var(--space-1);
}

.legend i {
  display: inline-block;
  border-radius: 2px;
}

.k-run {
  width: 24px;
  height: 12px;
  border: 1px solid var(--border-strong);
  background: var(--bg-raised);
}

.k-mark,
.k-todo,
.k-flag {
  width: 10px;
  height: 10px;
}

.k-mark {
  background: var(--accent);
}

.k-todo {
  border: 1.5px dashed var(--accent);
}

.k-flag {
  background: var(--status-degraded);
}

.k-gap {
  width: 24px;
  border-top: 1px dashed var(--border-strong);
}

.legend b {
  color: var(--accent);
  font-weight: var(--weight-regular);
}

.legend .hint {
  margin-left: auto;
}
</style>
