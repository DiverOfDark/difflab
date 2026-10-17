<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue'
import { outputUrl, readTimeline, type Production, type ShotInfo, type Timeline } from '../../api'
import { mmss, timecode, type StageId } from '../../production'
import ShotInspector, { type Tab } from '../shots/ShotInspector.vue'
import ShotTimeline, { type Zoom } from '../shots/ShotTimeline.vue'

const props = defineProps<{
  production: Production
  chapter: number | null
  focus: number | null
  /** A member to focus as the stage opens: its track and its references in the open shot. */
  member?: string | null
  /** Shots a STALE banner sent here to be shown: marked on the timeline until cleared. */
  affected: number[]
  busy: string
  act: (name: string, what: () => Promise<unknown>) => Promise<void>
}>()
const emit = defineEmits<{ go: [to: StageId, shot?: number | null]; unmark: [] }>()

const timeline = ref<Timeline | null>(null)
const failed = ref('')
async function load() {
  try {
    const found = await readTimeline(props.production.id, props.chapter)
    if (JSON.stringify(found) !== JSON.stringify(timeline.value)) timeline.value = found
    failed.value = ''
  } catch (error) {
    failed.value = String(error)
  }
}
watch(() => [props.production, props.chapter], load, { immediate: true })

const shots = computed(() => timeline.value?.shots ?? [])
const starts = computed(() => {
  let at = 0
  return shots.value.map((s) => {
    const here = at
    at += s.seconds
    return here
  })
})
const total = computed(() => shots.value.reduce((n, s) => n + s.seconds, 0))
const film = computed<ShotInfo[]>(() => props.production.chapters.flatMap((c) => c.shots))
const detail = computed(() => new Map(film.value.map((s) => [s.id, s])))
const inView = computed(() => new Set(shots.value.map((s) => s.id)))
const marked = computed(() => new Set(props.affected))
const markedLabels = computed(() => shots.value.filter((s) => marked.value.has(s.id)).map(label))

const openId = ref<number | null>(props.focus)
watch(
  () => props.focus,
  (id) => {
    if (id !== null && id !== openId.value) {
      openId.value = id
      playT.value = startOf(id)
    }
  },
)
const index = computed(() => Math.max(0, shots.value.findIndex((s) => s.id === openId.value)))
const light = computed(() => shots.value[index.value])
const shot = computed(() => (light.value ? detail.value.get(light.value.id) : undefined))
watch(
  () => light.value?.id,
  (id) => {
    if (id !== undefined && id !== props.focus) emit('go', 'shots', id)
  },
)
const startOf = (id: number) => starts.value[Math.max(0, shots.value.findIndex((s) => s.id === id))] ?? 0

const memberFocus = ref<string | null>(props.member ?? null)
const tab = ref<Tab>(props.production.story_kind === 'manga' ? 'panel' : 'refs')
const zoom = ref<Zoom>('fit')
const said = ref('')

// --- playback -----------------------------------------------------------------------------------
const playT = ref(0)
const clock = { get t() { return playT.value } }
const playing = ref(false)
const video = ref<HTMLVideoElement | null>(null)
const start = computed(() => starts.value[index.value] ?? 0)
const length = computed(() => light.value?.seconds ?? 0)
const inShot = computed(() => Math.min(Math.max(playT.value - start.value, 0), length.value))

function select(i: number, t = starts.value[i]) {
  if (!shots.value[i]) return
  openId.value = shots.value[i].id
  playT.value = t
}
let raf = 0
let last = 0
function tick(now: number) {
  const dt = last ? (now - last) / 1000 : 0
  last = now
  const s = light.value
  if (!playing.value || !s) return
  const v = video.value
  // A clip still loading, or one the browser cannot decode, is held like a still for the shot's length.
  const live = !!s.clip && !!v && v.readyState >= 2 && !v.error
  const t = live ? start.value + v!.currentTime : playT.value + dt
  const ended = live ? v!.ended || v!.currentTime >= clipEnd(v!, s.seconds) : t >= start.value + s.seconds
  if (ended) {
    if (index.value + 1 >= shots.value.length) {
      playing.value = false
      playT.value = total.value - 0.01
      return
    }
    select(index.value + 1)
  } else if (Math.floor(t * 24) !== Math.floor(playT.value * 24)) playT.value = t
  raf = requestAnimationFrame(tick)
}
/** A clip may run shorter than its shot: it ends at whichever comes first. */
const clipEnd = (v: HTMLVideoElement, length: number) => Math.min(v.duration || length, length) - 0.02
function toggle() {
  if (playing.value) {
    playing.value = false
    video.value?.pause()
    return
  }
  if (playT.value >= total.value - 0.05) select(0)
  playing.value = true
  last = 0
  const v = video.value
  // Played from its end, a video starts over: the next shot plays instead.
  if (v && light.value?.clip && v.currentTime >= clipEnd(v, light.value.seconds)) select(Math.min(index.value + 1, shots.value.length - 1))
  else v?.play().catch(() => undefined)
  raf = requestAnimationFrame(tick)
}
onUnmounted(() => cancelAnimationFrame(raf))
function loaded() {
  const v = video.value
  if (!v) return
  v.currentTime = inShot.value
  if (playing.value) v.play().catch(() => undefined)
}
watch(playing, (on) => {
  if (!on) video.value?.pause()
})

function seek(t: number, member: string | null) {
  const i = shots.value.findIndex((s, k) => t >= starts.value[k] && t < starts.value[k] + s.seconds)
  const k = i < 0 ? shots.value.length - 1 : i
  const same = k === index.value
  select(k, t)
  if (same && video.value && light.value?.clip) video.value.currentTime = t - starts.value[k]
  memberFocus.value = member
  if (member) tab.value = 'refs'
}
const step = (by: number) => select((index.value + by + shots.value.length) % shots.value.length)
const open = (id: number, member: string | null) => {
  const i = shots.value.findIndex((s) => s.id === id)
  if (i >= 0) select(i)
  memberFocus.value = member
  tab.value = 'refs'
}

// --- what the viewer says -----------------------------------------------------------------------
const before = computed(() => {
  const prev = shots.value[index.value - 1]
  return prev && light.value && prev.chapter === light.value.chapter ? prev : null
})
const label = (s: { chapter: number; position: number }) => `${String(s.chapter + 1).padStart(2, '0')}.${String(s.position + 1).padStart(3, '0')}`
const status = computed(() => {
  const s = light.value
  if (!s) return { text: '', color: '' }
  if (s.clip) return { text: `CLIP · ${Math.round(s.seconds)} S · WITH AUDIO`, color: 'var(--status-ready)' }
  if (s.still) return { text: 'KEYFRAME · NOT RENDERED', color: 'var(--text-secondary)' }
  return { text: 'NOT DRAWN', color: 'var(--text-tertiary)' }
})
const openTag = computed(() => {
  const s = light.value
  if (!s) return ''
  if (s.opening === 'chain') return `OPENS ON ${before.value ? label(before.value) : '—'} LAST FRAME`
  return s.opening === 'refs' ? 'REFERENCES ONLY' : 'KEYFRAME'
})
const openLabel = computed(() => {
  const s = light.value
  if (!s || s.clip || s.still) return ''
  if (s.opening === 'chain') return `last frame of ${before.value ? label(before.value) : '—'}`
  if (s.opening === 'refs') return 'no first frame'
  return s.waiting ? `waiting for ${s.waiting} state image${s.waiting > 1 ? 's' : ''}` : 'not composed yet'
})
const box = computed(() => {
  const s = light.value
  const dashed = s && (s.opening === 'refs' || (s.opening === 'keyframe' && s.waiting))
  return {
    borderStyle: dashed ? 'dashed' : 'solid',
    borderColor: s?.opening === 'chain' ? 'var(--accent)' : 'var(--border)',
    background: s?.clip ? 'var(--video)' : s?.opening === 'refs' ? 'transparent' : s?.opening === 'chain' || s?.waiting ? 'var(--bg-base)' : 'oklch(22% 0.02 260)',
  }
})
const renderSeconds = computed(() => {
  const r = props.production.timings?.render_shot
  if (!r || !light.value) return 0
  return Math.round(r.per_second ? r.per_second * light.value.seconds : r.seconds)
})
const note = computed(() =>
  light.value?.clip
    ? `Plays the rendered clip with its own sound. Changing anything in this shot re-renders it (~${renderSeconds.value} s).`
    : 'Shows the opening frame held for the shot length. Clips play once Render has run.',
)

const summary = computed(() => {
  const n = shots.value.length
  const rendered = shots.value.filter((s) => s.pipe[2] === 'done').length
  const toDraw = (timeline.value?.tracks ?? []).reduce((sum, t) => sum + t.marks.filter((m) => m.image === 'pending' || m.image === 'failed').length, 0)
  const scope = props.chapter === null ? (props.production.chapters.length > 1 ? 'all chapters' : `ch ${String((props.production.chapters[0]?.position ?? 0) + 1).padStart(2, '0')}`) : `ch ${String(props.chapter + 1).padStart(2, '0')}`
  return `${scope} · ${n} shots · ${mmss(total.value)} · ${rendered} clips rendered · ${toDraw} state images to draw`
})
</script>

<template>
  <p v-if="failed" class="band error">{{ failed }}</p>
  <p v-if="timeline && !shots.length" class="band hint">No shots yet. Plan the {{ chapter === null ? 'production' : 'chapter' }}: the director writes the shots and who is in each.</p>
  <div v-else-if="light" class="shots" data-screen-label="Shots timeline">
    <p v-if="said" class="band hint">{{ said }} <button class="link" @click="said = ''">dismiss</button></p>
    <p v-if="markedLabels.length" class="band hint marked">
      Marked on the timeline: {{ markedLabels.length }} affected shot{{ markedLabels.length > 1 ? 's' : '' }},
      {{ markedLabels.length > 1 ? `${markedLabels[0]} → ${markedLabels[markedLabels.length - 1]}` : markedLabels[0] }}.
      <button class="link" @click="emit('unmark')">clear</button>
    </p>
    <div class="upper">
      <section class="viewer">
        <div class="topbar">
          <span class="id">{{ label(light) }} <span class="dim">· {{ light.scene || 'no scene' }}</span></span>
          <span class="status" :style="{ color: status.color }">{{ status.text }}</span>
        </div>
        <div class="inside">
          <div class="screen" :style="box">
            <video
              v-if="light.clip"
              :key="light.clip"
              ref="video"
              class="media"
              :src="outputUrl(light.clip)"
              :poster="light.still ? outputUrl(light.still) : undefined"
              preload="auto"
              playsinline
              @loadedmetadata="loaded"
            />
            <img v-else-if="light.still" class="media" :src="outputUrl(light.still)" alt="" />
            <span v-else class="blank">{{ openLabel }}</span>
            <span class="tag">{{ openTag }}</span>
            <span class="tc">{{ timecode(inShot) }} / {{ mmss(light.seconds) }}</span>
          </div>
          <div class="progress"><i :style="{ width: `${length ? (inShot / length) * 100 : 0}%` }" /></div>
          <div class="controls">
            <div class="transport">
              <button class="step" title="Previous shot" @click="step(-1)">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="19 20 9 12 19 4 19 20" /><line x1="5" y1="19" x2="5" y2="5" /></svg>
              </button>
              <button class="play" :title="playing ? 'Pause' : 'Play'" @click="toggle">
                <svg v-if="playing" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16" /><rect x="14" y="4" width="4" height="16" /></svg>
                <svg v-else width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="6 4 20 12 6 20 6 4" /></svg>
              </button>
              <button class="step" title="Next shot" @click="step(1)">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 4 15 12 5 20 5 4" /><line x1="19" y1="5" x2="19" y2="19" /></svg>
              </button>
            </div>
            <span class="now">{{ timecode(playT) }} <span class="dim">/ {{ mmss(total) }}</span></span>
          </div>
          <span class="note">{{ note }}</span>
        </div>
      </section>

      <ShotInspector
        v-if="shot"
        :production="production"
        :shot="shot"
        :film="film"
        :in-view="inView"
        :focus="memberFocus"
        :tab="tab"
        :busy="busy"
        :act="act"
        @tab="tab = $event"
        @select="open"
        @next="step(1)"
        @said="said = $event"
      />
      <section v-else class="viewer"><p class="note pad">This shot is not in the production as last read; it shows after the next refresh.</p></section>
    </div>

    <ShotTimeline
      v-if="timeline"
      :shots="shots"
      :tracks="timeline.tracks"
      :starts="starts"
      :total="total"
      :open="index"
      :focus="memberFocus"
      :clock="clock"
      :summary="summary"
      :marked="marked"
      :zoom="zoom"
      @seek="seek"
      @zoom="zoom = $event"
    />
  </div>
</template>

<style scoped>
.shots {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.upper {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr));
  gap: var(--space-2);
  align-items: start;
}

.viewer {
  border: 1px solid var(--border);
  border-radius: var(--radius-card);
  background: var(--bg-surface);
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

.id {
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  white-space: nowrap;
}

.dim {
  color: var(--text-tertiary);
}

.status {
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.06em;
  white-space: nowrap;
}

.inside {
  padding: var(--space-2);
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

/* A portrait canvas is held inside 16:9 rather than let grow the page past the timeline. */
.screen {
  aspect-ratio: 16 / 9;
  border: 1px solid;
  border-radius: var(--radius-input);
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.media {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}

.blank {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-tertiary);
}

.tag {
  position: absolute;
  top: 8px;
  left: 8px;
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.06em;
  padding: 1px 6px;
  background: var(--bg-base);
  border: 1px solid var(--border);
  border-radius: var(--radius-input);
  color: var(--text-secondary);
}

.tc {
  position: absolute;
  bottom: 8px;
  right: 8px;
  font-family: var(--font-mono);
  font-size: 11px;
  padding: 1px 6px;
  background: var(--bg-base);
  border-radius: var(--radius-input);
  color: var(--text-primary);
}

.progress {
  height: 2px;
  background: var(--bg-raised);
  border-radius: 1px;
  overflow: hidden;
}

.progress i {
  display: block;
  height: 100%;
  background: var(--accent);
}

.controls {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--space-2);
  flex-wrap: wrap;
}

.transport {
  display: flex;
  gap: var(--space-1);
}

.transport button {
  display: flex;
  padding: 4px 8px;
  border-radius: var(--radius-btn);
  color: var(--text-secondary);
}

.transport .play {
  background: var(--accent);
  border: 0;
  padding: 4px 16px;
  color: var(--accent-fg);
}

.transport .play:hover {
  color: var(--accent-fg);
}

.now {
  font-family: var(--font-mono);
  font-size: var(--text-sm);
  white-space: nowrap;
}

.pad {
  padding: var(--space-2);
}

.note {
  font-size: var(--text-xs);
  font-weight: var(--weight-light);
  color: var(--text-secondary);
  line-height: 1.45;
}
</style>
