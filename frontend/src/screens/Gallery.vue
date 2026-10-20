<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { RouterLink } from 'vue-router'
import { deleteResult, listGallery, thumbUrl, type GalleryItem } from '../api'
import { state } from '../store'

const ARMED_MS = 4000
// Chrome drops all but the first of several downloads started in the same tick.
const DOWNLOAD_GAP_MS = 250

const items = ref<GalleryItem[]>([])
const loaded = ref(false)
const problem = ref('')
const only = ref<'all' | 'image' | 'video'>('all')
const opened = ref<number | null>(null)
const picked = ref(new Set<number>())
const anchor = ref<number | null>(null)
const hovered = ref<number | null>(null)
const armed = ref<'one' | 'many' | null>(null)
const deleting = ref(false)
let timer: ReturnType<typeof setTimeout> | undefined

async function load() {
  try {
    items.value = await listGallery()
    problem.value = ''
  } catch (failure) {
    problem.value = failure instanceof Error ? failure.message : String(failure)
  }
  loaded.value = true
  const present = new Set(items.value.map((one) => one.id))
  picked.value = new Set([...picked.value].filter((id) => present.has(id)))
}

const shown = computed(() => (only.value === 'all' ? items.value : items.value.filter((one) => one.kind === only.value)))
const counts = computed(() => ({
  all: items.value.length,
  image: items.value.filter((one) => one.kind === 'image').length,
  video: items.value.filter((one) => one.kind === 'video').length,
}))

const dayOf = (t: number) => new Date(t * 1000).toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'long', year: 'numeric' })
const days = computed(() => {
  const out: { day: string; entries: { item: GalleryItem; index: number }[] }[] = []
  shown.value.forEach((item, index) => {
    const day = dayOf(item.created)
    const last = out[out.length - 1]
    if (last?.day === day) last.entries.push({ item, index })
    else out.push({ day, entries: [{ item, index }] })
  })
  return out
})

const url = (item: GalleryItem) => `/output/${item.path}`
const nameOf = (item: GalleryItem) => item.path.split('/').pop() ?? item.path
const still = (item: GalleryItem) => (item.kind === 'image' ? item.path : item.poster)
const size = (bytes: number) => (bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${(bytes / 1e6).toFixed(1)} MB`)
const when = (t: number) => new Date(t * 1000).toLocaleString()

const current = computed(() => (opened.value === null ? null : (shown.value[opened.value] ?? null)))
const selecting = computed(() => picked.value.size > 0)
const pickedBytes = computed(() => items.value.filter((one) => picked.value.has(one.id)).reduce((sum, one) => sum + one.bytes, 0))

function toggle(index: number, range: boolean) {
  const next = new Set(picked.value)
  if (range && anchor.value !== null) {
    const [a, b] = [Math.min(anchor.value, index), Math.max(anchor.value, index)]
    for (const one of shown.value.slice(a, b + 1)) next.add(one.id)
  } else {
    const id = shown.value[index].id
    if (next.has(id)) next.delete(id)
    else next.add(id)
  }
  anchor.value = index
  picked.value = next
  disarm()
}

function click(index: number, event: MouseEvent) {
  if (selecting.value || event.shiftKey || event.ctrlKey || event.metaKey) toggle(index, event.shiftKey)
  else opened.value = index
}

function pickAll() {
  picked.value = new Set(shown.value.map((one) => one.id))
}

function clear() {
  picked.value = new Set()
  anchor.value = null
  disarm()
}

function step(by: number) {
  if (opened.value === null || !shown.value.length) return
  opened.value = (opened.value + by + shown.value.length) % shown.value.length
  disarm()
}

function close() {
  opened.value = null
  disarm()
}

function download(list: GalleryItem[]) {
  list.forEach((item, i) =>
    setTimeout(() => {
      const link = document.createElement('a')
      link.href = url(item)
      link.download = nameOf(item)
      document.body.appendChild(link)
      link.click()
      link.remove()
    }, i * DOWNLOAD_GAP_MS),
  )
}

function disarm() {
  armed.value = null
  clearTimeout(timer)
}

function arm(which: 'one' | 'many') {
  armed.value = which
  clearTimeout(timer)
  timer = setTimeout(disarm, ARMED_MS)
}

async function remove(list: GalleryItem[]) {
  deleting.value = true
  const failed: string[] = []
  const gone = new Set<number>()
  for (const item of list) {
    try {
      await deleteResult(item.id)
      gone.add(item.id)
    } catch (failure) {
      failed.push(`${nameOf(item)}: ${failure instanceof Error ? failure.message : String(failure)}`)
    }
  }
  const at = opened.value
  items.value = items.value.filter((one) => !gone.has(one.id))
  picked.value = new Set([...picked.value].filter((id) => !gone.has(id)))
  if (at !== null) opened.value = shown.value.length ? Math.min(at, shown.value.length - 1) : null
  problem.value = failed.join(' · ')
  deleting.value = false
}

async function deleteOne() {
  if (!current.value) return
  if (armed.value !== 'one') return arm('one')
  disarm()
  await remove([current.value])
}

async function deletePicked() {
  if (armed.value !== 'many') return arm('many')
  disarm()
  await remove(items.value.filter((one) => picked.value.has(one.id)))
}

function key(event: KeyboardEvent) {
  if (current.value) {
    if (event.key === 'Escape') close()
    else if (event.key === 'ArrowRight') step(1)
    else if (event.key === 'ArrowLeft') step(-1)
    else return
  } else if (event.key === 'Escape' && selecting.value) clear()
  else return
  event.preventDefault()
}

watch(only, () => {
  anchor.value = null
  opened.value = null
})
// A run that finishes while the page is open adds its pictures without a reload.
watch(() => state.runs.filter((one) => one.status === 'done').length, load)

onMounted(() => {
  load()
  window.addEventListener('keydown', key)
})
onBeforeUnmount(() => {
  window.removeEventListener('keydown', key)
  clearTimeout(timer)
})
</script>

<template>
  <main class="screen stack">
    <section class="card">
      <div class="head">
        <h5>Gallery</h5>
        <nav class="filter">
          <button :class="{ on: only === 'all' }" @click="only = 'all'">all <span class="mono">{{ counts.all }}</span></button>
          <button :class="{ on: only === 'image' }" @click="only = 'image'">pictures <span class="mono">{{ counts.image }}</span></button>
          <button :class="{ on: only === 'video' }" @click="only = 'video'">videos <span class="mono">{{ counts.video }}</span></button>
        </nav>
      </div>
      <div class="bar" :class="{ live: selecting }">
        <template v-if="selecting">
          <span class="mono">{{ picked.size }} selected · {{ size(pickedBytes) }}</span>
          <button @click="pickAll">select all {{ shown.length }}</button>
          <button @click="download(items.filter((one) => picked.has(one.id)))">download</button>
          <button class="kill" :class="{ armed: armed === 'many' }" :disabled="deleting" @click="deletePicked" @mouseleave="armed === 'many' && disarm()">
            {{ deleting ? 'deleting…' : armed === 'many' ? `confirm deleting ${picked.size}` : 'delete' }}
          </button>
          <button class="quiet" @click="clear">clear</button>
        </template>
        <span v-else class="note">click to view · ctrl- or shift-click to select · ← → to step through</span>
      </div>
      <div v-if="problem" class="band error">{{ problem }}</div>

      <div class="body days">
        <p v-if="loaded && !shown.length" class="note">Nothing here yet — finished pictures and videos land here.</p>
        <section v-for="group in days" :key="group.day" class="day">
          <h6 class="label">{{ group.day }} <span class="mono dim">{{ group.entries.length }}</span></h6>
          <div class="grid">
            <figure
              v-for="{ item, index } in group.entries"
              :key="item.id"
              class="tile"
              :class="{ picked: picked.has(item.id) }"
              :title="`${item.title || nameOf(item)} · ${when(item.created)}`"
              @click="click(index, $event)"
              @mouseenter="hovered = item.id"
              @mouseleave="hovered = null"
            >
              <video v-if="item.kind === 'video' && (hovered === item.id || !still(item))" :src="url(item)" :poster="still(item) ? thumbUrl(still(item)!) : undefined" muted loop playsinline :autoplay="hovered === item.id" preload="metadata" />
              <img v-else :src="thumbUrl(still(item)!)" alt="" loading="lazy" />
              <span v-if="item.kind === 'video'" class="badge mono">▶</span>
              <button class="pick" :aria-pressed="picked.has(item.id)" title="select" @click.stop="toggle(index, $event.shiftKey)">✓</button>
              <a class="get" :href="url(item)" :download="nameOf(item)" title="download" @click.stop>↓</a>
            </figure>
          </div>
        </section>
      </div>
    </section>

    <div v-if="current" class="viewer" @click.self="close">
      <header>
        <div class="about">
          <RouterLink v-if="current.run" :to="`/runs/${current.run}`" class="title">{{ current.title || current.run }}</RouterLink>
          <span v-else class="title">{{ nameOf(current) }}</span>
          <span class="mono dim">
            {{ when(current.created) }} · {{ size(current.bytes) }}{{ current.seed !== null ? ` · seed ${current.seed}` : '' }} · {{ (opened ?? 0) + 1 }} / {{ shown.length }}
          </span>
        </div>
        <a class="button" :href="url(current)" :download="nameOf(current)">download</a>
        <button class="kill" :class="{ armed: armed === 'one' }" :disabled="deleting" @click="deleteOne" @mouseleave="armed === 'one' && disarm()">
          {{ deleting ? 'deleting…' : armed === 'one' ? 'confirm deletion' : 'delete' }}
        </button>
        <button class="quiet" title="close (Esc)" @click="close">✕</button>
      </header>
      <button class="side prev" title="previous (←)" @click="step(-1)">‹</button>
      <video v-if="current.kind === 'video'" :key="current.id" class="full" :src="url(current)" controls autoplay loop playsinline />
      <img v-else :key="current.id" class="full" :src="url(current)" alt="" @click="close" />
      <button class="side next" title="next (→)" @click="step(1)">›</button>
    </div>
  </main>
</template>

<style scoped>
.filter {
  display: flex;
  gap: 4px;
}

.filter button,
.bar button,
.viewer header button,
.viewer header .button {
  font-size: var(--text-xs);
  padding: 2px 10px;
  border-color: var(--line);
  color: var(--muted);
}

.filter button.on {
  color: var(--accent);
  border-color: var(--accent);
}

.bar {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  min-height: 40px;
  padding: 0 var(--space-2);
  border-bottom: 1px solid var(--line);
  font-size: var(--text-xs);
  position: sticky;
  top: 56px;
  z-index: 2;
  background: var(--surface);
}

.bar.live .mono {
  color: var(--accent);
  margin-right: var(--space-1);
}

.kill:not(:disabled):hover {
  color: var(--status-error);
  border-color: var(--status-error);
}

.kill.armed,
.kill.armed:hover:not(:disabled) {
  color: var(--bg-base);
  background: var(--status-error);
  border-color: var(--status-error);
}

.days {
  gap: var(--space-3);
}

.day h6 {
  margin: 0 0 var(--space-1);
  font-size: var(--text-xs);
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 6px;
}

.tile {
  position: relative;
  margin: 0;
  aspect-ratio: 1;
  background: var(--video);
  border-radius: var(--radius-input);
  overflow: hidden;
  cursor: zoom-in;
  outline: 2px solid transparent;
  outline-offset: -2px;
}

.tile img,
.tile video {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.tile.picked {
  outline-color: var(--accent);
}

.tile.picked img,
.tile.picked video {
  opacity: 0.75;
}

.badge {
  position: absolute;
  left: 6px;
  bottom: 6px;
  font-size: 10px;
  padding: 1px 5px;
  border-radius: var(--radius-input);
  background: oklch(0% 0 0 / 0.6);
  color: var(--text-primary);
  pointer-events: none;
}

.pick,
.get {
  position: absolute;
  top: 6px;
  width: 24px;
  height: 24px;
  padding: 0;
  display: grid;
  place-items: center;
  font-size: 12px;
  line-height: 1;
  border-radius: 50%;
  border: 1px solid oklch(100% 0 0 / 0.5);
  background: oklch(0% 0 0 / 0.55);
  color: var(--text-primary);
  text-decoration: none;
  opacity: 0;
  transition: opacity 120ms ease;
}

.pick {
  left: 6px;
  color: transparent;
}

.get {
  right: 6px;
}

.tile:hover .pick,
.tile:hover .get,
.tile.picked .pick {
  opacity: 1;
}

.tile:hover .pick {
  color: var(--text-primary);
}

.tile.picked .pick {
  background: var(--accent);
  border-color: var(--accent);
  color: var(--accent-fg);
}

.get:hover,
.pick:hover {
  border-color: var(--accent);
}

.viewer {
  position: fixed;
  inset: 0;
  z-index: 20;
  background: oklch(0% 0 0 / 0.94);
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  grid-template-columns: 64px minmax(0, 1fr) 64px;
}

.viewer header {
  grid-column: 1 / -1;
  display: flex;
  align-items: center;
  gap: var(--space-1);
  padding: var(--space-1) var(--space-2);
}

.viewer .about {
  display: flex;
  flex-direction: column;
  min-width: 0;
  margin-right: auto;
}

.viewer .title {
  color: var(--text-primary);
  font-family: var(--font-heading);
  font-weight: var(--weight-semibold);
  text-decoration: none;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

a.title:hover {
  color: var(--accent);
}

.viewer .about .mono {
  font-size: 11px;
}

.viewer header .button {
  border: 1px solid var(--line);
  border-radius: var(--radius-btn);
  font-weight: var(--weight-medium);
  text-decoration: none;
  padding: 3px 10px;
}

.viewer header .button:hover {
  color: var(--accent);
  border-color: var(--accent);
}

.full {
  grid-row: 2;
  grid-column: 2;
  justify-self: center;
  align-self: center;
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  min-height: 0;
  padding-bottom: var(--space-2);
  box-sizing: border-box;
}

img.full {
  cursor: zoom-out;
}

.side {
  grid-row: 2;
  align-self: center;
  justify-self: center;
  border: none;
  font-size: 40px;
  line-height: 1;
  color: var(--muted);
  padding: var(--space-2) var(--space-1);
}

.side.prev {
  grid-column: 1;
}

.side.next {
  grid-column: 3;
}
</style>
