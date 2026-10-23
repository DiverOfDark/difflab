<script setup lang="ts">
/**
 * The order carries meaning — the first photo is what the take opens inside, and the rest are what
 * the director plans against — so nothing here sorts them. The browser is not on the machine with
 * the card, so a photo is uploaded first and the job names what was stored.
 */
import { ref } from 'vue'
import { listUploads, thumbUrl, upload, type Asset } from '../api'

const props = defineProps<{
  photos: string[]
  /** What each item is called in a caption; empty when a planned direction will cut its own crops
   *  and number them itself. */
  labels?: string[]
  /** Set when no item is the one the take opens on. */
  empty?: string
  /** Every item is a run of its own. */
  each?: boolean
  accept?: string
  /** Any one photo may be marked the start, which moves it first; `start` says whether the first is one. */
  startable?: boolean
  start?: boolean
  /** The model only edits: the start can be moved to another photo, never unmarked. */
  startFixed?: boolean
}>()
const emit = defineEmits<{ 'update:photos': [photos: string[]]; 'update:start': [start: boolean] }>()

const known = ref<Record<string, Asset>>({})
const busy = ref(false)
const problem = ref('')
const dragging = ref(false)
const held = ref(-1)

refresh()

async function refresh() {
  try {
    for (const asset of await listUploads()) known.value[asset.path] = asset
  } catch {
    /* the list is a convenience; a job still renders without it */
  }
}

async function add(files: File[]) {
  if (!files.length) return
  busy.value = true
  problem.value = ''
  try {
    const stored = await upload(files)
    for (const asset of stored) known.value[asset.path] = asset
    emit('update:photos', [...props.photos, ...stored.map((a) => a.path)])
  } catch (error) {
    problem.value = String(error)
  } finally {
    busy.value = false
  }
}

function onDrop(event: DragEvent) {
  dragging.value = false
  const files = [...(event.dataTransfer?.files ?? [])]
  if (files.length) add(files)
}

function onPick(event: Event) {
  const input = event.target as HTMLInputElement
  add([...(input.files ?? [])])
  input.value = ''
}

function remove(index: number) {
  if (index === 0 && props.start) emit('update:start', false)
  emit(
    'update:photos',
    props.photos.filter((_, i) => i !== index),
  )
}

function mark(index: number) {
  if (index === 0 && props.start) return props.startFixed || emit('update:start', false)
  const next = [...props.photos]
  next.unshift(...next.splice(index, 1))
  emit('update:photos', next)
  emit('update:start', true)
}

function drop(index: number) {
  if (held.value < 0 || held.value === index) return
  const next = [...props.photos]
  next.splice(index, 0, ...next.splice(held.value, 1))
  held.value = -1
  emit('update:photos', next)
}

const name = (path: string) => known.value[path]?.name ?? path.split('/').at(-1) ?? path

const copied = ref(-1)

/** Writing a prompt means typing these, so clicking one puts it on the clipboard. */
async function copy(index: number) {
  const label = props.labels?.[index]
  if (!label) return
  try {
    await navigator.clipboard.writeText(label)
    copied.value = index
    setTimeout(() => (copied.value = copied.value === index ? -1 : copied.value), 1200)
  } catch {
    /* a clipboard the browser will not give us is not worth an error */
  }
}
// A photo the pod already held has no stored thumb, so one is drawn on demand: a job written before
// uploads existed still shows its cast.
const thumb = (path: string) => known.value[path]?.thumb ?? thumbUrl(path)
</script>

<template>
  <div
    class="drop"
    :class="{ over: dragging }"
    @dragover.prevent="dragging = true"
    @dragleave="dragging = false"
    @drop.prevent="onDrop"
  >
    <ul v-if="photos.length" class="list">
      <li
        v-for="(path, index) in photos"
        :key="path + index"
        draggable="true"
        @dragstart="held = index"
        @dragover.prevent
        @drop.stop.prevent="drop(index)"
      >
        <img :src="thumb(path)" alt="" @error="($event.target as HTMLImageElement).style.visibility = 'hidden'" />
        <div class="what">
          <div class="name">{{ name(path) }}</div>
          <div v-if="index === 0 && !labels?.length && !empty" class="opens">opens the take</div>
          <div v-else-if="index === 0 && start" class="opens">the start · edited into the result</div>
        </div>
        <button
          v-if="startable"
          class="tag"
          :class="{ off: !(index === 0 && start) }"
          :title="index === 0 && start ? (startFixed ? 'this model edits the start: mark another to move it' : 'make it a reference again') : 'start from this photo'"
          @click.stop="mark(index)"
        >
          start
        </button>
        <button
          v-if="labels?.[index]"
          class="tag"
          :class="{ copied: copied === index }"
          :title="`call it ${labels[index]} in the prompt — click to copy`"
          @click.stop="copy(index)"
        >
          {{ copied === index ? 'copied' : labels[index] }}
        </button>
        <button class="quiet" title="remove" @click="remove(index)">✕</button>
      </li>
    </ul>
    <p v-else class="dim empty">{{ empty ?? 'Drop photos here, or choose them. The first one is what the take opens on.' }}</p>

    <div class="row">
      <label class="choose">
        <input type="file" multiple :accept="accept ?? 'image/*,video/*'" hidden @change="onPick" />
        <span>{{ busy ? 'uploading…' : 'choose…' }}</span>
      </label>
      <span v-if="each" class="dim small">one run each · {{ photos.length }}</span>
      <span v-else class="dim small">drag to reorder</span>
    </div>
    <p v-if="problem" class="band error">{{ problem }}</p>
  </div>
</template>

<style scoped>
.drop {
  border: 1px dashed var(--line);
  border-radius: var(--radius);
  padding: 8px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  transition: border-color 120ms ease, background 120ms ease;
}

.drop.over {
  border-color: var(--live);
  background: var(--live-back);
}

.list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

li {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px;
  border-radius: var(--radius-card);
  background: var(--surface);
  cursor: grab;
}

img {
  width: 36px;
  height: 56px;
  object-fit: cover;
  border-radius: var(--radius-input);
  background: var(--line);
  flex: 0 0 auto;
}

.what {
  min-width: 0;
  flex: 1;
}

.name {
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.opens {
  font-size: 11px;
  color: var(--live);
}

.tag {
  font-family: var(--mono);
  font-size: 11px;
  padding: 2px 8px;
  border-radius: var(--radius-input);
  background: var(--live-back);
  color: var(--live);
  border: none;
  flex: 0 0 auto;
}

.tag:hover {
  background: var(--bg-overlay);
}

.tag.off {
  background: none;
  color: var(--dim);
  border: 1px solid var(--line);
}

.tag.copied {
  background: var(--done-back);
  color: var(--done);
}

.empty {
  font-size: 12px;
  margin: 8px 4px;
}

.row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.choose span {
  display: inline-block;
  background: var(--raised);
  border: 1px solid var(--line);
  border-radius: var(--radius-btn);
  padding: 5px 12px;
  cursor: pointer;
  font-size: 13px;
}

.small {
  font-size: 12px;
}
</style>
