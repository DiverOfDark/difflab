<script setup lang="ts">
/** A chat with the VLM. Each reply is a run on the card, so it waits behind a render; the page polls it as it writes. */
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import {
  chatAgain,
  chatRewind,
  chatStop,
  deleteChat,
  listChats,
  patchChat,
  readChat,
  sayInChat,
  startChat,
  thumbUrl,
  upload,
  type Asset,
  type Chat,
  type ChatMessage,
  type ChatSummary,
} from '../api'
import { state } from '../store'

const POLL_MS = 300
const VIDEO = /\.(mp4|mov|mkv|webm)$/i
const SOUND = /\.(wav|mp3|m4a|flac|ogg)$/i

const props = defineProps<{ id?: number }>()
const router = useRouter()

const list = ref<ChatSummary[]>([])
const chat = ref<Chat | null>(null)
const text = ref('')
const attached = ref<Asset[]>([])
const uploading = ref(0)
const busy = ref(false)
const failure = ref('')
const dragging = ref(false)
const armed = ref<number | null>(null)
const thread = ref<HTMLElement | null>(null)
const picker = ref<HTMLInputElement | null>(null)
const box = ref<HTMLTextAreaElement | null>(null)
let timer: ReturnType<typeof setTimeout> | undefined

const pending = computed(() => chat.value?.messages.some((m) => m.state === 'queued' || m.state === 'writing') ?? false)
const last = computed(() => chat.value?.messages.at(-1) ?? null)
const canSend = computed(() => !!chat.value && !pending.value && !busy.value && !uploading.value && (!!text.value.trim() || attached.value.length > 0))

const fileUrl = (path: string) => `/uploads/${path.split('/').pop()}`
const nameOf = (path: string) => (path.split('/').pop() ?? path).replace(/^[0-9a-f]{16}-/, '')
const words = (s: string) => s.trim().split(/\s+/).filter(Boolean).length

async function act<T>(work: () => Promise<T>): Promise<T | undefined> {
  failure.value = ''
  busy.value = true
  try {
    return await work()
  } catch (error) {
    failure.value = String(error instanceof Error ? error.message : error)
    return undefined
  } finally {
    busy.value = false
  }
}

async function loadList() {
  list.value = await listChats().catch(() => list.value)
}

function show(found: Chat) {
  const near = !thread.value || thread.value.scrollHeight - thread.value.scrollTop - thread.value.clientHeight < 120
  const grew = (found.messages.length !== chat.value?.messages.length || found.messages.at(-1)?.text !== chat.value?.messages.at(-1)?.text)
  chat.value = found
  if (near && grew) nextTick(() => thread.value?.scrollTo({ top: thread.value.scrollHeight }))
  clearTimeout(timer)
  if (pending.value) timer = setTimeout(poll, POLL_MS)
  else loadList()
}

async function poll() {
  if (!chat.value) return
  const id = chat.value.id
  const found = await readChat(id).catch(() => null)
  if (found && chat.value?.id === id) show(found)
  else if (chat.value?.id === id) timer = setTimeout(poll, POLL_MS * 3)
}

watch(
  () => props.id,
  async (id) => {
    clearTimeout(timer)
    failure.value = ''
    await loadList()
    if (id === undefined) {
      chat.value = null
      const newest = list.value[0]
      if (newest) router.replace(`/chat/${newest.id}`)
      return
    }
    const found = await act(() => readChat(id))
    if (found) {
      chat.value = null
      show(found)
      nextTick(() => box.value?.focus())
    }
  },
  { immediate: true },
)

onBeforeUnmount(() => clearTimeout(timer))

async function fresh() {
  const made = await act(startChat)
  if (made) {
    await loadList()
    router.push(`/chat/${made.id}`)
  }
}

async function remove(id: number) {
  if (armed.value !== id) {
    armed.value = id
    setTimeout(() => armed.value === id && (armed.value = null), 4000)
    return
  }
  armed.value = null
  await act(() => deleteChat(id))
  await loadList()
  if (props.id === id) router.replace(list.value[0] ? `/chat/${list.value[0].id}` : '/chat')
}

async function attach(files: File[]) {
  if (!files.length) return
  uploading.value += files.length
  failure.value = ''
  try {
    attached.value = [...attached.value, ...(await upload(files))]
  } catch (error) {
    failure.value = String(error instanceof Error ? error.message : error)
  } finally {
    uploading.value -= files.length
  }
}

function picked(event: Event) {
  const input = event.target as HTMLInputElement
  attach([...(input.files ?? [])])
  input.value = ''
}

function pasted(event: ClipboardEvent) {
  const files = [...(event.clipboardData?.files ?? [])]
  if (files.length) {
    event.preventDefault()
    attach(files)
  }
}

function dropped(event: DragEvent) {
  dragging.value = false
  attach([...(event.dataTransfer?.files ?? [])])
}

async function send() {
  if (!canSend.value || !chat.value) return
  const id = chat.value.id
  const found = await act(() => sayInChat(id, text.value, attached.value.map((a) => a.path)))
  if (found) {
    text.value = ''
    attached.value = []
    show(found)
    nextTick(() => thread.value?.scrollTo({ top: thread.value.scrollHeight }))
  }
}

function keyed(event: KeyboardEvent) {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault()
    send()
  }
}

async function stop() {
  if (!chat.value) return
  const id = chat.value.id
  const found = await act(() => chatStop(id))
  if (found) show(found)
}

async function again() {
  if (!chat.value) return
  const id = chat.value.id
  const found = await act(() => chatAgain(id))
  if (found) show(found)
}

/** The conversation back to before this message, its words and attachments back in the box. */
async function edit(message: ChatMessage) {
  if (!chat.value) return
  const id = chat.value.id
  const found = await act(() => chatRewind(id, message.id))
  if (!found) return
  text.value = message.text
  attached.value = message.media.map((path) => ({ path, name: nameOf(path), url: fileUrl(path), bytes: 0 }))
  show(found)
  nextTick(() => box.value?.focus())
}

async function rename(event: Event) {
  if (!chat.value) return
  const title = (event.target as HTMLInputElement).value.trim()
  if (title === chat.value.title) return
  const id = chat.value.id
  const found = await act(() => patchChat(id, { title }))
  if (found) show(found)
}

async function setSystem(event: Event) {
  if (!chat.value) return
  const system = (event.target as HTMLTextAreaElement).value.trim()
  if (system === (chat.value.system || chat.value.default_system).trim()) return
  const id = chat.value.id
  const found = await act(() => patchChat(id, { system: system === chat.value?.default_system.trim() ? '' : system }))
  if (found) show(found)
}

const copied = ref<number | null>(null)
async function copy(message: ChatMessage) {
  await navigator.clipboard.writeText(message.text).catch(() => undefined)
  copied.value = message.id
  setTimeout(() => copied.value === message.id && (copied.value = null), 1500)
}

/** What a reply in flight is doing, in the words the header uses for the card. */
function waiting(message: ChatMessage): string {
  if (message.state === 'queued') {
    const other = state.gpu?.run && state.gpu.run !== message.run ? state.gpu : null
    return other ? `waiting for the card · ${other.title}` : 'waiting for the card'
  }
  if (!message.text) return message.thinking ? `thinking · ${words(message.thinking)} words` : 'loading the model…'
  return ''
}
</script>

<template>
  <main class="chat-screen">
    <aside class="side">
      <button class="primary" :disabled="busy" @click="fresh">New chat</button>
      <nav class="list">
        <div v-for="one in list" :key="one.id" class="item" :class="{ on: one.id === props.id }">
          <RouterLink :to="`/chat/${one.id}`" class="name">{{ one.title || 'new chat' }}</RouterLink>
          <button class="quiet del" :class="{ armed: armed === one.id }" :title="armed === one.id ? 'click again to delete' : 'delete'" @click="remove(one.id)">
            {{ armed === one.id ? 'delete?' : '×' }}
          </button>
        </div>
        <p v-if="!list.length" class="muted empty">No chats yet.</p>
      </nav>
    </aside>

    <section v-if="chat" class="main" :class="{ dragging }" @dragover.prevent="dragging = true" @dragleave.self="dragging = false" @drop.prevent="dropped">
      <header class="head">
        <input class="title" :value="chat.title" placeholder="untitled" @change="rename" @keydown.enter="($event.target as HTMLInputElement).blur()" />
        <details class="system">
          <summary class="label">system prompt{{ chat.system ? ' · own' : '' }}</summary>
          <textarea rows="5" :value="chat.system || chat.default_system" @change="setSystem" />
          <p class="muted hint">Applies from the next reply. Clear it, or put the default back, to follow the default.</p>
        </details>
      </header>

      <div ref="thread" class="thread">
        <p v-if="!chat.messages.length" class="muted empty">
          Ask anything. Attach pictures or videos with the clip, by pasting, or by dropping them here — a video is shown to the model as a frame a second.
        </p>
        <article v-for="message in chat.messages" :key="message.id" class="message" :class="message.role">
          <div v-if="message.media.length" class="media">
            <template v-for="path in message.media" :key="path">
              <video v-if="VIDEO.test(path)" :src="fileUrl(path)" controls preload="metadata" />
              <audio v-else-if="SOUND.test(path)" :src="fileUrl(path)" controls />
              <a v-else :href="fileUrl(path)" target="_blank"><img :src="thumbUrl(path)" :alt="nameOf(path)" loading="lazy" /></a>
            </template>
          </div>
          <details v-if="message.thinking" class="thinking" :open="message.state === 'writing' && !message.text">
            <summary class="label">thinking · {{ words(message.thinking) }} words</summary>
            <div class="words muted">{{ message.thinking }}</div>
          </details>
          <div v-if="message.text" class="words">{{ message.text }}</div>
          <p v-if="waiting(message)" class="status mono muted"><i class="dot running" />{{ waiting(message) }}</p>
          <p v-if="message.state === 'stopped'" class="status mono dim">stopped</p>
          <p v-if="message.state === 'failed'" class="band error">{{ message.error || 'failed' }}</p>
          <div v-if="message.state !== 'queued' && message.state !== 'writing'" class="actions">
            <button v-if="message.role === 'assistant' && message.text" class="quiet" @click="copy(message)">{{ copied === message.id ? 'copied' : 'copy' }}</button>
            <button v-if="message.role === 'assistant' && message === last" class="quiet" :disabled="pending || busy" @click="again">again</button>
            <button v-if="message.role === 'user'" class="quiet" :disabled="pending || busy" title="drop this and everything after it, and say it differently" @click="edit(message)">edit</button>
            <RouterLink v-if="message.run" class="run mono" :to="`/runs/${message.run}`">run</RouterLink>
          </div>
        </article>
      </div>

      <p v-if="failure" class="band error">{{ failure }}</p>
      <footer class="composer">
        <div v-if="attached.length || uploading" class="attached">
          <span v-for="(asset, i) in attached" :key="asset.path" class="chip">
            <img v-if="!VIDEO.test(asset.path) && !SOUND.test(asset.path)" :src="thumbUrl(asset.path)" alt="" />
            <span class="mono">{{ asset.name }}</span>
            <button class="quiet" title="remove" @click="attached.splice(i, 1)">×</button>
          </span>
          <span v-if="uploading" class="chip mono muted">uploading {{ uploading }}…</span>
        </div>
        <div class="row">
          <button class="clip" title="attach pictures or videos" @click="picker?.click()">＋</button>
          <input ref="picker" type="file" multiple accept="image/*,video/*,audio/*" hidden @change="picked" />
          <textarea ref="box" v-model="text" rows="3" placeholder="Message · Enter sends, Shift+Enter for a new line" @keydown="keyed" @paste="pasted" />
          <button v-if="pending" class="danger" @click="stop">Stop</button>
          <button v-else class="primary" :disabled="!canSend" @click="send">Send</button>
        </div>
      </footer>
    </section>
    <section v-else class="main">
      <p v-if="failure" class="band error">{{ failure }}</p>
      <p class="muted empty">Start a chat with the button on the left.</p>
    </section>
  </main>
</template>

<style scoped>
.chat-screen {
  display: grid;
  grid-template-columns: 240px 1fr;
  height: calc(100vh - 56px);
}

.side {
  border-right: 1px solid var(--line);
  padding: var(--space-2);
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  min-height: 0;
}

.list {
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.item {
  display: flex;
  align-items: center;
  border-radius: var(--radius-input);
}

.item.on,
.item:hover {
  background: var(--bg-raised);
}

.item .name {
  flex: 1;
  padding: 6px 8px;
  color: var(--text);
  text-decoration: none;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-size: var(--text-sm);
}

.item .del {
  visibility: hidden;
}

.item:hover .del,
.item .del.armed {
  visibility: visible;
}

.del.armed {
  color: var(--fail);
}

.main {
  display: flex;
  flex-direction: column;
  min-height: 0;
  position: relative;
}

.main.dragging::after {
  content: 'drop to attach';
  position: absolute;
  inset: var(--space-2);
  display: grid;
  place-items: center;
  border: 2px dashed var(--accent);
  border-radius: var(--radius-card);
  background: oklch(13% 0.01 260 / 0.8);
  color: var(--accent);
  pointer-events: none;
}

.head {
  padding: var(--space-2) var(--space-3);
  border-bottom: 1px solid var(--line);
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.title {
  background: none;
  border-color: transparent;
  font-size: var(--text-lg, 18px);
  font-weight: var(--weight-semibold);
  padding: 4px 0;
}

.title:hover,
.title:focus {
  border-color: var(--line);
  padding: 4px 8px;
}

.system summary {
  cursor: pointer;
}

.system textarea {
  margin-top: var(--space-1);
  font-size: var(--text-sm);
}

.hint {
  font-size: var(--text-xs);
  margin: 4px 0 0;
}

.thread {
  flex: 1;
  overflow-y: auto;
  padding: var(--space-3);
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.empty {
  margin: var(--space-3);
}

.message {
  max-width: min(860px, 85%);
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.message.user {
  align-self: flex-end;
  align-items: flex-end;
}

.message.user .words {
  background: var(--bg-raised);
  border-radius: var(--radius-card);
  padding: 8px 12px;
}

.words {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  line-height: 1.55;
}

.thinking summary {
  cursor: pointer;
}

.thinking .words {
  font-size: var(--text-sm);
  border-left: 2px solid var(--line);
  padding-left: var(--space-1);
  margin-top: 4px;
}

.media {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  justify-content: flex-end;
}

.media img,
.media video {
  max-height: 220px;
  max-width: 320px;
  border-radius: var(--radius-card);
  display: block;
}

.status {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: var(--text-xs);
  margin: 0;
}

.dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--status-building);
  animation: pulse 1.2s ease-in-out infinite;
}

@keyframes pulse {
  50% {
    opacity: 0.3;
  }
}

.actions {
  display: flex;
  gap: 2px;
  align-items: center;
  opacity: 0;
  transition: opacity 120ms ease;
}

.message:hover .actions,
.message:last-child .actions {
  opacity: 1;
}

.actions .quiet {
  font-size: var(--text-xs);
}

.run {
  font-size: var(--text-xs);
  color: var(--dim);
  padding: 4px 6px;
}

.band {
  margin: 0 var(--space-3) var(--space-1);
}

.composer {
  border-top: 1px solid var(--line);
  padding: var(--space-2) var(--space-3);
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.attached {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--bg-raised);
  border-radius: var(--radius-input);
  padding: 2px 2px 2px 6px;
  font-size: var(--text-xs);
}

.chip img {
  height: 28px;
  border-radius: 2px;
}

.row {
  display: flex;
  gap: var(--space-1);
  align-items: flex-end;
}

.row textarea {
  flex: 1;
  resize: none;
}

.clip {
  padding: 7px 11px;
}
</style>
