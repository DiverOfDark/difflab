<script setup lang="ts">
/** A chat with the VLM, which acts through difflab's tools. Each reply and each render it asks for is a run on the
 *  card, so it waits behind others; the page polls while anything is in flight. */
import DOMPurify from 'dompurify'
import { marked } from 'marked'
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import {
  chatAgain,
  chatAllow,
  chatCompact,
  chatRewind,
  chatStop,
  deleteChat,
  listChats,
  outputUrl,
  patchChat,
  readChat,
  sayInChat,
  startChat,
  thumbUrl,
  upload,
  uploadFrom,
  type Asset,
  type Chat,
  type ChatMessage,
  type ChatSummary,
} from '../api'
import { state } from '../store'

const POLL_MS = 300
const LIST_MS = 5000
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
/** The newest message the summary stands in for: the divider goes after it. */
const summarisedUpTo = computed(() =>
  Math.max(0, ...(chat.value?.messages.filter((m) => m.id <= (chat.value?.summary_through ?? 0)).map((m) => m.id) ?? [])),
)
const thousands = (n: number) => `${Math.round(n / 1000)}k`
const canSend = computed(() => !!chat.value && !pending.value && !busy.value && !uploading.value && (!!text.value.trim() || attached.value.length > 0))

// Absolute is an upload or a fetched picture, under assets/; relative is a render's, under output/.
const fileUrl = (path: string) => (path.startsWith('/') ? `/uploads/${path.split('/').pop()}` : outputUrl(path))
const nameOf = (path: string) => (path.split('/').pop() ?? path).replace(/^[0-9a-f]{16}-/, '')
const words = (s: string) => s.trim().split(/\s+/).filter(Boolean).length
const rendered = (s: string) => DOMPurify.sanitize(marked.parse(s, { async: false, gfm: true, breaks: true }))

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
  const added = found.id !== chat.value?.id || found.messages.length !== chat.value?.messages.length
  chat.value = found
  if (near && grew) nextTick(() => thread.value?.scrollTo({ top: thread.value.scrollHeight }))
  clearTimeout(timer)
  if (pending.value) timer = setTimeout(poll, POLL_MS)
  if (added || !pending.value) loadList()
}

async function poll() {
  if (!chat.value) return
  const id = chat.value.id
  const found = await readChat(id).catch(() => null)
  if (found && chat.value?.id === id) show(found)
  else if (chat.value?.id === id) timer = setTimeout(poll, POLL_MS * 3)
}

/** Every picture in the chat in order, so the viewer steps through them all. */
const pictures = computed(() =>
  (chat.value?.messages ?? []).flatMap((m) => m.media.filter((path) => !VIDEO.test(path) && !SOUND.test(path))),
)
// Above the watch: it runs at once, and clears `viewing` before the first chat loads.
const viewing = ref<number | null>(null)
const onScreen = computed(() => (viewing.value === null ? null : (pictures.value[viewing.value] ?? null)))

watch(
  () => props.id,
  async (id) => {
    clearTimeout(timer)
    failure.value = ''
    viewing.value = null
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

function view(path: string) {
  viewing.value = pictures.value.indexOf(path)
}

function step(by: number) {
  if (viewing.value === null || !pictures.value.length) return
  viewing.value = (viewing.value + by + pictures.value.length) % pictures.value.length
}

function key(event: KeyboardEvent) {
  if (onScreen.value === null) return
  if (event.key === 'Escape') viewing.value = null
  else if (event.key === 'ArrowRight') step(1)
  else if (event.key === 'ArrowLeft') step(-1)
  else return
  event.preventDefault()
}

// Replies in chats other than the open one move them up too.
let listing: ReturnType<typeof setInterval>
onMounted(() => {
  window.addEventListener('keydown', key)
  listing = setInterval(loadList, LIST_MS)
})
onBeforeUnmount(() => {
  window.removeEventListener('keydown', key)
  clearTimeout(timer)
  clearInterval(listing)
})

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

/** A picture dragged from another tab often carries only its address, which the server then fetches. */
async function dropped(event: DragEvent) {
  dragging.value = false
  const files = [...(event.dataTransfer?.files ?? [])]
  if (files.length) return attach(files)
  const html = event.dataTransfer?.getData('text/html') ?? ''
  const url =
    new DOMParser().parseFromString(html, 'text/html').querySelector('img')?.src ||
    (event.dataTransfer?.getData('text/uri-list') ?? '').split('\n').find((line) => line && !line.startsWith('#')) ||
    ''
  if (!/^https?:/.test(url)) return
  uploading.value += 1
  failure.value = ''
  try {
    attached.value = [...attached.value, await uploadFrom(url)]
  } catch (error) {
    failure.value = String(error instanceof Error ? error.message : error)
  } finally {
    uploading.value -= 1
  }
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

async function compact() {
  if (!chat.value) return
  const id = chat.value.id
  const found = await act(() => chatCompact(id))
  if (found) show(found)
}

async function allow(message: ChatMessage, allowed: boolean) {
  if (!chat.value) return
  const id = chat.value.id
  const found = await act(() => chatAllow(id, message.id, allowed))
  if (found) show(found)
}

/** A call as one line: its tool and its arguments, long values cut. */
function called(message: ChatMessage): string {
  const call = message.calls[0]
  if (!call) return ''
  const args = Object.entries(call.arguments).map(([key, value]) => {
    const said = typeof value === 'string' ? value : JSON.stringify(value)
    return `${key}: ${said.length > 80 ? `${said.slice(0, 80)}…` : said}`
  })
  return args.join(' · ')
}

/** Where a tool call's run is: rendering now on the card, or waiting behind something else. */
function rendering(message: ChatMessage): string {
  if (message.calls[0]?.follows) return 'following the run'
  return state.gpu?.run === message.run ? `${state.gpu?.state ?? 'running'} · ${state.gpu?.title ?? ''}` : 'waiting for the card'
}

// Bound to drafts: a re-render (the list reloads every LIST_MS) writes a bound :value back over what is being typed.
const titleDraft = ref('')
const systemDraft = ref('')
watch(
  () => [chat.value?.id, chat.value?.title, chat.value?.system, chat.value?.default_system],
  () => {
    titleDraft.value = chat.value?.title ?? ''
    systemDraft.value = chat.value ? chat.value.system || chat.value.default_system : ''
  },
  { immediate: true },
)

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
  if (message.state !== 'queued' && message.state !== 'writing') return ''
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
        <input v-model="titleDraft" class="title" placeholder="untitled" @change="rename" @keydown.enter="($event.target as HTMLInputElement).blur()" />
        <div class="fill mono muted" title="the newest reply's prompt against what the model holds; past the mark it summarises the chat first">
          <span>context {{ thousands(chat.context.used) }} / {{ thousands(chat.context.total) }}</span>
          <span class="meter"><i :style="{ width: `${Math.min(100, (100 * chat.context.used) / chat.context.total)}%` }" /><b :style="{ left: `${(100 * chat.context.compacts_at) / chat.context.total}%` }" /></span>
          <span v-if="chat.compact_next">summarises before the next reply</span>
          <button v-else class="quiet" :disabled="busy || pending || !chat.messages.length" title="the next reply summarises the chat first, keeping its newest messages" @click="compact">compact</button>
        </div>
        <details class="system">
          <summary class="label">system prompt{{ chat.system ? ' · own' : '' }}</summary>
          <textarea v-model="systemDraft" rows="5" @change="setSystem" />
          <p class="muted hint">Applies from the next reply. Clear it, or put the default back, to follow the default.</p>
        </details>
      </header>

      <div ref="thread" class="thread">
        <p v-if="!chat.messages.length" class="muted empty">
          Ask anything, or ask for a picture or a clip: the model makes them itself, finds references on the web, and works on a
          production you name. Attach pictures or videos with the clip, by pasting, or by dropping them here —
          a video is shown to the model as a frame a second.
        </p>
        <template v-for="message in chat.messages" :key="message.id">
        <article class="message" :class="[message.role, { summarised: message.id <= chat.summary_through }]">
          <div v-if="message.role === 'tool'" class="call mono">
            <span class="tool">{{ message.calls[0]?.name }}</span>
            <span class="args muted">{{ called(message) }}</span>
          </div>
          <p v-if="message.role === 'tool' && message.state === 'writing'" class="status mono muted"><i class="dot running" />{{ rendering(message) }}</p>
          <div v-if="message.state === 'asking'" class="ask">
            <p>{{ message.text }}</p>
            <div class="buttons">
              <button class="primary" :disabled="busy" @click="allow(message, true)">Allow</button>
              <button :disabled="busy" @click="allow(message, false)">Decline</button>
            </div>
          </div>
          <div v-if="message.media.length" class="media">
            <template v-for="path in message.media" :key="path">
              <video v-if="VIDEO.test(path)" :src="fileUrl(path)" controls preload="metadata" />
              <audio v-else-if="SOUND.test(path)" :src="fileUrl(path)" controls />
              <img v-else :src="thumbUrl(path)" :alt="nameOf(path)" loading="lazy" @click="view(path)" />
            </template>
          </div>
          <details v-if="message.thinking" class="thinking" :open="message.state === 'writing' && !message.text">
            <summary class="label">thinking · {{ words(message.thinking) }} words</summary>
            <div class="words muted">{{ message.thinking }}</div>
          </details>
          <div v-if="message.text && message.role === 'assistant'" class="words md" v-html="rendered(message.text)" />
          <details v-else-if="message.text && message.role === 'tool' && message.state !== 'asking'" class="result">
            <summary class="label">result</summary>
            <pre class="mono">{{ message.text }}</pre>
          </details>
          <div v-else-if="message.text && message.role === 'user'" class="words">{{ message.text }}</div>
          <p v-if="message.role !== 'tool' && waiting(message)" class="status mono muted"><i class="dot running" />{{ waiting(message) }}</p>
          <p v-if="message.state === 'stopped'" class="status mono dim">stopped</p>
          <p v-if="message.state === 'failed'" class="band error">{{ message.error || 'failed' }}</p>
          <div v-if="message.state !== 'queued' && message.state !== 'writing' && message.state !== 'asking'" class="actions">
            <button v-if="message.role === 'assistant' && message.text" class="quiet" @click="copy(message)">{{ copied === message.id ? 'copied' : 'copy' }}</button>
            <button v-if="message.role === 'assistant' && message === last" class="quiet" :disabled="pending || busy" @click="again">again</button>
            <button v-if="message.role === 'user'" class="quiet" :disabled="pending || busy" title="drop this and everything after it, and say it differently" @click="edit(message)">edit</button>
            <RouterLink v-if="message.run" class="run mono" :to="`/runs/${message.run}`">run</RouterLink>
          </div>
        </article>
        <details v-if="chat.summary && message.id === summarisedUpTo" class="divider">
          <summary class="label">the model has a summary of the messages above · {{ words(chat.summary) }} words</summary>
          <div class="words muted">{{ chat.summary }}</div>
        </details>
        </template>
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

    <div v-if="onScreen" class="viewer" @click.self="viewing = null">
      <header>
        <div class="about">
          <span class="title">{{ nameOf(onScreen) }}</span>
          <span class="mono dim">{{ (viewing ?? 0) + 1 }} / {{ pictures.length }}</span>
        </div>
        <a class="button" :href="fileUrl(onScreen)" :download="nameOf(onScreen)">download</a>
        <button class="quiet" title="close (Esc)" @click="viewing = null">✕</button>
      </header>
      <button class="side prev" title="previous (←)" @click="step(-1)">‹</button>
      <img :key="onScreen" class="full" :src="fileUrl(onScreen)" alt="" @click="viewing = null" />
      <button class="side next" title="next (→)" @click="step(1)">›</button>
    </div>
  </main>
</template>

<style scoped>
.chat-screen {
  display: grid;
  grid-template-columns: 240px minmax(0, 1fr);
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

.fill {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: var(--text-xs);
}

.meter {
  position: relative;
  width: 120px;
  height: 4px;
  background: var(--bg-raised);
  border-radius: 2px;
}

.meter i {
  position: absolute;
  inset: 0 auto 0 0;
  background: var(--accent);
  border-radius: 2px;
}

.meter b {
  position: absolute;
  top: -3px;
  width: 1px;
  height: 10px;
  background: var(--text-secondary);
}

.message.summarised {
  opacity: 0.5;
}

.divider {
  border-top: 1px dashed var(--line);
  padding-top: var(--space-1);
}

.divider summary {
  cursor: pointer;
}

.divider .words {
  font-size: var(--text-sm);
  margin-top: 4px;
}

.message.tool {
  gap: 4px;
  border-left: 2px solid var(--line);
  padding-left: var(--space-1);
}

.message.tool .media {
  justify-content: flex-start;
}

.call {
  display: flex;
  gap: 8px;
  font-size: var(--text-xs);
  align-items: baseline;
  min-width: 0;
}

.call .tool {
  color: var(--accent);
}

.call .args {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.ask {
  background: var(--bg-raised);
  border-radius: var(--radius-card);
  padding: 8px 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.ask p {
  margin: 0;
}

.ask .buttons {
  display: flex;
  gap: 6px;
}

.result summary {
  cursor: pointer;
  font-size: var(--text-xs);
}

.result pre {
  font-size: var(--text-xs);
  max-height: 260px;
  overflow: auto;
  background: var(--bg-raised);
  border-radius: var(--radius-input);
  padding: 8px 12px;
  margin: 4px 0 0;
  white-space: pre-wrap;
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

.md {
  white-space: normal;
}

.md :deep(:first-child) {
  margin-top: 0;
}

.md :deep(:last-child) {
  margin-bottom: 0;
}

.md :deep(p),
.md :deep(ul),
.md :deep(ol),
.md :deep(pre),
.md :deep(blockquote),
.md :deep(table) {
  margin: 0 0 0.7em;
}

.md :deep(h1),
.md :deep(h2),
.md :deep(h3),
.md :deep(h4) {
  font-size: 1em;
  font-weight: var(--weight-semibold);
  margin: 1em 0 0.4em;
}

.md :deep(h1),
.md :deep(h2) {
  font-size: 1.15em;
}

.md :deep(ul),
.md :deep(ol) {
  padding-left: 1.4em;
}

.md :deep(li + li) {
  margin-top: 0.2em;
}

.md :deep(code) {
  font-family: var(--mono);
  font-size: 0.9em;
  background: var(--bg-raised);
  border-radius: 3px;
  padding: 1px 4px;
}

.md :deep(pre) {
  background: var(--bg-raised);
  border-radius: var(--radius-input);
  padding: 8px 12px;
  overflow-x: auto;
}

.md :deep(pre code) {
  background: none;
  padding: 0;
  white-space: pre;
}

.md :deep(blockquote) {
  border-left: 2px solid var(--line);
  padding-left: var(--space-1);
  color: var(--text-secondary);
}

.md :deep(table) {
  border-collapse: collapse;
  display: block;
  overflow-x: auto;
}

.md :deep(th),
.md :deep(td) {
  border: 1px solid var(--line);
  padding: 4px 8px;
  text-align: left;
}

.md :deep(hr) {
  border: none;
  border-top: 1px solid var(--line);
  margin: 1em 0;
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

.media img {
  cursor: zoom-in;
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
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
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

.viewer .full {
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
  cursor: zoom-out;
}

.viewer .side {
  grid-row: 2;
  align-self: center;
  justify-self: center;
  border: none;
  font-size: 40px;
  line-height: 1;
  color: var(--muted);
  padding: var(--space-2) var(--space-1);
}

.viewer .prev {
  grid-column: 1;
}

.viewer .next {
  grid-column: 3;
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
