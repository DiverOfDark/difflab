<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, reactive, ref, watch } from 'vue'
import { checkModel, resolveModel, type ModelChoice, type ModelFit } from '../api'
import { ROLES } from '../roles'

const props = defineProps<{
  value: string | null
  fallback: string
  codeDefault: string
  remembered: string[]
  role: string
  empty?: string
}>()
const emit = defineEmits<{ pick: [value: string]; forget: [value: string] }>()

const open = ref(false)
const custom = ref('')
const problem = ref('')
const root = ref<HTMLElement | null>(null)
const field = ref<HTMLInputElement | null>(null)

const REF = /^([\w.-]+\/[\w.-]+@[0-9a-f]{40}(:.+)?|civitai:\d+@[0-9a-f]{64}:.+)$/

function parts(value: string) {
  if (!value) return { name: props.empty ?? 'none', where: props.empty ? 'Hugging Face or Civitai' : 'made on first load' }
  const civitai = value.match(/^civitai:(\d+)@[0-9a-f]{64}:(.+)$/)
  if (civitai) return { name: civitai[2].replace(/\.(safetensors|pth|bin)$/, ''), where: `civitai · version ${civitai[1]}` }
  const [head, files = ''] = value.split(/:(.*)/s, 2)
  const [repo, commit = ''] = head.split('@')
  const file = files
    .split(',')
    .map((f) => f.split('/').pop() ?? f)
    .find((f) => f && !f.endsWith('.json'))
  return { name: (file ?? repo.split('/').pop() ?? repo).replace(/\.(safetensors|pth|bin)$/, ''), where: `${repo} · ${commit.slice(0, 7)}` }
}

const current = computed(() => props.value ?? props.fallback)
const shown = computed(() => parts(current.value))
const options = computed(() => {
  const all = [...(props.codeDefault ? [props.codeDefault] : []), ...props.remembered.filter((v) => v !== props.codeDefault)]
  if (current.value && !all.includes(current.value)) all.push(current.value)
  return all
})

const fit = reactive<Record<string, ModelFit>>({})
function ask(value: string) {
  if (!value || value in fit) return
  checkModel(props.role, value).then(
    (found) => (fit[value] = found),
    (error) => (fit[value] = { level: 'unknown', summary: `not checked: ${String(error).replace(/^Error: /, '')}`, notes: [], like: '' }),
  )
}
watch(current, ask, { immediate: true })
watch(open, (now) => now && options.value.forEach(ask))
const MARK = { ok: '✓', warn: '⚠', bad: '✗', unknown: '?' }
const said = (found: ModelFit) => (found.like ? `looks like a file for ${ROLES[found.like] ?? found.like}` : found.summary)
const told = (value: string) => {
  const found = fit[value]
  return found ? [said(found), ...found.notes].join('\n') : 'checking…'
}

function choose(value: string) {
  emit('pick', value)
  open.value = false
}

const resolving = ref(false)
/** The files a pasted repo or Civitai version holds, while one is being picked. */
const files = ref<ModelChoice[]>([])

async function useCustom() {
  const value = custom.value.trim()
  problem.value = ''
  files.value = []
  if (REF.test(value)) return take(value)
  resolving.value = true
  try {
    const found = await resolveModel(value, props.role)
    if (found.choices?.length) {
      files.value = found.choices
      found.choices.forEach((c) => ask(c.value))
    }
    else if (found.value) take(found.value)
  } catch (error) {
    problem.value = `${String(error).replace(/^Error: /, '')} — paste a Hugging Face or Civitai link, or org/repo@<commit>[:file]`
  } finally {
    resolving.value = false
  }
}

function take(value: string) {
  custom.value = ''
  files.value = []
  choose(value)
}

const size = (bytes: number) => (!bytes ? '' : bytes >= 2 ** 30 ? `${(bytes / 2 ** 30).toFixed(1)} GB` : `${Math.max(1, Math.round(bytes / 2 ** 20))} MB`)

async function toggle() {
  open.value = !open.value
  files.value = []
  if (open.value) {
    await nextTick()
    field.value?.focus()
  }
}

function outside(event: MouseEvent) {
  if (open.value && root.value && !root.value.contains(event.target as Node)) open.value = false
}
document.addEventListener('mousedown', outside)
onBeforeUnmount(() => document.removeEventListener('mousedown', outside))
</script>

<template>
  <div ref="root" class="picker" @keydown.esc="open = false">
    <button class="current" :class="{ open }" :title="current" @click="toggle">
      <span class="name">{{ shown.name }}</span>
      <span class="where">{{ shown.where }}</span>
      <span class="caret">▾</span>
    </button>
    <p v-if="fit[current] && fit[current].level !== 'ok'" class="fit" :class="fit[current].level" :title="told(current)">
      {{ MARK[fit[current].level] }} {{ said(fit[current]) }}
    </p>
    <div v-if="open" class="panel">
      <div
        v-for="option in options"
        :key="option"
        class="option"
        :class="{ on: option === current }"
        :title="option"
        @click="choose(option)"
      >
        <div>
          <span class="name">{{ parts(option).name }}</span>
          <span class="where">{{ parts(option).where }}</span>
        </div>
        <span v-if="option && fit[option]" class="mark" :class="fit[option].level" :title="told(option)">{{ MARK[fit[option].level] }}</span>
        <span v-if="option === codeDefault" class="tag">default</span>
        <button v-else-if="option && remembered.includes(option)" class="link" title="forget this model" @click.stop="emit('forget', option)">
          ✕
        </button>
      </div>
      <form class="custom" @submit.prevent="useCustom">
        <input ref="field" v-model="custom" placeholder="a Hugging Face or Civitai link, or org/repo@commit[:file]" spellcheck="false" />
        <button type="submit" :disabled="!custom.trim() || resolving">{{ resolving ? 'Pinning…' : 'Use' }}</button>
      </form>
      <div v-if="files.length" class="files">
        <p class="pick">{{ files.length }} files here — pick one</p>
        <div v-for="f in files" :key="f.value" class="option" :title="f.value" @click="take(f.value)">
          <div>
            <span class="name">{{ f.name.split('/').pop() }}</span>
            <span class="where">{{ f.name.includes('/') ? f.name.slice(0, f.name.lastIndexOf('/')) : parts(f.value).where }}</span>
          </div>
          <span class="tag">
            <span v-if="fit[f.value]" class="mark" :class="fit[f.value].level" :title="told(f.value)">{{ MARK[fit[f.value].level] }}</span>
            {{ size(f.size) }}
          </span>
        </div>
      </div>
      <p v-if="problem" class="problem">{{ problem }}</p>
    </div>
  </div>
</template>

<style scoped>
.picker {
  position: relative;
  min-width: 0;
}

.current {
  width: 100%;
  display: grid;
  grid-template-columns: 1fr auto;
  grid-template-rows: auto auto;
  column-gap: var(--space-1);
  text-align: left;
  padding: 6px 10px;
  background: var(--raised);
  border: 1px solid var(--border);
  border-radius: var(--radius-input);
}

.current:hover,
.current.open {
  border-color: var(--border-strong);
}

.current .name {
  grid-column: 1;
}

.current .where {
  grid-column: 1;
}

.current .caret {
  grid-column: 2;
  grid-row: 1 / span 2;
  align-self: center;
  color: var(--dim);
}

.name {
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  color: var(--text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  display: block;
}

.where {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--dim);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  display: block;
}

.panel {
  position: absolute;
  z-index: 20;
  top: calc(100% + 4px);
  right: 0;
  width: max(100%, 440px);
  background: var(--bg-overlay);
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-card);
  box-shadow: var(--shadow-float);
  padding: 4px;
}

.option {
  display: grid;
  grid-template-columns: 1fr auto auto;
  align-items: center;
  gap: var(--space-1);
  padding: 6px 8px;
  border-radius: var(--radius-input);
  cursor: pointer;
}

.option > div {
  min-width: 0;
}

.option:hover {
  background: var(--raised);
}

.option.on .name {
  color: var(--accent);
}

.tag {
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--dim);
}

.custom {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 4px;
  padding: 6px 4px 2px;
  margin-top: 4px;
  border-top: 1px solid var(--border);
}

.custom input {
  font-size: var(--text-xs);
  text-align: left;
}

.files {
  max-height: 320px;
  overflow-y: auto;
  margin-top: 4px;
  border-top: 1px solid var(--border);
}

.pick {
  margin: 6px 8px 2px;
  font-size: 11px;
  color: var(--dim);
}

.fit {
  margin: 4px 2px 0;
  font-size: 11px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  cursor: help;
}

.mark {
  font-size: 12px;
  cursor: help;
}

.ok {
  color: var(--done);
}

.warn,
.unknown {
  color: var(--warn);
}

.bad {
  color: var(--status-error);
}

.problem {
  margin: 4px;
  font-size: 11px;
  color: var(--status-error);
}
</style>
