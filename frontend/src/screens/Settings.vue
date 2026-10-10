<script setup lang="ts">
/**
 * Every tool's settings: the code default, what this machine overrides, what one production
 * overrides. Models are set once per role across the tools that load them; a change is priced as it
 * is made - the keyed fields that move, and the made things in scope that go stale for it.
 */
import { computed, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import LoraList from '../components/LoraList.vue'
import ModelPicker from '../components/ModelPicker.vue'
import {
  forgetModel,
  listAccounts,
  listModels,
  listTools,
  putSettings,
  saveAccount,
  settingsImpact,
  type Account,
  type SettingField,
  type ToolInfo,
} from '../api'
import { state } from '../store'

const ROLES: Record<string, string> = {
  vlm: 'VLM',
  editor: 'Image editor',
  'editor-21': 'Qwen-Image 2.1',
  'editor-21-loras': 'Qwen-Image 2.1 LoRAs',
  'editor-21-text-encoder': 'Qwen-Image 2.1 text encoder',
  'video-model': 'Video model',
  'video-loras': 'LoRAs',
  'video-text-encoder': 'Video text encoder',
  'video-vae': 'Video VAE',
  'audio-vae': 'Audio VAE',
  'preview-decoder': 'Preview decoder',
  'pixel-upscaler': 'Pixel upscaler',
  restorer: 'Restorer',
  'restorer-vae': 'Restorer VAE',
  'frame-interpolation': 'Frame interpolation',
}
const ROLE_NOTES: Record<string, string> = {
  vlm: 'every question, prompt and plan, and the judge; loaded as published if it is already 4-bit, quantized to 4-bit as it loads if it is not quantized',
  editor: 'edits photos and draws keyframes',
  'editor-21': 'the experimental engine: draws in its place when the engine is set to qwen-image-2.1',
  'editor-21-loras': 'stacked on Qwen-Image 2.1 in order; strength 0 turns one off without removing it',
  'editor-21-text-encoder': 'reads the instruction and the photos for Qwen-Image 2.1',
  'video-model': 'samples every clip',
  'video-loras': 'stacked on the video model in order; a video model without a turbo fused in needs a turbo LoRA first; strength 0 turns one off without removing it',
  'video-text-encoder': 'reads the caption for the video model',
  'video-vae': 'turns sampled latents into frames',
  'audio-vae': 'turns sampled latents into sound',
  'preview-decoder': 'the quick picture of each step while a clip samples',
  'pixel-upscaler': 'enlarges finished frames',
  restorer: 'restores a finished chapter at half its 4K size, before the upscaler doubles it',
  'restorer-vae': 'takes frames into and out of the restorer',
  'frame-interpolation': 'draws the frames between frames, for 48 or 60 fps',
}
const GROUPS: { title: string; tools: [string, string][] }[] = [
  { title: 'Pictures', tools: [['edit_image', 'Editing a photo'], ['compose_keyframe', 'Drawing a keyframe']] },
  {
    title: 'Video',
    tools: [['render_chunk', 'Rendering'], ['handoff', 'Joining shots'], ['restore', 'Finishing at 4K'], ['retime', 'Retiming to 60 fps']],
  },
  { title: 'Checking', tools: [['judge', 'Judge']] },
  {
    title: 'Writing',
    tools: [['ask_vlm', 'Questions'], ['write_instruction', 'Edit instructions'], ['write_video_prompt', 'Video captions']],
  },
]

const route = useRoute()
const scope = ref<'machine' | 'production'>(route.query.production ? 'production' : 'machine')
const productionId = ref<number | null>(route.query.production ? Number(route.query.production) : (state.productions[0]?.id ?? null))
const tools = ref<ToolInfo[]>([])
// null: nothing set at this scope, so the value comes from the layer below.
const drafts = ref<Record<string, Record<string, string | null>>>({})
const remembered = ref<Record<string, string[]>>({})
const stale = ref<{ changed: string[]; stale: Record<string, number> } | null>(null)
const failure = ref('')
const saving = ref(false)
const advanced = ref(localStorage.getItem('settings.advanced') === '1')
watch(advanced, (on) => localStorage.setItem('settings.advanced', on ? '1' : '0'))

const scopeId = computed(() => (scope.value === 'production' && productionId.value !== null ? String(productionId.value) : ''))
const byName = computed(() => Object.fromEntries(tools.value.map((t) => [t.name, t])))

const own = (tool: ToolInfo) => (scope.value === 'machine' ? tool.machine : tool.production) ?? {}

async function load() {
  const [loaded, models] = await Promise.all([
    listTools(scope.value === 'production' && productionId.value !== null ? productionId.value : undefined),
    listModels(),
  ])
  tools.value = loaded
  remembered.value = models
  drafts.value = Object.fromEntries(
    loaded.map((tool) => [tool.name, Object.fromEntries(tool.settings.map((f) => [f.name, own(tool)[f.name] === undefined ? null : String(own(tool)[f.name])]))]),
  )
  stale.value = null
}
watch([scope, productionId], load, { immediate: true })

// What a field comes to with nothing set at this scope.
function inherited(tool: ToolInfo, f: SettingField): string {
  if (scope.value === 'production' && tool.machine?.[f.name] !== undefined) return String(tool.machine[f.name])
  return String(f.default)
}

const value = (tool: ToolInfo, f: SettingField) => drafts.value[tool.name]?.[f.name] ?? inherited(tool, f)
const setHere = (tool: ToolInfo, f: SettingField) => drafts.value[tool.name]?.[f.name] != null

function set(tool: ToolInfo, f: SettingField, raw: string) {
  drafts.value[tool.name][f.name] = raw === inherited(tool, f) ? null : raw
}

function overrides(tool: ToolInfo): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const f of tool.settings) {
    const raw = drafts.value[tool.name]?.[f.name]
    if (raw == null) continue
    out[f.name] = f.type === 'int' || f.type === 'float' ? Number(raw) : f.type === 'bool' ? raw === 'true' : raw
  }
  return out
}

const same = (a: Record<string, unknown>, b: Record<string, unknown>) =>
  JSON.stringify(Object.entries(a).sort()) === JSON.stringify(Object.entries(b).sort())
const dirty = computed(() => tools.value.filter((tool) => !same(overrides(tool), own(tool))))
const changes = computed(() =>
  dirty.value.reduce((n, tool) => n + new Set([...Object.keys(overrides(tool)), ...Object.keys(own(tool))].filter((k) => overrides(tool)[k] !== own(tool)[k])).size, 0),
)

let pricing = 0
watch(
  () => JSON.stringify(drafts.value),
  () => {
    const ticket = ++pricing
    setTimeout(async () => {
      if (ticket !== pricing) return
      if (!dirty.value.length) return void (stale.value = null)
      try {
        const results = await Promise.all(dirty.value.map((t) => settingsImpact(t.name, scope.value, scopeId.value, overrides(t))))
        if (ticket !== pricing) return
        const total: Record<string, number> = {}
        for (const r of results) for (const [kind, n] of Object.entries(r.stale)) total[kind] = (total[kind] ?? 0) + n
        stale.value = { changed: [...new Set(results.flatMap((r) => r.changed))], stale: total }
      } catch (error) {
        failure.value = String(error)
      }
    }, 300)
  },
)

async function save() {
  failure.value = ''
  saving.value = true
  try {
    const notes: string[] = []
    for (const tool of dirty.value) notes.push(...(await putSettings(tool.name, scope.value, scopeId.value, overrides(tool))).notes)
    if (notes.length) failure.value = notes.join('; ')
    await load()
  } catch (error) {
    failure.value = String(error)
  } finally {
    saving.value = false
  }
}

// --- models: one row per role, set on every tool that loads it --------------------------------
const roles = computed(() => {
  const found = new Map<string, { tool: ToolInfo; field: SettingField }[]>()
  for (const tool of tools.value)
    for (const field of tool.settings) if (field.role) found.set(field.role, [...(found.get(field.role) ?? []), { tool, field }])
  return [...found.entries()]
    .map(([role, uses]) => ({ role, uses, advanced: uses.every((u) => u.field.advanced) }))
    .filter((r) => advanced.value || !r.advanced)
    .sort((a, b) => Object.keys(ROLES).indexOf(a.role) - Object.keys(ROLES).indexOf(b.role))
})

function roleValue(uses: { tool: ToolInfo; field: SettingField }[]) {
  const values = new Set(uses.map((u) => value(u.tool, u.field)))
  return values.size === 1 ? [...values][0] : null
}

function pickRole(uses: { tool: ToolInfo; field: SettingField }[], picked: string) {
  for (const u of uses) set(u.tool, u.field, picked)
}

async function forget(role: string, model: string) {
  remembered.value = await forgetModel(role, model)
}

// --- accounts: tokens for downloads, checked with the service as they are saved -----------------
const SERVICES: { name: string; title: string; note: string; where: string }[] = [
  { name: 'huggingface', title: 'Hugging Face', note: 'gated and private repos', where: 'https://huggingface.co/settings/tokens' },
  { name: 'civitai', title: 'Civitai', note: 'most Civitai files will not download without one', where: 'https://civitai.com/user/account' },
]
const accounts = ref<Record<string, Account>>({})
const typed = ref<Record<string, string>>({})
const checking = ref('')
listAccounts().then((a) => (accounts.value = a))

async function saveToken(service: string, token: string) {
  failure.value = ''
  checking.value = service
  try {
    accounts.value = await saveAccount(service, token)
    typed.value[service] = ''
  } catch (error) {
    failure.value = String(error).replace(/^Error: /, '')
  } finally {
    checking.value = ''
  }
}

function signedIn(a: Account | undefined) {
  if (!a?.source) return 'no token: public files only'
  if (a.source === 'settings') return `saved · ${a.account || 'account unknown'} · …${a.tail}`
  return `from ${a.source} · …${a.tail}`
}

// --- knobs -------------------------------------------------------------------------------------
const knobs = (tool: ToolInfo) => tool.settings.filter((f) => !f.role && (advanced.value || !f.advanced))
const groups = computed(() =>
  GROUPS.map((g) => ({
    title: g.title,
    tools: g.tools
      .map(([name, title]) => ({ tool: byName.value[name], title }))
      .filter((t) => t.tool && knobs(t.tool).length),
  })).filter((g) => g.tools.length),
)
const hidden = computed(() => tools.value.reduce((n, t) => n + t.settings.filter((f) => !f.role && f.advanced).length, 0))

const describe = (f: SettingField) =>
  [f.hint, f.unit, f.low !== null || f.high !== null ? `${f.low ?? '…'} – ${f.high ?? '…'}` : ''].filter(Boolean).join(' · ')
</script>

<template>
  <main class="screen settings">
    <header class="top">
      <div class="seg">
        <button :class="{ on: scope === 'machine' }" @click="scope = 'machine'">This machine</button>
        <button :class="{ on: scope === 'production' }" :disabled="!state.productions.length" @click="scope = 'production'">A production</button>
      </div>
      <select v-if="scope === 'production'" v-model.number="productionId" class="production">
        <option v-for="p in state.productions" :key="p.id" :value="p.id">{{ p.title }}</option>
      </select>
      <span class="note grow">
        {{ scope === 'machine' ? 'Every run on this machine, unless a production says otherwise.' : 'Only this production; the rest comes from this machine.' }}
      </span>
      <label class="toggle"><input v-model="advanced" type="checkbox" /> Advanced{{ advanced ? '' : ` (${hidden} more)` }}</label>
    </header>

    <section class="card">
      <div class="head"><h5>Models</h5><span class="note">pick one, paste another, or forget one you no longer use</span></div>
      <div class="rows">
        <div v-for="r in roles" :key="r.role" class="row" :class="{ list: r.uses[0].field.multiple }">
          <div class="what">
            <span class="title">
              {{ ROLES[r.role] ?? r.role }}
              <button v-if="r.uses.some((u) => setHere(u.tool, u.field))" class="link" title="back to what the layer below says" @click="pickRole(r.uses, inherited(r.uses[0].tool, r.uses[0].field))">
                reset
              </button>
            </span>
            <span class="hint">{{ ROLE_NOTES[r.role] ?? r.uses[0].field.hint }}</span>
          </div>
          <LoraList
            v-if="r.uses[0].field.multiple"
            :value="roleValue(r.uses) ?? ''"
            :remembered="remembered[r.role] ?? []"
            @pick="(v) => pickRole(r.uses, v)"
            @forget="(v) => forget(r.role, v)"
          />
          <ModelPicker
            v-else
            :value="roleValue(r.uses)"
            :fallback="String(r.uses[0].field.default)"
            :code-default="String(r.uses[0].field.default)"
            :remembered="remembered[r.role] ?? []"
            @pick="(v) => pickRole(r.uses, v)"
            @forget="(v) => forget(r.role, v)"
          />
        </div>
      </div>
    </section>

    <section class="card">
      <div class="head"><h5>Accounts</h5><span class="note">tokens for downloading models; kept on this machine, never shown again</span></div>
      <div class="rows">
        <div v-for="svc in SERVICES" :key="svc.name" class="row">
          <div class="what">
            <span class="title">
              {{ svc.title }}
              <button v-if="accounts[svc.name]?.source === 'settings'" class="link" :disabled="checking === svc.name" @click="saveToken(svc.name, '')">remove</button>
            </span>
            <span class="hint">{{ signedIn(accounts[svc.name]) }} · {{ svc.note }} · <a :href="svc.where" target="_blank" rel="noopener">get a token</a></span>
          </div>
          <form class="token" @submit.prevent="saveToken(svc.name, typed[svc.name] ?? '')">
            <input v-model="typed[svc.name]" type="password" autocomplete="off" placeholder="paste a token" spellcheck="false" />
            <button type="submit" :disabled="!typed[svc.name]?.trim() || checking === svc.name">{{ checking === svc.name ? 'Checking…' : 'Save' }}</button>
          </form>
        </div>
      </div>
    </section>

    <section v-for="g in groups" :key="g.title" class="card">
      <div class="head"><h5>{{ g.title }}</h5></div>
      <div v-for="t in g.tools" :key="t.tool.name" class="tool">
        <div class="tool-head">
          <span class="title">{{ t.title }}</span><span class="note">{{ t.tool.hint }}</span>
        </div>
        <div v-for="f in knobs(t.tool)" :key="f.name" class="row" :class="{ here: setHere(t.tool, f) }">
          <div class="what">
            <span class="title">
              <i class="mark" :title="scope === 'machine' ? 'set for this machine' : 'set for this production'" />
              {{ f.label || f.name }}
              <button v-if="setHere(t.tool, f)" class="link" @click="drafts[t.tool.name][f.name] = null">reset</button>
            </span>
            <span class="hint">{{ describe(f) }}</span>
          </div>
          <select v-if="f.choices.length" :value="value(t.tool, f)" @change="set(t.tool, f, ($event.target as HTMLSelectElement).value)">
            <option v-for="c in f.choices" :key="String(c)" :value="String(c)">{{ String(c) }}</option>
          </select>
          <label v-else-if="f.type === 'bool'" class="switch">
            <input type="checkbox" :checked="value(t.tool, f) === 'true'" @change="set(t.tool, f, String(($event.target as HTMLInputElement).checked))" />
            {{ value(t.tool, f) === 'true' ? 'on' : 'off' }}
          </label>
          <input
            v-else
            :value="value(t.tool, f)"
            :type="f.type === 'str' ? 'text' : 'number'"
            :min="f.low ?? undefined"
            :max="f.high ?? undefined"
            :step="f.type === 'float' ? 'any' : 1"
            @input="set(t.tool, f, ($event.target as HTMLInputElement).value)"
          />
        </div>
      </div>
    </section>

    <p v-if="failure" class="band error">{{ failure }}</p>

    <footer v-if="dirty.length" class="foot">
      <span class="note">
        {{ changes }} change{{ changes === 1 ? '' : 's' }}
        <template v-if="stale">
          <template v-if="stale.changed.length">
            · re-makes what depends on {{ stale.changed.join(', ') }}
            <template v-for="(n, kind) in stale.stale" :key="kind"> · {{ n }} {{ kind }} go stale</template>
          </template>
          <template v-else> · nothing already made goes stale</template>
        </template>
      </span>
      <div class="buttons">
        <button :disabled="saving" @click="load">Discard</button>
        <button class="primary" :disabled="saving" @click="save">{{ saving ? 'Saving…' : 'Save' }}</button>
      </div>
    </footer>
  </main>
</template>

<style scoped>
.settings {
  display: grid;
  gap: var(--space-3);
  align-content: start;
  max-width: 960px;
  margin: 0 auto;
  width: 100%;
}

.top {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  flex-wrap: wrap;
}

.top .grow {
  flex: 1;
  min-width: 200px;
}

.production {
  width: auto;
  min-width: 220px;
  background: var(--raised);
}

.toggle,
.switch {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  font-size: var(--text-sm);
  white-space: nowrap;
  cursor: pointer;
}

.switch {
  justify-content: flex-end;
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  color: var(--muted);
}

.rows,
.tool {
  padding: 0 var(--space-2);
}

.tool + .tool {
  border-top: 1px solid var(--line);
}

.tool-head {
  display: flex;
  align-items: baseline;
  gap: var(--space-1);
  padding: var(--space-2) 0 var(--space-1);
}

.tool-head .title {
  font-size: var(--text-sm);
  font-weight: var(--weight-medium);
}

.row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 280px;
  gap: var(--space-2);
  align-items: center;
  padding: 10px 0;
}

.row + .row {
  border-top: 1px dashed var(--line);
}

.tool .row {
  grid-template-columns: minmax(0, 1fr) 160px;
}

.token {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 4px;
}

.row .token input {
  text-align: left;
}

.row.list {
  grid-template-columns: minmax(0, 1fr) 380px;
  align-items: start;
}

.what {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}

.what .title {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  font-size: var(--text-sm);
}

.what .hint {
  font-size: 11px;
  font-weight: var(--weight-light);
  color: var(--muted);
  line-height: 1.4;
}

.mark {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: transparent;
  flex-shrink: 0;
  margin-left: -12px;
}

.here .mark {
  background: var(--accent);
}

.row input,
.row select {
  text-align: right;
  font-size: var(--text-xs);
  background: var(--raised);
  font-family: var(--font-mono);
}

.here input,
.here select {
  border-color: var(--accent-muted);
}

.foot .note {
  font-size: var(--text-sm);
}
</style>
