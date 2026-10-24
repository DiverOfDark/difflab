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
import { ROLES } from '../roles'
import { state } from '../store'

const ROLE_NOTES: Record<string, string> = {
  vlm: 'every question, prompt and plan, and the judge; loaded as published if it is already 4-bit, quantized to 4-bit as it loads if it is not quantized',
  editor: 'edits photos and draws keyframes',
  'editor-21': 'the experimental engine: draws in its place when the engine is set to qwen-image-2.1',
  'editor-21-loras': 'stacked on Qwen-Image 2.1 in order; strength 0 turns one off without removing it',
  'editor-21-text-encoder': 'reads the instruction and the photos for Qwen-Image 2.1',
  'editor-21-vae': "takes photos into and out of Qwen-Image 2.1; diffusers' layout, as Qwen ships it",
  'video-model': 'samples every clip',
  'video-loras': 'stacked on the video model in order; a video model without a turbo fused in needs a turbo LoRA first; strength 0 turns one off without removing it',
  'video-text-encoder': 'reads the caption for the video model',
  'video-vae': 'turns sampled latents into frames',
  'audio-vae': 'turns sampled latents into sound',
  'preview-decoder': 'the quick picture of each step while a clip samples',
  'ltx-model': 'samples a take when its video model is LTX: a 2.5 or a 2.3 model, read from the file',
  'ltx-loras': 'stacked on LTX-2.5 when a take has references; the first carrying slot embeddings (MSR) places them',
  'ltx-text-encoder': 'reads the prompt for LTX-2.5: its own Gemma 4 with the projections',
  'ltx23-text-encoder': 'reads the prompt for LTX-2.3: Gemma 3 12B; the projections come with the model',
  'ltx23-loras': 'stacked on an LTX-2.3 model for every clip: a DMD LoRA, a style; 2.3 places no references',
  'ltx-video-vae': 'takes frames into and out of LTX, unless the model file carries its own',
  'ltx-audio-vae': 'turns LTX latents into sound, with its vocoder, unless the model file carries its own',
  'pixel-upscaler': 'enlarges finished frames',
  restorer: 'restores a finished chapter at half its 4K size, before the upscaler doubles it',
  'restorer-vae': 'takes frames into and out of the restorer',
  'frame-interpolation': 'draws the frames between frames, for 48 or 60 fps',
}
// Tools whose knobs share every name are drawn as one table, a column each.
const TABS: { key: string; title: string; roles: string[]; tools: [string, string][] }[] = [
  {
    key: 'writing',
    title: 'Writing',
    roles: ['vlm'],
    tools: [['ask_vlm', 'Questions'], ['write_instruction', 'Edit instructions'], ['write_video_prompt', 'H3 captions'], ['write_ltx_prompt', 'LTX prompts'], ['judge', 'Judge']],
  },
  {
    key: 'pictures',
    title: 'Pictures',
    roles: ['editor', 'editor-21', 'editor-21-loras', 'editor-21-text-encoder', 'editor-21-vae'],
    tools: [['edit_image', 'Editing a photo'], ['compose_keyframe', 'Drawing a keyframe']],
  },
  {
    key: 'h3',
    title: 'MiniMax H3',
    roles: ['video-model', 'video-loras', 'video-text-encoder', 'video-vae', 'audio-vae', 'preview-decoder'],
    tools: [['render_chunk', 'Rendering'], ['handoff', 'Joining shots']],
  },
  {
    key: 'ltx',
    title: 'LTX',
    roles: ['ltx-model', 'ltx-loras', 'ltx-text-encoder', 'ltx23-text-encoder', 'ltx23-loras', 'ltx-video-vae', 'ltx-audio-vae'],
    tools: [['render_ltx', 'Rendering']],
  },
  {
    key: 'finishing',
    title: 'Finishing',
    roles: ['pixel-upscaler', 'restorer', 'restorer-vae', 'frame-interpolation'],
    tools: [['restore', 'At 4K'], ['retime', 'To 60 fps']],
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

// --- models: one row per role, set on every tool that loads it. Every one is listed: a model hidden behind
// Advanced, and left out of its count, read as one that could not be changed.
const roles = computed(() => {
  const found = new Map<string, { tool: ToolInfo; field: SettingField }[]>()
  for (const tool of tools.value)
    for (const field of tool.settings) if (field.role) found.set(field.role, [...(found.get(field.role) ?? []), { tool, field }])
  return [...found.entries()]
    .map(([role, uses]) => ({ role, uses }))
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
const fieldOf = (tool: ToolInfo, name: string) => tool.settings.find((f) => f.name === name)!
const differs = (tool: ToolInfo, f: SettingField) =>
  (drafts.value[tool.name]?.[f.name] ?? null) !== (own(tool)[f.name] === undefined ? null : String(own(tool)[f.name]))

type Column = { tool: ToolInfo; title: string }
function blocks(names: [string, string][]) {
  const out: { columns: Column[]; fields: SettingField[] }[] = []
  for (const [name, title] of names) {
    const tool = byName.value[name]
    const fields = tool ? knobs(tool) : []
    if (!fields.length) continue
    const last = out.at(-1)
    if (last && last.fields.map((f) => f.name).join() === fields.map((f) => f.name).join()) last.columns.push({ tool, title })
    else out.push({ columns: [{ tool, title }], fields })
  }
  return out
}

const placed = new Set(TABS.flatMap((t) => t.roles))
const tabs = computed(() => {
  const loose = roles.value.map((r) => r.role).filter((r) => !placed.has(r))
  return [...TABS, ...(loose.length ? [{ key: 'other', title: 'Other', roles: loose, tools: [] }] : [])].map((t) => {
    const models = t.roles.flatMap((role) => roles.value.filter((r) => r.role === role))
    const tools = t.tools.map(([name]) => byName.value[name]).filter(Boolean)
    const fields = [...models.flatMap((r) => r.uses), ...tools.flatMap((tool) => tool.settings.filter((f) => !f.role).map((field) => ({ tool, field })))]
    return {
      ...t,
      models,
      blocks: blocks(t.tools),
      here: models.filter((r) => r.uses.some((u) => setHere(u.tool, u.field))).length + fields.filter((u) => !u.field.role && setHere(u.tool, u.field)).length,
      changed: fields.some((u) => differs(u.tool, u.field)),
      hidden: tools.reduce((n, tool) => n + tool.settings.filter((f) => !f.role && f.advanced).length, 0),
    }
  })
})
const tab = ref(localStorage.getItem('settings.tab') ?? TABS[0].key)
watch(tab, (t) => localStorage.setItem('settings.tab', t))
const current = computed(() => tabs.value.find((t) => t.key === tab.value) ?? tabs.value[0])
const hidden = computed(() => tabs.value.reduce((n, t) => n + t.hidden, 0))

const describe = (f: SettingField) =>
  [f.hint, f.unit, f.low !== null || f.high !== null ? `${f.low ?? '…'} – ${f.high ?? '…'}` : ''].filter(Boolean).join(' · ')
</script>

<template>
  <main class="screen settings">
    <aside class="rail">
      <div class="seg">
        <button :class="{ on: scope === 'machine' }" @click="scope = 'machine'">This machine</button>
        <button :class="{ on: scope === 'production' }" :disabled="!state.productions.length" @click="scope = 'production'">A production</button>
      </div>
      <select v-if="scope === 'production'" v-model.number="productionId" class="production">
        <option v-for="p in state.productions" :key="p.id" :value="p.id">{{ p.title }}</option>
      </select>
      <p class="note">
        {{ scope === 'machine' ? 'Every run on this machine, unless a production says otherwise.' : 'Only this production; the rest comes from this machine.' }}
      </p>

      <nav class="tabs">
        <button v-for="t in tabs" :key="t.key" class="tab" :class="{ on: current.key === t.key }" @click="tab = t.key">
          <span>{{ t.title }}</span>
          <i v-if="t.changed" class="unsaved" title="unsaved changes" />
          <span v-if="t.here" class="count" :title="`${t.here} set ${scope === 'machine' ? 'for this machine' : 'for this production'}`">{{ t.here }}</span>
        </button>
        <button class="tab" :class="{ on: tab === 'accounts' }" @click="tab = 'accounts'">
          <span>Accounts</span>
        </button>
      </nav>

      <label class="toggle"><input v-model="advanced" type="checkbox" /> Advanced{{ advanced ? '' : ` (${hidden} more)` }}</label>
    </aside>

    <section class="pane">
      <div class="body">
        <section v-if="tab === 'accounts'" class="card">
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

        <div v-else class="split" :class="{ wide: current.models.length < 3 }">
          <section class="card">
            <div class="head"><h5>Models</h5><span class="note">pick one, paste another, or forget one you no longer use</span></div>
            <div class="rows">
              <div v-for="r in current.models" :key="r.role" class="row model" :class="{ list: r.uses[0].field.multiple }">
                <div class="what">
                  <span class="title">
                    {{ ROLES[r.role] ?? r.role }}
                    <button v-if="r.uses.some((u) => setHere(u.tool, u.field))" class="link" title="back to what the layer below says" @click="pickRole(r.uses, inherited(r.uses[0].tool, r.uses[0].field))">
                      reset
                    </button>
                  </span>
                  <span class="hint" :title="ROLE_NOTES[r.role] ?? r.uses[0].field.hint">{{ ROLE_NOTES[r.role] ?? r.uses[0].field.hint }}</span>
                </div>
                <LoraList
                  v-if="r.uses[0].field.multiple"
                  :value="roleValue(r.uses) ?? ''"
                  :role="r.role"
                  :remembered="remembered[r.role] ?? []"
                  @pick="(v) => pickRole(r.uses, v)"
                  @forget="(v) => forget(r.role, v)"
                />
                <ModelPicker
                  v-else
                  :value="roleValue(r.uses)"
                  :fallback="String(r.uses[0].field.default)"
                  :code-default="String(r.uses[0].field.default)"
                  :role="r.role"
                  :remembered="remembered[r.role] ?? []"
                  @pick="(v) => pickRole(r.uses, v)"
                  @forget="(v) => forget(r.role, v)"
                />
              </div>
            </div>
          </section>

          <section class="card">
            <div class="head"><h5>Settings</h5><span v-if="!advanced && current.hidden" class="note">{{ current.hidden }} more under Advanced</span></div>
            <p v-if="!current.blocks.length" class="empty note">nothing to tune here{{ advanced || !current.hidden ? '' : ' without Advanced' }}</p>
            <div v-for="b in current.blocks" :key="b.columns[0].tool.name" class="tool" :style="{ '--columns': b.columns.length }">
              <div class="grid tool-head">
                <span v-if="b.columns.length === 1" class="title">{{ b.columns[0].title }} <span class="note">{{ b.columns[0].tool.hint }}</span></span>
                <template v-else>
                  <span />
                  <span v-for="c in b.columns" :key="c.tool.name" class="title column" :title="c.tool.hint">{{ c.title }}</span>
                </template>
              </div>
              <div v-for="f in b.fields" :key="f.name" class="grid row" :class="{ here: b.columns.some((c) => setHere(c.tool, fieldOf(c.tool, f.name))) }">
                <div class="what">
                  <span class="title">
                    <i class="mark" :title="scope === 'machine' ? 'set for this machine' : 'set for this production'" />
                    {{ f.label || f.name }}
                    <button v-if="b.columns.some((c) => setHere(c.tool, fieldOf(c.tool, f.name)))" class="link" @click="b.columns.forEach((c) => (drafts[c.tool.name][f.name] = null))">reset</button>
                  </span>
                  <span class="hint" :title="describe(f)">{{ describe(f) }}</span>
                </div>
                <div v-for="c in b.columns" :key="c.tool.name" class="cell" :class="{ set: setHere(c.tool, fieldOf(c.tool, f.name)) }">
                  <select v-if="f.choices.length" :value="value(c.tool, fieldOf(c.tool, f.name))" @change="set(c.tool, fieldOf(c.tool, f.name), ($event.target as HTMLSelectElement).value)">
                    <option v-for="o in f.choices" :key="String(o)" :value="String(o)">{{ String(o) }}</option>
                  </select>
                  <label v-else-if="f.type === 'bool'" class="switch">
                    <input
                      type="checkbox"
                      :checked="value(c.tool, fieldOf(c.tool, f.name)) === 'true'"
                      @change="set(c.tool, fieldOf(c.tool, f.name), String(($event.target as HTMLInputElement).checked))"
                    />
                    {{ value(c.tool, fieldOf(c.tool, f.name)) === 'true' ? 'on' : 'off' }}
                  </label>
                  <input
                    v-else
                    :value="value(c.tool, fieldOf(c.tool, f.name))"
                    :type="f.type === 'str' ? 'text' : 'number'"
                    :min="f.low ?? undefined"
                    :max="f.high ?? undefined"
                    :step="f.type === 'float' ? 'any' : 1"
                    @input="set(c.tool, fieldOf(c.tool, f.name), ($event.target as HTMLInputElement).value)"
                  />
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>

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
    </section>
  </main>
</template>

<style scoped>
.settings {
  display: grid;
  grid-template-columns: 220px minmax(0, 1fr);
  grid-template-rows: minmax(0, 1fr);
  gap: var(--space-3);
  height: calc(100vh - 56px);
  overflow: hidden;
}

.rail {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  min-height: 0;
}

.rail .seg {
  display: grid;
  grid-template-columns: 1fr 1fr;
}

.rail .seg > * {
  padding-inline: 4px;
}

.rail > .note {
  margin: 0;
  font-size: 11px;
  line-height: 1.4;
}

.production {
  background: var(--raised);
  font-size: var(--text-sm);
}

.tabs {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding-top: var(--space-1);
  border-top: 1px solid var(--line);
}

.tab {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  padding: 8px 10px;
  border: 1px solid transparent;
  border-radius: var(--radius-input);
  background: transparent;
  color: var(--muted);
  font-size: var(--text-sm);
  text-align: left;
  text-transform: none;
  letter-spacing: normal;
}

.tab > span:first-child {
  flex: 1;
}

.tab:hover {
  color: var(--text);
  background: var(--surface);
}

.tab.on {
  color: var(--text);
  background: var(--surface);
  border-color: var(--line);
  box-shadow: inset 2px 0 0 var(--accent);
}

.count {
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--accent);
}

.unsaved {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--accent);
}

.toggle {
  margin-top: auto;
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

.pane {
  display: flex;
  flex-direction: column;
  min-height: 0;
  min-width: 0;
}

.pane > .body {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
}

.pane > .band {
  margin: var(--space-2) 0 0;
}

.split {
  display: grid;
  grid-template-columns: minmax(0, 3fr) minmax(0, 2fr);
  gap: var(--space-3);
  align-items: start;
}

.split.wide {
  grid-template-columns: minmax(0, 1fr);
}

.rows,
.tool {
  padding: 0 var(--space-2);
}

.tool + .tool {
  border-top: 1px solid var(--line);
}

.empty {
  padding: var(--space-2);
  margin: 0;
}

.grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) repeat(var(--columns), 120px);
  gap: var(--space-2);
  align-items: center;
}

.tool-head {
  padding: var(--space-2) 0 var(--space-1);
}

.tool-head .title {
  grid-column: 1 / -1;
  font-size: var(--text-sm);
  font-weight: var(--weight-medium);
}

.tool-head .title.column {
  grid-column: auto;
  font-size: var(--text-xs);
  text-align: right;
}

.tool-head .note {
  font-weight: normal;
}

.row {
  padding: 6px 0;
}

.row + .row {
  border-top: 1px dashed var(--line);
}

.row.model {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 300px;
  gap: var(--space-2);
  align-items: center;
  padding: 8px 0;
}

.row.model.list {
  grid-template-columns: minmax(0, 1fr) 360px;
  align-items: start;
}

.wide .row.model {
  grid-template-columns: minmax(0, 1fr) 420px;
}

.rows > .row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 320px;
  gap: var(--space-2);
  align-items: center;
}

.token {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 4px;
}

.row .token input {
  text-align: left;
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
  overflow: hidden;
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 1;
}

.row.model .hint {
  -webkit-line-clamp: 2;
}

.row:focus-within .hint {
  -webkit-line-clamp: unset;
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

.cell input,
.cell select,
.row .token input {
  font-size: var(--text-xs);
  background: var(--raised);
  font-family: var(--font-mono);
}

.cell input:not([type='checkbox']),
.cell select {
  text-align: right;
  width: 100%;
}

.cell.set input,
.cell.set select {
  border-color: var(--accent-muted);
}

.foot .note {
  font-size: var(--text-sm);
}
</style>
