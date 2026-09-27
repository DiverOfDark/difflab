<script setup lang="ts">
/**
 * Every tool's settings: the code default, what this machine overrides, what one production
 * overrides, and what a job of it would run with. A change is priced before it is saved - the keyed
 * fields that move, and the made things in scope that go stale for it.
 */
import { computed, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { listTools, putSettings, settingsImpact, type SettingField, type ToolInfo } from '../api'
import { state } from '../store'

const route = useRoute()
const scope = ref<'machine' | 'production'>(route.query.production ? 'production' : 'machine')
const productionId = ref<number | null>(route.query.production ? Number(route.query.production) : null)
const tools = ref<ToolInfo[]>([])
const drafts = ref<Record<string, Record<string, string>>>({})
const impact = ref<Record<string, { changed: string[]; stale: Record<string, number> }>>({})
const failure = ref('')
const advanced = ref(false)

const scopeId = computed(() => (scope.value === 'production' && productionId.value !== null ? String(productionId.value) : ''))

async function load() {
  tools.value = await listTools(scope.value === 'production' && productionId.value !== null ? productionId.value : undefined)
  drafts.value = {}
  for (const tool of tools.value) {
    const own = (scope.value === 'machine' ? tool.machine : tool.production) ?? {}
    drafts.value[tool.name] = Object.fromEntries(tool.settings.map((f) => [f.name, own[f.name] === undefined ? '' : String(own[f.name])]))
  }
  impact.value = {}
}
watch([scope, productionId], load, { immediate: true })

const shown = computed(() => tools.value.filter((tool) => tool.settings.length))
const fields = (tool: ToolInfo) => tool.settings.filter((f) => advanced.value || !f.advanced)

function overrides(tool: ToolInfo): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const f of tool.settings) {
    const raw = drafts.value[tool.name]?.[f.name] ?? ''
    if (raw.trim() === '') continue
    out[f.name] = f.type === 'int' || f.type === 'float' ? Number(raw) : f.type === 'bool' ? raw === 'true' : raw
  }
  return out
}

async function price(tool: ToolInfo) {
  failure.value = ''
  try {
    impact.value[tool.name] = await settingsImpact(tool.name, scope.value, scopeId.value, overrides(tool))
  } catch (error) {
    failure.value = String(error)
  }
}

async function save(tool: ToolInfo) {
  failure.value = ''
  try {
    const result = await putSettings(tool.name, scope.value, scopeId.value, overrides(tool))
    if (result.notes.length) failure.value = result.notes.join('; ')
    await load()
  } catch (error) {
    failure.value = String(error)
  }
}

const effective = (tool: ToolInfo, f: SettingField) => String(tool.effective?.[f.name] ?? f.default)
const describe = (f: SettingField) => [f.hint, f.unit ? `in ${f.unit}` : '', f.low !== null || f.high !== null ? `${f.low ?? '…'}..${f.high ?? '…'}` : ''].filter(Boolean).join(' · ')
</script>

<template>
  <main class="screen settings">
    <section class="card">
      <div class="head">
        <h5>Settings</h5>
        <span class="note">the code defaults, overridden for this machine, then for one production; a job records what it ran with</span>
      </div>
      <div class="scope">
        <label>
          <input v-model="scope" type="radio" value="machine" /> this machine
        </label>
        <label>
          <input v-model="scope" type="radio" value="production" /> a production
          <select v-model.number="productionId" :disabled="scope !== 'production'">
            <option v-for="p in state.productions" :key="p.id" :value="p.id">{{ p.title }}</option>
          </select>
        </label>
        <label><input v-model="advanced" type="checkbox" /> show the checkpoints</label>
      </div>
      <p v-if="failure" class="band error">{{ failure }}</p>
    </section>

    <section v-for="tool in shown" :key="tool.name" class="card">
      <div class="head">
        <h5 class="mono">{{ tool.name }}</h5>
        <span class="note">{{ tool.hint }} · key v{{ tool.key_version }}</span>
      </div>
      <div class="knobs">
        <div v-for="f in fields(tool)" :key="f.name" class="knob">
          <div>
            <span class="name">{{ f.name }}<span v-if="f.keyed" class="keyed" title="part of the key: changing it re-makes things"> · keyed</span></span>
            <span class="hint">{{ describe(f) }}</span>
            <span class="hint mono">default {{ String(f.default) }} · effective {{ effective(tool, f) }}</span>
          </div>
          <select v-if="f.choices.length" v-model="drafts[tool.name][f.name]">
            <option value="">— as above —</option>
            <option v-for="c in f.choices" :key="String(c)" :value="String(c)">{{ String(c) }}</option>
          </select>
          <select v-else-if="f.type === 'bool'" v-model="drafts[tool.name][f.name]">
            <option value="">— as above —</option>
            <option value="true">true</option>
            <option value="false">false</option>
          </select>
          <input v-else v-model="drafts[tool.name][f.name]" :type="f.type === 'str' ? 'text' : 'number'" placeholder="—" />
        </div>
      </div>
      <p v-if="impact[tool.name]" class="note">
        <template v-if="impact[tool.name].changed.length">
          moves {{ impact[tool.name].changed.join(', ') }}
          <template v-for="(n, kind) in impact[tool.name].stale" :key="kind"> · {{ n }} {{ kind }} go stale</template>
        </template>
        <template v-else>changes nothing a key includes</template>
      </p>
      <div class="buttons">
        <button @click="price(tool)">What would change</button>
        <button class="primary" @click="save(tool)">Save</button>
      </div>
    </section>
  </main>
</template>

<style scoped>
.settings {
  display: grid;
  gap: var(--space-3);
  align-content: start;
}

.scope {
  display: flex;
  gap: var(--space-3);
  align-items: center;
  flex-wrap: wrap;
  font-size: var(--text-sm);
  padding: var(--space-2);
}

.scope label {
  display: flex;
  gap: var(--space-1);
  align-items: center;
  white-space: nowrap;
}

.scope select {
  width: auto;
  min-width: 200px;
  background: var(--bg-raised);
}

.card > .band,
.card > .note {
  margin: 0 var(--space-2) var(--space-2);
}

.knobs {
  padding: 0 var(--space-2);
}

.knob {
  grid-template-columns: 1fr 160px;
}

.keyed {
  color: var(--accent);
  font-size: var(--text-xs);
}

.buttons {
  display: flex;
  justify-content: flex-end;
  gap: var(--space-1);
  padding: var(--space-2);
}
</style>
