<script setup lang="ts">
import { computed, ref } from 'vue'
import { deleteDownload, listDownloads, type Download, type DownloadUse } from '../api'
import { ROLES } from '../roles'
import DeleteButton from './DeleteButton.vue'

const emit = defineEmits<{ failed: [message: string] }>()

const entries = ref<Download[] | null>(null)
const show = ref<'all' | 'used' | 'unused'>('all')
const open = ref<Set<string>>(new Set())

const load = async () => (entries.value = await listDownloads())
load().catch((error) => emit('failed', String(error)))

// "remembered" is the picker's history: it names a file without anything loading it.
const named = (uses: DownloadUse[]) => uses.filter((u) => u.where !== 'remembered')
const usesOf = (d: Download) => d.files.flatMap((f) => f.uses)
const used = (d: Download) => named(usesOf(d)).length > 0

const total = (list: Download[]) => list.reduce((n, d) => n + d.size, 0)
const counts = computed(() => {
  const all = entries.value ?? []
  const inUse = all.filter(used)
  const idle = all.filter((d) => !used(d))
  return { all: [all.length, total(all)], used: [inUse.length, total(inUse)], unused: [idle.length, total(idle)] }
})
const shown = computed(() => (entries.value ?? []).filter((d) => show.value === 'all' || used(d) === (show.value === 'used')))

function size(bytes: number) {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`
  if (bytes >= 1e6) return `${Math.round(bytes / 1e6)} MB`
  return `${Math.max(1, Math.round(bytes / 1e3))} KB`
}

function labels(uses: DownloadUse[]) {
  const out = new Map<string, string[]>()
  for (const u of named(uses)) {
    const what = u.role ? (ROLES[u.role] ?? u.role) : 'built in'
    const where = u.where === 'this machine' || u.where === 'built in' ? [] : [u.where]
    out.set(what, [...new Set([...(out.get(what) ?? []), ...where])])
  }
  return [...out].map(([what, where]) => (where.length ? `${what} · ${where.join(', ')}` : what))
}

const offered = (uses: DownloadUse[]) => [...new Set(uses.filter((u) => u.where === 'remembered').map((u) => ROLES[u.role] ?? u.role))]

function toggle(id: string) {
  const next = new Set(open.value)
  if (!next.delete(id)) next.add(id)
  open.value = next
}

async function remove(d: Download, files: string[] = []) {
  entries.value = await deleteDownload(d.id, files)
}

const SOURCE: Record<Download['source'], string> = { huggingface: 'Hugging Face', civitai: 'Civitai', url: 'URL' }
</script>

<template>
  <section class="card downloads">
    <div class="head">
      <h5>Downloads</h5>
      <div class="seg">
        <button :class="{ on: show === 'all' }" @click="show = 'all'">All · {{ size(counts.all[1]) }}</button>
        <button :class="{ on: show === 'used' }" @click="show = 'used'">In use · {{ size(counts.used[1]) }}</button>
        <button :class="{ on: show === 'unused' }" @click="show = 'unused'">Not used · {{ size(counts.unused[1]) }}</button>
      </div>
    </div>
    <p class="note lead">
      Model files on this disk. In use means a setting on this machine or a production names it, or difflab
      loads it itself; deleting one of those means it downloads again the next time it is needed.
    </p>

    <p v-if="!entries" class="note lead">reading the caches…</p>
    <p v-else-if="!shown.length" class="note lead">nothing here</p>

    <div v-for="d in shown" :key="d.id" class="entry" :class="{ idle: !used(d) }">
      <div class="line">
        <button v-if="d.files.length > 1" class="caret" :class="{ open: open.has(d.id) }" :title="`${d.files.length} files`" @click="toggle(d.id)">▸</button>
        <span v-else class="caret" />
        <div class="what">
          <span class="name" :title="d.name">{{ d.name }}</span>
          <span class="sub">{{ SOURCE[d.source] }} · {{ d.revision.slice(0, 7) }}{{ d.files.length > 1 ? ` · ${d.files.length} files` : '' }}</span>
        </div>
        <div class="uses">
          <span v-for="l in labels(usesOf(d))" :key="l" class="chip">{{ l }}</span>
          <span v-if="!used(d)" class="none">{{ offered(usesOf(d)).length ? `not used · in the ${offered(usesOf(d)).join(', ')} picker` : 'not used' }}</span>
        </div>
        <span class="size">{{ size(d.size) }}</span>
        <DeleteButton
          :remove="() => remove(d)"
          :what="`${d.name} (${size(d.size)})${used(d) ? '; it downloads again when next needed' : ''}`"
          @failed="(m) => emit('failed', m)"
        />
      </div>
      <template v-if="open.has(d.id)">
        <div v-for="f in d.files" :key="f.name" class="line file" :class="{ idle: !named(f.uses).length }">
          <span class="caret" />
          <span class="name" :title="f.name">{{ f.name }}</span>
          <div class="uses">
            <span v-for="l in labels(f.uses)" :key="l" class="chip">{{ l }}</span>
            <span v-if="!named(f.uses).length" class="none">not used</span>
          </div>
          <span class="size">{{ size(f.size) }}</span>
          <DeleteButton
            v-if="d.source === 'huggingface'"
            :remove="() => remove(d, [f.name])"
            :what="`${f.name} (${size(f.size)})`"
            @failed="(m) => emit('failed', m)"
          />
          <span v-else />
        </div>
      </template>
    </div>
  </section>
</template>

<style scoped>
.lead {
  margin: 0;
  padding: var(--space-2) var(--space-2) var(--space-1);
  font-size: 11px;
}

.entry {
  padding: 0 var(--space-2);
}

.entry + .entry {
  border-top: 1px dashed var(--line);
}

.line {
  display: grid;
  grid-template-columns: 16px minmax(0, 1.1fr) minmax(0, 1fr) 72px 128px;
  gap: var(--space-2);
  align-items: center;
  padding: 7px 0;
}

.line.file {
  padding: 3px 0;
  font-size: var(--text-xs);
}

.caret {
  border: 0;
  background: none;
  padding: 0;
  color: var(--muted);
  font-size: 11px;
  transition: transform 0.12s;
}

.caret.open {
  transform: rotate(90deg);
}

.what {
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.name {
  font-size: var(--text-sm);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file .name {
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  color: var(--muted);
}

.sub {
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--muted);
}

.uses {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  min-width: 0;
}

.chip {
  font-size: 11px;
  padding: 1px 6px;
  border: 1px solid var(--accent-muted);
  border-radius: var(--radius-input);
  color: var(--accent);
  white-space: nowrap;
}

.none {
  font-size: 11px;
  color: var(--muted);
}

.size {
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  text-align: right;
}

.line > :deep(.delete) {
  justify-self: end;
}

.idle .name {
  color: var(--muted);
}
</style>
