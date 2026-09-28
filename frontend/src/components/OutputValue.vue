<script setup lang="ts">
/** A job's outputs are whatever its tool returned, so this draws JSON by shape rather than by tool. */
import { computed, ref } from 'vue'

defineOptions({ name: 'OutputValue' })
const props = defineProps<{ value: unknown }>()

const PICTURE = /\.(png|jpe?g|webp)$/i
const isPicture = (v: unknown) => typeof v === 'string' && PICTURE.test(v)
const isPlain = (v: unknown) => v === null || ['string', 'number', 'boolean'].includes(typeof v)
const human = (key: string) => key.replace(/_/g, ' ')

const kind = computed(() => {
  const v = props.value
  if (isPicture(v)) return 'picture'
  if (typeof v === 'string') return v.length > 90 || v.includes('\n') ? 'prose' : 'word'
  if (isPlain(v)) return 'word'
  if (Array.isArray(v)) {
    if (!v.length) return 'empty'
    if (v.every(isPicture)) return 'pictures'
    if (v.every((x) => isPlain(x) && String(x).length <= 60)) return 'chips'
    return 'list'
  }
  return Object.keys(v as object).length ? 'fields' : 'empty'
})

const src = (path: string) => (path.startsWith('/') ? path : `/output/${path}`)

const copied = ref(false)
async function copy() {
  await navigator.clipboard.writeText(String(props.value))
  copied.value = true
  setTimeout(() => (copied.value = false), 1200)
}
</script>

<template>
  <a v-if="kind === 'picture'" class="picture" :href="src(value as string)" target="_blank">
    <img :src="src(value as string)" :title="value as string" alt="" />
  </a>
  <div v-else-if="kind === 'pictures'" class="pictures">
    <a v-for="p in value as string[]" :key="p" class="picture" :href="src(p)" target="_blank">
      <img :src="src(p)" :title="p" alt="" />
    </a>
  </div>
  <div v-else-if="kind === 'prose'" class="prose">
    <p>{{ value }}</p>
    <button class="copy" @click="copy">{{ copied ? 'copied' : 'copy' }}</button>
  </div>
  <span v-else-if="kind === 'word'" class="mono word">{{ value === null ? '—' : String(value) }}</span>
  <span v-else-if="kind === 'empty'" class="mono dim">—</span>
  <div v-else-if="kind === 'chips'" class="chips">
    <span v-for="(x, i) in value as unknown[]" :key="i" class="mono chip">{{ x }}</span>
  </div>
  <ol v-else-if="kind === 'list'" class="list">
    <li v-for="(x, i) in value as unknown[]" :key="i"><OutputValue :value="x" /></li>
  </ol>
  <dl v-else class="fields">
    <template v-for="(x, key) in value as Record<string, unknown>" :key="key">
      <dt class="label">{{ human(String(key)) }}</dt>
      <dd><OutputValue :value="x" /></dd>
    </template>
  </dl>
</template>

<style scoped>
.fields {
  display: grid;
  grid-template-columns: minmax(80px, max-content) minmax(0, 1fr);
  gap: 6px var(--space-2);
  margin: 0;
  align-items: baseline;
}

.fields dd {
  margin: 0;
  min-width: 0;
}

.fields dt {
  font-size: 11px;
}

.word {
  font-size: var(--text-xs);
  overflow-wrap: anywhere;
}

.prose {
  position: relative;
  background: var(--raised);
  border-radius: var(--radius-input);
  padding: var(--space-1) var(--space-2);
}

.prose p {
  margin: 0;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  font-size: var(--text-sm);
  line-height: var(--leading-body);
  padding-right: 48px;
}

.copy {
  position: absolute;
  top: 6px;
  right: 6px;
  font-size: 11px;
  padding: 1px 6px;
  border-color: var(--line);
  color: var(--muted);
}

.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.chip {
  font-size: 11px;
  padding: 1px 6px;
  border: 1px solid var(--line);
  border-radius: var(--radius-input);
}

.list {
  margin: 0;
  padding-left: 1.4em;
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.list > li::marker {
  color: var(--dim);
  font-family: var(--font-mono);
  font-size: 11px;
}

.pictures {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-1);
}

.picture img {
  display: block;
  height: 120px;
  max-width: 200px;
  object-fit: contain;
  background: var(--video);
  border: 1px solid var(--line);
  border-radius: var(--radius-input);
}
</style>
