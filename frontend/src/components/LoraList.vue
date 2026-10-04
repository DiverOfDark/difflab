<script setup lang="ts">
import { computed } from 'vue'
import ModelPicker from './ModelPicker.vue'

const props = defineProps<{ value: string; remembered: string[] }>()
const emit = defineEmits<{ pick: [value: string]; forget: [value: string] }>()

type Row = { ref: string; strength: string }

const rows = computed<Row[]>(() =>
  props.value
    .split('\n')
    .filter((line) => line.trim())
    .map((line) => {
      const [ref, strength = '1'] = line.split('*')
      return { ref: ref.trim(), strength: strength.trim() }
    }),
)

function write(next: Row[]) {
  emit('pick', next.map((r) => (Number(r.strength) === 1 ? r.ref : `${r.ref}*${r.strength}`)).join('\n'))
}

const change = (i: number, patch: Partial<Row>) => write(rows.value.map((r, j) => (j === i ? { ...r, ...patch } : r)))
const remove = (i: number) => write(rows.value.filter((_, j) => j !== i))
const add = (ref: string) => write([...rows.value, { ref, strength: '1' }])
</script>

<template>
  <div class="loras">
    <div v-for="(r, i) in rows" :key="i" class="lora" :class="{ off: Number(r.strength) === 0 }">
      <ModelPicker :value="r.ref" :fallback="r.ref" code-default="" :remembered="remembered" @pick="(v) => change(i, { ref: v })" @forget="(v) => emit('forget', v)" />
      <input
        :value="r.strength"
        type="number"
        min="-2"
        max="2"
        step="0.05"
        title="strength; 0 turns it off without removing it"
        @input="change(i, { strength: ($event.target as HTMLInputElement).value || '0' })"
      />
      <button class="link" title="remove" @click="remove(i)">✕</button>
    </div>
    <ModelPicker :value="''" fallback="" code-default="" empty="+ add a LoRA" :remembered="remembered" @pick="add" @forget="(v) => emit('forget', v)" />
  </div>
</template>

<style scoped>
.loras {
  display: grid;
  gap: 6px;
  min-width: 0;
}

.lora {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 72px auto;
  gap: 6px;
  align-items: center;
}

.lora.off :deep(.name) {
  color: var(--dim);
  text-decoration: line-through;
}

.lora input {
  text-align: right;
  font-size: var(--text-xs);
  background: var(--raised);
  font-family: var(--font-mono);
}
</style>
