<script setup lang="ts">
/** Delete a run in two clicks: the first arms the button, the second within ARMED_MS deletes. */
import { onBeforeUnmount, ref } from 'vue'
import { deleteRun } from '../api'

const ARMED_MS = 4000

const props = defineProps<{ id: string; disabled?: boolean }>()
const emit = defineEmits<{ deleted: [] }>()

const armed = ref(false)
const deleting = ref(false)
const failed = ref('')
let timer: ReturnType<typeof setTimeout> | undefined

function disarm() {
  armed.value = false
  clearTimeout(timer)
}

async function click() {
  failed.value = ''
  if (!armed.value) {
    armed.value = true
    timer = setTimeout(disarm, ARMED_MS)
    return
  }
  disarm()
  deleting.value = true
  try {
    await deleteRun(props.id)
    emit('deleted')
  } catch (problem) {
    failed.value = problem instanceof Error ? problem.message : String(problem)
  } finally {
    deleting.value = false
  }
}

onBeforeUnmount(() => clearTimeout(timer))
</script>

<template>
  <button
    class="delete"
    :class="{ armed, failed: !!failed }"
    :disabled="disabled || deleting"
    :title="failed || (disabled ? 'cancel the run first' : armed ? 'click again to delete the run and the files it made' : 'delete the run and the files it made')"
    @click="click"
    @mouseleave="disarm"
    @blur="disarm"
  >
    {{ deleting ? 'deleting…' : armed ? 'confirm deletion' : failed ? 'not deleted' : 'delete' }}
  </button>
</template>

<style scoped>
.delete {
  font-size: var(--text-xs);
  padding: 2px 8px;
  border-color: var(--line);
  color: var(--muted);
  white-space: nowrap;
}

.delete:not(:disabled):hover,
.delete.failed {
  color: var(--status-error);
  border-color: var(--status-error);
}

.delete.armed {
  color: var(--bg-base);
  background: var(--status-error);
  border-color: var(--status-error);
}
</style>
