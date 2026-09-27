/**
 * A form field over a value the page re-reads every few seconds. The poll hands back a fresh object
 * each time, so a field that copies from it on every read loses what was typed mid-sentence; this one
 * takes the server's value only when that value itself moved and nothing typed is waiting.
 */
import { computed, reactive, ref, watch, type Ref } from 'vue'

export function useDraft<T>(source: () => T, same: (a: T, b: T) => boolean = (a, b) => JSON.stringify(a) === JSON.stringify(b)) {
  const draft = ref(source()) as Ref<T>
  let base = source()
  const dirty = computed(() => !same(draft.value, source()))
  watch(
    () => JSON.stringify(source()),
    () => {
      const now = source()
      if (same(draft.value, base)) draft.value = now
      base = now
    },
  )
  const reset = () => {
    base = source()
    draft.value = base
  }
  return { draft, dirty, reset }
}

/** Per-row drafts for a list of inline fields: `value(key, server)` shows what is typed, else the server's. */
export function useRowDrafts() {
  const typed = reactive<Record<string, string>>({})
  return {
    value: (key: string, server: string) => (key in typed ? typed[key] : server),
    input: (key: string, event: Event) => (typed[key] = (event.target as HTMLInputElement).value),
    drop: (key: string) => delete typed[key],
    typed,
  }
}
