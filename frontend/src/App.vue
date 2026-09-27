<script setup lang="ts">
import { computed, onMounted, onUnmounted } from 'vue'
import { RouterLink, RouterView, useRoute } from 'vue-router'
import { halt, live, queued, start, state } from './store'

const route = useRoute()
const SCREENS = [
  { to: '/quick', label: 'Take', here: (p: string) => p.startsWith('/quick') },
  { to: '/', label: 'Production', here: (p: string) => p === '/' || p.startsWith('/productions') },
  { to: '/runs', label: 'Runs', here: (p: string) => p.startsWith('/runs') },
  { to: '/settings', label: 'Settings', here: (p: string) => p.startsWith('/settings') },
]

const playing = computed(() => state.runs.find((one) => one.status === 'live'))
const line = computed(() => {
  const r = playing.value
  if (!r) return queued.value ? `${queued.value} queued` : ''
  return `${r.title} · job ${r.jobs_done} / ${r.jobs_total}${queued.value ? ` · ${queued.value} queued` : ''}`
})
const target = computed(() => (playing.value ? `/runs/${playing.value.id}` : '/runs'))

onMounted(start)
onUnmounted(halt)
</script>

<template>
  <div class="app">
    <header class="top">
      <div class="left">
        <!-- Drawn rather than fetched: one 24px glyph is not worth a request. -->
        <RouterLink to="/" class="brand">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <rect x="1" y="4" width="22" height="16" rx="2" stroke="var(--accent)" stroke-width="1.5" />
            <line x1="7" y1="4" x2="7" y2="20" stroke="var(--accent)" stroke-width="1.5" />
            <line x1="17" y1="4" x2="17" y2="20" stroke="var(--accent)" stroke-width="1.5" />
            <rect x="1" y="4" width="6" height="16" fill="var(--accent)" />
            <path d="M10.5 9v6l4.5-3z" fill="var(--accent)" />
          </svg>
          <span>DiffLab</span>
        </RouterLink>
        <nav class="seg">
          <RouterLink v-for="s in SCREENS" :key="s.to" :to="s.to" :class="{ on: s.here(route.path) }">{{ s.label }}</RouterLink>
        </nav>
      </div>
      <RouterLink v-if="line" class="line mono" :to="target" title="open the run in progress">
        <i v-if="playing" class="dot running" /><span>{{ line }}</span><span class="go">→</span>
      </RouterLink>
      <span v-else />
      <span class="gpu mono">
        <span>24 GB</span>
        <span class="state"><i class="dot" :class="live ? 'running' : 'done'" />{{ live ? 'rendering' : 'idle' }}</span>
      </span>
    </header>

    <RouterView />
  </div>
</template>

<style scoped>
.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.top {
  height: 56px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--space-2);
  padding: 0 var(--space-3);
  border-bottom: 1px solid var(--border);
  background: var(--bg-base);
  position: sticky;
  top: 0;
  z-index: 5;
}

.left {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  flex-shrink: 0;
}

.brand {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  color: var(--text-primary);
  text-decoration: none;
  font-family: var(--font-heading);
  font-weight: var(--weight-bold);
  letter-spacing: -0.01em;
  font-size: 15px;
}

.line {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  font-size: var(--text-xs);
  color: var(--text-secondary);
  text-decoration: none;
  white-space: nowrap;
  min-width: 0;
  padding: 4px var(--space-1);
  border: 1px solid transparent;
  border-radius: var(--radius-btn);
}

.line > span:not(.go) {
  overflow: hidden;
  text-overflow: ellipsis;
}

.line:hover {
  color: var(--accent);
  border-color: var(--border-strong);
}

.go {
  color: var(--accent);
}

.gpu {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  font-size: var(--text-xs);
  color: var(--text-secondary);
  white-space: nowrap;
  flex-shrink: 0;
}

.state {
  display: flex;
  align-items: center;
  gap: var(--space-1);
}

@media (max-width: 900px) {
  .top {
    height: auto;
    flex-wrap: wrap;
    padding: var(--space-1) var(--space-2);
  }
}
</style>
