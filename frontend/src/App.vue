<script setup lang="ts">
import { computed, onMounted, onUnmounted } from 'vue'
import { RouterLink, RouterView, useRoute } from 'vue-router'
import { gb, mmss } from './production'
import { halt, queued, start, state } from './store'

const route = useRoute()
const SCREENS = [
  { to: '/quick', label: 'Take', here: (p: string) => p.startsWith('/quick') },
  { to: '/productions', label: 'Production', here: (p: string) => p.startsWith('/productions') },
  { to: '/chat', label: 'Chat', here: (p: string) => p.startsWith('/chat') },
  { to: '/gallery', label: 'Gallery', here: (p: string) => p.startsWith('/gallery') },
  { to: '/runs', label: 'Runs', here: (p: string) => p.startsWith('/runs') },
  { to: '/settings', label: 'Settings', here: (p: string) => p.startsWith('/settings') },
]

const playing = computed(() => state.runs.find((one) => one.status === 'live'))
const onProduction = computed(() => route.path.startsWith('/productions'))
const line = computed(() => {
  if (onProduction.value) return state.heading
  const r = playing.value
  if (!r) return queued.value ? `${queued.value} queued` : ''
  return `${r.title} · job ${r.jobs_done} / ${r.jobs_total}${queued.value ? ` · ${queued.value} queued` : ''}`
})
const gpu = computed(() => {
  const g = state.gpu
  const used = g?.vram_used === undefined ? '' : ` · ${g.vram_used.toFixed(1)} GB`
  return {
    total: g?.vram_total === undefined ? '' : `${Math.round(g.vram_total)} GB`,
    text: fetching.value?.text ?? `${g?.state ?? 'idle'}${used}`,
    busy: !!g && g.state !== 'idle',
    run: g?.run ?? null,
    title: fetching.value ? `${fetching.value.title} · open the run` : g?.run ? `${g.title} · ${g.tool} · open the run` : 'nothing is running on the card',
  }
})

const fetching = computed(() => {
  const d = state.gpu?.download
  if (!d) return null
  const rate = d.seconds > 0 ? d.done / d.seconds : 0
  const left = rate > 0 ? ` · ${mmss((d.total - d.done) / rate)} left` : ''
  return {
    text: `downloading · ${gb(d.done)} / ${gb(d.total)}${rate ? ` · ${Math.round(rate / 1e6)} MB/s` : ''}${left}`,
    share: d.total ? Math.min(1, d.done / d.total) : 0,
    title: d.label,
  }
})

onMounted(start)
onUnmounted(halt)
</script>

<template>
  <div class="app">
    <header class="top">
      <div class="left">
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
      <span v-if="onProduction" class="line mono">{{ line }}</span>
      <RouterLink v-else-if="line" class="line mono link" :to="playing ? `/runs/${playing.id}` : '/runs'" title="open the run in progress">
        <span>{{ line }}</span><span class="go">→</span>
      </RouterLink>
      <span v-else />
      <span class="gpu mono">
        <span>{{ gpu.total }}</span>
        <component :is="gpu.run ? RouterLink : 'span'" class="state" :to="gpu.run ? `/runs/${gpu.run}` : undefined" :title="gpu.title">
          <i class="dot" :class="gpu.busy ? 'running' : 'idle'" />{{ gpu.text }}
        </component>
      </span>
      <div v-if="fetching" class="fetch" :title="fetching.title"><i :style="{ width: `${fetching.share * 100}%` }" /></div>
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

span.line {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
}

.line.link:hover {
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
  color: inherit;
  text-decoration: none;
}

a.state:hover {
  color: var(--accent);
}

.fetch {
  position: absolute;
  left: 0;
  right: 0;
  bottom: -1px;
  height: 2px;
}

.fetch > i {
  display: block;
  height: 100%;
  background: var(--accent);
  transition: width 0.5s linear;
}

.dot.idle {
  background: var(--status-stopped);
}

@media (max-width: 900px) {
  .top {
    height: auto;
    flex-wrap: wrap;
    padding: var(--space-1) var(--space-2);
  }
}
</style>
