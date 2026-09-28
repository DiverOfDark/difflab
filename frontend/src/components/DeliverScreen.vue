<script setup lang="ts">
/**
 * The Finish card reads rather than writes: `core/events` has no "finish this run again" command,
 * so it says what the run delivered with and the button that would re-run it is not drawn.
 */
import { computed, ref } from 'vue'
import { runFile, type Job, type RunRecord } from '../api'
import { duration, madeNoFiles, type Item, type RunState } from '../run'
import JobResults from './JobResults.vue'

const props = defineProps<{ run: RunState; record: RunRecord | null }>()
const emit = defineEmits<{ again: [spec: Job] }>()

interface Output {
  path: string
  name: string
  desc: string
  meta: string
  media: 'video' | 'image'
}

/** Everything worth sending someone, best first. */
const outputs = computed<Output[]>(() => {
  const out: Output[] = []
  if (props.run.finalVideo)
    out.push({
      path: props.run.finalVideo,
      name: props.run.finalVideo.split('/').pop() ?? props.run.finalVideo,
      desc: 'the joined take',
      meta: 'delivered',
      media: 'video',
    })
  for (const made of [...props.run.artifacts].reverse())
    if (made.path !== props.run.finalVideo)
      out.push({ path: made.path, name: made.name, desc: made.media, meta: '', media: made.media })
  return out
})

const textual = computed(() => madeNoFiles(props.run) && jobs.value.length > 0)

const chosen = ref(0)
const current = computed<Output | null>(() => outputs.value[chosen.value] ?? null)
const src = computed(() => (current.value ? `/output/${current.value.path}` : ''))

/** What you type into a file manager. */
const folder = computed(() => {
  const path = current.value?.path ?? ''
  const cut = path.lastIndexOf('/')
  return cut < 0 ? 'output/' : `output/${path.slice(0, cut)}/`
})

/**
 * What the run recorded when it was asked for, never the form: the person goes on editing the job,
 * and a panel that answers "what made this file" out of the box they are typing in is not provenance.
 */
const spec = computed<Job | null>(() => props.record?.spec ?? null)
const jobs = computed(() => props.record?.jobs ?? [])
const took = (seconds: number | null | undefined) => (seconds == null ? '' : duration(seconds))

const finishes = computed(() => {
  const j = spec.value
  return [
    {
      title: 'Decode size',
      key: 'finish_megapixels',
      value: !j ? '—' : j.finish_megapixels ? `${j.finish_megapixels} MP` : 'as sampled',
      hint: 'Decoded larger than it sampled. The latents are unchanged, so this costs a decode, not a render.',
    },
    {
      title: 'Colour',
      key: 'grade_strength',
      value: !j ? '—' : j.grade_strength ? String(j.grade_strength) : 'as rendered',
      hint: 'Pulled back toward the colour of its own first second, to take out the drift a chain inherits.',
    },
    {
      title: 'Frame rate',
      key: 'retime_fps',
      value: !j ? '—' : j.retime_fps ? `${j.retime_fps} fps` : '24 · as rendered',
      hint: 'RIFE drew the frames between. The 24 fps file is kept beside it and shares every chunk on disk.',
    },
  ]
})

/** The delivered clip is joined, so these are how you find which chunk to re-roll without scrubbing. */
const poster = (it: Item) => {
  const relative = it.poster || it.opening
  return relative ? runFile(props.run.id, relative) : ''
}

/** The seed and the canvas live in whichever settings object the mode actually sampled from. */
const sampled = computed<{ seed: number; megapixels: number } | null>(() => {
  const j = spec.value
  if (!j) return null
  if (j.mode === 'edit') return j.edit
  return j.video
})

const model = computed(() => {
  const found = jobs.value.find((j) => typeof j.settings?.checkpoint === 'string')
  const checkpoint = found ? String(found.settings.checkpoint) : ''
  return checkpoint.split('@')[0].split('/').pop() ?? ''
})

const textFacts = computed<[string, string][]>(() => [
  ['RUN', props.run.id],
  ['JOBS', `${jobs.value.filter((j) => j.state === 'done' || j.state === 'skipped').length} / ${jobs.value.length}`],
  ['ELAPSED', duration(props.run.elapsed)],
  ['MODEL', model.value || '—'],
  ['STATUS', props.run.status],
])

const facts = computed<[string, string][]>(() => [
  ['RUN', props.run.id],
  ['CHUNKS', String(props.run.items.length)],
  ['ELAPSED', duration(props.run.elapsed)],
  ['SEED', sampled.value ? String(sampled.value.seed) : '—'],
  ['CANVAS', sampled.value ? `${sampled.value.megapixels} MP` : '—'],
  ['STATUS', props.run.status],
])
</script>

<template>
  <main v-if="textual" class="screen deliver">
    <section class="stack">
      <div class="top">
        <div class="named">
          <span class="label" :class="run.status === 'done' ? 'ready' : 'bad'"
            >● {{ run.status }} · {{ jobs.length }} jobs · {{ duration(run.elapsed) }}</span
          >
          <h3>{{ run.title }}</h3>
        </div>
        <router-link v-if="record?.production != null" :to="`/productions/${record.production}`" class="button"
          >Open production</router-link
        >
      </div>
      <div v-if="run.error" class="band error">{{ run.error }}</div>
      <JobResults :jobs="jobs" />
    </section>

    <aside class="stack">
      <div class="card">
        <div class="head"><h5>Run</h5></div>
        <div class="knobs">
          <div v-for="[k, v] in textFacts" :key="k" class="fact">
            <span class="label">{{ k }}</span>
            <span class="mono">{{ v }}</span>
          </div>
        </div>
      </div>
    </aside>
  </main>

  <main v-else class="screen deliver">
    <section class="stack">
      <div class="top">
        <div class="named">
          <span class="label" :class="run.status === 'done' ? 'ready' : 'bad'"
            >● {{ run.status }} · {{ run.items.length }} chunks · {{ duration(run.elapsed) }}</span
          >
          <h3>{{ run.title }}</h3>
        </div>
        <span class="mono muted">{{ folder }}</span>
      </div>

      <!-- A failed run lands here too, being no longer live. Without this it showed an empty black
           player and nothing else, which is how a crash gets reported as "the model returned black". -->
      <div v-if="run.error" class="band error">{{ run.error }}</div>

      <div class="card picker">
        <div class="player">
          <video v-if="current?.media === 'video'" :key="src" :src="src" controls autoplay muted loop playsinline />
          <img v-else-if="current" :src="src" alt="" />
          <p v-else class="note">
            {{ run.error ? 'This run failed before it wrote anything.' : 'This run delivered no file.' }}
          </p>
        </div>
        <div class="files">
          <div class="named">
            <span class="label">selected file</span>
            <span class="mono path">{{ current?.path ?? '—' }}</span>
          </div>
          <div class="list">
            <button v-for="(o, i) in outputs" :key="o.path" class="file" :class="{ on: i === chosen }" @click="chosen = i">
              <i class="dot" :class="{ done: i === chosen }" />
              <span class="what">
                <span class="mono name">{{ o.name }}</span>
                <span class="note">{{ o.desc }}</span>
              </span>
              <span class="mono meta">{{ o.meta }}</span>
            </button>
          </div>
          <div class="buttons">
            <a v-if="current" class="primary button" :href="src" download>Download</a>
            <button
              :disabled="!spec"
              :title="spec ? '' : 'this run recorded no spec to start from'"
              @click="spec && emit('again', spec)"
            >
              New take from this
            </button>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="head">
          <h5>Chunks</h5>
          <span class="note">re-rolling one re-joins the take; the finish is not in the cache key</span>
        </div>
        <div class="grid">
          <div v-for="it in run.items" :key="it.index" class="cell">
            <div class="frame">
              <img v-if="poster(it)" :src="poster(it)" alt="" />
              <span>chunk {{ it.index + 1 }}</span>
            </div>
            <div class="facts">
              <span>seed {{ it.seed ?? '—' }}</span>
              <span>{{ it.version > 1 ? `v${it.version}` : '' }}</span>
            </div>
          </div>
        </div>
      </div>
    </section>

    <aside class="stack">
      <div v-if="jobs.length" class="card">
        <div class="head">
          <h5>Jobs</h5>
          <span class="note">what this run was made of, in order</span>
        </div>
        <div class="knobs">
          <div v-for="one in jobs" :key="one.id" class="fact">
            <span class="label">{{ one.position + 1 }} · {{ one.label || one.tool }}</span>
            <span class="mono" :class="one.state">
              {{ one.state }}{{ one.started && one.ended ? ` · ${took(one.ended - one.started)}` : '' }}
            </span>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="head block">
          <h5>Finish</h5>
          <p class="note">
            {{
              spec
                ? 'Ran after sampling and wrote beside its input. What this delivery used:'
                : 'This run recorded no spec — nothing on disk says what it was asked for.'
            }}
          </p>
        </div>
        <div class="knobs">
          <div v-for="f in finishes" :key="f.key" class="knob">
            <div>
              <span class="name">{{ f.key }}</span>
              <span class="hint">{{ f.hint }}</span>
            </div>
            <span class="mono value">{{ f.value }}</span>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="head"><h5>Run</h5></div>
        <div class="knobs">
          <div v-for="[k, v] in facts" :key="k" class="fact">
            <span class="label">{{ k }}</span>
            <span class="mono">{{ v }}</span>
          </div>
        </div>
      </div>
    </aside>
  </main>
</template>

<style scoped>
.deliver {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(280px, 340px);
  gap: var(--space-3);
  align-items: start;
}

.top {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: var(--space-3);
  flex-wrap: wrap;
}

.named {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}

.top h3 {
  font-size: var(--text-xl);
  margin: 0;
}

.label.ready {
  color: var(--done);
}

.label.bad {
  color: var(--fail);
}

/* Wrapped rather than truncated: the line of a traceback that matters is usually the last one. */
.band.error {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  max-height: 260px;
  overflow: auto;
  font-family: var(--font-mono);
  font-size: var(--text-xs);
}

.picker {
  padding: var(--space-3);
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-3);
}

.player {
  width: 240px;
  flex-shrink: 0;
  aspect-ratio: 9 / 16;
  background: var(--video);
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-input);
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.player video,
.player img {
  max-width: 100%;
  max-height: 100%;
}

.files {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  min-width: 0;
  flex: 1 1 280px;
}

.path {
  font-size: var(--text-sm);
  overflow-wrap: anywhere;
}

.list {
  display: flex;
  flex-direction: column;
}

.file {
  display: grid;
  grid-template-columns: 16px minmax(0, 1fr) auto;
  gap: var(--space-2);
  align-items: center;
  padding: var(--space-1) 0;
  border: 0;
  border-bottom: 1px solid var(--line);
  border-radius: 0;
  background: none;
  text-align: left;
}

.file .dot {
  justify-self: center;
}

.file .what {
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.file .name {
  font-size: var(--text-xs);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.file.on .name {
  color: var(--accent);
}

.file .meta {
  font-size: 11px;
  color: var(--dim);
  white-space: nowrap;
}

.buttons {
  display: flex;
  gap: var(--space-1);
  flex-wrap: wrap;
}

a.button {
  display: inline-block;
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-btn);
  padding: 7px 14px;
  font-size: var(--text-sm);
  color: var(--text-primary);
  text-decoration: none;
}

/* A real link, so the browser saves the file rather than the page fetching and revoking a blob. */
a.primary.button {
  display: inline-block;
  border: 1px solid var(--accent);
  border-radius: var(--radius-btn);
  padding: 7px 14px;
  font-size: var(--text-sm);
  font-weight: var(--weight-semibold);
  background: var(--accent);
  color: var(--accent-fg);
  text-decoration: none;
}

.picks {
  display: flex;
  flex-direction: column;
  padding: var(--space-1) var(--space-2) var(--space-2);
  gap: 4px;
}

.picks button {
  text-align: left;
  font-family: var(--mono);
  font-size: var(--text-xs);
  border-color: var(--line);
}

.picks button.on {
  border-color: var(--accent);
  color: var(--accent);
}

.head.block {
  display: block;
}

.head.block p {
  margin-top: 4px;
}

.knobs {
  padding: var(--space-1) var(--space-2) var(--space-2);
  display: flex;
  flex-direction: column;
}

/* Wider than the shared knob's 104px, which wraps every one of these words ("as rendered"). */
.knob {
  grid-template-columns: 1fr 108px;
  gap: var(--space-2);
}

.knob .value {
  text-align: right;
  font-size: var(--text-xs);
  color: var(--muted);
}

.fact .mono {
  text-align: right;
  overflow-wrap: anywhere;
}

.fact {
  display: flex;
  justify-content: space-between;
  gap: var(--space-2);
  padding: var(--space-1) 0;
  border-bottom: 1px solid var(--line);
  font-size: var(--text-xs);
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: var(--space-1);
  padding: var(--space-2);
}

.cell {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.frame {
  aspect-ratio: 9 / 16;
  background: var(--video);
  border: 1px solid var(--line);
  border-radius: var(--radius-input);
  display: flex;
  align-items: flex-end;
  justify-content: center;
  padding-bottom: 4px;
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--dim);
  position: relative;
  overflow: hidden;
}

.frame img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.frame span {
  position: relative;
  text-shadow: 0 1px 2px #000;
}

.facts {
  display: flex;
  justify-content: space-between;
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--dim);
}

@media (max-width: 960px) {
  .deliver {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>
