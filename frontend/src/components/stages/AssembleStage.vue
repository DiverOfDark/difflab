<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { RouterLink } from 'vue-router'
import {
  listTools,
  outputUrl,
  putSettings,
  readRun,
  type Assembled,
  type ChapterMaster,
  type Production,
  type SettingField,
  type ShotInfo,
  type ToolInfo,
} from '../../api'
import { ago, plural, seconds } from '../../production'

const props = defineProps<{
  production: Production
  shots: ShotInfo[]
  chapter: number | null
  runs: Production['runs']
  assembled: Assembled | null
  /** The chapter position being previewed, -1 for the film, null for none. */
  preview: number | null
  act: (name: string, what: () => Promise<unknown>) => Promise<void>
}>()
const emit = defineEmits<{ preview: [at: number | null] }>()

const two = (n: number) => String(n + 1).padStart(2, '0')
const inView = computed(() => (props.assembled?.chapters ?? []).filter((c) => props.chapter === null || c.position === props.chapter))
/** The finished master once there is one, the native one until then. */
const best = (c: Assembled['chapters'][number]): ChapterMaster | undefined => c.finished ?? c.native
const kind = (m: ChapterMaster) => (m.finished ? 'finished' : 'native')

const uses = computed(() => {
  const n = props.shots.length
  const clips = props.shots.filter((s) => s.clip.present).length
  return [
    { a: 'Clips with audio', h: `${clips}${clips < n ? ` / ${n}` : ''} · ${props.production.hashes.render}` },
    { a: 'Shot list', h: `${n} · ${props.production.hashes.shots}` },
  ]
})
const produces = computed(() => {
  const film = props.assembled?.film
  const lines = inView.value.reduce((sum, c) => sum + c.lines, 0)
  return [
    ...inView.value.map((c) => {
      const m = best(c)
      return { a: `Chapter ${two(c.position)}`, h: m ? `${kind(m)} · ${seconds(m.seconds)} · ${m.hash}${m.current ? '' : ' · older clips'}` : '—' }
    }),
    { a: 'Full film', h: film ? `${film.chapters} ch · ${seconds(film.seconds)} · ${film.hash}` : '—' },
    { a: 'Subtitles', h: film || inView.value.some(best) ? `${lines} line${lines === 1 ? '' : 's'}` : '—' },
  ]
})

const tool = ref<ToolInfo | null>(null)
const failed = ref('')
const loadTool = () =>
  listTools(props.production.id)
    .then((tools) => (tool.value = tools.find((t) => t.name === 'assemble') ?? null))
    .catch((e) => (failed.value = String(e)))
loadTool()
const value = (f: SettingField) => tool.value?.effective?.[f.name] ?? f.default
async function set(f: SettingField, raw: string) {
  const v = f.type === 'bool' ? raw === 'on' : f.type === 'float' || f.type === 'int' ? Number(raw) : raw
  await props.act('knob', async () => {
    await putSettings('assemble', 'production', String(props.production.id), { ...(tool.value?.production ?? {}), [f.name]: v })
    await loadTool()
  })
}

const playing = computed(() => {
  const at = props.preview
  if (at === null || !props.assembled) return null
  if (at === -1) return props.assembled.film ? { title: 'Full film', made: props.assembled.film, lufs: null as number | null } : null
  const c = props.assembled.chapters.find((x) => x.position === at)
  const m = c && best(c)
  return m ? { title: `Chapter ${two(at)} · ${kind(m)}`, made: m, lufs: m.lufs } : null
})

interface Joined {
  run: string
  state: string
  created: number
  video: string
  seconds: number | null
}
const joined = ref<Joined[]>([])
const known = new Map<string, Joined>()
// Only a finished run is cached: a live one has not written its join's outputs yet.
watch(
  () => props.runs.map((r) => `${r.id}:${r.state}`).join(),
  async () => {
    const out: Joined[] = []
    for (const r of props.runs) {
      let found = known.get(r.id)
      if (!found) {
        const record = await readRun(r.id).catch(() => null)
        const job =
          record?.jobs.find((j) => j.tool === 'finish' && typeof j.outputs.video === 'string') ??
          record?.jobs.find((j) => j.tool === 'assemble')
        found = {
          run: r.id,
          state: r.state,
          created: r.created,
          video: typeof job?.outputs.video === 'string' ? job.outputs.video : '',
          seconds: typeof job?.outputs.seconds === 'number' ? job.outputs.seconds : null,
        }
        if (r.state === 'done' || r.state === 'failed') known.set(r.id, found)
      }
      out.push(found)
    }
    joined.value = out
  },
  { immediate: true },
)
const file = (path: string) => path.split('/').at(-1)
</script>

<template>
  <div class="assemble">
    <section v-if="playing" class="card player">
      <div class="head slim">
        <h5>{{ playing.title }}</h5>
        <span class="mono dim small">
          {{ seconds(playing.made.seconds) }}<template v-if="playing.lufs !== null"> · {{ playing.lufs.toFixed(1) }} LUFS</template> ·
          {{ playing.made.lines ?? 0 }} subtitle line{{ playing.made.lines === 1 ? '' : 's' }}
          <button class="link" @click="emit('preview', null)">close</button>
        </span>
      </div>
      <div class="body">
        <video :key="playing.made.video" :src="outputUrl(playing.made.video)" controls autoplay preload="metadata">
          <track v-if="playing.made.vtt" kind="subtitles" label="Dialogue" :src="outputUrl(playing.made.vtt)" default />
        </video>
      </div>
    </section>

    <div class="pair">
      <section class="card">
        <div class="head slim"><h5>Uses</h5></div>
        <div v-for="r in uses" :key="r.a" class="line"><span>{{ r.a }}</span><span class="h">{{ r.h }}</span></div>
        <div class="head slim sub"><h5>Produces</h5></div>
        <div v-for="w in produces" :key="w.a" class="line"><span>{{ w.a }}</span><span class="h">{{ w.h }}</span></div>
      </section>

      <section class="card">
        <div class="head slim"><h5>Knobs</h5></div>
        <div class="knobs">
          <div v-for="f in tool?.settings ?? []" :key="f.name" class="knob">
            <div>
              <span class="name">{{ f.label ? f.label[0].toUpperCase() + f.label.slice(1) : f.name }}</span>
              <span class="hint">{{ f.hint }}</span>
            </div>
            <select v-if="f.type === 'bool'" :value="value(f) ? 'on' : 'off'" @change="set(f, ($event.target as HTMLSelectElement).value)">
              <option>on</option>
              <option>off</option>
            </select>
            <select v-else-if="f.choices.length" :value="value(f)" @change="set(f, ($event.target as HTMLSelectElement).value)">
              <option v-for="c in f.choices" :key="String(c)">{{ c }}</option>
            </select>
            <input
              v-else
              type="number"
              :step="f.type === 'int' ? 1 : 0.05"
              :min="f.low ?? undefined"
              :max="f.high ?? undefined"
              :value="value(f)"
              @change="set(f, ($event.target as HTMLInputElement).value)"
            />
          </div>
          <p v-if="failed" class="note">{{ failed }}</p>
          <p class="note">The crossfade is for joins where the scene changes; a chained shot keeps its seam blend, any other join is a cut.</p>
        </div>
      </section>
    </div>

    <section class="card">
      <div class="head slim">
        <h5>Outputs</h5>
        <span class="mono dim small">{{ assembled?.busy ? `${assembled.busy} step${assembled.busy === 1 ? '' : 's'} under way` : 'newest of each' }}</span>
      </div>
      <div v-for="c in inView" :key="c.id" class="out">
        <span class="mono">Chapter {{ two(c.position) }}</span>
        <span class="dim small">{{ c.shots }} shots · {{ c.chained }} chained · {{ c.boundaries }} scene crossfade{{ c.boundaries === 1 ? '' : 's' }} · {{ c.lines }} lines</span>
        <span class="files">
          <template v-for="m in [c.finished, c.native].filter((x): x is ChapterMaster => !!x)" :key="m.job">
            <span class="ph" :class="m.current ? 'done' : 'stale'" :title="m.current ? 'made from the clips as they are' : 'made from an earlier set of clips'">{{ kind(m) }}</span>
            <a class="mono small" :href="outputUrl(m.video)" target="_blank">{{ file(m.video) }}</a>
            <a v-if="m.srt" class="mono small" :href="outputUrl(m.srt)" download>srt</a>
            <a v-if="m.vtt" class="mono small" :href="outputUrl(m.vtt)" download>vtt</a>
            <button class="link small" @click="emit('preview', c.position)">play</button>
          </template>
          <span v-if="!c.finished && !c.native" class="mono dim small">not assembled yet</span>
        </span>
      </div>
      <div class="out">
        <span class="mono">Full film</span>
        <span class="dim small">{{ assembled?.film ? `${plural(assembled.film.chapters, 'chapter')} · ${plural(assembled.film.lines ?? 0, 'line')}` : 'every chapter joined, with chapter markers' }}</span>
        <span class="files">
          <template v-if="assembled?.film">
            <a class="mono small" :href="outputUrl(assembled.film.video)" target="_blank">{{ file(assembled.film.video) }}</a>
            <a v-if="assembled.film.srt" class="mono small" :href="outputUrl(assembled.film.srt)" download>srt</a>
            <a v-if="assembled.film.vtt" class="mono small" :href="outputUrl(assembled.film.vtt)" download>vtt</a>
            <button class="link small" @click="emit('preview', -1)">play</button>
          </template>
          <span v-else class="mono dim small">not assembled yet</span>
        </span>
      </div>
    </section>

    <section class="card">
      <div class="head slim"><h5>Chapter runs</h5><span class="mono dim small">each ends in a native join; Finish enlarges and retimes it</span></div>
      <div v-for="j in joined" :key="j.run" class="row">
        <RouterLink class="mono" :to="`/runs/${j.run}`">{{ j.run }}</RouterLink>
        <span class="ph" :class="j.state === 'done' ? 'done' : j.state === 'failed' ? 'stale' : 'running'">{{ j.state.toUpperCase() }}</span>
        <a v-if="j.video" class="mono small" :href="outputUrl(j.video)" target="_blank">{{ file(j.video) }}</a>
        <span v-else class="mono dim small">—</span>
        <span class="mono dim small">{{ ago(j.created) }}</span>
      </div>
      <p v-if="!joined.length" class="note pad">No chapter run yet.</p>
    </section>
  </div>
</template>

<style scoped>
.assemble {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.head {
  align-items: center !important;
  flex-wrap: wrap;
}

.head.sub {
  border-top: 0;
}

.small {
  font-size: 11px;
}

.player video {
  width: 100%;
  max-height: 70vh;
  background: var(--video);
  border-radius: 2px;
}

.pair {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 400px), 1fr));
  gap: var(--space-3);
  align-items: start;
}

.line {
  display: flex;
  justify-content: space-between;
  gap: var(--space-2);
  padding: var(--space-1) var(--space-2);
  border-bottom: 1px solid var(--border);
  font-family: var(--font-mono);
  font-size: var(--text-xs);
}

.line .h {
  color: var(--text-tertiary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.knobs {
  padding: var(--space-1) var(--space-2) var(--space-2);
  display: flex;
  flex-direction: column;
}

.knob {
  grid-template-columns: minmax(0, 1fr) 148px;
  gap: var(--space-1) var(--space-2);
}

.knobs .note {
  margin-top: var(--space-1);
}

.out {
  display: grid;
  grid-template-columns: 96px minmax(0, 1fr);
  gap: 4px var(--space-2);
  align-items: baseline;
  padding: var(--space-1) var(--space-2);
  border-bottom: 1px solid var(--border);
  font-size: var(--text-xs);
}

.out .files {
  grid-column: 2;
  display: flex;
  flex-wrap: wrap;
  gap: 4px var(--space-1);
  align-items: baseline;
}

.out .files a {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 100%;
}

.row {
  display: grid;
  grid-template-columns: 150px 72px minmax(0, 1fr) 48px;
  gap: var(--space-2);
  align-items: center;
  padding: var(--space-1) var(--space-2);
  border-bottom: 1px solid var(--border);
  font-size: var(--text-xs);
}

.row a {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.pad {
  padding: var(--space-1) var(--space-2) var(--space-2);
}
</style>
