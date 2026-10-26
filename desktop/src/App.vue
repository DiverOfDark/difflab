<script setup lang="ts">
import "../../frontend/src/style.css";
import { computed, nextTick, onMounted, reactive, ref } from "vue";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { check, type Update } from "@tauri-apps/plugin-updater";
import { exit, relaunch } from "@tauri-apps/plugin-process";
import * as api from "./api";
import { usd, type Hardware, type Kind, type Offer, type Prefs, type LogLine, type Status, type Versions } from "./api";
import Wizard from "./Wizard.vue";

type Tab = "run" | "results" | "settings";

const loaded = ref(false);
const wizard = ref(false);
const tab = ref<Tab>("run");
const status = ref<Status>({ phase: "none", kind: null, gpu: "", usd: 0, usd_per_hour: 0, uptime_s: 0, draining: false, message: "", progress: null, slow: false, layers: [] });
const prefs = reactive<Prefs>({ idle_minutes: 5, stuck_minutes: 15, spend_limit_usd: 5, exit_when_done: false, auto_download: true, results_dir: "", setup_done: false, gpus: [] });
const hasKey = ref(false);
const hasLocal = ref(false);
const balance = ref<number | null>(null);
const resultsDir = ref("");
const hardware = ref<Hardware | null>(null);
const secrets = reactive({ runpodKey: "" });
const busy = ref("");
const error = ref("");
const saved = ref(false);
const sync = ref<{ done: number; total: number; file: string } | null>(null);
const syncNote = ref("");
const lastSynced = ref("");
const versions = ref<Versions | null>(null);
const logLines = ref<LogLine[]>([]);
const logBox = ref<HTMLElement | null>(null);
const logsOpen = ref(true);
const LOG_SHOWN = 1000;
const offers = ref<Offer[] | null>(null);
const offersNote = ref("");

// Follows the bottom unless the reader has scrolled up to read something.
function addLog(line: LogLine) {
  const box = logBox.value;
  const atBottom = !box || box.scrollHeight - box.scrollTop - box.clientHeight < 24;
  logLines.value.push(line);
  if (logLines.value.length > LOG_SHOWN) logLines.value.splice(0, logLines.value.length - LOG_SHOWN);
  if (atBottom) nextTick(() => logBox.value && (logBox.value.scrollTop = logBox.value.scrollHeight));
}
const appUpdate = ref<{ version: string; ready: boolean } | null>(null);
const appUpdateNote = ref("");
let pending: Update | null = null;
const CHECK_EVERY_MS = 6 * 3600 * 1000;

const running = computed(() => ["starting", "booting", "ready"].includes(status.value.phase));
const uptime = computed(() => {
  const m = Math.floor(status.value.uptime_s / 60);
  return m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : `${m} min`;
});
const phase = computed(
  () =>
    ({
      none: { label: "Not running", dot: "" },
      starting: { label: "Starting", dot: "running" },
      booting: { label: "Booting", dot: "running" },
      ready: { label: "Running", dot: "done" },
      ended: { label: "Stopped", dot: "" },
      error: { label: "Problem", dot: "failed" },
    })[status.value.phase],
);
const imageStale = computed(() => {
  const v = versions.value;
  return !!v?.image_latest && !!v.image_local && v.image_latest.digest !== v.image_local.digest;
});
const updateBadge = computed(() => !!appUpdate.value?.ready || imageStale.value);
const imageLabel = (v: { revision: string; created: string } | null | undefined) => (v ? `${v.revision || "?"} · ${v.created || "?"}` : "—");

// Downloaded quietly; installed when the window closes, or now if asked.
async function checkApp() {
  try {
    const found = await check();
    if (!found) {
      appUpdateNote.value = "up to date";
      return;
    }
    appUpdate.value = { version: found.version, ready: false };
    await found.download();
    pending = found;
    appUpdate.value = { version: found.version, ready: true };
  } catch (e) {
    appUpdateNote.value = `could not check (${String(e)})`;
  }
}

async function installNow() {
  if (!pending) return;
  await pending.install();
  await relaunch();
}

async function checkNow() {
  versions.value = await api.versions(true);
  await checkApp();
}

const LAYER_PILL: Record<string, string> = { done: "done", downloaded: "done", downloading: "live", extracting: "live", retrying: "failed", waiting: "stopped" };
const size = (bytes: number) => (bytes >= 1e9 ? `${(bytes / 1e9).toFixed(2)} GB` : bytes >= 1e6 ? `${Math.round(bytes / 1e6)} MB` : bytes ? `${Math.round(bytes / 1e3)} kB` : "");

const where = computed(() => (status.value.kind === "local" ? "this PC" : "RunPod"));

async function reload() {
  const got = await api.load();
  Object.assign(prefs, got.prefs);
  hasKey.value = got.has_runpod_key;
  hasLocal.value = got.has_local_results;
  status.value = got.status;
  resultsDir.value = got.results_dir;
  balance.value = got.balance;
  lastSynced.value = got.last_synced;
}

async function act(label: string, fn: () => Promise<unknown>) {
  busy.value = label;
  error.value = "";
  try {
    await fn();
  } catch (e) {
    error.value = String(e);
  } finally {
    busy.value = "";
  }
}

async function finishSetup(kind: Kind) {
  wizard.value = false;
  tab.value = "run";
  await reload();
  if (kind === "runpod") balance.value = await api.balance().catch(() => null);
}

const saveSettings = () =>
  act("save", async () => {
    await api.saveSettings({ ...prefs }, secrets);
    const newKey = !!secrets.runpodKey;
    secrets.runpodKey = "";
    await reload();
    if (newKey) balance.value = await api.balance();
    saved.value = true;
    setTimeout(() => (saved.value = false), 2000);
  });

async function loadOffers() {
  offersNote.value = "";
  try {
    offers.value = await api.offers();
  } catch (e) {
    offers.value = null;
    offersNote.value = String(e);
  }
}

const STOCK_PILL: Record<string, string> = { High: "done", Medium: "live", Low: "stopped" };
const shotUsd = (o: Offer) => (o.usd_per_hour * o.shot_minutes) / 60;
// Among what RunPod has now, so a tag never points at a card that cannot be rented.
const bestValue = computed(() => offers.value?.filter((o) => o.stock).reduce<Offer | null>((a, o) => (!a || shotUsd(o) < shotUsd(a) ? o : a), null)?.id);
const fastest = computed(() => offers.value?.filter((o) => o.stock).reduce<Offer | null>((a, o) => (!a || o.shot_minutes < a.shot_minutes ? o : a), null)?.id);

async function chooseFolder() {
  const picked = await open({ directory: true, defaultPath: resultsDir.value });
  if (typeof picked === "string") {
    prefs.results_dir = picked;
    resultsDir.value = picked;
  }
}

const start = (kind: Kind) =>
  act(kind, async () => {
    logLines.value = [];
    await api.start(kind);
  });
const stopNow = () => act("stop", () => api.stop(true));
const stopWhenDone = () => act("drain", () => api.stop(false));
const retryMachine = () =>
  act("retry", async () => {
    logLines.value = [];
    await api.retryMachine();
  });
const openUi = () => act("open", () => api.openUi());
const openUiBrowser = () => act("browser", () => api.openUiBrowser());
const openResults = () => act("folder", () => api.openResults());
const download = () =>
  act("download", async () => {
    syncNote.value = await api.syncResults();
  });

onMounted(async () => {
  await listen<Status>("status", (e) => (status.value = e.payload));
  await listen<LogLine>("log", (e) => addLog(e.payload));
  logLines.value = (await api.logs()).slice(-LOG_SHOWN);
  await listen<number | null>("balance", (e) => (balance.value = e.payload));
  await listen<string>("synced", (e) => (lastSynced.value = e.payload));
  await listen<{ done: number; total: number; file: string }>("sync", (e) => (sync.value = e.payload));
  await listen<Omit<Versions, "app" | "running_image">>("versions", (e) => {
    if (versions.value) versions.value = { ...versions.value, ...e.payload };
  });
  // Closing the window only hides it to the tray; Quit in the tray menu ends the app, installing a
  // downloaded update on the way out.
  await listen("quit-requested", async () => {
    if (pending) await pending.install().catch(() => undefined);
    await exit(0);
  });
  await reload();
  wizard.value = !prefs.setup_done && !running.value;
  loaded.value = true;
  hardware.value = await api.hardware();
  versions.value = await api.versions(false);
  checkApp();
  setInterval(checkApp, CHECK_EVERY_MS);
});
</script>

<template>
  <div v-if="loaded" class="app">
    <header class="top">
      <span class="brand">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <rect x="1" y="4" width="22" height="16" rx="2" stroke="var(--accent)" stroke-width="1.5" />
          <line x1="7" y1="4" x2="7" y2="20" stroke="var(--accent)" stroke-width="1.5" />
          <line x1="17" y1="4" x2="17" y2="20" stroke="var(--accent)" stroke-width="1.5" />
          <rect x="1" y="4" width="6" height="16" fill="var(--accent)" />
          <path d="M10.5 9v6l4.5-3z" fill="var(--accent)" />
        </svg>
        <span>DiffLab</span>
      </span>
      <span class="state mono">
        <span v-if="balance !== null" title="your RunPod balance">{{ usd(balance) }}</span>
        <button v-if="updateBadge" class="link update" title="an update is ready" @click="tab = 'settings'; wizard = false; loadOffers()">update</button>
        <span class="phase"><i class="dot" :class="phase.dot" />{{ phase.label.toLowerCase() }}</span>
      </span>
    </header>

    <main class="screen">
      <Wizard v-if="wizard" :prefs="prefs" :hardware="hardware" :has-key="hasKey" @done="finishSetup" />

      <div v-else class="stack">
        <nav class="seg">
          <button :class="{ on: tab === 'run' }" @click="tab = 'run'">Run</button>
          <button :class="{ on: tab === 'results' }" @click="tab = 'results'">Results</button>
          <button :class="{ on: tab === 'settings' }" @click="tab = 'settings'; loadOffers()">Settings</button>
        </nav>

        <template v-if="tab === 'run'">
        <section class="card">
          <div class="head">
            <h5>{{ phase.label }}<span v-if="status.kind" class="muted"> · {{ where }}</span></h5>
            <span v-if="running && status.kind === 'runpod'" class="mono muted">{{ usd(status.usd_per_hour) }}/h</span>
          </div>
          <div class="body">
            <div v-if="running" class="facts">
              <div v-if="status.gpu"><span class="label">GPU</span><span class="mono">{{ status.gpu }}</span></div>
              <div v-if="status.kind === 'runpod'"><span class="label">Spent</span><span class="mono">{{ usd(status.usd) }}</span></div>
              <div v-if="status.kind === 'runpod' && balance !== null"><span class="label">Balance</span><span class="mono">{{ usd(balance) }}</span></div>
              <div><span class="label">Up</span><span class="mono">{{ uptime }}</span></div>
            </div>
            <p v-if="status.draining" class="band hint">It stops as soon as the queue is done.</p>
            <div v-if="running && status.layers.length" class="layer-list mono">
              <div v-for="l in status.layers" :key="l.id">
                <span class="dim">{{ l.id }}</span>
                <span class="num">{{ size(l.bytes) }}</span>
                <span class="pill" :class="LAYER_PILL[l.state]">{{ l.state }}</span>
                <span class="dim">{{ l.retries ? `retried ${l.retries}×` : "" }}</span>
              </div>
            </div>
            <div v-if="status.progress !== null" class="bar"><i :style="{ width: `${status.progress * 100}%` }" /></div>
            <p v-if="status.message" class="band" :class="status.phase === 'error' ? 'error' : 'hint'">{{ status.message }}</p>

            <div v-if="running" class="buttons">
              <button v-if="status.slow" class="primary" :disabled="!!busy" @click="retryMachine"><i v-if="busy === 'retry'" class="spin" />{{ busy === "retry" ? "Starting another…" : "Try another machine" }}</button>
              <button class="primary" :disabled="status.phase !== 'ready' || !!busy" @click="openUi"><i v-if="busy === 'open'" class="spin" />Open DiffLab</button>
              <button class="link" :disabled="status.phase !== 'ready' || !!busy" @click="openUiBrowser">in browser →</button>
              <button v-if="status.kind === 'runpod' && !status.draining" :disabled="!!busy" @click="stopWhenDone"><i v-if="busy === 'drain'" class="spin" />Stop when queue is done</button>
              <button class="danger" :disabled="!!busy" @click="stopNow"><i v-if="busy === 'stop'" class="spin" />{{ busy === "stop" ? "Stopping…" : "Stop now" }}</button>
            </div>
            <template v-else>
              <div class="buttons">
                <button class="primary" :disabled="!hasKey || !!busy" @click="start('runpod')"><i v-if="busy === 'runpod'" class="spin" />{{ busy === "runpod" ? "Starting…" : "Run on RunPod" }}</button>
                <button :disabled="!hardware?.ok || !!busy" @click="start('local')"><i v-if="busy === 'local'" class="spin" />{{ busy === "local" ? "Starting…" : "Run on this PC" }}</button>
              </div>
              <p v-if="!hasKey" class="note">Connect a RunPod account in Settings to run in the cloud.</p>
              <p v-if="!hardware" class="note">Checking this PC…</p>
              <p v-else-if="hardware.ok" class="note">This PC: {{ hardware.gpu }}, {{ hardware.ram_gb.toFixed(0) }} GB RAM, {{ hardware.runtime }}.</p>
              <p v-else class="note">This PC: {{ hardware.reason }}.</p>
            </template>
          </div>
        </section>

        <section v-if="logLines.length" class="card logs">
          <div class="head slim">
            <h5>Log</h5>
            <button class="link" @click="logsOpen = !logsOpen">{{ logsOpen ? "hide" : "show" }}</button>
          </div>
          <div v-show="logsOpen" ref="logBox" class="lines mono">
            <div v-for="(l, i) in logLines" :key="i" :class="l.source">{{ l.line }}</div>
          </div>
        </section>
        </template>

        <section v-else-if="tab === 'results'" class="card">
          <div class="head"><h5>Results</h5><button class="link" @click="openResults">open folder →</button></div>
          <div class="body">
            <p class="mono muted path">{{ resultsDir }}</p>
            <div class="buttons">
              <button class="primary" :disabled="(!hasLocal && !(running && status.kind === 'runpod')) || !!busy" @click="download"><i v-if="busy === 'download'" class="spin" />{{ busy === "download" ? "Downloading…" : "Download results" }}</button>
            </div>
            <template v-if="busy === 'download' && sync">
              <div class="bar"><i :style="{ width: `${sync.total ? (sync.done / sync.total) * 100 : 0}%` }" /></div>
              <p class="note mono">{{ sync.done }} / {{ sync.total }} · {{ sync.file }}</p>
            </template>
            <p v-else-if="syncNote" class="band hint">{{ syncNote }}</p>
            <p v-if="lastSynced" class="note mono">{{ lastSynced }}</p>
            <p class="note">Results made on this PC are copied from Docker when you press Download. RunPod's download {{ prefs.auto_download ? "by themselves every minute while a GPU runs, and once more before it stops" : "while a GPU runs there" }}.</p>
          </div>
        </section>

        <section v-else class="card">
          <div class="head"><h5>Settings</h5><button class="link" @click="wizard = true">run setup again →</button></div>
          <div class="body">
            <label class="field"><span class="label">RunPod API key</span>
              <input v-model="secrets.runpodKey" type="password" :placeholder="hasKey ? 'saved; paste a new one to replace it' : 'rpa_…'" />
            </label>
            <div>
              <div class="knob">
                <div><span class="name">Nothing to do</span><span class="hint">Minutes idle with the app closed before the GPU is deleted.</span></div>
                <input v-model.number="prefs.idle_minutes" type="number" min="2" max="30" />
              </div>
              <div class="knob">
                <div><span class="name">Spend limit</span><span class="hint">Dollars per session.</span></div>
                <input v-model.number="prefs.spend_limit_usd" type="number" min="1" step="1" />
              </div>
              <div class="knob">
                <div><span class="name">Stuck</span><span class="hint">Minutes a render may go without progress.</span></div>
                <input v-model.number="prefs.stuck_minutes" type="number" min="5" max="60" />
              </div>
            </div>
            <div class="field">
              <span class="label">RunPod GPUs</span>
              <span class="note">It rents whichever ticked GPU is free first. A 15 s shot is estimated from each card's speed against a 3090's ~10 minutes; what it costs is that time at the hourly price. Stock counts machines with the 64 GB of RAM every pod needs.</span>
              <div v-if="offers" class="offers mono">
                <div class="th">
                  <span />
                  <span>GPU</span>
                  <span class="num">VRAM</span>
                  <span class="num">per hour</span>
                  <span class="num">15 s shot</span>
                  <span class="num">costs</span>
                  <span>now</span>
                </div>
                <label v-for="o in offers" :key="o.id" :class="{ dim: !o.stock }">
                  <input v-model="prefs.gpus" type="checkbox" :value="o.id" />
                  <span>{{ o.name }}<span v-if="o.id === bestValue" class="tag">best value</span><span v-if="o.id === fastest" class="tag">fastest</span></span>
                  <span class="num">{{ o.vram_gb }} GB</span>
                  <span class="num">{{ usd(o.usd_per_hour) }}</span>
                  <span class="num">~{{ Math.round(o.shot_minutes) }} min</span>
                  <span class="num">~{{ usd(shotUsd(o)) }}</span>
                  <span><span class="pill" :class="o.stock ? STOCK_PILL[o.stock] : 'failed'">{{ o.stock ? o.stock.toLowerCase() : "none" }}</span></span>
                </label>
              </div>
              <p v-else class="note">{{ offersNote || "Asking RunPod for prices…" }}</p>
            </div>
            <label class="check"><input v-model="prefs.exit_when_done" type="checkbox" /> Stop as soon as the queue is done</label>
            <label class="check"><input v-model="prefs.auto_download" type="checkbox" /> Download RunPod results automatically</label>
            <label class="field"><span class="label">Results folder</span>
              <div class="row"><input :value="resultsDir" readonly /><button @click="chooseFolder">Choose…</button></div>
            </label>
            <div class="buttons">
              <button class="primary" :disabled="!!busy" @click="saveSettings"><i v-if="busy === 'save'" class="spin" />{{ saved ? "Saved" : "Save" }}</button>
            </div>
          </div>
        </section>

        <section v-if="tab === 'settings' && versions" class="card">
          <div class="head"><h5>Versions</h5><button class="link" @click="checkNow">check now →</button></div>
          <div class="body">
            <div class="knob">
              <div><span class="name">App {{ versions.app }}</span>
                <span v-if="appUpdate?.ready" class="hint">{{ appUpdate.version }} is downloaded and installs when you quit from the tray.</span>
                <span v-else-if="appUpdate" class="hint">Downloading {{ appUpdate.version }}…</span>
                <span v-else class="hint">{{ appUpdateNote || "checking…" }}</span>
              </div>
              <button v-if="appUpdate?.ready" class="primary" @click="installNow">Restart to update</button>
            </div>
            <div class="knob">
              <div><span class="name">DiffLab image</span>
                <span class="hint">Latest: {{ imageLabel(versions.image_latest) }}</span>
                <span class="hint">On this PC: {{ versions.image_local ? imageLabel(versions.image_local) : "not downloaded" }}<template v-if="versions.updating"> · downloading the update in the background…</template><template v-else-if="imageStale"> · an update is available</template></span>
                <span v-if="versions.running_image && versions.image_latest && versions.running_image !== versions.image_latest.revision" class="hint">Running {{ versions.running_image }}; stop and start again to use the latest.</span>
                <span class="hint">RunPod always starts the latest.</span>
              </div>
            </div>
          </div>
        </section>

        <p v-if="error" class="band error">{{ error }}</p>
      </div>
    </main>
  </div>
</template>

<style scoped>
.app {
  height: 100vh;
  display: flex;
  flex-direction: column;
}
.top {
  height: 56px;
  flex: 0 0 auto;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 var(--space-3);
  border-bottom: 1px solid var(--border);
  background: var(--bg-base);
}
.brand {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  font-family: var(--font-heading);
  font-weight: var(--weight-bold);
  letter-spacing: -0.01em;
  font-size: 15px;
}
.state {
  display: flex;
  gap: var(--space-2);
  font-size: var(--text-xs);
  color: var(--text-secondary);
}
button {
  display: inline-flex;
  align-items: center;
  gap: var(--space-1);
}
.spin {
  width: 12px;
  height: 12px;
  border: 1.5px solid currentColor;
  border-right-color: transparent;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
.layer-list {
  display: grid;
  gap: 4px;
  font-size: 11px;
}
.layer-list > div {
  display: grid;
  grid-template-columns: 110px 70px 110px 1fr;
  align-items: center;
}
.layer-list .num {
  text-align: right;
  padding-right: var(--space-2);
}
.logs .lines {
  max-height: 260px;
  overflow-y: auto;
  padding: var(--space-1) var(--space-2);
  font-size: 11px;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-all;
  background: var(--bg-base);
}
.logs .system {
  color: var(--text-tertiary);
}
.update {
  color: var(--accent);
}
.phase {
  display: flex;
  align-items: center;
  gap: var(--space-1);
}
.seg {
  align-self: flex-start;
}
.seg > button {
  background: transparent;
}
.facts {
  display: grid;
  gap: 6px;
}
.facts > div {
  display: grid;
  grid-template-columns: 72px 1fr;
  align-items: baseline;
}
.buttons {
  display: flex;
  gap: var(--space-1);
  flex-wrap: wrap;
}
.band {
  margin: 0;
}
.field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.row {
  display: flex;
  gap: var(--space-1);
}
.check {
  display: flex;
  gap: var(--space-1);
  align-items: center;
  color: var(--text-secondary);
}
.offers {
  display: grid;
  gap: 4px;
  font-size: var(--text-xs);
}
.offers > label,
.offers > .th {
  display: grid;
  grid-template-columns: 20px 1fr 52px 64px 68px 56px 64px;
  align-items: center;
}
.offers > .th {
  color: var(--text-tertiary);
}
.offers .tag {
  display: block;
  font-size: 10px;
  color: var(--accent);
}
.offers .num {
  text-align: right;
  padding-right: var(--space-2);
}
.path {
  margin: 0;
  font-size: var(--text-xs);
  word-break: break-all;
}
</style>
