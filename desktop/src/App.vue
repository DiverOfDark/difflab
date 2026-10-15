<script setup lang="ts">
import "../../frontend/src/style.css";
import { computed, onMounted, reactive, ref } from "vue";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { check, type Update } from "@tauri-apps/plugin-updater";
import { relaunch } from "@tauri-apps/plugin-process";
import { getCurrentWindow } from "@tauri-apps/api/window";
import * as api from "./api";
import { usd, type Hardware, type Kind, type Prefs, type Status, type Versions } from "./api";
import Wizard from "./Wizard.vue";

type Tab = "run" | "results" | "settings";

const loaded = ref(false);
const wizard = ref(false);
const tab = ref<Tab>("run");
const status = ref<Status>({ phase: "none", kind: null, gpu: "", usd: 0, usd_per_hour: 0, uptime_s: 0, draining: false, message: "", progress: null });
const prefs = reactive<Prefs>({ datacenter: "EU-RO-1", volume_id: "", idle_minutes: 5, stuck_minutes: 15, spend_limit_usd: 5, exit_when_done: false, results_dir: "", setup_done: false });
const datacenters = ref<string[]>([]);
const hasKey = ref(false);
const hasS3 = ref(false);
const balance = ref<number | null>(null);
const resultsDir = ref("");
const hardware = ref<Hardware | null>(null);
const secrets = reactive({ runpodKey: "", s3Secret: "" });
const busy = ref("");
const error = ref("");
const saved = ref(false);
const sync = ref<{ done: number; total: number; file: string } | null>(null);
const syncNote = ref("");
const versions = ref<Versions | null>(null);
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

const where = computed(() => (status.value.kind === "local" ? "this PC" : "RunPod"));

async function reload() {
  const got = await api.load();
  Object.assign(prefs, got.prefs);
  hasKey.value = got.has_runpod_key;
  hasS3.value = got.has_s3;
  datacenters.value = got.datacenters;
  status.value = got.status;
  resultsDir.value = got.results_dir;
  balance.value = got.balance;
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
  act("Saving", async () => {
    await api.saveSettings({ ...prefs }, secrets);
    const newKey = !!secrets.runpodKey;
    secrets.runpodKey = secrets.s3Secret = "";
    await reload();
    if (newKey) balance.value = await api.balance();
    saved.value = true;
    setTimeout(() => (saved.value = false), 2000);
  });

async function chooseFolder() {
  const picked = await open({ directory: true, defaultPath: resultsDir.value });
  if (typeof picked === "string") {
    prefs.results_dir = picked;
    resultsDir.value = picked;
  }
}

const start = (kind: Kind) => act("Starting", () => api.start(kind));
const stopNow = () => act("Stopping", () => api.stop(true));
const stopWhenDone = () => act("Stopping", () => api.stop(false));
const openUi = () => act("Opening", () => api.openUi());
const openResults = () => act("Opening", () => api.openResults());
const download = () =>
  act("Downloading", async () => {
    syncNote.value = await api.syncResults();
  });

onMounted(async () => {
  await listen<Status>("status", (e) => (status.value = e.payload));
  await listen<number | null>("balance", (e) => (balance.value = e.payload));
  await listen<{ done: number; total: number; file: string }>("sync", (e) => (sync.value = e.payload));
  await listen<Omit<Versions, "app" | "running_image">>("versions", (e) => {
    if (versions.value) versions.value = { ...versions.value, ...e.payload };
  });
  await getCurrentWindow().onCloseRequested(async (event) => {
    if (!pending) return;
    event.preventDefault();
    await pending.install().catch(() => undefined);
    await getCurrentWindow().destroy();
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
        <button v-if="updateBadge" class="link update" title="an update is ready" @click="tab = 'settings'; wizard = false">update</button>
        <span class="phase"><i class="dot" :class="phase.dot" />{{ phase.label.toLowerCase() }}</span>
      </span>
    </header>

    <main class="screen">
      <Wizard v-if="wizard" :prefs="prefs" :hardware="hardware" :datacenters="datacenters" :has-key="hasKey" @done="finishSetup" />

      <div v-else class="stack">
        <nav class="seg">
          <button :class="{ on: tab === 'run' }" @click="tab = 'run'">Run</button>
          <button :class="{ on: tab === 'results' }" @click="tab = 'results'">Results</button>
          <button :class="{ on: tab === 'settings' }" @click="tab = 'settings'">Settings</button>
        </nav>

        <section v-if="tab === 'run'" class="card">
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
            <div v-if="status.progress !== null" class="bar"><i :style="{ width: `${status.progress * 100}%` }" /></div>
            <p v-if="status.message" class="band" :class="status.phase === 'error' ? 'error' : 'hint'">{{ status.message }}</p>

            <div v-if="running" class="buttons">
              <button class="primary" :disabled="status.phase !== 'ready'" @click="openUi">Open DiffLab</button>
              <button v-if="status.kind === 'runpod' && !status.draining" @click="stopWhenDone">Stop when queue is done</button>
              <button class="danger" @click="stopNow">Stop now</button>
            </div>
            <template v-else>
              <div class="buttons">
                <button class="primary" :disabled="!hasKey || !!busy" @click="start('runpod')">Run on RunPod</button>
                <button :disabled="!hardware?.ok || !!busy" @click="start('local')">Run on this PC</button>
              </div>
              <p v-if="!hasKey" class="note">Connect a RunPod account in Settings to run in the cloud.</p>
              <p v-if="!hardware" class="note">Checking this PC…</p>
              <p v-else-if="hardware.ok" class="note">This PC: {{ hardware.gpu }}, {{ hardware.ram_gb.toFixed(0) }} GB RAM, {{ hardware.runtime }}.</p>
              <p v-else class="note">This PC: {{ hardware.reason }}.</p>
            </template>
          </div>
        </section>

        <section v-else-if="tab === 'results'" class="card">
          <div class="head"><h5>Results</h5><button class="link" @click="openResults">open folder →</button></div>
          <div class="body">
            <p class="mono muted path">{{ resultsDir }}</p>
            <div class="buttons">
              <button class="primary" :disabled="!hasS3 || !!busy" @click="download">Download results</button>
            </div>
            <template v-if="busy === 'Downloading' && sync">
              <div class="bar"><i :style="{ width: `${sync.total ? (sync.done / sync.total) * 100 : 0}%` }" /></div>
              <p class="note mono">{{ sync.done }} / {{ sync.total }} · {{ sync.file }}</p>
            </template>
            <p v-else-if="syncNote" class="band hint">{{ syncNote }}</p>
            <p v-if="!hasS3" class="note">Add an S3 key in Settings to download results, even with no GPU running.</p>
          </div>
        </section>

        <section v-else class="card">
          <div class="head"><h5>Settings</h5><button class="link" @click="wizard = true">run setup again →</button></div>
          <div class="body">
            <label class="field"><span class="label">RunPod API key</span>
              <input v-model="secrets.runpodKey" type="password" :placeholder="hasKey ? 'saved; paste a new one to replace it' : 'rpa_…'" />
            </label>
            <label class="field"><span class="label">S3 key</span>
              <input v-model="secrets.s3Secret" type="password" :placeholder="hasS3 ? 'saved; paste a new one to replace it' : 'rps_…'" />
            </label>
            <label class="field"><span class="label">Region</span>
              <select v-model="prefs.datacenter">
                <option v-for="dc in datacenters" :key="dc" :value="dc">{{ dc }}</option>
              </select>
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
            <label class="check"><input v-model="prefs.exit_when_done" type="checkbox" /> Stop as soon as the queue is done</label>
            <label class="field"><span class="label">Results folder</span>
              <div class="row"><input :value="resultsDir" readonly /><button @click="chooseFolder">Choose…</button></div>
            </label>
            <div class="buttons">
              <button class="primary" :disabled="!!busy" @click="saveSettings">{{ saved ? "Saved" : "Save" }}</button>
            </div>
          </div>
        </section>

        <section v-if="tab === 'settings' && versions" class="card">
          <div class="head"><h5>Versions</h5><button class="link" @click="checkNow">check now →</button></div>
          <div class="body">
            <div class="knob">
              <div><span class="name">App {{ versions.app }}</span>
                <span v-if="appUpdate?.ready" class="hint">{{ appUpdate.version }} is downloaded and installs when you close the app.</span>
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
.path {
  margin: 0;
  font-size: var(--text-xs);
  word-break: break-all;
}
</style>
