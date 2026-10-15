<script setup lang="ts">
import { computed, onMounted, reactive, ref } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";

type Kind = "runpod" | "local";
interface Status {
  phase: "none" | "starting" | "booting" | "ready" | "ended" | "error";
  kind: Kind | null;
  gpu: string;
  usd: number;
  usd_per_hour: number;
  uptime_s: number;
  draining: boolean;
  message: string;
}
interface Prefs {
  datacenter: string;
  volume_id: string;
  idle_minutes: number;
  stuck_minutes: number;
  spend_limit_usd: number;
  exit_when_done: boolean;
  results_dir: string;
}
interface Hardware {
  ok: boolean;
  reason: string;
  gpu: string;
  vram_gb: number;
  ram_gb: number;
  runtime: string;
}

const status = ref<Status>({ phase: "none", kind: null, gpu: "", usd: 0, usd_per_hour: 0, uptime_s: 0, draining: false, message: "" });
const prefs = reactive<Prefs>({ datacenter: "EU-RO-1", volume_id: "", idle_minutes: 5, stuck_minutes: 15, spend_limit_usd: 5, exit_when_done: false, results_dir: "" });
const datacenters = ref<string[]>([]);
const hasKey = ref(false);
const hasS3 = ref(false);
const resultsDir = ref("");
const hardware = ref<Hardware | null>(null);
const secrets = reactive({ runpod: "", s3Access: "", s3Secret: "" });
const settingsOpen = ref(false);
const busy = ref("");
const error = ref("");
const sync = ref<{ done: number; total: number; file: string } | null>(null);
const syncNote = ref("");

const running = computed(() => ["starting", "booting", "ready"].includes(status.value.phase));
const uptime = computed(() => {
  const m = Math.floor(status.value.uptime_s / 60);
  return m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : `${m} min`;
});
const phaseLabel = computed(() => ({
  none: "Not running",
  starting: "Starting",
  booting: "Booting",
  ready: "Running",
  ended: "Stopped",
  error: "Problem",
})[status.value.phase]);

async function load() {
  const loaded = await invoke<{ prefs: Prefs; has_runpod_key: boolean; has_s3: boolean; datacenters: string[]; status: Status; results_dir: string }>("load");
  Object.assign(prefs, loaded.prefs);
  hasKey.value = loaded.has_runpod_key;
  hasS3.value = loaded.has_s3;
  datacenters.value = loaded.datacenters;
  status.value = loaded.status;
  resultsDir.value = loaded.results_dir;
  settingsOpen.value = !loaded.has_runpod_key;
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

const saveSettings = () =>
  act("Saving", async () => {
    await invoke("save_settings", {
      prefs: { ...prefs },
      runpodKey: secrets.runpod || null,
      s3Access: secrets.s3Access || null,
      s3Secret: secrets.s3Secret || null,
    });
    secrets.runpod = secrets.s3Access = secrets.s3Secret = "";
    await load();
    settingsOpen.value = false;
  });

async function chooseFolder() {
  const picked = await open({ directory: true, defaultPath: resultsDir.value });
  if (typeof picked === "string") {
    prefs.results_dir = picked;
    resultsDir.value = picked;
  }
}

const start = (kind: Kind) => act("Starting", () => invoke("start", { kind }));
const stopNow = () => act("Stopping", () => invoke("stop", { immediately: true }));
const stopWhenDone = () => act("Stopping", () => invoke("stop", { immediately: false }));
const openUi = () => act("Opening", () => invoke("open_ui"));
const openResults = () => act("Opening", () => invoke("open_results"));
const download = () =>
  act("Downloading", async () => {
    syncNote.value = await invoke<string>("sync_results");
  });

onMounted(async () => {
  await listen<Status>("status", (e) => (status.value = e.payload));
  await listen<{ done: number; total: number; file: string }>("sync", (e) => (sync.value = e.payload));
  await load();
  hardware.value = await invoke<Hardware>("hardware");
});
</script>

<template>
  <main>
    <header>
      <h1>DiffLab</h1>
      <button class="link" @click="settingsOpen = !settingsOpen">{{ settingsOpen ? "Close settings" : "Settings" }}</button>
    </header>

    <section v-if="settingsOpen" class="card">
      <h2>Settings</h2>
      <label>RunPod API key
        <input v-model="secrets.runpod" type="password" :placeholder="hasKey ? 'saved — paste to replace' : 'rpa_…'" />
      </label>
      <label>S3 access key <small>(RunPod → Settings → S3 API Keys; lets you download results with no pod running)</small>
        <input v-model="secrets.s3Access" type="password" :placeholder="hasS3 ? 'saved — paste to replace' : 'user_…'" />
      </label>
      <label>S3 secret
        <input v-model="secrets.s3Secret" type="password" :placeholder="hasS3 ? 'saved — paste to replace' : 'rps_…'" />
      </label>
      <label>Region of your storage
        <select v-model="prefs.datacenter">
          <option v-for="dc in datacenters" :key="dc" :value="dc">{{ dc }}</option>
        </select>
      </label>
      <div class="row">
        <label>Stop after idle (min)<input v-model.number="prefs.idle_minutes" type="number" min="2" max="30" /></label>
        <label>Spend limit per session ($)<input v-model.number="prefs.spend_limit_usd" type="number" min="1" step="1" /></label>
      </div>
      <label class="check"><input v-model="prefs.exit_when_done" type="checkbox" /> Stop the pod as soon as the queue is done</label>
      <label>Results folder
        <div class="row"><input :value="resultsDir" readonly /><button @click="chooseFolder">Choose…</button></div>
      </label>
      <button class="primary" :disabled="!!busy" @click="saveSettings">Save</button>
    </section>

    <section class="card status" :class="status.phase">
      <div class="phase"><span class="dot"></span>{{ phaseLabel }}<span v-if="status.kind" class="muted"> · {{ status.kind === "runpod" ? "RunPod" : "this PC" }}</span></div>
      <div v-if="running" class="facts">
        <div v-if="status.gpu"><span class="muted">GPU</span>{{ status.gpu }}</div>
        <div v-if="status.kind === 'runpod'"><span class="muted">Spent</span>${{ status.usd.toFixed(2) }} <small class="muted">at ${{ status.usd_per_hour.toFixed(2) }}/h</small></div>
        <div><span class="muted">Up</span>{{ uptime }}</div>
      </div>
      <p v-if="status.draining" class="note">Stops when the queue is done.</p>
      <p v-if="status.message" class="note">{{ status.message }}</p>

      <div v-if="running" class="actions">
        <button class="primary" :disabled="status.phase !== 'ready'" @click="openUi">Open DiffLab</button>
        <button v-if="status.kind === 'runpod' && !status.draining" @click="stopWhenDone">Stop when queue is done</button>
        <button class="danger" @click="stopNow">Stop now</button>
      </div>
      <div v-else class="actions">
        <button class="primary" :disabled="!hasKey || !!busy" @click="start('runpod')">Run on RunPod</button>
        <button :disabled="!hardware?.ok || !!busy" @click="start('local')">Run on this PC</button>
      </div>
      <p v-if="!running && !hasKey" class="muted small">Add your RunPod API key in Settings to run in the cloud.</p>
      <p v-if="!running && hardware && !hardware.ok" class="muted small">This PC: {{ hardware.reason }}.</p>
      <p v-if="!running && hardware?.ok" class="muted small">This PC: {{ hardware.gpu }}, {{ hardware.ram_gb.toFixed(0) }} GB RAM, {{ hardware.runtime }}.</p>
    </section>

    <section class="card">
      <h2>Results</h2>
      <p class="muted small">{{ resultsDir }}</p>
      <div class="actions">
        <button :disabled="!hasS3 || !!busy" @click="download">Download results</button>
        <button @click="openResults">Open folder</button>
      </div>
      <p v-if="busy === 'Downloading' && sync" class="note">{{ sync.done }} / {{ sync.total }} {{ sync.file }}</p>
      <p v-else-if="syncNote" class="note">{{ syncNote }}</p>
      <p v-if="!hasS3" class="muted small">Add your S3 key in Settings to download results.</p>
    </section>

    <p v-if="error" class="error">{{ error }}</p>
    <p v-if="busy" class="muted small">{{ busy }}…</p>
  </main>
</template>

<style>
:root { color-scheme: dark; font-family: system-ui, sans-serif; color: #e6e6e6; background: #15171a; font-size: 14px; }
body { margin: 0; }
main { padding: 16px 20px; display: flex; flex-direction: column; gap: 12px; }
header { display: flex; align-items: baseline; justify-content: space-between; }
h1 { font-size: 20px; margin: 0; }
h2 { font-size: 14px; margin: 0 0 8px; text-transform: uppercase; letter-spacing: .06em; color: #9aa0a6; }
.card { background: #1d2025; border: 1px solid #2b2f36; border-radius: 10px; padding: 14px; display: flex; flex-direction: column; gap: 10px; }
label { display: flex; flex-direction: column; gap: 4px; color: #b8bcc2; }
label.check { flex-direction: row; align-items: center; gap: 8px; }
input, select { background: #121418; color: #e6e6e6; border: 1px solid #343944; border-radius: 6px; padding: 7px 8px; font: inherit; }
.row { display: flex; gap: 8px; }
.row > * { flex: 1; }
button { background: #2a2f37; color: #e6e6e6; border: 1px solid #3a404b; border-radius: 6px; padding: 8px 12px; font: inherit; cursor: pointer; }
button:disabled { opacity: .45; cursor: default; }
button.primary { background: #3b6ef5; border-color: #3b6ef5; }
button.danger { background: #7a2630; border-color: #96303c; }
button.link { background: none; border: none; color: #8ab4f8; padding: 0; }
.actions { display: flex; gap: 8px; flex-wrap: wrap; }
.phase { font-size: 16px; font-weight: 600; display: flex; align-items: center; gap: 8px; }
.dot { width: 10px; height: 10px; border-radius: 50%; background: #6b7078; }
.ready .dot { background: #3ccf6e; }
.starting .dot, .booting .dot { background: #f2b33d; }
.error .dot { background: #e5534b; }
.facts { display: grid; gap: 4px; }
.facts .muted { display: inline-block; width: 60px; }
.muted { color: #8b9098; }
.small { font-size: 12px; margin: 0; }
.note { margin: 0; color: #c9ccd1; }
.error { color: #ff7b72; margin: 0; }
</style>
