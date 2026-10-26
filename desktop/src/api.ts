import { invoke } from "@tauri-apps/api/core";

export type Kind = "runpod" | "local";

export interface Status {
  phase: "none" | "starting" | "booting" | "ready" | "ended" | "error";
  kind: Kind | null;
  gpu: string;
  usd: number;
  usd_per_hour: number;
  uptime_s: number;
  draining: boolean;
  message: string;
  progress: number | null;
  slow: boolean;
  layers: Layer[];
}

export interface Layer {
  id: string;
  bytes: number;
  state: "waiting" | "downloading" | "downloaded" | "extracting" | "retrying" | "done";
  retries: number;
}

export interface Prefs {
  idle_minutes: number;
  stuck_minutes: number;
  spend_limit_usd: number;
  exit_when_done: boolean;
  auto_download: boolean;
  results_dir: string;
  setup_done: boolean;
  gpus: string[];
}

export interface Offer {
  id: string;
  name: string;
  vram_gb: number;
  usd_per_hour: number;
  shot_minutes: number;
  stock: "High" | "Medium" | "Low" | null;
}

export interface Hardware {
  ok: boolean;
  reason: string;
  gpu: string;
  vram_gb: number;
  ram_gb: number;
  runtime: string;
}

export interface Loaded {
  prefs: Prefs;
  has_runpod_key: boolean;
  has_local_results: boolean;
  status: Status;
  results_dir: string;
  balance: number | null;
  last_synced: string;
}

export interface Secrets {
  runpodKey?: string;
}

export const load = () => invoke<Loaded>("load");
export const hardware = () => invoke<Hardware>("hardware");
export const offers = () => invoke<Offer[]>("offers");
export const balance = (key?: string) => invoke<number | null>("balance", { key: key || null });
export const saveSettings = (prefs: Prefs, secrets: Secrets = {}) =>
  invoke("save_settings", {
    prefs,
    runpodKey: secrets.runpodKey || null,
  });
export const start = (kind: Kind) => invoke("start", { kind });
export const stop = (immediately: boolean) => invoke("stop", { immediately });
export const retryMachine = () => invoke("retry_machine");
export const openUi = () => invoke("open_ui");
export const openUiBrowser = () => invoke("open_ui_browser");
export const syncResults = () => invoke<string>("sync_results");
export const openResults = () => invoke("open_results");
export const openLink = (url: string) => invoke("open_link", { url });

export const usd = (n: number) => `$${n.toFixed(2)}`;

export interface ImageVersion {
  digest: string;
  revision: string;
  created: string;
}

export interface Versions {
  app: string;
  running_image: string;
  image_latest: ImageVersion | null;
  image_local: ImageVersion | null;
  updating: boolean;
  checked_at: number;
}

export const versions = (refresh = false) => invoke<Versions>("versions", { refresh });

export interface LogLine {
  source: string;
  line: string;
  ts: string;
}

export const logs = () => invoke<LogLine[]>("logs");
