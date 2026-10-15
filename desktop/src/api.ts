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
}

export interface Prefs {
  datacenter: string;
  volume_id: string;
  idle_minutes: number;
  stuck_minutes: number;
  spend_limit_usd: number;
  exit_when_done: boolean;
  results_dir: string;
  setup_done: boolean;
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
  has_s3: boolean;
  has_local_results: boolean;
  datacenters: string[];
  status: Status;
  results_dir: string;
  balance: number | null;
}

export interface Secrets {
  runpodKey?: string;
  s3Secret?: string;
}

export const load = () => invoke<Loaded>("load");
export const hardware = () => invoke<Hardware>("hardware");
export const balance = (key?: string) => invoke<number | null>("balance", { key: key || null });
export const saveSettings = (prefs: Prefs, secrets: Secrets = {}) =>
  invoke("save_settings", {
    prefs,
    runpodKey: secrets.runpodKey || null,
    s3Secret: secrets.s3Secret || null,
  });
export const start = (kind: Kind) => invoke("start", { kind });
export const stop = (immediately: boolean) => invoke("stop", { immediately });
export const openUi = () => invoke("open_ui");
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
