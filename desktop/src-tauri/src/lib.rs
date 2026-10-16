mod local;
mod runpod;
mod store;

use rand::RngCore;
use serde::Serialize;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tauri::webview::DownloadEvent;
use tauri::{AppHandle, Emitter, Manager, State, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_notification::NotificationExt;
#[cfg(not(target_os = "linux"))]
use tauri_plugin_opener::OpenerExt;
use tokio::sync::Mutex;

use store::{Kind, Prefs, Session, Store, RUNPOD_KEY};

// Also the status refresh; the pod's watchdog counts the app as gone after DIFFLAB_IDLE_MINUTES without one.
const HEARTBEAT: Duration = Duration::from_secs(10);
const BALANCE_EVERY: i64 = 60;
const VERSIONS_EVERY: i64 = 6 * 3600;
const LOG_KEEP: usize = 2000;
const SYNC_EVERY: i64 = 60;
// How long Stop waits for results before deleting the pod; the rest stay on the volume for the next session.
const STOP_SYNC_LIMIT: Duration = Duration::from_secs(60);
// A result download that receives nothing for this long is dropped: RunPod's proxy can stall a stream
// without closing it, and the sync it belongs to holds every later one, Stop's included.
const STALL: Duration = Duration::from_secs(60);
// A first pod answered 118 s after creation (4090, EU-RO-1); one whose host retried a 2 GB layer for
// 7 minutes (3090, EU-CZ-1) never got there. Past this the user is offered another machine.
const SLOW_BOOT_S: i64 = 480;
const NOISE: [&str; 2] = ["/api/pod/heartbeat", "GET /api/gpu"];
const RESULT_PREFIXES: [&str; 2] = ["images/", "videos/"];

#[derive(Serialize, Clone, Default)]
struct Status {
    phase: String,
    kind: Option<Kind>,
    gpu: String,
    usd: f64,
    usd_per_hour: f64,
    uptime_s: i64,
    draining: bool,
    message: String,
    progress: Option<f64>,
    slow: bool,
    layers: Vec<local::Layer>,
}

#[derive(Serialize, Clone, Default)]
struct Versions {
    image_latest: Option<local::ImageVersion>,
    image_local: Option<local::ImageVersion>,
    updating: bool,
    checked_at: i64,
}

struct Inner {
    store: Store,
    balance: Mutex<(i64, Option<f64>)>,
    status: Mutex<Status>,
    session: Mutex<Option<Session>>,
    draining: Mutex<bool>,
    versions: Mutex<Versions>,
    http: reqwest::Client,
    // std rather than tokio locks: the log readers fill them from synchronous callbacks.
    logs: std::sync::Mutex<std::collections::VecDeque<runpod::LogLine>>,
    boot: std::sync::Mutex<Option<local::Pulled>>,
    // The host's registry pull failed and is being retried, or the container has started.
    retrying: std::sync::Mutex<bool>,
    container_started: std::sync::Mutex<bool>,
    follower: std::sync::Mutex<Option<tauri::async_runtime::JoinHandle<()>>>,
    tray_status: std::sync::Mutex<Option<tauri::menu::MenuItem<tauri::Wry>>>,
    syncing: Mutex<()>,
    synced_at: std::sync::Mutex<i64>,
    last_synced: std::sync::Mutex<String>,
}

type Shared = Arc<Inner>;

fn now() -> i64 {
    chrono::Utc::now().timestamp()
}

impl Inner {
    fn runpod(&self) -> Result<runpod::Client, String> {
        let key = self
            .store
            .secret(RUNPOD_KEY)
            .ok_or("add your RunPod API key first")?;
        Ok(runpod::Client::new(key))
    }

    async fn refresh_balance(&self, app: &AppHandle, force: bool) -> Option<f64> {
        let mut held = self.balance.lock().await;
        if !force && now() - held.0 < BALANCE_EVERY {
            return held.1;
        }
        held.0 = now();
        let client = self.runpod().ok()?;
        held.1 = client.balance().await.ok();
        let _ = app.emit("balance", held.1);
        held.1
    }

    async fn set(&self, app: &AppHandle, status: Status) {
        let line = match status.phase.as_str() {
            "ready" if status.usd_per_hour > 0.0 => {
                format!("Running on RunPod · ${:.2}", status.usd)
            }
            "ready" => "Running on this PC".to_string(),
            "starting" | "booting" => "Starting…".to_string(),
            "error" => "Problem: open DiffLab".to_string(),
            _ => "Not running".to_string(),
        };
        if let Some(item) = self.tray_status.lock().unwrap().as_ref() {
            let _ = item.set_text(&line);
        }
        if let Some(tray) = app.tray_by_id("tray") {
            let _ = tray.set_tooltip(Some(format!("DiffLab: {line}")));
        }
        *self.status.lock().await = status.clone();
        let _ = app.emit("status", status);
    }

    async fn end(&self, app: &AppHandle, message: String) {
        if self.session.lock().await.is_none() && !message.is_empty() {
            // Already ended by someone else (Stop, or another tick): keep what they said.
            return;
        }
        for window in studio_windows(app) {
            let _ = window.destroy();
        }
        if let Some(follower) = self.follower.lock().unwrap().take() {
            follower.abort();
        }
        *self.boot.lock().unwrap() = None;
        *self.session.lock().await = None;
        *self.draining.lock().await = false;
        let _ = self.store.save_session(None);
        self.set(
            app,
            Status {
                phase: "ended".into(),
                message,
                ..Default::default()
            },
        )
        .await;
    }

    async fn heartbeat(&self, session: &Session, rate: f64) -> Option<Value> {
        let draining = *self.draining.lock().await;
        let mut body = json!({});
        if rate > 0.0 {
            body["usd_per_hour"] = json!(rate);
        }
        if draining {
            body["drain"] = json!(true);
        }
        if self.syncing.try_lock().is_err() {
            body["hold"] = json!(true);
        }
        let response = self
            .http
            .post(format!("{}/api/pod/heartbeat", session.base_url))
            .bearer_auth(&session.token)
            .json(&body)
            .timeout(Duration::from_secs(15))
            .send()
            .await
            .ok()?;
        if !response.status().is_success() {
            return None;
        }
        response.json().await.ok()
    }

    /// Why the pod ended, as its watchdog announced in the heartbeat a tick before deleting it.
    async fn ended_reason(&self, session: &Session) -> String {
        let usd = self.status.lock().await.usd;
        if session.ending.is_empty() {
            "The pod is gone: it shut down while the app was closed, or was deleted outside it."
                .into()
        } else {
            format!("Pod shut down: {}, ${usd:.2}.", session.ending)
        }
    }

    fn push_log(&self, app: &AppHandle, line: runpod::LogLine) {
        // The server's access log of the app's own heartbeat and the page's card poll, every few seconds.
        if line.source == "container" && NOISE.iter().any(|n| line.line.contains(n)) {
            return;
        }
        let mut logs = self.logs.lock().unwrap();
        // RunPod repeats a layer's "Downloading" several times a second.
        if line.source == "system"
            && logs
                .back()
                .is_some_and(|last| last.source == "system" && last.line == line.line)
        {
            return;
        }
        if logs.len() >= LOG_KEEP {
            logs.pop_front();
        }
        logs.push_back(line.clone());
        let _ = app.emit("log", line);
    }

    fn ensure_follower(&self, app: &AppHandle, session: &Session) {
        let mut held = self.follower.lock().unwrap();
        if held.as_ref().is_some_and(|f| !f.inner().is_finished()) {
            return;
        }
        let app = app.clone();
        let session = session.clone();
        *held = Some(tauri::async_runtime::spawn(async move {
            let inner = app.state::<Shared>().inner().clone();
            match session.kind {
                Kind::Runpod => follow_pod(&app, &inner, &session).await,
                Kind::Local => follow_container(&app, &inner).await,
            }
        }));
    }

    async fn tick(&self, app: &AppHandle) {
        let Some(mut session) = self.session.lock().await.clone() else {
            return;
        };
        let draining = *self.draining.lock().await;
        self.ensure_follower(app, &session);
        let mut status = Status {
            phase: "booting".into(),
            kind: Some(session.kind.clone()),
            gpu: session.gpu.clone(),
            usd_per_hour: session.usd_per_hour,
            uptime_s: now() - session.started_at,
            draining,
            ..Default::default()
        };
        match session.kind {
            Kind::Runpod => {
                let client = match self.runpod() {
                    Ok(c) => c,
                    Err(e) => {
                        return self
                            .set(
                                app,
                                Status {
                                    phase: "error".into(),
                                    message: e,
                                    ..status
                                },
                            )
                            .await
                    }
                };
                match client.pod(&session.pod_id).await {
                    // A pod just created may not be listed yet.
                    Ok(None) if status.uptime_s < 120 => {}
                    Ok(None) => {
                        let reason = self.ended_reason(&session).await;
                        return self.end(app, reason).await;
                    }
                    Ok(Some(pod)) if pod.desired_status == "TERMINATED" => {
                        let reason = self.ended_reason(&session).await;
                        return self.end(app, reason).await;
                    }
                    // Its container exited: the watchdog is gone with it, and the pod still holds its disk.
                    Ok(Some(pod)) if pod.desired_status == "EXITED" => {
                        let _ = client.delete_pod(&session.pod_id).await;
                        let reason = self.ended_reason(&session).await;
                        let reason = if reason.starts_with("Pod shut down") {
                            reason
                        } else {
                            "The server on the pod exited; the app deleted the pod.".into()
                        };
                        return self.end(app, reason).await;
                    }
                    Ok(Some(pod)) => {
                        if pod.cost_per_hr > 0.0
                            && (pod.cost_per_hr != session.usd_per_hour || session.gpu.is_empty())
                        {
                            session.usd_per_hour = pod.cost_per_hr;
                            session.gpu = match pod.gpu_name() {
                                // The REST pod has no gpu; GraphQL's machine does.
                                name if name.is_empty() => {
                                    client.gpu_name(&pod.id).await.unwrap_or_default()
                                }
                                name => name,
                            };
                            let _ = self.store.save_session(Some(&session));
                            *self.session.lock().await = Some(session.clone());
                            status.usd_per_hour = session.usd_per_hour;
                            status.gpu = session.gpu.clone();
                        }
                    }
                    Err(e) => status.message = e,
                }
            }
            Kind::Local => {
                if !local::running().await {
                    return self.end(app, "The local container stopped.".into()).await;
                }
            }
        }
        if let Some(answer) = self.heartbeat(&session, session.usd_per_hour).await {
            status.phase = "ready".into();
            status.usd = answer["usd"].as_f64().unwrap_or(0.0);
            let ending = answer["ending"].as_str().unwrap_or("").to_string();
            let due = now() - *self.synced_at.lock().unwrap() >= SYNC_EVERY;
            // The watchdog's announcement is the last tick the pod's files can be fetched in.
            if session.kind == Kind::Runpod
                && (due || (!ending.is_empty() && session.ending.is_empty()))
            {
                sync_soon(app, app.state::<Shared>().inner());
            }
            if ending != session.ending {
                session.ending = ending;
                let _ = self.store.save_session(Some(&session));
                *self.session.lock().await = Some(session.clone());
            }
            if !session.ending.is_empty() {
                status.message = format!("Shutting down in under a minute: {}.", session.ending);
            }
            *self.boot.lock().unwrap() = None;
        } else {
            if session.kind == Kind::Runpod {
                status.usd = status.uptime_s as f64 / 3600.0 * session.usd_per_hour;
            }
            if let Some(boot) = self.boot.lock().unwrap().as_ref() {
                status.layers = boot.layers.clone();
            }
            let pulling = session.kind == Kind::Runpod && !*self.container_started.lock().unwrap();
            let retrying = *self.retrying.lock().unwrap();
            if pulling && (retrying || status.uptime_s > SLOW_BOOT_S) {
                status.slow = true;
                status.message = format!(
                    "This machine is slow to download DiffLab ({} min{}). A new pod usually lands on another machine and is up in about 2 minutes.",
                    status.uptime_s / 60,
                    if retrying { ", its download keeps failing and retrying" } else { "" }
                );
            }
            if status.message.is_empty() {
                let boot = self.boot.lock().unwrap().clone();
                (status.message, status.progress) = match (boot, &session.kind) {
                    (Some(b), _) => {
                        status.layers = b.layers;
                        (b.message, b.progress)
                    }
                    (None, Kind::Runpod) => {
                        ("Waiting for RunPod to set up the machine…".into(), None)
                    }
                    (None, Kind::Local) => ("Starting the server (about a minute)…".into(), None),
                };
            }
        }
        // Stop may have ended this session while the tick waited on RunPod or the heartbeat; a late
        // write would put "booting" over "stopped" for good, as no later tick has a session to fix it.
        if self
            .session
            .lock()
            .await
            .as_ref()
            .map(|s| s.pod_id.as_str())
            != Some(session.pod_id.as_str())
        {
            return;
        }
        self.set(app, status).await;
    }
}

async fn follow_pod(app: &AppHandle, inner: &Inner, session: &Session) {
    let Ok(client) = inner.runpod() else { return };
    let mut tracker = local::PullTracker::new(
        local::layer_sizes().await.unwrap_or_default(),
        "onto the RunPod machine",
    );
    let mut cursor: Option<String> = None;
    loop {
        let from = cursor.clone();
        let _ = client
            .follow_logs(&session.pod_id, from.as_deref(), |id, line| {
                cursor = Some(id);
                if line.source == "system" {
                    if line.line.contains("Retrying in") {
                        *inner.retrying.lock().unwrap() = true;
                    }
                    if let Some(pulled) = tracker.feed(&line.line) {
                        *inner.boot.lock().unwrap() = Some(pulled);
                    }
                } else {
                    *inner.container_started.lock().unwrap() = true;
                    *inner.retrying.lock().unwrap() = false;
                    *inner.boot.lock().unwrap() = Some(local::Pulled {
                        message: "Starting the server…".into(),
                        progress: None,
                        layers: Vec::new(),
                    });
                }
                inner.push_log(app, line);
            })
            .await;
        tokio::time::sleep(Duration::from_secs(3)).await;
    }
}

async fn follow_container(app: &AppHandle, inner: &Inner) {
    use tokio::io::{AsyncBufReadExt, BufReader};
    let Ok(mut child) = local::follow_logs(300) else {
        return;
    };
    let (Some(out), Some(err)) = (child.stdout.take(), child.stderr.take()) else {
        return;
    };
    let (mut out, mut err) = (BufReader::new(out).lines(), BufReader::new(err).lines());
    let (mut out_open, mut err_open) = (true, true);
    while out_open || err_open {
        let line = tokio::select! {
            l = out.next_line(), if out_open => l.ok().flatten().or_else(|| { out_open = false; None }),
            l = err.next_line(), if err_open => l.ok().flatten().or_else(|| { err_open = false; None }),
        };
        if let Some(line) = line {
            let ts = chrono::Utc::now().to_rfc3339();
            inner.push_log(
                app,
                runpod::LogLine {
                    source: "container".into(),
                    line,
                    ts,
                },
            );
        }
    }
}

fn token() -> String {
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    hex::encode(bytes)
}

#[derive(Serialize)]
struct Loaded {
    prefs: Prefs,
    has_runpod_key: bool,
    has_local_results: bool,
    status: Status,
    results_dir: String,
    balance: Option<f64>,
    last_synced: String,
}

fn results_dir(app: &AppHandle, prefs: &Prefs) -> PathBuf {
    if !prefs.results_dir.is_empty() {
        return PathBuf::from(&prefs.results_dir);
    }
    app.path().home_dir().unwrap_or_default().join("DiffLab")
}

#[tauri::command]
async fn load(app: AppHandle, inner: State<'_, Shared>) -> Result<Loaded, String> {
    let prefs = inner.store.prefs();
    Ok(Loaded {
        has_runpod_key: inner.store.secret(RUNPOD_KEY).is_some(),
        has_local_results: local::has_results().await,
        status: inner.status.lock().await.clone(),
        results_dir: results_dir(&app, &prefs).display().to_string(),
        balance: inner.balance.lock().await.1,
        last_synced: inner.last_synced.lock().unwrap().clone(),
        prefs,
    })
}

/// A key typed into the wizard is checked before it is saved; with none, the saved one.
#[tauri::command]
async fn balance(
    app: AppHandle,
    inner: State<'_, Shared>,
    key: Option<String>,
) -> Result<Option<f64>, String> {
    match key.map(|k| k.trim().to_string()).filter(|k| !k.is_empty()) {
        Some(key) => runpod::Client::new(key).balance().await.map(Some),
        None => Ok(inner.refresh_balance(&app, true).await),
    }
}

/// What ghcr has as `latest`, and what this PC last pulled; a stale local image is pulled in the
/// background so the next local start does not wait for it.
async fn check_versions(app: &AppHandle, inner: &Shared) {
    let latest = local::latest_image().await;
    let here = local::local_image().await;
    let stale = matches!((&latest, &here), (Some(l), Some(h)) if l.digest != h.digest);
    {
        let mut held = inner.versions.lock().await;
        held.checked_at = now();
        held.image_latest = latest;
        held.image_local = here;
        if stale && !held.updating {
            held.updating = true;
            let (app, inner) = (app.clone(), inner.clone());
            tauri::async_runtime::spawn(async move {
                let (tx, _) = tokio::sync::mpsc::unbounded_channel();
                let _ = local::pull(tx).await;
                let mut held = inner.versions.lock().await;
                held.updating = false;
                held.image_local = local::local_image().await;
                let _ = app.emit("versions", held.clone());
            });
        }
        let _ = app.emit("versions", held.clone());
    }
}

#[derive(Serialize)]
struct VersionInfo {
    app: String,
    running_image: String,
    #[serde(flatten)]
    versions: Versions,
}

#[tauri::command]
async fn versions(
    app: AppHandle,
    inner: State<'_, Shared>,
    refresh: bool,
) -> Result<VersionInfo, String> {
    if refresh {
        check_versions(&app, inner.inner()).await;
    }
    Ok(VersionInfo {
        app: app.package_info().version.to_string(),
        running_image: inner
            .session
            .lock()
            .await
            .as_ref()
            .map(|s| s.image.clone())
            .unwrap_or_default(),
        versions: inner.versions.lock().await.clone(),
    })
}

#[tauri::command]
fn logs(inner: State<'_, Shared>) -> Vec<runpod::LogLine> {
    inner.logs.lock().unwrap().iter().cloned().collect()
}

#[tauri::command]
async fn hardware() -> Result<local::Hardware, String> {
    Ok(local::check().await)
}

#[tauri::command]
async fn save_settings(
    inner: State<'_, Shared>,
    prefs: Prefs,
    runpod_key: Option<String>,
) -> Result<(), String> {
    inner.store.save_prefs(&prefs)?;
    if let Some(key) = runpod_key
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty())
    {
        inner.store.set_secret(RUNPOD_KEY, &key)?;
    }
    Ok(())
}

#[tauri::command]
async fn start(app: AppHandle, inner: State<'_, Shared>, kind: Kind) -> Result<(), String> {
    if inner.session.lock().await.is_some() {
        return Err("difflab is already running".into());
    }
    inner.logs.lock().unwrap().clear();
    *inner.boot.lock().unwrap() = None;
    *inner.retrying.lock().unwrap() = false;
    *inner.container_started.lock().unwrap() = false;
    let starting = Status {
        phase: "starting".into(),
        kind: Some(kind.clone()),
        ..Default::default()
    };
    inner.set(&app, starting.clone()).await;
    let result = launch(&app, &inner, kind).await;
    match result {
        Ok(session) => {
            inner.store.save_session(Some(&session))?;
            *inner.session.lock().await = Some(session);
            inner.tick(&app).await;
            Ok(())
        }
        Err(e) => {
            inner
                .set(
                    &app,
                    Status {
                        phase: "error".into(),
                        message: e,
                        ..Default::default()
                    },
                )
                .await;
            // The status card already says it; an Err would show it a second time.
            Ok(())
        }
    }
}

async fn launch(app: &AppHandle, inner: &Inner, kind: Kind) -> Result<Session, String> {
    let token = token();
    let prefs = inner.store.prefs();
    match kind {
        Kind::Local => {
            let found = local::check().await;
            if !found.ok {
                return Err(found.reason);
            }
            inner
                .set(
                    app,
                    Status {
                        phase: "starting".into(),
                        kind: Some(Kind::Local),
                        message: "Checking for a newer DiffLab…".into(),
                        ..Default::default()
                    },
                )
                .await;
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
            let pulling = tokio::spawn(local::pull(tx));
            while let Some(update) = rx.recv().await {
                inner
                    .set(
                        app,
                        Status {
                            phase: "starting".into(),
                            kind: Some(Kind::Local),
                            message: update.message,
                            progress: update.progress,
                            layers: update.layers,
                            ..Default::default()
                        },
                    )
                    .await;
            }
            pulling.await.map_err(|e| e.to_string())??;
            inner
                .set(
                    app,
                    Status {
                        phase: "starting".into(),
                        kind: Some(Kind::Local),
                        message: "Starting the container…".into(),
                        ..Default::default()
                    },
                )
                .await;
            let port = local::start(&token, &found.runtime).await?;
            Ok(Session {
                kind,
                pod_id: local::CONTAINER.into(),
                token,
                base_url: local::url(port),
                started_at: now(),
                usd_per_hour: 0.0,
                gpu: found.gpu,
                image: local::local_image()
                    .await
                    .map(|v| v.revision)
                    .unwrap_or_default(),
                ending: String::new(),
            })
        }
        Kind::Runpod => {
            let client = inner.runpod()?;
            let key = inner.store.secret(RUNPOD_KEY).unwrap_or_default();
            let volume = client.global_volume().await?;
            let mut env = vec![
                ("DIFFLAB_TOKEN", token.clone()),
                ("DIFFLAB_RUNPOD_API_KEY", key),
                ("DIFFLAB_IDLE_MINUTES", prefs.idle_minutes.to_string()),
                ("DIFFLAB_STUCK_MINUTES", prefs.stuck_minutes.to_string()),
                ("DIFFLAB_SPEND_LIMIT_USD", prefs.spend_limit_usd.to_string()),
                (
                    "DIFFLAB_EXIT_WHEN_DONE",
                    if prefs.exit_when_done { "1" } else { "" }.to_string(),
                ),
            ];
            env.extend(runpod::MODELS.map(|(key, value)| (key, value.to_string())));
            let latest = local::latest_image().await;
            let image = match &latest {
                Some(v) => format!("{}@{}", runpod::IMAGE.trim_end_matches(":latest"), v.digest),
                None => runpod::IMAGE.to_string(),
            };
            let pod = client.create_pod(&volume, &image, &env).await?;
            Ok(Session {
                kind,
                base_url: runpod::proxy_url(&pod.id),
                gpu: pod.gpu_name(),
                image: latest.map(|v| v.revision).unwrap_or_default(),
                usd_per_hour: pod.cost_per_hr,
                pod_id: pod.id,
                token,
                started_at: now(),
                ending: String::new(),
            })
        }
    }
}

#[tauri::command]
async fn stop(app: AppHandle, inner: State<'_, Shared>, immediately: bool) -> Result<(), String> {
    stop_now(&app, inner.inner(), immediately).await
}

async fn stop_now(app: &AppHandle, inner: &Shared, immediately: bool) -> Result<(), String> {
    let app = app.clone();
    let Some(session) = inner.session.lock().await.clone() else {
        return Ok(());
    };
    if !immediately && session.kind == Kind::Runpod {
        *inner.draining.lock().await = true;
        inner.tick(&app).await;
        return Ok(());
    }
    let mut note = String::new();
    if session.kind == Kind::Runpod
        && inner.status.lock().await.phase == "ready"
        && inner.store.prefs().auto_download
    {
        inner
            .set(
                &app,
                Status {
                    message: "Downloading results before stopping…".into(),
                    ..inner.status.lock().await.clone()
                },
            )
            .await;
        match tokio::time::timeout(STOP_SYNC_LIMIT, sync_pod(&app, inner, false)).await {
            Ok(Ok(_)) => {}
            Ok(Err(problem)) => note = format!(" Results could not be downloaded first: {problem}"),
            Err(_) => {
                note =
                    " Some results were still downloading; the next session downloads them.".into()
            }
        }
    }
    match session.kind {
        Kind::Runpod => inner.runpod()?.delete_pod(&session.pod_id).await?,
        Kind::Local => local::stop().await?,
    }
    let usd = inner.status.lock().await.usd;
    inner
        .end(&app, format!("Stopped by you, ${usd:.2}.{note}"))
        .await;
    Ok(())
}

/// Deletes a pod whose host is failing to pull the image and asks RunPod for another.
#[tauri::command]
async fn retry_machine(app: AppHandle, inner: State<'_, Shared>) -> Result<(), String> {
    if let Some(session) = inner
        .session
        .lock()
        .await
        .clone()
        .filter(|s| s.kind == Kind::Runpod)
    {
        inner.runpod()?.delete_pod(&session.pod_id).await?;
    }
    inner.end(&app, String::new()).await;
    start(app, inner, Kind::Runpod).await
}

#[tauri::command]
async fn open_ui(app: AppHandle, inner: State<'_, Shared>) -> Result<(), String> {
    open_studio(&app, inner.inner()).await
}

async fn open_studio(app: &AppHandle, inner: &Shared) -> Result<(), String> {
    let app = app.clone();
    let session = inner
        .session
        .lock()
        .await
        .clone()
        .ok_or("difflab is not running")?;
    // A studio window closed by its title bar can linger under its label without being on screen,
    // so only a visible one is reused; the rest are dropped and the new one takes a label of its own.
    for window in studio_windows(&app) {
        if window.is_visible().unwrap_or(false) {
            bring_forward(&window);
            return Ok(());
        }
        let _ = window.destroy();
    }
    let url = format!("{}/auth?token={}", session.base_url, session.token);
    let dir = results_dir(&app, &inner.store.prefs());
    let notifier = app.clone();
    let base = session.base_url.clone();
    let opener = app.clone();
    WebviewWindowBuilder::new(
        &app,
        format!("ui-{}", chrono::Utc::now().timestamp_millis()),
        WebviewUrl::External(url.parse().map_err(|e| format!("{e}"))?),
    )
    .title("DiffLab")
    .inner_size(1440.0, 900.0)
    // Tauri takes file drops for itself by default, and the page's own drop zones never see one.
    .disable_drag_drop_handler()
    // A web view drops downloads unless the app says where they go; a save dialog here would block
    // the window's own event loop, so they go to the results folder and a notification says so.
    .on_download(move |_, event| {
        match event {
            DownloadEvent::Requested { url, destination } => {
                let name = destination
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .or_else(|| {
                        url.path_segments()
                            .and_then(|mut s| s.next_back())
                            .map(str::to_string)
                    })
                    .filter(|n| !n.is_empty())
                    .unwrap_or_else(|| "download".into());
                *destination = free_name(&dir, &name);
            }
            DownloadEvent::Finished { path, success, .. } => {
                let shown = path
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| dir.display().to_string());
                let (title, body) = if success {
                    ("Downloaded", shown)
                } else {
                    ("Download failed", shown)
                };
                let _ = notifier
                    .notification()
                    .builder()
                    .title(title)
                    .body(body)
                    .show();
            }
            _ => {}
        }
        true
    })
    .on_navigation(move |to| {
        if to.as_str().starts_with(&base)
            || to.scheme() == "blob"
            || to.scheme() == "data"
            || to.scheme() == "about"
        {
            return true;
        }
        let _ = open_external(&opener, to.as_str());
        false
    })
    .build()
    .map(|_| ())
    .map_err(|e| e.to_string())
}

#[derive(Serialize, Clone)]
struct SyncProgress {
    done: usize,
    total: usize,
    file: String,
}

/// Copies what the running pod made and this PC lacks into the results folder's RunPod/. One at a time:
/// the minute timer, the shutdown announcement and Stop can all ask at once.
async fn sync_pod(app: &AppHandle, inner: &Shared, report: bool) -> Result<usize, String> {
    let _one = inner.syncing.lock().await;
    let Some(session) = inner
        .session
        .lock()
        .await
        .clone()
        .filter(|s| s.kind == Kind::Runpod)
    else {
        return Ok(0);
    };
    let dir = results_dir(app, &inner.store.prefs()).join("RunPod");
    let listed: Vec<Value> = inner
        .http
        .get(format!("{}/api/results", session.base_url))
        .bearer_auth(&session.token)
        .timeout(Duration::from_secs(30))
        .send()
        .await
        .map_err(|e| e.to_string())?
        .json()
        .await
        .map_err(|e| format!("the pod's results list: {e}"))?;
    let wanted: Vec<(String, PathBuf)> = listed
        .iter()
        .filter_map(|r| {
            let (key, size) = (r["path"].as_str()?, r["size"].as_u64()?);
            let to = dir.join(key);
            (std::fs::metadata(&to).map(|m| m.len()).ok() != Some(size))
                .then(|| (key.to_string(), to))
        })
        .collect();
    let total = wanted.len();
    for (done, (key, to)) in wanted.iter().enumerate() {
        if report {
            let _ = app.emit(
                "sync",
                SyncProgress {
                    done,
                    total,
                    file: key.clone(),
                },
            );
        }
        download(
            &inner.http,
            &format!("{}/output/{key}", session.base_url),
            &session.token,
            to,
        )
        .await?;
    }
    if total > 0 {
        *inner.last_synced.lock().unwrap() = format!(
            "{total} new from RunPod at {}",
            chrono::Local::now().format("%H:%M")
        );
        let _ = app.emit("synced", inner.last_synced.lock().unwrap().clone());
        if !report {
            let body = format!(
                "{total} new file{} in {}",
                if total == 1 { "" } else { "s" },
                dir.display()
            );
            let _ = app
                .notification()
                .builder()
                .title("DiffLab results downloaded")
                .body(body)
                .show();
        }
    }
    Ok(total)
}

/// In the background, so the tick's heartbeat never waits on a download.
fn sync_soon(app: &AppHandle, inner: &Shared) {
    if !inner.store.prefs().auto_download {
        return;
    }
    *inner.synced_at.lock().unwrap() = now();
    let (app, inner) = (app.clone(), inner.clone());
    tauri::async_runtime::spawn(async move {
        if let Err(problem) = sync_pod(&app, &inner, false).await {
            *inner.last_synced.lock().unwrap() = format!("Automatic download failed: {problem}");
            let _ = app.emit("synced", inner.last_synced.lock().unwrap().clone());
        }
    });
}

#[tauri::command]
async fn sync_results(app: AppHandle, inner: State<'_, Shared>) -> Result<String, String> {
    let dir = results_dir(&app, &inner.store.prefs());
    // Kept apart from RunPod/, so the two can never overwrite each other's file of the same name.
    let here = dir.join("This PC");
    let missing = |base: &std::path::Path, key: &str, size: u64| {
        let to = base.join(key);
        let have = std::fs::metadata(&to).map(|m| m.len()).ok();
        (have != Some(size)).then_some((key.to_string(), to))
    };
    let progress = |done: usize, total: usize, file: &str| {
        let _ = app.emit(
            "sync",
            SyncProgress {
                done,
                total,
                file: file.to_string(),
            },
        );
    };
    let mut copied = 0;
    if local::has_results().await {
        let wanted: Vec<_> = local::list_results(&RESULT_PREFIXES)
            .await?
            .into_iter()
            .filter_map(|(key, size)| missing(&here, &key, size))
            .collect();
        let total = wanted.len();
        local::copy_results(&wanted, |done, key| progress(done, total, key)).await?;
        copied += total;
    }
    // A global volume has no S3 API (checked: its id is refused like a bucket that does not exist), so
    // RunPod's results come from the pod while it is up.
    let pod_up = inner.status.lock().await.phase == "ready"
        && inner
            .session
            .lock()
            .await
            .as_ref()
            .is_some_and(|s| s.kind == Kind::Runpod);
    if pod_up {
        copied += sync_pod(&app, inner.inner(), true).await?;
    } else if !local::has_results().await {
        return Err(
            "Nothing to download yet. RunPod results download while a pod is running.".into(),
        );
    }
    progress(copied, copied, "");
    Ok(if copied == 0 {
        "Everything is already downloaded.".into()
    } else {
        format!("Downloaded {copied} files.")
    })
}

#[tauri::command]
async fn open_results(app: AppHandle, inner: State<'_, Shared>) -> Result<(), String> {
    let dir = results_dir(&app, &inner.store.prefs());
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|e| e.to_string())?;
    #[cfg(target_os = "linux")]
    {
        let _ = app;
        local::command("xdg-open")
            .arg(&dir)
            .spawn()
            .map(|_| ())
            .map_err(|e| format!("could not open {}: {e}", dir.display()))
    }
    #[cfg(not(target_os = "linux"))]
    app.opener()
        .open_path(dir.display().to_string(), None::<&str>)
        .map_err(|e| e.to_string())
}

async fn download(
    http: &reqwest::Client,
    url: &str,
    token: &str,
    to: &std::path::Path,
) -> Result<(), String> {
    use futures_util::StreamExt;
    use tokio::io::AsyncWriteExt;
    let response = http
        .get(url)
        .bearer_auth(token)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !response.status().is_success() {
        return Err(format!("{url}: {}", response.status()));
    }
    if let Some(parent) = to.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| e.to_string())?;
    }
    let partial = to.with_file_name(format!(
        "{}.part",
        to.file_name().unwrap_or_default().to_string_lossy()
    ));
    let mut file = tokio::fs::File::create(&partial)
        .await
        .map_err(|e| e.to_string())?;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = tokio::time::timeout(STALL, stream.next())
        .await
        .map_err(|_| format!("{url}: no data for {} s", STALL.as_secs()))?
    {
        file.write_all(&chunk.map_err(|e| e.to_string())?)
            .await
            .map_err(|e| e.to_string())?;
    }
    file.flush().await.map_err(|e| e.to_string())?;
    tokio::fs::rename(&partial, to)
        .await
        .map_err(|e| e.to_string())
}

/// `name` in `dir`, or "name (2).ext" and so on when that is taken.
fn free_name(dir: &std::path::Path, name: &str) -> PathBuf {
    let _ = std::fs::create_dir_all(dir);
    let path = std::path::Path::new(name);
    let (stem, ext) = (
        path.file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default(),
        path.extension()
            .map(|e| format!(".{}", e.to_string_lossy()))
            .unwrap_or_default(),
    );
    let mut candidate = dir.join(name);
    let mut n = 2;
    while candidate.exists() {
        candidate = dir.join(format!("{stem} ({n}){ext}"));
        n += 1;
    }
    candidate
}

fn open_external(app: &AppHandle, url: &str) -> Result<(), String> {
    #[cfg(target_os = "linux")]
    {
        let _ = app;
        local::std_command("xdg-open")
            .arg(url)
            .spawn()
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
    #[cfg(not(target_os = "linux"))]
    app.opener()
        .open_url(url, None::<&str>)
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn open_link(app: AppHandle, url: String) -> Result<(), String> {
    if !url.starts_with("https://") {
        return Err("only https links open".into());
    }
    #[cfg(target_os = "linux")]
    {
        let _ = app;
        local::command("xdg-open")
            .arg(&url)
            .spawn()
            .map(|_| ())
            .map_err(|e| format!("could not open {url}: {e}"))
    }
    #[cfg(not(target_os = "linux"))]
    app.opener()
        .open_url(url, None::<&str>)
        .map_err(|e| e.to_string())
}

fn studio_windows(app: &AppHandle) -> Vec<tauri::WebviewWindow> {
    app.webview_windows()
        .into_iter()
        .filter(|(label, _)| label.starts_with("ui"))
        .map(|(_, window)| window)
        .collect()
}

fn show_main(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        bring_forward(&window);
    }
}

/// KDE on Wayland refuses focus to a window that asks without an activation token, which the tray
/// does not hand over, so a window already open stays behind. Mapping it again puts it in front.
fn bring_forward(window: &tauri::WebviewWindow) {
    if window.is_visible().unwrap_or(false) {
        let _ = window.hide();
    }
    let _ = window.unminimize();
    let _ = window.show();
    let _ = window.set_focus();
}

/// Like JetBrains Toolbox: closing the window leaves DiffLab in the tray, where a running pod keeps
/// its heartbeat; Quit is the menu's.
fn tray(app: &tauri::App, inner: &Shared) -> tauri::Result<()> {
    use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
    use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
    let status = MenuItem::with_id(app, "status", "Not running", false, None::<&str>)?;
    let show = MenuItem::with_id(app, "show", "Show DiffLab", true, None::<&str>)?;
    let studio = MenuItem::with_id(app, "studio", "Open studio", true, None::<&str>)?;
    let stop = MenuItem::with_id(app, "stop", "Stop now", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
    let menu = Menu::with_items(
        app,
        &[
            &status,
            &PredefinedMenuItem::separator(app)?,
            &show,
            &studio,
            &stop,
            &PredefinedMenuItem::separator(app)?,
            &quit,
        ],
    )?;
    *inner.tray_status.lock().unwrap() = Some(status);
    let mut builder = TrayIconBuilder::with_id("tray")
        .tooltip("DiffLab")
        .menu(&menu)
        .show_menu_on_left_click(false);
    if let Some(icon) = app.default_window_icon() {
        builder = builder.icon(icon.clone());
    }
    builder
        .on_menu_event(|app, event| {
            let app = app.clone();
            match event.id.as_ref() {
                "show" => show_main(&app),
                // Nothing on screen says these failed otherwise: the window may be hidden.
                "studio" => {
                    tauri::async_runtime::spawn(async move {
                        let inner = app.state::<Shared>().inner().clone();
                        if let Err(problem) = open_studio(&app, &inner).await {
                            let _ = app
                                .notification()
                                .builder()
                                .title("Could not open the studio")
                                .body(problem)
                                .show();
                        }
                    });
                }
                "stop" => {
                    tauri::async_runtime::spawn(async move {
                        let inner = app.state::<Shared>().inner().clone();
                        let _ = app
                            .notification()
                            .builder()
                            .title("Stopping DiffLab")
                            .body("Downloading results first, then stopping.")
                            .show();
                        match stop_now(&app, &inner, true).await {
                            Ok(()) => {
                                let said = inner.status.lock().await.message.clone();
                                let _ = app
                                    .notification()
                                    .builder()
                                    .title("DiffLab stopped")
                                    .body(said)
                                    .show();
                            }
                            Err(problem) => {
                                let _ = app
                                    .notification()
                                    .builder()
                                    .title("Could not stop DiffLab")
                                    .body(problem)
                                    .show();
                            }
                        }
                    });
                }
                "quit" => {
                    // The page installs a downloaded update first, then exits; this is the fallback.
                    let _ = app.emit("quit-requested", ());
                    tauri::async_runtime::spawn(async move {
                        tokio::time::sleep(Duration::from_secs(20)).await;
                        app.exit(0);
                    });
                }
                _ => {}
            }
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                show_main(tray.app_handle());
            }
        })
        .build(app)?;
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_notification::init())
        .setup(|app| {
            #[cfg(target_os = "linux")]
            local::integrate();
            let dir = app.path().app_config_dir()?;
            let store = Store::new(dir);
            let session = store.session();
            let inner: Shared = Arc::new(Inner {
                balance: Mutex::new((0, None)),
                status: Mutex::new(Status {
                    phase: if session.is_some() { "booting" } else { "none" }.into(),
                    ..Default::default()
                }),
                session: Mutex::new(session),
                draining: Mutex::new(false),
                versions: Mutex::new(Versions::default()),
                logs: std::sync::Mutex::new(Default::default()),
                boot: std::sync::Mutex::new(None),
                retrying: std::sync::Mutex::new(false),
                container_started: std::sync::Mutex::new(false),
                follower: std::sync::Mutex::new(None),
                tray_status: std::sync::Mutex::new(None),
                syncing: Mutex::new(()),
                synced_at: std::sync::Mutex::new(0),
                last_synced: std::sync::Mutex::new(String::new()),
                store,
                http: reqwest::Client::new(),
            });
            app.manage(inner.clone());
            tray(app, &inner)?;
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                loop {
                    inner.refresh_balance(&handle, false).await;
                    if now() - inner.versions.lock().await.checked_at > VERSIONS_EVERY {
                        check_versions(&handle, &inner).await;
                    }
                    inner.tick(&handle).await;
                    tokio::time::sleep(HEARTBEAT).await;
                }
            });
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                if window.label() == "main" {
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            load,
            balance,
            versions,
            logs,
            hardware,
            save_settings,
            start,
            stop,
            retry_machine,
            open_ui,
            sync_results,
            open_results,
            open_link
        ])
        .run(tauri::generate_context!())
        .expect("error while running DiffLab");
}
