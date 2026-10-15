mod local;
mod runpod;
mod s3;
mod store;

use rand::RngCore;
use serde::Serialize;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager, State, WebviewUrl, WebviewWindowBuilder};
#[cfg(not(target_os = "linux"))]
use tauri_plugin_opener::OpenerExt;
use tokio::sync::Mutex;

use store::{Kind, Prefs, Session, Store, RUNPOD_KEY, S3_ACCESS, S3_SECRET};

// Also the status refresh; the pod's watchdog counts the app as gone after DIFFLAB_IDLE_MINUTES without one.
const HEARTBEAT: Duration = Duration::from_secs(10);
const BALANCE_EVERY: i64 = 60;
const VERSIONS_EVERY: i64 = 6 * 3600;
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

    async fn volume(&self) -> Option<s3::Volume> {
        let prefs = self.store.prefs();
        let id = Some(prefs.volume_id).filter(|v| !v.is_empty())?;
        let secret = self.store.secret(S3_SECRET)?;
        let access = match self.store.secret(S3_ACCESS) {
            Some(access) => access,
            None => {
                let access = self.runpod().ok()?.user_id().await.ok()?;
                let _ = self.store.set_secret(S3_ACCESS, &access);
                access
            }
        };
        Some(s3::Volume {
            datacenter: prefs.datacenter,
            id,
            access,
            secret,
        })
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
        *self.status.lock().await = status.clone();
        let _ = app.emit("status", status);
    }

    async fn end(&self, app: &AppHandle, message: String) {
        if let Some(window) = app.get_webview_window("ui") {
            let _ = window.close();
        }
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

    /// Why the pod ended, from what its watchdog wrote on the volume.
    async fn ended_reason(&self, pod_id: &str) -> String {
        let Some(volume) = self.volume().await else {
            return "The pod is gone.".into();
        };
        match volume.get("pod-ended.json").await {
            Ok(Some(bytes)) => match serde_json::from_slice::<Value>(&bytes) {
                Ok(v) if v["pod"].as_str() == Some(pod_id) => format!(
                    "Pod shut down: {}, ${:.2}.",
                    v["reason"].as_str().unwrap_or("no reason recorded"),
                    v["usd"].as_f64().unwrap_or(0.0)
                ),
                _ => "The pod was deleted outside the app.".into(),
            },
            _ => "The pod is gone.".into(),
        }
    }

    async fn tick(&self, app: &AppHandle) {
        let Some(mut session) = self.session.lock().await.clone() else {
            return;
        };
        let draining = *self.draining.lock().await;
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
                        let reason = self.ended_reason(&session.pod_id).await;
                        return self.end(app, reason).await;
                    }
                    Ok(Some(pod)) if pod.desired_status == "TERMINATED" => {
                        let reason = self.ended_reason(&session.pod_id).await;
                        return self.end(app, reason).await;
                    }
                    // Its container exited: the watchdog is gone with it, and the pod still holds its disk.
                    Ok(Some(pod)) if pod.desired_status == "EXITED" => {
                        let _ = client.delete_pod(&session.pod_id).await;
                        let reason = self.ended_reason(&session.pod_id).await;
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
                            session.gpu = pod.gpu_name();
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
        } else if session.kind == Kind::Local {
            status.message = "Starting the server (about a minute)…".into();
        } else if session.kind == Kind::Runpod {
            status.usd = status.uptime_s as f64 / 3600.0 * session.usd_per_hour;
            if status.message.is_empty() {
                status.message = "Starting: pulling the image and booting the server (a few minutes on a new pod).".into();
            }
        }
        self.set(app, status).await;
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
    has_s3: bool,
    has_local_results: bool,
    datacenters: Vec<&'static str>,
    status: Status,
    results_dir: String,
    balance: Option<f64>,
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
        has_s3: inner.store.secret(S3_SECRET).is_some(),
        has_local_results: local::has_results().await,
        datacenters: runpod::DATACENTERS.to_vec(),
        status: inner.status.lock().await.clone(),
        results_dir: results_dir(&app, &prefs).display().to_string(),
        balance: inner.balance.lock().await.1,
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
async fn hardware() -> Result<local::Hardware, String> {
    Ok(local::check().await)
}

#[tauri::command]
async fn save_settings(
    inner: State<'_, Shared>,
    prefs: Prefs,
    runpod_key: Option<String>,
    s3_secret: Option<String>,
) -> Result<(), String> {
    let mut prefs = prefs;
    let old = inner.store.prefs();
    if old.datacenter != prefs.datacenter {
        prefs.volume_id = String::new();
    }
    inner.store.save_prefs(&prefs)?;
    for (name, value) in [(RUNPOD_KEY, runpod_key.clone()), (S3_SECRET, s3_secret)] {
        if let Some(v) = value
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
        {
            inner.store.set_secret(name, &v)?;
        }
    }
    if runpod_key.is_some_and(|k| !k.trim().is_empty()) {
        if let Ok(id) = inner.runpod()?.user_id().await {
            inner.store.set_secret(S3_ACCESS, &id)?;
        }
    }
    Ok(())
}

#[tauri::command]
async fn start(app: AppHandle, inner: State<'_, Shared>, kind: Kind) -> Result<(), String> {
    if inner.session.lock().await.is_some() {
        return Err("difflab is already running".into());
    }
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
            })
        }
        Kind::Runpod => {
            let client = inner.runpod()?;
            let key = inner.store.secret(RUNPOD_KEY).unwrap_or_default();
            let volume = client.volume(&prefs.datacenter).await?;
            let mut prefs = prefs;
            prefs.volume_id = volume.id.clone();
            inner.store.save_prefs(&prefs)?;
            let env = json!({
                "DIFFLAB_TOKEN": token,
                "DIFFLAB_RUNPOD_API_KEY": key,
                "DIFFLAB_IDLE_MINUTES": prefs.idle_minutes.to_string(),
                "DIFFLAB_STUCK_MINUTES": prefs.stuck_minutes.to_string(),
                "DIFFLAB_SPEND_LIMIT_USD": prefs.spend_limit_usd.to_string(),
                "DIFFLAB_EXIT_WHEN_DONE": if prefs.exit_when_done { "1" } else { "" },
            });
            let latest = local::latest_image().await;
            let image = match &latest {
                Some(v) => format!("{}@{}", runpod::IMAGE.trim_end_matches(":latest"), v.digest),
                None => runpod::IMAGE.to_string(),
            };
            let pod = client.create_pod(&volume.id, &image, env).await?;
            Ok(Session {
                kind,
                base_url: runpod::proxy_url(&pod.id),
                gpu: pod.gpu_name(),
                image: latest.map(|v| v.revision).unwrap_or_default(),
                usd_per_hour: pod.cost_per_hr,
                pod_id: pod.id,
                token,
                started_at: now(),
            })
        }
    }
}

#[tauri::command]
async fn stop(app: AppHandle, inner: State<'_, Shared>, immediately: bool) -> Result<(), String> {
    let Some(session) = inner.session.lock().await.clone() else {
        return Ok(());
    };
    if !immediately && session.kind == Kind::Runpod {
        *inner.draining.lock().await = true;
        inner.tick(&app).await;
        return Ok(());
    }
    match session.kind {
        Kind::Runpod => inner.runpod()?.delete_pod(&session.pod_id).await?,
        Kind::Local => local::stop().await?,
    }
    let usd = inner.status.lock().await.usd;
    inner.end(&app, format!("Stopped by you, ${usd:.2}.")).await;
    Ok(())
}

#[tauri::command]
async fn open_ui(app: AppHandle, inner: State<'_, Shared>) -> Result<(), String> {
    let session = inner
        .session
        .lock()
        .await
        .clone()
        .ok_or("difflab is not running")?;
    if let Some(window) = app.get_webview_window("ui") {
        return window.set_focus().map_err(|e| e.to_string());
    }
    let url = format!("{}/auth?token={}", session.base_url, session.token);
    WebviewWindowBuilder::new(
        &app,
        "ui",
        WebviewUrl::External(url.parse().map_err(|e| format!("{e}"))?),
    )
    .title("DiffLab")
    .inner_size(1440.0, 900.0)
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

#[tauri::command]
async fn sync_results(app: AppHandle, inner: State<'_, Shared>) -> Result<String, String> {
    let dir = results_dir(&app, &inner.store.prefs());
    let missing = |key: &str, size: u64| {
        let to = dir.join(key);
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
            .filter_map(|(key, size)| missing(&key, size))
            .collect();
        let total = wanted.len();
        local::copy_results(&wanted, |done, key| progress(done, total, key)).await?;
        copied += total;
    }
    if let Some(volume) = inner.volume().await {
        let mut wanted = Vec::new();
        for prefix in RESULT_PREFIXES {
            for object in volume.list(prefix).await? {
                if !object.key.ends_with('/') {
                    wanted.extend(missing(&object.key, object.size));
                }
            }
        }
        let total = wanted.len();
        for (done, (key, to)) in wanted.into_iter().enumerate() {
            progress(done, total, &key);
            volume.download(&key, &to).await?;
        }
        copied += total;
    } else if !local::has_results().await {
        return Err(
            "Nothing to download yet: run DiffLab here first, or add your S3 key for RunPod."
                .into(),
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
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
                store,
                http: reqwest::Client::new(),
            });
            app.manage(inner.clone());
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
        .invoke_handler(tauri::generate_handler![
            load,
            balance,
            versions,
            hardware,
            save_settings,
            start,
            stop,
            open_ui,
            sync_results,
            open_results,
            open_link
        ])
        .run(tauri::generate_context!())
        .expect("error while running DiffLab");
}
