use serde::Serialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;

use crate::runpod::IMAGE;

pub const CONTAINER: &str = "difflab-desktop";
pub const VOLUME: &str = "difflab";
// The pipeline offloads for a 24 GB card and assumes a 58 GB host (`core/models.py`); 56 lets a
// "64 GB" machine through after what firmware and the iGPU reserve.
const MIN_VRAM_MIB: u64 = 23_000;
const MIN_RAM_GB: f64 = 56.0;
// Turing is dropped from the image's kernel builds (Dockerfile COMFY_CUDA_ARCHS).
const MIN_COMPUTE: f64 = 8.0;

#[derive(Serialize, Clone, Default)]
pub struct Hardware {
    pub ok: bool,
    pub reason: String,
    pub gpu: String,
    pub vram_gb: f64,
    pub ram_gb: f64,
    pub runtime: String,
}

// Where nvidia-smi lives besides PATH: WSL, and the host's /usr from inside a toolbox or Flatpak.
const SMI: [&str; 4] = [
    "nvidia-smi",
    "/usr/bin/nvidia-smi",
    "/usr/lib/wsl/lib/nvidia-smi",
    "/run/host/usr/bin/nvidia-smi",
];

// What an AppImage's AppRun points at its own bundle: inherited by docker, nvidia-smi or xdg-open,
// they load the bundle's libraries and Qt plugins instead of the system's (xdg-open then fails silently).
const APPIMAGE_VARS: [&str; 11] = [
    "LD_LIBRARY_PATH",
    "QT_PLUGIN_PATH",
    "PYTHONHOME",
    "PYTHONPATH",
    "PERLLIB",
    "GDK_PIXBUF_MODULE_FILE",
    "GSETTINGS_SCHEMA_DIR",
    "GI_TYPELIB_PATH",
    "GTK_PATH",
    "GIO_MODULE_DIR",
    "GTK_EXE_PREFIX",
];

pub fn command(program: &str) -> Command {
    let mut command = Command::new(program);
    if let Some(dir) = std::env::var_os("APPDIR").filter(|_| std::env::var_os("APPIMAGE").is_some())
    {
        let dir = dir.to_string_lossy().trim_end_matches('/').to_string();
        for name in APPIMAGE_VARS {
            command.env_remove(name);
        }
        for name in ["PATH", "XDG_DATA_DIRS"] {
            if let Ok(value) = std::env::var(name) {
                let kept: Vec<&str> = value
                    .split(':')
                    .filter(|p| !p.is_empty() && !p.starts_with(&dir))
                    .collect();
                command.env(name, kept.join(":"));
            }
        }
    }
    command
}

async fn run(program: &str, args: &[&str]) -> Result<String, String> {
    let out = command(program)
        .args(args)
        .output()
        .await
        .map_err(|e| format!("{program}: {e}"))?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        Err(format!(
            "{program}: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ))
    }
}

async fn output(program: &str, args: &[&str]) -> Option<String> {
    run(program, args).await.ok()
}

pub async fn check() -> Hardware {
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    let mut found = Hardware {
        ram_gb: system.total_memory() as f64 / 1e9,
        ..Default::default()
    };
    let mut smi = Err(String::new());
    for program in SMI {
        smi = run(
            program,
            &[
                "--query-gpu=name,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
        )
        .await;
        if smi.is_ok() {
            break;
        }
    }
    let line = match &smi {
        Ok(text) => text.lines().next().unwrap_or("").to_string(),
        Err(problem) => {
            found.reason = format!("no NVIDIA GPU found ({problem})");
            return found;
        }
    };
    let fields: Vec<&str> = line.split(',').map(str::trim).collect();
    found.gpu = fields.first().unwrap_or(&"").to_string();
    let vram_mib: u64 = fields.get(1).and_then(|v| v.parse().ok()).unwrap_or(0);
    let compute: f64 = fields.get(2).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    found.vram_gb = vram_mib as f64 / 1024.0;
    found.runtime = runtime().await.unwrap_or_default();
    found.reason = if vram_mib < MIN_VRAM_MIB {
        format!(
            "{} has {:.0} GB of VRAM; difflab needs 24 GB",
            found.gpu, found.vram_gb
        )
    } else if compute < MIN_COMPUTE {
        format!(
            "{} is older than the image's kernels support (Ampere or newer)",
            found.gpu
        )
    } else if found.ram_gb < MIN_RAM_GB {
        format!("{:.0} GB of RAM; difflab needs 64 GB", found.ram_gb)
    } else if found.runtime.is_empty() {
        "Docker is not installed or not running".into()
    } else {
        String::new()
    };
    found.ok = found.reason.is_empty();
    found
}

async fn runtime() -> Option<String> {
    let version = output("docker", &["--version"]).await?;
    output("docker", &["info", "--format", "{{.ServerVersion}}"]).await?;
    Some(if version.to_lowercase().contains("podman") {
        "podman".into()
    } else {
        "docker".into()
    })
}

const REGISTRY: &str = "https://ghcr.io";
const REPOSITORY: &str = "diverofdark/difflab";
const ACCEPT: &str = "application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, \
                      application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json";

struct Registry {
    http: reqwest::Client,
    token: String,
}

impl Registry {
    async fn open() -> Option<Self> {
        let http = reqwest::Client::new();
        let token: Value = http
            .get(format!(
                "{REGISTRY}/token?scope=repository:{REPOSITORY}:pull"
            ))
            .send()
            .await
            .ok()?
            .json()
            .await
            .ok()?;
        let token = token["token"].as_str()?.to_string();
        Some(Registry { http, token })
    }

    async fn get(&self, path: &str, accept: &str) -> Option<reqwest::Response> {
        let response = self
            .http
            .get(format!("{REGISTRY}/v2/{REPOSITORY}/{path}"))
            .bearer_auth(&self.token)
            .header("Accept", accept)
            .send()
            .await
            .ok()?;
        response.status().is_success().then_some(response)
    }

    /// The digest `latest` points at (what `docker pull` records), and the linux/amd64 manifest under it.
    async fn latest(&self) -> Option<(String, Value)> {
        let response = self.get("manifests/latest", ACCEPT).await?;
        let digest = response
            .headers()
            .get("docker-content-digest")?
            .to_str()
            .ok()?
            .to_string();
        let mut manifest: Value = response.json().await.ok()?;
        if let Some(list) = manifest["manifests"].as_array() {
            let platform = list.iter().find(|m| {
                m["platform"]["os"] == "linux" && m["platform"]["architecture"] == "amd64"
            })?["digest"]
                .as_str()?
                .to_string();
            manifest = self
                .get(&format!("manifests/{platform}"), ACCEPT)
                .await?
                .json()
                .await
                .ok()?;
        }
        Some((digest, manifest))
    }
}

#[derive(Serialize, Clone, Default)]
pub struct ImageVersion {
    pub digest: String,
    pub revision: String,
    pub created: String,
}

fn version_from(digest: String, labels: &Value) -> ImageVersion {
    ImageVersion {
        digest,
        revision: labels["org.opencontainers.image.revision"]
            .as_str()
            .unwrap_or("")
            .chars()
            .take(7)
            .collect(),
        created: labels["org.opencontainers.image.created"]
            .as_str()
            .unwrap_or("")
            .chars()
            .take(10)
            .collect(),
    }
}

/// What `latest` is on ghcr now: the image CI built from the newest commit on main.
pub async fn latest_image() -> Option<ImageVersion> {
    let registry = Registry::open().await?;
    let (digest, manifest) = registry.latest().await?;
    let config = manifest["config"]["digest"].as_str()?;
    let config: Value = registry
        .get(&format!("blobs/{config}"), "*/*")
        .await?
        .json()
        .await
        .ok()?;
    Some(version_from(digest, &config["config"]["Labels"]))
}

/// The image this PC would start, when it has pulled one.
pub async fn local_image() -> Option<ImageVersion> {
    let found = output(
        "docker",
        &["image", "inspect", IMAGE, "--format", "{{json .}}"],
    )
    .await?;
    let found: Value = serde_json::from_str(&found).ok()?;
    let digest = found["RepoDigests"]
        .as_array()?
        .iter()
        .filter_map(Value::as_str)
        .find_map(|d| d.split_once('@').map(|(_, digest)| digest.to_string()))?;
    Some(version_from(digest, &found["Config"]["Labels"]))
}

/// Each layer's compressed size by the 12-character id `docker pull` prints: without a terminal it
/// prints no byte counts, only which layers are done.
pub async fn layer_sizes() -> Option<HashMap<String, u64>> {
    let (_, manifest) = Registry::open().await?.latest().await?;
    let sizes = manifest["layers"]
        .as_array()?
        .iter()
        .filter_map(|layer| {
            let digest = layer["digest"].as_str()?.strip_prefix("sha256:")?;
            Some((digest.get(..12)?.to_string(), layer["size"].as_u64()?))
        })
        .collect::<HashMap<_, _>>();
    (!sizes.is_empty()).then_some(sizes)
}

pub struct Pulled {
    pub message: String,
    pub progress: Option<f64>,
}

/// `docker pull`, reporting what it has fetched as it goes.
pub async fn pull(progress: mpsc::UnboundedSender<Pulled>) -> Result<(), String> {
    let sizes = layer_sizes().await.unwrap_or_default();
    let total: u64 = sizes.values().sum();
    let mut child = command("docker")
        .args(["pull", IMAGE])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("docker pull: {e}"))?;
    let mut lines = BufReader::new(child.stdout.take().expect("piped")).lines();
    let (mut seen, mut fetched, mut unpacked) = (HashSet::new(), HashSet::new(), HashSet::new());
    while let Ok(Some(line)) = lines.next_line().await {
        let Some((id, state)) = line.split_once(": ") else {
            continue;
        };
        if id.len() != 12 {
            continue;
        }
        seen.insert(id.to_string());
        if matches!(
            state,
            "Download complete" | "Already exists" | "Pull complete"
        ) {
            fetched.insert(id.to_string());
        }
        if matches!(state, "Already exists" | "Pull complete") {
            unpacked.insert(id.to_string());
        }
        let got: u64 = fetched.iter().filter_map(|id| sizes.get(id)).sum();
        let update = if fetched.len() < seen.len() && total > 0 {
            Pulled {
                message: format!(
                    "Downloading DiffLab: {:.1} / {:.1} GB",
                    got as f64 / 1e9,
                    total as f64 / 1e9
                ),
                progress: Some(got as f64 / total as f64),
            }
        } else if fetched.len() < seen.len() {
            Pulled {
                message: format!(
                    "Downloading DiffLab: {} / {} parts",
                    fetched.len(),
                    seen.len()
                ),
                progress: Some(fetched.len() as f64 / seen.len() as f64),
            }
        } else {
            Pulled {
                message: format!("Unpacking: {} / {} parts", unpacked.len(), seen.len()),
                progress: Some(unpacked.len() as f64 / seen.len() as f64),
            }
        };
        let _ = progress.send(update);
    }
    let out = child.wait_with_output().await.map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!(
            "docker pull failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(())
}

/// Returns the host port it listens on: whatever the OS hands out, since any fixed one may be taken.
pub async fn start(token: &str, runtime: &str) -> Result<u16, String> {
    let free = std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map_err(|e| format!("no free port: {e}"))?
        .port();
    let _ = command("docker")
        .args(["rm", "-f", CONTAINER])
        .output()
        .await;
    // Podman reaches the card through CDI; Docker through the NVIDIA container toolkit's --gpus.
    let gpus: &[&str] = if runtime == "podman" {
        &["--device", "nvidia.com/gpu=all"]
    } else {
        &["--gpus", "all"]
    };
    let port = format!("127.0.0.1:{free}:8765");
    let volume = format!("{VOLUME}:/app/output");
    let token = format!("DIFFLAB_TOKEN={token}");
    let mut args = vec![
        "run", "-d", "--name", CONTAINER, "-p", &port, "-v", &volume, "-e", &token,
    ];
    args.extend_from_slice(gpus);
    args.push(IMAGE);
    let out = command("docker")
        .args(&args)
        .output()
        .await
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let said = stderr
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.contains("--help"))
            .collect::<Vec<_>>()
            .join("\n");
        return Err(format!(
            "Docker could not start DiffLab: {}",
            said.trim_start_matches("docker: ")
        ));
    }
    Ok(free)
}

pub async fn running() -> bool {
    output(
        "docker",
        &["inspect", "-f", "{{.State.Running}}", CONTAINER],
    )
    .await
    .as_deref()
        == Some("true")
}

pub async fn stop() -> Result<(), String> {
    let out = command("docker")
        .args(["rm", "-f", CONTAINER])
        .output()
        .await
        .map_err(|e| e.to_string())?;
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).trim().to_string())
    }
}

pub fn url(port: u16) -> String {
    format!("http://127.0.0.1:{port}")
}

const RESULTS_HELPER: &str = "difflab-desktop-results";

pub async fn has_results() -> bool {
    output("docker", &["volume", "inspect", VOLUME])
        .await
        .is_some()
}

/// Every file under `prefixes` in the Docker volume, as (path under output/, bytes).
pub async fn list_results(prefixes: &[&str]) -> Result<Vec<(String, u64)>, String> {
    let volume = format!("{VOLUME}:/app/output:ro");
    let mut args = vec!["run", "--rm", "--entrypoint", "find", "-v", &volume, IMAGE];
    let roots: Vec<String> = prefixes
        .iter()
        .map(|p| format!("/app/output/{}", p.trim_end_matches('/')))
        .collect();
    args.extend(roots.iter().map(String::as_str));
    args.extend(["-type", "f", "-printf", "%s %p\\n"]);
    let out = command("docker")
        .args(&args)
        .output()
        .await
        .map_err(|e| e.to_string())?;
    // find exits non-zero when one of the folders does not exist yet; what it did list still counts.
    let text = String::from_utf8_lossy(&out.stdout);
    let mut found = Vec::new();
    for line in text.lines() {
        let Some((size, path)) = line.split_once(' ') else {
            continue;
        };
        let Ok(size) = size.parse() else { continue };
        if let Some(key) = path.strip_prefix("/app/output/") {
            found.push((key.to_string(), size));
        }
    }
    if found.is_empty()
        && !out.status.success()
        && !String::from_utf8_lossy(&out.stderr).contains("No such file")
    {
        return Err(format!(
            "could not read the difflab volume: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(found)
}

/// `docker cp` rather than a bind mount: the CLI writes the files itself, so they belong to this user
/// and need no SELinux label on the target folder.
pub async fn copy_results(
    files: &[(String, std::path::PathBuf)],
    mut each: impl FnMut(usize, &str),
) -> Result<(), String> {
    let _ = command("docker")
        .args(["rm", "-f", RESULTS_HELPER])
        .output()
        .await;
    let volume = format!("{VOLUME}:/app/output");
    run(
        "docker",
        &["create", "--name", RESULTS_HELPER, "-v", &volume, IMAGE],
    )
    .await?;
    let mut result = Ok(());
    for (done, (key, to)) in files.iter().enumerate() {
        each(done, key);
        if let Some(parent) = to.parent() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }
        let from = format!("{RESULTS_HELPER}:/app/output/{key}");
        if let Err(e) = run("docker", &["cp", &from, &to.display().to_string()]).await {
            result = Err(e);
            break;
        }
    }
    let _ = command("docker")
        .args(["rm", "-f", RESULTS_HELPER])
        .output()
        .await;
    result
}

/// An AppImage installs nothing, so without this the taskbar finds no icon for the window's app id.
#[cfg(target_os = "linux")]
pub fn integrate() {
    let (Some(appimage), Some(home)) = (std::env::var_os("APPIMAGE"), std::env::var_os("HOME"))
    else {
        return;
    };
    let data = std::env::var_os("XDG_DATA_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(&home).join(".local/share"));
    let icons = data.join("icons/hicolor");
    for (size, bytes) in [
        ("32x32", &include_bytes!("../icons/32x32.png")[..]),
        ("128x128", &include_bytes!("../icons/128x128.png")[..]),
        ("256x256", &include_bytes!("../icons/128x128@2x.png")[..]),
    ] {
        let dir = icons.join(size).join("apps");
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(dir.join("difflab-desktop.png"), bytes);
    }
    let entry = format!(
        "[Desktop Entry]\nType=Application\nName=DiffLab\nComment=Run DiffLab on this PC or on RunPod\nExec=\"{}\"\nIcon=difflab-desktop\nStartupWMClass=difflab-desktop\nCategories=Graphics;Video;\nTerminal=false\n",
        appimage.to_string_lossy()
    );
    let apps = data.join("applications");
    let _ = std::fs::create_dir_all(&apps);
    let path = apps.join("difflab-desktop.desktop");
    if std::fs::read_to_string(&path).ok().as_deref() != Some(entry.as_str()) {
        let _ = std::fs::write(path, entry);
    }
}
