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

fn command(program: &str) -> Command {
    let mut command = Command::new(program);
    // An AppImage points LD_LIBRARY_PATH at its own bundled libraries, which docker and nvidia-smi
    // must not load.
    if std::env::var_os("APPIMAGE").is_some() {
        command.env_remove("LD_LIBRARY_PATH");
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

/// Each layer's compressed size by the 12-character id `docker pull` prints: without a terminal it
/// prints no byte counts, only which layers are done.
pub async fn layer_sizes() -> Option<HashMap<String, u64>> {
    let http = reqwest::Client::new();
    let token: Value = http
        .get(format!("{REGISTRY}/token?scope=repository:{REPOSITORY}:pull"))
        .send()
        .await
        .ok()?
        .json()
        .await
        .ok()?;
    let token = token["token"].as_str()?.to_string();
    let get = |reference: String| {
        http.get(format!("{REGISTRY}/v2/{REPOSITORY}/manifests/{reference}"))
            .bearer_auth(&token)
            .header("Accept", ACCEPT)
            .send()
    };
    let mut manifest: Value = get("latest".into()).await.ok()?.json().await.ok()?;
    if let Some(list) = manifest["manifests"].as_array() {
        let digest = list
            .iter()
            .find(|m| m["platform"]["os"] == "linux" && m["platform"]["architecture"] == "amd64")?["digest"]
            .as_str()?
            .to_string();
        manifest = get(digest).await.ok()?.json().await.ok()?;
    }
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
        if matches!(state, "Download complete" | "Already exists" | "Pull complete") {
            fetched.insert(id.to_string());
        }
        if matches!(state, "Already exists" | "Pull complete") {
            unpacked.insert(id.to_string());
        }
        let got: u64 = fetched.iter().filter_map(|id| sizes.get(id)).sum();
        let update = if fetched.len() < seen.len() && total > 0 {
            Pulled {
                message: format!("Downloading DiffLab: {:.1} / {:.1} GB", got as f64 / 1e9, total as f64 / 1e9),
                progress: Some(got as f64 / total as f64),
            }
        } else if fetched.len() < seen.len() {
            Pulled {
                message: format!("Downloading DiffLab: {} / {} parts", fetched.len(), seen.len()),
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
        return Err(format!("docker pull failed: {}", String::from_utf8_lossy(&out.stderr).trim()));
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
        let said = stderr.lines().filter(|l| !l.trim().is_empty() && !l.contains("--help")).collect::<Vec<_>>().join("\n");
        return Err(format!("Docker could not start DiffLab: {}", said.trim_start_matches("docker: ")));
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
