use serde::Serialize;
use tokio::process::Command;

use crate::runpod::IMAGE;

pub const CONTAINER: &str = "difflab-desktop";
pub const VOLUME: &str = "difflab";
// Not 8765: a difflab started by hand on this machine already holds it.
pub const PORT: u16 = 18765;
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

pub async fn start(token: &str, runtime: &str) -> Result<(), String> {
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
    let port = format!("127.0.0.1:{PORT}:8765");
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
        return Err(format!(
            "docker run failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(())
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

pub fn url() -> String {
    format!("http://127.0.0.1:{PORT}")
}
