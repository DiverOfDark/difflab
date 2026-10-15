use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const SERVICE: &str = "difflab";
pub const RUNPOD_KEY: &str = "runpod_api_key";
pub const S3_ACCESS: &str = "s3_access_key";
pub const S3_SECRET: &str = "s3_secret";

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Prefs {
    pub datacenter: String,
    pub volume_id: String,
    pub idle_minutes: f64,
    pub stuck_minutes: f64,
    pub spend_limit_usd: f64,
    pub exit_when_done: bool,
    pub results_dir: String,
    pub setup_done: bool,
}

impl Default for Prefs {
    fn default() -> Self {
        Prefs {
            datacenter: "EU-RO-1".into(),
            volume_id: String::new(),
            idle_minutes: 5.0,
            stuck_minutes: 15.0,
            spend_limit_usd: 5.0,
            exit_when_done: false,
            results_dir: String::new(),
            setup_done: false,
        }
    }
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Runpod,
    Local,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Session {
    pub kind: Kind,
    pub pod_id: String,
    pub token: String,
    pub base_url: String,
    pub started_at: i64,
    pub usd_per_hour: f64,
    pub gpu: String,
    #[serde(default)]
    pub image: String,
}

pub struct Store {
    dir: PathBuf,
}

impl Store {
    pub fn new(dir: PathBuf) -> Self {
        let _ = std::fs::create_dir_all(&dir);
        Store { dir }
    }

    pub fn prefs(&self) -> Prefs {
        read(&self.dir.join("prefs.json")).unwrap_or_default()
    }

    pub fn save_prefs(&self, prefs: &Prefs) -> Result<(), String> {
        write(&self.dir.join("prefs.json"), prefs)
    }

    pub fn session(&self) -> Option<Session> {
        read(&self.dir.join("session.json"))
    }

    pub fn save_session(&self, session: Option<&Session>) -> Result<(), String> {
        let path = self.dir.join("session.json");
        match session {
            Some(s) => write(&path, s),
            None => {
                let _ = std::fs::remove_file(&path);
                Ok(())
            }
        }
    }

    pub fn secret(&self, name: &str) -> Option<String> {
        if let Ok(value) = keyring::Entry::new(SERVICE, name).and_then(|e| e.get_password()) {
            return Some(value);
        }
        self.fallback().remove(name).filter(|v| !v.is_empty())
    }

    // The OS keychain first; a desktop with no Secret Service (a bare window manager) gets a 0600 file.
    pub fn set_secret(&self, name: &str, value: &str) -> Result<(), String> {
        if keyring::Entry::new(SERVICE, name)
            .and_then(|e| e.set_password(value))
            .is_ok()
        {
            return Ok(());
        }
        let mut all = self.fallback();
        all.insert(name.into(), value.into());
        let path = self.dir.join("secrets.json");
        write(&path, &all)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
        }
        Ok(())
    }

    fn fallback(&self) -> HashMap<String, String> {
        read(&self.dir.join("secrets.json")).unwrap_or_default()
    }
}

fn read<T: for<'de> Deserialize<'de>>(path: &Path) -> Option<T> {
    serde_json::from_slice(&std::fs::read(path).ok()?).ok()
}

fn write<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let body = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
    std::fs::write(path, body).map_err(|e| format!("could not write {}: {e}", path.display()))
}
