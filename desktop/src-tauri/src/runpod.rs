use serde::Deserialize;
use serde_json::{json, Value};

const API: &str = "https://rest.runpod.io/v1";
pub const IMAGE: &str = "ghcr.io/diverofdark/difflab:latest";
pub const VOLUME_NAME: &str = "difflab";
// ~96 GB of weights, the 40 GiB KV cache (`VLM_DISK_CACHE_BYTES`), the database and finals.
pub const VOLUME_GB: u32 = 150;
// The pipeline offloads for a 24 GB card; README "Serving it" and `core/models.py` assume a 58 GB host.
pub const GPUS: [&str; 3] = [
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 5090",
];
pub const MIN_RAM_GB: u32 = 64;
// The unpacked image is ~10 GB; triton's kernels and HF caches live on the volume.
pub const CONTAINER_DISK_GB: u32 = 40;
pub const PORT: u16 = 8765;

// Network volumes reachable through RunPod's S3 API; a volume elsewhere could not be read from here.
pub const DATACENTERS: [&str; 15] = [
    "EU-CZ-1", "EU-RO-1", "EUR-IS-1", "EUR-NO-1", "US-CA-2", "US-GA-2", "US-IL-1", "US-KS-2",
    "US-MD-1", "US-MO-1", "US-MO-2", "US-NC-1", "US-NC-2", "US-NE-1", "US-WA-1",
];

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Volume {
    pub id: String,
    pub name: String,
    pub data_center_id: String,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase", default)]
pub struct Pod {
    pub id: String,
    pub desired_status: String,
    pub cost_per_hr: f64,
    pub gpu: Option<Value>,
}

impl Pod {
    pub fn gpu_name(&self) -> String {
        self.gpu
            .as_ref()
            .and_then(|g| g.get("displayName").or_else(|| g.get("id")))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    }
}

pub struct Client {
    http: reqwest::Client,
    key: String,
}

impl Client {
    pub fn new(key: String) -> Self {
        Client {
            http: reqwest::Client::new(),
            key,
        }
    }

    async fn call(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<Value>,
    ) -> Result<Option<Value>, String> {
        let mut request = self
            .http
            .request(method, format!("{API}{path}"))
            .bearer_auth(&self.key);
        if let Some(body) = body {
            request = request.json(&body);
        }
        let response = request
            .send()
            .await
            .map_err(|e| format!("RunPod is unreachable: {e}"))?;
        let status = response.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        let text = response.text().await.unwrap_or_default();
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err("RunPod rejected the API key".into());
        }
        if !status.is_success() {
            return Err(format!(
                "RunPod answered {status}: {}",
                text.chars().take(400).collect::<String>()
            ));
        }
        Ok(Some(serde_json::from_str(&text).unwrap_or(Value::Null)))
    }

    pub async fn volumes(&self) -> Result<Vec<Volume>, String> {
        let found = self
            .call(reqwest::Method::GET, "/networkvolumes", None)
            .await?
            .unwrap_or(Value::Null);
        let list = match found {
            Value::Object(fields) => fields
                .into_iter()
                .map(|(_, v)| v)
                .find(Value::is_array)
                .unwrap_or_default(),
            other => other,
        };
        serde_json::from_value(list).map_err(|e| format!("unexpected volume list: {e}"))
    }

    /// The volume named difflab in this datacenter, created when there is none.
    pub async fn volume(&self, datacenter: &str) -> Result<Volume, String> {
        let mut ours: Vec<Volume> = self
            .volumes()
            .await?
            .into_iter()
            .filter(|v| v.name == VOLUME_NAME && v.data_center_id == datacenter)
            .collect();
        if !ours.is_empty() {
            return Ok(ours.remove(0));
        }
        let made = self
            .call(
                reqwest::Method::POST,
                "/networkvolumes",
                Some(json!({"name": VOLUME_NAME, "size": VOLUME_GB, "dataCenterId": datacenter})),
            )
            .await?
            .ok_or("RunPod did not create the volume")?;
        serde_json::from_value(made).map_err(|e| format!("unexpected volume: {e}"))
    }

    pub async fn create_pod(&self, volume_id: &str, env: Value) -> Result<Pod, String> {
        let body = json!({
            "name": "difflab",
            "imageName": IMAGE,
            "gpuTypeIds": GPUS,
            "gpuCount": 1,
            "cloudType": "SECURE",
            "networkVolumeId": volume_id,
            "volumeMountPath": "/app/output",
            "containerDiskInGb": CONTAINER_DISK_GB,
            "minRAMPerGPU": MIN_RAM_GB,
            "ports": [format!("{PORT}/http")],
            "env": env,
        });
        let made = self
            .call(reqwest::Method::POST, "/pods", Some(body))
            .await?
            .ok_or("RunPod did not create the pod")?;
        serde_json::from_value(made).map_err(|e| format!("unexpected pod: {e}"))
    }

    pub async fn pod(&self, id: &str) -> Result<Option<Pod>, String> {
        match self
            .call(reqwest::Method::GET, &format!("/pods/{id}"), None)
            .await?
        {
            Some(found) => Ok(serde_json::from_value(found).ok()),
            None => Ok(None),
        }
    }

    pub async fn delete_pod(&self, id: &str) -> Result<(), String> {
        self.call(reqwest::Method::DELETE, &format!("/pods/{id}"), None)
            .await
            .map(|_| ())
    }
}

pub fn proxy_url(pod_id: &str) -> String {
    format!("https://{pod_id}-{PORT}.proxy.runpod.net")
}
