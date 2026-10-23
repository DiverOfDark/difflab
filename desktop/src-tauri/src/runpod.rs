use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const API: &str = "https://rest.runpod.io/v1";
// Global volumes, pods that mount one, and the balance are only on GraphQL (RunPod's SDK uses it for them).
const GRAPHQL: &str = "https://api.runpod.io/graphql";
// Pod logs are on the v2 API (what `runpodctl pod logs` reads): server-sent events of
// {source: system|container, line, ts}, the system lines narrating the image pull layer by layer.
const LOGS: &str = "https://api.runpod.io/v2";
pub const IMAGE: &str = "ghcr.io/diverofdark/difflab:latest";
pub const VOLUME_NAME: &str = "difflab";
// The pipeline offloads for a 24 GB card; README "Serving it" and `core/models.py` assume a 58 GB host.
pub const GPUS: [&str; 3] = [
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 5090",
];
pub const MIN_RAM_GB: u32 = 64;
// The unpacked image (~10 GB) and every weight (~96 GB), which the image loads from /models, not from the
// global volume (see the Dockerfile), with room to spare. The Hub delivered 28.4 GB in 83 s.
pub const CONTAINER_DISK_GB: u32 = 150;
// Where `difflab/core/modelstore.py` keeps a plain copy of the weights for the next pod.
pub const MODEL_STORE: &str = "/app/output/.model-store";
pub const PORT: u16 = 8765;

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

#[derive(Deserialize, Serialize, Clone)]
pub struct LogLine {
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub line: String,
    #[serde(default)]
    pub ts: String,
}

pub struct Client {
    http: reqwest::Client,
    key: String,
}

impl Client {
    pub fn new(key: String) -> Self {
        Client {
            // Without one a hung request held Stop on "Stopping…" for good.
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .unwrap_or_default(),
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

    async fn graphql(&self, query: &str, variables: Value) -> Result<Value, String> {
        let response = self
            .http
            .post(GRAPHQL)
            .query(&[("api_key", &self.key)])
            .json(&json!({ "query": query, "variables": variables }))
            .send()
            .await
            .map_err(|e| format!("RunPod is unreachable: {e}"))?;
        let status = response.status();
        let body: Value = response.json().await.unwrap_or(Value::Null);
        if let Some(problem) = body["errors"][0]["message"].as_str() {
            return Err(format!("RunPod: {problem}"));
        }
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err("RunPod rejected the API key".into());
        }
        Ok(body["data"].clone())
    }

    /// The global volume named difflab, created when there is none. It is not in any datacenter, so a
    /// pod on it can start wherever a GPU is free; there is nothing to size, it grows as it is used.
    pub async fn global_volume(&self) -> Result<String, String> {
        let found = self
            .graphql(
                "query { myself { globalStoreBuckets { id name } } }",
                json!({}),
            )
            .await?;
        let existing = found["myself"]["globalStoreBuckets"]
            .as_array()
            .and_then(|all| all.iter().find(|b| b["name"] == VOLUME_NAME))
            .and_then(|b| b["id"].as_str());
        if let Some(id) = existing {
            return Ok(id.to_string());
        }
        let made = self
            .graphql(
                "mutation make($input: GlobalStoreBucketCreateInput!) { globalStoreBucketCreate(input: $input) { id } }",
                json!({ "input": { "name": VOLUME_NAME } }),
            )
            .await?;
        made["globalStoreBucketCreate"]["id"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| "RunPod did not create the volume".into())
    }

    /// Through GraphQL, as RunPod's SDK does: the REST pod has no way to mount a global volume.
    /// `image` is pinned by digest, as a host that cached an older `latest` would otherwise start that.
    pub async fn create_pod(
        &self,
        volume_id: &str,
        image: &str,
        env: &[(&str, String)],
    ) -> Result<Pod, String> {
        let input = json!({
            "name": "difflab",
            "imageName": image,
            "gpuTypeIdList": GPUS,
            "gpuCount": 1,
            "cloudType": "SECURE",
            "containerDiskInGb": CONTAINER_DISK_GB,
            "volumeInGb": 0,
            "minMemoryInGb": MIN_RAM_GB,
            "ports": format!("{PORT}/http"),
            "env": env.iter().map(|(key, value)| json!({ "key": key, "value": value })).collect::<Vec<_>>(),
            "volumeMounts": [{ "volumeId": volume_id, "volumeType": "OBJECT_STORE_VOLUME", "mountPath": "/app/output" }],
        });
        let made = self
            .graphql(
                "mutation deploy($input: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $input) { id costPerHr machine { gpuDisplayName } } }",
                json!({ "input": input }),
            )
            .await?;
        let pod = &made["podFindAndDeployOnDemand"];
        let id = pod["id"].as_str().ok_or(
            "RunPod found no free GPU with 24 GB and 64 GB of RAM; try again in a few minutes",
        )?;
        Ok(Pod {
            id: id.to_string(),
            cost_per_hr: pod["costPerHr"].as_f64().unwrap_or(0.0),
            gpu: Some(json!({ "displayName": pod["machine"]["gpuDisplayName"] })),
            ..Default::default()
        })
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

    pub async fn balance(&self) -> Result<f64, String> {
        self.graphql("query { myself { clientBalance } }", json!({}))
            .await?
            .pointer("/myself/clientBalance")
            .and_then(Value::as_f64)
            .ok_or_else(|| "RunPod rejected the API key".into())
    }

    pub async fn gpu_name(&self, id: &str) -> Option<String> {
        let found = self
            .graphql(
                "query pod($id: String!) { pod(input: {podId: $id}) { machine { gpuDisplayName } } }",
                json!({ "id": id }),
            )
            .await
            .ok()?;
        found["pod"]["machine"]["gpuDisplayName"]
            .as_str()
            .map(str::to_string)
    }

    /// Streams the pod's log from after `cursor` (or its last `tail` lines) until the connection
    /// drops; the caller reconnects with the last id it was handed.
    pub async fn follow_logs(
        &self,
        id: &str,
        cursor: Option<&str>,
        mut each: impl FnMut(String, LogLine),
    ) -> Result<(), String> {
        let mut request = self
            .http
            .get(format!("{LOGS}/pods/{id}/logs"))
            .bearer_auth(&self.key)
            .header("Accept", "text/event-stream");
        request = match cursor {
            Some(cursor) => request.header("Last-Event-ID", cursor),
            None => request.query(&[("tail", "300")]),
        };
        let response = request.send().await.map_err(|e| e.to_string())?;
        if !response.status().is_success() {
            return Err(format!("logs answered {}", response.status()));
        }
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        while let Some(chunk) = stream.next().await {
            buffer.push_str(&String::from_utf8_lossy(&chunk.map_err(|e| e.to_string())?));
            while let Some(end) = buffer.find("\n\n") {
                let event: String = buffer.drain(..end + 2).collect();
                let (mut event_id, mut data) = (String::new(), String::new());
                for line in event.lines() {
                    if let Some(v) = line.strip_prefix("id:") {
                        event_id = v.trim().to_string();
                    } else if let Some(v) = line.strip_prefix("data:") {
                        data.push_str(v.trim_start());
                    }
                }
                if let Ok(line) = serde_json::from_str::<LogLine>(&data) {
                    each(event_id, line);
                }
            }
        }
        Ok(())
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
