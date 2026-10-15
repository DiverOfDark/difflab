//! RunPod's S3 API over a network volume: the bucket is the volume id, the region its datacenter.

use futures_util::StreamExt;
use hmac::{Hmac, Mac};
use percent_encoding::{utf8_percent_encode, AsciiSet, NON_ALPHANUMERIC};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::path::Path;
use tokio::io::AsyncWriteExt;

const UNRESERVED: &AsciiSet = &NON_ALPHANUMERIC.remove(b'-').remove(b'_').remove(b'.').remove(b'~');
// SHA-256 of an empty body: every request here is a GET.
const EMPTY: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

pub struct Volume {
    pub datacenter: String,
    pub id: String,
    pub access: String,
    pub secret: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct Listing {
    #[serde(default)]
    contents: Vec<Object>,
    #[serde(default)]
    is_truncated: bool,
    next_continuation_token: Option<String>,
}

#[derive(Deserialize, Clone)]
#[serde(rename_all = "PascalCase")]
pub struct Object {
    pub key: String,
    pub size: u64,
}

impl Volume {
    fn host(&self) -> String {
        format!("s3api-{}.runpod.io", self.datacenter.to_lowercase())
    }

    fn signed(&self, path: &str, query: &[(&str, &str)]) -> reqwest::RequestBuilder {
        let (url, stamp, authorization) = self.sign(path, query, chrono::Utc::now());
        reqwest::Client::new()
            .get(url)
            .header("x-amz-date", stamp)
            .header("x-amz-content-sha256", EMPTY)
            .header("authorization", authorization)
    }

    fn sign(&self, path: &str, query: &[(&str, &str)], now: chrono::DateTime<chrono::Utc>) -> (String, String, String) {
        let stamp = now.format("%Y%m%dT%H%M%SZ").to_string();
        let day = now.format("%Y%m%d").to_string();
        let host = self.host();
        let mut pairs: Vec<(String, String)> = query
            .iter()
            .map(|(k, v)| (enc(k), enc(v)))
            .collect();
        pairs.sort();
        let canonical_query = pairs.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("&");
        let canonical = format!(
            "GET\n{path}\n{canonical_query}\nhost:{host}\nx-amz-content-sha256:{EMPTY}\nx-amz-date:{stamp}\n\nhost;x-amz-content-sha256;x-amz-date\n{EMPTY}"
        );
        let scope = format!("{day}/{}/s3/aws4_request", self.datacenter);
        let to_sign = format!("AWS4-HMAC-SHA256\n{stamp}\n{scope}\n{}", hex::encode(Sha256::digest(canonical)));
        let mut key = hmac(format!("AWS4{}", self.secret).as_bytes(), day.as_bytes());
        for part in [self.datacenter.as_str(), "s3", "aws4_request"] {
            key = hmac(&key, part.as_bytes());
        }
        let signature = hex::encode(hmac(&key, to_sign.as_bytes()));
        let url = if canonical_query.is_empty() {
            format!("https://{host}{path}")
        } else {
            format!("https://{host}{path}?{canonical_query}")
        };
        let authorization = format!(
            "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature={signature}",
            self.access
        );
        (url, stamp, authorization)
    }

    fn object_path(&self, key: &str) -> String {
        let key = key.split('/').map(enc).collect::<Vec<_>>().join("/");
        format!("/{}/{key}", self.id)
    }

    pub async fn list(&self, prefix: &str) -> Result<Vec<Object>, String> {
        let mut all = Vec::new();
        let mut token: Option<String> = None;
        loop {
            let mut query = vec![("list-type", "2"), ("prefix", prefix)];
            if let Some(t) = token.as_deref() {
                query.push(("continuation-token", t));
            }
            let response = self
                .signed(&format!("/{}", self.id), &query)
                .send()
                .await
                .map_err(|e| format!("the volume is unreachable: {e}"))?;
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            if !status.is_success() {
                return Err(format!("the volume answered {status}: {}", body.chars().take(300).collect::<String>()));
            }
            let page: Listing = quick_xml::de::from_str(&body).map_err(|e| format!("unexpected listing: {e}"))?;
            all.extend(page.contents);
            match (page.is_truncated, page.next_continuation_token) {
                (true, Some(next)) => token = Some(next),
                _ => return Ok(all),
            }
        }
    }

    pub async fn get(&self, key: &str) -> Result<Option<Vec<u8>>, String> {
        let response = self.signed(&self.object_path(key), &[]).send().await.map_err(|e| e.to_string())?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !response.status().is_success() {
            return Err(format!("the volume answered {}", response.status()));
        }
        response.bytes().await.map(|b| Some(b.to_vec())).map_err(|e| e.to_string())
    }

    pub async fn download(&self, key: &str, to: &Path) -> Result<(), String> {
        let response = self.signed(&self.object_path(key), &[]).send().await.map_err(|e| e.to_string())?;
        if !response.status().is_success() {
            return Err(format!("{key}: the volume answered {}", response.status()));
        }
        if let Some(parent) = to.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| e.to_string())?;
        }
        let partial = to.with_file_name(format!("{}.part", to.file_name().unwrap_or_default().to_string_lossy()));
        let mut file = tokio::fs::File::create(&partial).await.map_err(|e| e.to_string())?;
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            file.write_all(&chunk.map_err(|e| e.to_string())?).await.map_err(|e| e.to_string())?;
        }
        file.flush().await.map_err(|e| e.to_string())?;
        tokio::fs::rename(&partial, to).await.map_err(|e| e.to_string())
    }
}

fn enc(s: &str) -> String {
    utf8_percent_encode(s, UNRESERVED).to_string()
}

fn hmac(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("hmac takes any key length");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

