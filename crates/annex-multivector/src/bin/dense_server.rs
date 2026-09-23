use std::{
    fs,
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, RwLock},
};

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    routing::{get, post},
};
use clap::Parser;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};
use annex::{
    utils::types::DistanceMetric,
    vector::hnsw::{HNSWIndex, SearchRuntimeOptions},
};

#[derive(Parser)]
#[command(version)]
struct Args {
    #[arg(long)]
    dimension: usize,
    #[arg(long, default_value = "benchmark/results/dense-vectordb")]
    path: PathBuf,
    #[arg(long, default_value = "127.0.0.1:18081")]
    listen: SocketAddr,
}

struct DenseState {
    dimension: usize,
    path: PathBuf,
    ids: Vec<String>,
    vectors: Vec<Vec<f32>>,
    ann: Option<Arc<HNSWIndex>>,
}

#[derive(Deserialize)]
struct Document {
    id: String,
    vector: Vec<f32>,
}
#[derive(Deserialize)]
struct UpsertRequest {
    documents: Vec<Document>,
}
#[derive(Deserialize)]
struct BuildRequest {
    #[serde(default = "sixteen")]
    m: usize,
    #[serde(default = "two_fifty_six")]
    ef_construct: usize,
}
#[derive(Deserialize)]
struct QueryRequest {
    vector: Vec<f32>,
    #[serde(default = "ten")]
    top_k: usize,
    #[serde(default = "exact")]
    backend: String,
    #[serde(default = "two_fifty_six")]
    ef_search: usize,
}

fn ten() -> usize {
    10
}
fn sixteen() -> usize {
    16
}
fn two_fifty_six() -> usize {
    256
}
fn exact() -> String {
    "exact".into()
}
fn normalize(vector: &[f32]) -> Vec<f32> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm <= f32::EPSILON {
        vector.to_vec()
    } else {
        vector.iter().map(|value| value / norm).collect()
    }
}
fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

async fn health() -> Json<Value> {
    Json(json!({"status":"ok", "version": env!("CARGO_PKG_VERSION")}))
}
async fn stats(State(state): State<Arc<RwLock<DenseState>>>) -> Json<Value> {
    let state = state.read().unwrap();
    Json(json!({
        "documents": state.ids.len(),
        "dimension": state.dimension,
        "hnsw_nodes": state.ann.as_ref().map_or(0, |index| index.len()),
    }))
}
async fn upsert(
    State(state): State<Arc<RwLock<DenseState>>>,
    Json(body): Json<UpsertRequest>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let mut state = state.write().unwrap();
    for document in body.documents {
        if document.vector.len() != state.dimension {
            return Err((StatusCode::BAD_REQUEST, "vector dimension mismatch".into()));
        }
        state.ids.push(document.id);
        state.vectors.push(normalize(&document.vector));
    }
    state.ann = None;
    Ok(Json(json!({"documents": state.ids.len()})))
}
async fn build(
    State(state): State<Arc<RwLock<DenseState>>>,
    Json(body): Json<BuildRequest>,
) -> Result<Json<Value>, (StatusCode, String)> {
    if body.m == 0 || body.ef_construct == 0 {
        return Err((
            StatusCode::BAD_REQUEST,
            "m and ef_construct must be positive".into(),
        ));
    }
    let state_read = state.read().unwrap();
    let mut index = HNSWIndex::new(
        DistanceMetric::Dot,
        body.m,
        body.ef_construct,
        16,
        state_read.dimension,
    );
    for (point, vector) in state_read.vectors.iter().enumerate() {
        index
            .insert(point as u64, vector.clone())
            .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
    }
    fs::create_dir_all(&state_read.path)
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
    index
        .save_to_path(state_read.path.join("hnsw.bin"))
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
    fs::write(
        state_read.path.join("ids.json"),
        serde_json::to_vec(&state_read.ids).unwrap(),
    )
    .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
    drop(state_read);
    let mut state = state.write().unwrap();
    let nodes = index.len();
    state.ann = Some(Arc::new(index));
    Ok(Json(json!({"nodes": nodes})))
}
async fn query(
    State(state): State<Arc<RwLock<DenseState>>>,
    Json(body): Json<QueryRequest>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let state = state.read().unwrap();
    if body.vector.len() != state.dimension || body.top_k == 0 {
        return Err((StatusCode::BAD_REQUEST, "invalid query".into()));
    }
    let query = normalize(&body.vector);
    let matches = match body.backend.as_str() {
        "exact" => {
            let mut scored: Vec<_> = state
                .vectors
                .par_iter()
                .enumerate()
                .map(|(index, vector)| (index, dot(&query, vector)))
                .collect();
            scored.par_sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
            scored
                .into_iter()
                .take(body.top_k)
                .map(|(index, score)| json!({"id": state.ids[index], "score": score}))
                .collect::<Vec<_>>()
        }
        "hnsw" => {
            let index = state.ann.as_ref().ok_or_else(|| {
                (
                    StatusCode::BAD_REQUEST,
                    "HNSW index has not been built".into(),
                )
            })?;
            index
                .search_with_options(
                    &query,
                    body.top_k,
                    &SearchRuntimeOptions {
                        ef_search: Some(body.ef_search.max(body.top_k)),
                        ..SearchRuntimeOptions::default()
                    },
                )
                .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?
                .into_iter()
                .map(|point| json!({"id": state.ids[point.id as usize], "score": point.raw_score}))
                .collect()
        }
        _ => return Err((StatusCode::BAD_REQUEST, "unknown backend".into())),
    };
    Ok(Json(json!({"matches": matches})))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let state = Arc::new(RwLock::new(DenseState {
        dimension: args.dimension,
        path: args.path,
        ids: Vec::new(),
        vectors: Vec::new(),
        ann: None,
    }));
    let app = Router::new()
        .route("/healthz", get(health))
        .route("/v1/stats", get(stats))
        .route("/v1/vectors/upsert", post(upsert))
        .route("/v1/index", post(build))
        .route("/v1/query", post(query))
        .layer(DefaultBodyLimit::max(256 * 1024 * 1024))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(args.listen).await?;
    println!(
        "dense baseline listening on http://{}",
        listener.local_addr()?
    );
    axum::serve(listener, app).await?;
    Ok(())
}
