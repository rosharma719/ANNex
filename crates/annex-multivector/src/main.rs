use std::{net::SocketAddr, path::PathBuf, sync::Arc};

use axum::{
    Json, Router,
    extract::DefaultBodyLimit,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use clap::Parser;
use multivector::{IndexConfig, IndexError, MultiVectorIndex, UpsertDocument};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Parser)]
#[command(version)]
struct Args {
    #[arg(long, default_value = "./data")]
    path: PathBuf,
    #[arg(long)]
    dimension: usize,
    #[arg(long, default_value_t = 64)]
    centroids: usize,
    #[arg(long, default_value_t = 2)]
    residual_bits: u8,
    #[arg(long, default_value_t = 4)]
    probes: usize,
    #[arg(long, default_value_t = 20)]
    fde_repetitions: usize,
    #[arg(long, default_value_t = 4)]
    fde_ksim: usize,
    #[arg(long, default_value_t = 8)]
    fde_projected: usize,
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: SocketAddr,
}

#[derive(Deserialize)]
struct Document {
    id: String,
    vectors: Vec<Vec<f32>>,
    #[serde(default = "null")]
    metadata: Value,
}
#[derive(Deserialize)]
struct UpsertRequest {
    documents: Vec<Document>,
}
#[derive(Deserialize)]
struct QueryRequest {
    vectors: Vec<Vec<f32>>,
    #[serde(default = "ten")]
    top_k: usize,
    candidates: Option<usize>,
    rerank_candidates: Option<usize>,
    probes: Option<usize>,
    candidate_backend: Option<String>,
    ef_search: Option<usize>,
}
#[derive(Deserialize)]
struct CandidateRequest {
    vectors: Vec<Vec<f32>>,
    count: usize,
    #[serde(default = "default_candidate_backend")]
    candidate_backend: String,
    #[serde(default = "two_fifty_six")]
    ef_search: usize,
}
#[derive(Deserialize)]
struct TrainRequest {
    vectors: Vec<Vec<f32>>,
    #[serde(default = "twenty")]
    iterations: usize,
}
#[derive(Deserialize)]
struct ScoreRequest {
    query: Vec<Vec<f32>>,
    document: Option<Vec<Vec<f32>>>,
    id: Option<String>,
}
#[derive(Deserialize)]
struct BuildAnnRequest {
    #[serde(default = "sixteen")]
    m: usize,
    #[serde(default = "two_fifty_six")]
    ef_construct: usize,
}
fn ten() -> usize {
    10
}
fn null() -> Value {
    Value::Null
}
fn twenty() -> usize {
    20
}
fn sixteen() -> usize {
    16
}
fn two_fifty_six() -> usize {
    256
}
fn default_candidate_backend() -> String {
    "muvera".into()
}

struct ApiError(IndexError);
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self.0 {
            IndexError::Invalid(_) => StatusCode::BAD_REQUEST,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, Json(json!({"error": self.0.to_string()}))).into_response()
    }
}
impl From<IndexError> for ApiError {
    fn from(value: IndexError) -> Self {
        Self(value)
    }
}

async fn health() -> Json<Value> {
    Json(json!({"status":"ok", "version": env!("CARGO_PKG_VERSION")}))
}
async fn stats(State(index): State<Arc<MultiVectorIndex>>) -> Json<Value> {
    Json(serde_json::to_value(index.stats()).unwrap())
}
async fn upsert(
    State(index): State<Arc<MultiVectorIndex>>,
    Json(body): Json<UpsertRequest>,
) -> Result<Json<Value>, ApiError> {
    let count = body.documents.len();
    index.upsert_batch(
        body.documents
            .into_iter()
            .map(|document| UpsertDocument {
                id: document.id,
                vectors: document.vectors,
                metadata: document.metadata,
            })
            .collect(),
    )?;
    Ok(Json(json!({"upserted": count})))
}
async fn query(
    State(index): State<Arc<MultiVectorIndex>>,
    Json(body): Json<QueryRequest>,
) -> Result<Json<Value>, ApiError> {
    let matches = match body.candidate_backend.as_deref() {
        Some("hnsw") => match body.rerank_candidates {
            Some(rerank_candidates) => index.query_with_fde_ann_and_pruning(
                &body.vectors,
                body.top_k,
                body.candidates.unwrap_or(rerank_candidates),
                rerank_candidates,
                body.ef_search.unwrap_or(256),
            )?,
            None => index.query_with_fde_ann(
                &body.vectors,
                body.top_k,
                body.candidates,
                body.ef_search.unwrap_or(256),
            )?,
        },
        Some("muvera") | None => match (body.probes, body.rerank_candidates) {
            (None, Some(rerank_candidates)) => index.query_with_centroid_pruning(
                &body.vectors,
                body.top_k,
                body.candidates.unwrap_or(rerank_candidates),
                rerank_candidates,
            )?,
            (Some(probes), None) => {
                index.query_with_probes(&body.vectors, body.top_k, body.candidates, probes)?
            }
            (None, None) => index.query(&body.vectors, body.top_k, body.candidates)?,
            (Some(_), Some(_)) => {
                return Err(ApiError(IndexError::Invalid(
                    "rerank_candidates cannot be combined with probes".into(),
                )));
            }
        },
        Some(other) => {
            return Err(ApiError(IndexError::Invalid(format!(
                "unknown candidate backend: {other}"
            ))));
        }
    };
    Ok(Json(json!({"matches": matches})))
}
async fn train(
    State(index): State<Arc<MultiVectorIndex>>,
    Json(body): Json<TrainRequest>,
) -> Result<Json<Value>, ApiError> {
    let samples = body.vectors.len();
    index.train(&body.vectors, body.iterations)?;
    Ok(Json(json!({"trained_on": samples})))
}
async fn score(
    State(index): State<Arc<MultiVectorIndex>>,
    Json(body): Json<ScoreRequest>,
) -> Result<Json<Value>, ApiError> {
    let score = match (body.document, body.id) {
        (Some(document), None) => index.score_uncompressed(&body.query, &document)?,
        (None, Some(id)) => index.score_compressed(&body.query, &id)?,
        _ => {
            return Err(ApiError(IndexError::Invalid(
                "provide exactly one of document or id".into(),
            )));
        }
    };
    Ok(Json(json!({"score": score})))
}
async fn build_ann(
    State(index): State<Arc<MultiVectorIndex>>,
    Json(body): Json<BuildAnnRequest>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        json!({"nodes": index.build_fde_ann(body.m, body.ef_construct)?}),
    ))
}
async fn candidates(
    State(index): State<Arc<MultiVectorIndex>>,
    Json(body): Json<CandidateRequest>,
) -> Result<Json<Value>, ApiError> {
    let candidates = match body.candidate_backend.as_str() {
        "muvera" => index.exact_fde_candidates(&body.vectors, body.count)?,
        "hnsw" => index.ann_fde_candidates(&body.vectors, body.count, body.ef_search)?,
        other => {
            return Err(ApiError(IndexError::Invalid(format!(
                "unknown candidate backend: {other}"
            ))));
        }
    };
    Ok(Json(json!({"candidates": candidates})))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let config = IndexConfig {
        dimension: args.dimension,
        centroids: args.centroids,
        residual_bits: args.residual_bits,
        probes: args.probes,
        fde_repetitions: args.fde_repetitions,
        fde_ksim: args.fde_ksim,
        fde_projected: args.fde_projected,
    };
    let index = Arc::new(MultiVectorIndex::open(args.path, config)?);
    let app = Router::new()
        .route("/healthz", get(health))
        .route("/v1/stats", get(stats))
        .route("/v1/train", post(train))
        .route("/v1/debug/score", post(score))
        .route("/v1/debug/candidates", post(candidates))
        .route("/v1/fde/index", post(build_ann))
        .route("/v1/vectors/upsert", post(upsert))
        .route("/v1/query", post(query))
        // ColBERT batches are legitimately large: 100 documents can contain
        // millions of JSON floats. Keep the limit explicit and configurable at
        // the reverse proxy in deployed environments.
        .layer(DefaultBodyLimit::max(256 * 1024 * 1024))
        .with_state(index);
    let listener = tokio::net::TcpListener::bind(args.listen).await?;
    println!("multivector listening on http://{}", listener.local_addr()?);
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
