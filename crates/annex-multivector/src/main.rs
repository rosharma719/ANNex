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
    /// When true, the response includes a `stats` object with per-query
    /// timing, candidate counts, and FDE-vs-MaxSim rank-agreement signals.
    /// The primitive under EXPLAIN SEARCH — surface the "why" of a query
    /// alongside the "what", so downstream tuning, SLO enforcement, and
    /// confidence output all have data to consume.
    #[serde(default)]
    explain: bool,
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
    let explain = body.explain;
    let t0 = std::time::Instant::now();
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
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if !explain {
        return Ok(Json(json!({"matches": matches})));
    }

    // ── EXPLAIN block: cheap derived stats from the returned matches ──
    // Everything below is computed server-side so callers don't have to
    // duplicate the logic (see benchmark/headtohead.py for the reference
    // implementation). Kept close to the primitive: no model inference,
    // no calibration lookup — just what falls out of the hit list.
    let top_score = matches.first().map(|h| h.score).unwrap_or(0.0);
    let second_score = matches.get(1).map(|h| h.score).unwrap_or(top_score);
    let (fde_top_score, fde_top_rank_in_fde, fde_agreement) = {
        let fde_scores: Vec<f32> = matches.iter().map(|h| h.fde_score.unwrap_or(0.0)).collect();
        if matches.len() > 1 {
            let mut fde_order: Vec<usize> = (0..matches.len()).collect();
            fde_order.sort_by(|&a, &b| fde_scores[b].total_cmp(&fde_scores[a]));
            let fde_top_rank = fde_order.iter().position(|&i| i == 0).unwrap_or(0);
            let mut fde_rank_by_pos = vec![0usize; matches.len()];
            for (rank, &pos) in fde_order.iter().enumerate() {
                fde_rank_by_pos[pos] = rank;
            }
            let agree = fde_rank_by_pos
                .iter()
                .enumerate()
                .filter(|(i, r)| r.abs_diff(*i) <= 3)
                .count();
            (
                fde_scores.first().copied().unwrap_or(0.0),
                fde_top_rank,
                agree as f32 / matches.len() as f32,
            )
        } else {
            (fde_scores.first().copied().unwrap_or(0.0), 0usize, 1.0f32)
        }
    };

    Ok(Json(json!({
        "matches": matches,
        "stats": {
            "elapsed_ms": elapsed_ms,
            "matches_returned": matches.len(),
            "top_k_requested": body.top_k,
            "candidates_requested": body.candidates,
            "candidate_backend": body.candidate_backend,
            "top_score": top_score,
            "top_minus_second": top_score - second_score,
            "fde_top_score": fde_top_score,
            "fde_top_rank_in_fde": fde_top_rank_in_fde,
            "fde_maxsim_agreement": fde_agreement,
        }
    })))
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
