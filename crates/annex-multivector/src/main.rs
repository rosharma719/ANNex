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
use multivector::{
    Collections, Durability, IndexConfig, IndexError, MultiVectorIndex, RetrievalDocument,
    RetrievalPlan, RetrievalResponse, RetrieveRequest, TextAnalyzer,
};
use serde::Deserialize;
use serde_json::{Value, json};
use tower::ServiceExt;

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
    /// Fsync acknowledges durable commits; buffered only promises atomic visibility.
    #[arg(long, default_value = "fsync", value_parser = ["fsync", "buffered"])]
    durability: String,
    /// Text analysis policy for the lexical field of the default index.
    /// Collections created over HTTP configure the analyzer in their JSON config.
    #[arg(long, default_value = "plain", value_parser = ["plain", "english"])]
    analyzer: String,
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: SocketAddr,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UpsertRequest {
    documents: Vec<RetrievalDocument>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeleteRequest {
    id: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct QueryRequest {
    vectors: Vec<Vec<f32>>,
    #[serde(default = "ten")]
    top_k: usize,
    candidates: Option<usize>,
    rerank_candidates: Option<usize>,
    probes: Option<usize>,
    candidate_backend: Option<String>,
    ef_search: Option<usize>,
    #[serde(default)]
    explain: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateRequest {
    vectors: Vec<Vec<f32>>,
    count: usize,
    #[serde(default = "default_candidate_backend")]
    candidate_backend: String,
    ef_search: Option<usize>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TrainRequest {
    vectors: Vec<Vec<f32>>,
    #[serde(default = "twenty")]
    iterations: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScoreRequest {
    query: Vec<Vec<f32>>,
    document: Option<Vec<Vec<f32>>>,
    id: Option<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BuildAnnRequest {
    #[serde(default = "sixteen")]
    m: usize,
    #[serde(default = "two_fifty_six")]
    ef_construct: usize,
}
fn ten() -> usize {
    10
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

/// CLI presets; HTTP collection configs carry the full analyzer object.
fn analyzer_preset(name: &str) -> TextAnalyzer {
    match name {
        "english" => TextAnalyzer::english(),
        _ => TextAnalyzer::plain(),
    }
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

fn bounded(name: &str, value: usize, max: usize) -> Result<(), ApiError> {
    if value == 0 || value > max {
        return Err(IndexError::Invalid(format!("{name} must be between 1 and {max}")).into());
    }
    Ok(())
}

async fn delete(
    State(index): State<Arc<MultiVectorIndex>>,
    Json(body): Json<DeleteRequest>,
) -> Result<Json<Value>, ApiError> {
    bounded("document id bytes", body.id.len(), 4096)?;
    Ok(Json(json!({"deleted": index.delete(&body.id)?})))
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
    bounded("documents", body.documents.len(), 1024)?;
    let mut tokens = 0;
    let mut values = 0;
    for document in &body.documents {
        bounded("document id bytes", document.id.len(), 4096)?;
        if !document.vectors.is_empty() {
            bounded("document tokens", document.vectors.len(), 8192)?;
        }
        tokens += document.vectors.len();
        values += document.vectors.iter().map(Vec::len).sum::<usize>();
    }
    if tokens > 0 {
        bounded("batch tokens", tokens, 131_072)?;
    }
    if values > 0 {
        bounded("batch vector values", values, 16_777_216)?;
    }
    let count = body.documents.len();
    tokio::task::spawn_blocking(move || index.upsert_records(body.documents))
        .await
        .map_err(|e| ApiError(IndexError::Invalid(format!("ingest task failed: {e}"))))??;
    Ok(Json(json!({"upserted": count})))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DenseAnnRequest {
    field: String,
    #[serde(default = "sixteen")]
    m: usize,
    #[serde(default = "two_fifty_six")]
    ef_construct: usize,
}
async fn build_dense(
    State(index): State<Arc<MultiVectorIndex>>,
    Json(body): Json<DenseAnnRequest>,
) -> Result<Json<Value>, ApiError> {
    let nodes = tokio::task::spawn_blocking(move || {
        index.build_dense_ann(&body.field, body.m, body.ef_construct)
    })
    .await
    .map_err(|e| ApiError(IndexError::Invalid(e.to_string())))??;
    Ok(Json(json!({"nodes":nodes})))
}

async fn compact(State(index): State<Arc<MultiVectorIndex>>) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        tokio::task::spawn_blocking(move || index.compact())
            .await
            .map_err(|e| ApiError(IndexError::Invalid(format!("compaction task failed: {e}"))))??,
    ))
}

async fn retrieve(
    State(index): State<Arc<MultiVectorIndex>>,
    Json(body): Json<RetrieveRequest>,
) -> Result<Json<RetrievalResponse>, ApiError> {
    let result = tokio::task::spawn_blocking(move || index.retrieve(&body))
        .await
        .map_err(|e| ApiError(IndexError::Invalid(format!("query task failed: {e}"))))??;
    Ok(Json(result))
}

async fn plan(
    State(index): State<Arc<MultiVectorIndex>>,
    Json(body): Json<RetrieveRequest>,
) -> Result<Json<RetrievalPlan>, ApiError> {
    let result = tokio::task::spawn_blocking(move || index.plan(&body))
        .await
        .map_err(|e| ApiError(IndexError::Invalid(format!("planning task failed: {e}"))))??;
    Ok(Json(result))
}

async fn query(
    State(index): State<Arc<MultiVectorIndex>>,
    Json(body): Json<QueryRequest>,
) -> Result<Json<Value>, ApiError> {
    bounded("query tokens", body.vectors.len(), 1024)?;
    bounded("top_k", body.top_k, 10_000)?;
    for (name, value, limit) in [
        ("candidates", body.candidates, 100_000),
        ("rerank_candidates", body.rerank_candidates, 100_000),
        ("probes", body.probes, 4096),
        ("ef_search", body.ef_search, 65_536),
    ] {
        if let Some(value) = value {
            bounded(name, value, limit)?;
        }
    }
    let explain = body.explain;
    let t0 = std::time::Instant::now();
    // Existing clients used these knobs with the implicit MUVERA backend.
    // Requests without those legacy knobs opt into automatic dispatch.
    let backend = body.candidate_backend.as_deref().unwrap_or_else(|| {
        if body.probes.is_some() || body.rerank_candidates.is_some() {
            "muvera"
        } else {
            "auto"
        }
    });
    if (backend == "hnsw" && body.probes.is_some())
        || (backend == "muvera" && body.ef_search.is_some())
    {
        return Err(IndexError::Invalid(
            "probes requires muvera; ef_search requires hnsw or auto".into(),
        )
        .into());
    }
    let mut executed_backend = backend;
    let matches = match backend {
        "hnsw" => match body.rerank_candidates {
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
        // "auto" is the production default: HNSW when built + fresh,
        // exact FDE otherwise. Callers pin behavior with "hnsw" or "muvera".
        "auto" => {
            if body.probes.is_some() || body.rerank_candidates.is_some() {
                return Err(ApiError(IndexError::Invalid(
                    "auto backend does not accept probes/rerank_candidates; \
                     specify candidate_backend=muvera or hnsw explicitly"
                        .into(),
                )));
            }
            let (matches, selected) = index.query_auto_with_backend(
                &body.vectors,
                body.top_k,
                body.candidates,
                body.ef_search.unwrap_or(256),
            )?;
            executed_backend = selected;
            matches
        }
        "muvera" => match (body.probes, body.rerank_candidates) {
            (None, Some(rerank_candidates)) => index.query_with_centroid_pruning(
                &body.vectors,
                body.top_k,
                body.candidates.unwrap_or(rerank_candidates),
                rerank_candidates,
            )?,
            (Some(probes), None) => {
                executed_backend = "centroid";
                index.query_with_probes(&body.vectors, body.top_k, body.candidates, probes)?
            }
            (None, None) => index.query(&body.vectors, body.top_k, body.candidates)?,
            (Some(_), Some(_)) => {
                return Err(ApiError(IndexError::Invalid(
                    "rerank_candidates cannot be combined with probes".into(),
                )));
            }
        },
        other => {
            return Err(ApiError(IndexError::Invalid(format!(
                "unknown candidate backend: {other}"
            ))));
        }
    };
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if !explain {
        return Ok(Json(json!({"matches": matches})));
    }

    let fde_available = !matches.is_empty() && matches.iter().all(|hit| hit.fde_score.is_some());
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
            "candidate_backend": executed_backend,
            "candidate_backend_requested": body.candidate_backend,
            "top_score": top_score,
            "top_minus_second": top_score - second_score,
            "fde_top_score": if fde_available { Some(fde_top_score) } else { None },
            "fde_top_rank_in_fde": if fde_available { Some(fde_top_rank_in_fde) } else { None },
            "fde_maxsim_agreement": if fde_available { Some(fde_agreement) } else { None },
        }
    })))
}
async fn train(
    State(index): State<Arc<MultiVectorIndex>>,
    Json(body): Json<TrainRequest>,
) -> Result<Json<Value>, ApiError> {
    bounded("training samples", body.vectors.len(), 65_536)?;
    bounded("training iterations", body.iterations, 100)?;
    bounded(
        "training vector values",
        body.vectors.iter().map(Vec::len).sum(),
        16_777_216,
    )?;
    let samples = body.vectors.len();
    index.train(&body.vectors, body.iterations)?;
    Ok(Json(json!({"trained_on": samples})))
}
async fn score(
    State(index): State<Arc<MultiVectorIndex>>,
    Json(body): Json<ScoreRequest>,
) -> Result<Json<Value>, ApiError> {
    bounded("query tokens", body.query.len(), 1024)?;
    if let Some(document) = &body.document {
        bounded("document tokens", document.len(), 8192)?;
    }
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
    bounded("query tokens", body.vectors.len(), 1024)?;
    bounded("count", body.count, 100_000)?;
    if let Some(ef_search) = body.ef_search {
        bounded("ef_search", ef_search, 65_536)?;
        if body.candidate_backend != "hnsw" {
            return Err(IndexError::Invalid("ef_search requires hnsw".into()).into());
        }
    }
    let candidates = match body.candidate_backend.as_str() {
        "muvera" => index.exact_fde_candidates(&body.vectors, body.count)?,
        "hnsw" => {
            index.ann_fde_candidates(&body.vectors, body.count, body.ef_search.unwrap_or(256))?
        }
        other => {
            return Err(ApiError(IndexError::Invalid(format!(
                "unknown candidate backend: {other}"
            ))));
        }
    };
    Ok(Json(json!({"candidates": candidates})))
}

fn index_router(index: Arc<MultiVectorIndex>) -> Router {
    Router::new()
        .route("/healthz", get(health))
        .route("/v1/stats", get(stats))
        .route("/v1/train", post(train))
        .route("/v1/debug/score", post(score))
        .route("/v1/debug/candidates", post(candidates))
        .route("/v1/fde/index", post(build_ann))
        .route("/v1/dense/index", post(build_dense))
        .route("/v1/vectors/upsert", post(upsert))
        .route("/v1/vectors/delete", post(delete))
        .route("/v1/query", post(query))
        .route("/v1/plan", post(plan))
        .route("/v1/retrieve", post(retrieve))
        .route("/v1/compact", post(compact))
        // ColBERT batches are legitimately large: 100 documents can contain
        // millions of JSON floats. Keep the limit explicit and configurable at
        // the reverse proxy in deployed environments.
        .layer(DefaultBodyLimit::max(256 * 1024 * 1024))
        .with_state(index)
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CreateCollection {
    name: String,
    config: IndexConfig,
}
async fn list_collections(State(collections): State<Arc<Collections>>) -> Json<Value> {
    Json(json!({"collections":collections.names()}))
}
async fn create_collection(
    State(collections): State<Arc<Collections>>,
    Json(body): Json<CreateCollection>,
) -> Result<Json<Value>, ApiError> {
    tokio::task::spawn_blocking(move || collections.create(&body.name, body.config))
        .await
        .map_err(|e| ApiError(IndexError::Invalid(e.to_string())))??;
    Ok(Json(json!({"created":true})))
}
async fn collection_request(
    State(collections): State<Arc<Collections>>,
    axum::extract::Path((name, operation)): axum::extract::Path<(String, String)>,
    mut request: axum::extract::Request,
) -> Response {
    let Some(index) = collections.get(&name) else {
        return (StatusCode::NOT_FOUND, "collection not found").into_response();
    };
    let suffix = request
        .uri()
        .query()
        .map(|s| format!("?{s}"))
        .unwrap_or_default();
    let Ok(uri) = format!("/v1/{operation}{suffix}").parse() else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    *request.uri_mut() = uri;
    match index_router(index).oneshot(request).await {
        Ok(response) => response,
        Err(never) => match never {},
    }
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
        analyzer: analyzer_preset(&args.analyzer),
    };
    let durability = if args.durability == "fsync" {
        Durability::Fsync
    } else {
        Durability::Buffered
    };
    let collection_root = args.path.join("collections");
    let index = Arc::new(MultiVectorIndex::open_with_durability(
        args.path, config, durability,
    )?);
    let collections = Arc::new(Collections::open(collection_root, durability)?);
    let collection_routes = Router::new()
        .route(
            "/v1/collections",
            get(list_collections).post(create_collection),
        )
        .route(
            "/v1/collections/{name}/{*operation}",
            axum::routing::any(collection_request),
        )
        .with_state(collections);
    let app = index_router(index).merge(collection_routes);
    let listener = tokio::net::TcpListener::bind(args.listen).await?;
    println!("multivector listening on http://{}", listener.local_addr()?);
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (tempfile::TempDir, Arc<MultiVectorIndex>) {
        let directory = tempfile::tempdir().unwrap();
        let config = IndexConfig {
            dimension: 2,
            centroids: 2,
            residual_bits: 2,
            probes: 2,
            fde_repetitions: 2,
            fde_ksim: 2,
            fde_projected: 2,
            analyzer: TextAnalyzer::plain(),
        };
        let index = Arc::new(MultiVectorIndex::open(directory.path(), config).unwrap());
        index
            .train(
                &[vec![1., 0.], vec![0., 1.], vec![-1., 0.], vec![0., -1.]],
                2,
            )
            .unwrap();
        index.upsert("a", vec![vec![1., 0.]], json!({})).unwrap();
        index.upsert("b", vec![vec![0., 1.]], json!({})).unwrap();
        (directory, index)
    }

    async fn query_value(index: &Arc<MultiVectorIndex>, body: Value) -> Value {
        query(
            State(Arc::clone(index)),
            Json(serde_json::from_value(body).unwrap()),
        )
        .await
        .unwrap_or_else(|error| panic!("query failed: {}", error.0))
        .0
    }

    #[tokio::test]
    async fn debug_candidates_default_to_exact_without_graph() {
        let (_directory, index) = fixture();
        let body: CandidateRequest = serde_json::from_value(json!({
            "vectors": [[1., 0.]], "count": 2,
        }))
        .unwrap();
        assert_eq!(body.candidate_backend, "muvera");
        let expected = index
            .exact_fde_candidates(&body.vectors, body.count)
            .unwrap();
        let actual = candidates(State(index), Json(body))
            .await
            .unwrap_or_else(|error| panic!("candidate request failed: {}", error.0));
        assert_eq!(actual.0, json!({"candidates": expected}));
    }

    #[tokio::test]
    async fn plan_endpoint_compiles_without_running_retrieval() {
        let (_directory, index) = fixture();
        let body: RetrieveRequest = serde_json::from_value(json!({
            "prefetch": [{
                "kind": "multivector",
                "vectors": [[1.0, 0.0]],
                "limit": 2
            }],
            "limit": 1
        }))
        .unwrap();
        let actual = plan(State(Arc::clone(&index)), Json(body.clone()))
            .await
            .unwrap_or_else(|error| panic!("planning failed: {}", error.0));
        assert_eq!(
            actual.0.parallel_channels()[0].operator,
            multivector::PhysicalOperator::ExactFde
        );
        assert_eq!(
            actual.0.parallel_channels()[0].reason,
            multivector::PlanReason::AnnUnavailable
        );
        assert_eq!(actual.0.stats.documents, 2);

        index.build_fde_ann(4, 16).unwrap();
        let actual = plan(State(index), Json(body))
            .await
            .unwrap_or_else(|error| panic!("planning failed: {}", error.0));
        assert_eq!(
            actual.0.parallel_channels()[0].operator,
            multivector::PhysicalOperator::HnswFde
        );
        assert_eq!(
            actual.0.parallel_channels()[0].reason,
            multivector::PlanReason::AnnReady
        );
    }

    #[tokio::test]
    async fn omitted_backend_preserves_probe_and_pruning_requests() {
        let (_directory, index) = fixture();
        index.build_fde_ann(4, 16).unwrap();
        for knob in ["probes", "rerank_candidates"] {
            let mut body = json!({
                "vectors": [[1., 0.]], "top_k": 1, "candidates": 2,
            });
            body[knob] = json!(2);
            let implicit = query_value(&index, body.clone()).await;
            body["candidate_backend"] = json!("muvera");
            let explicit = query_value(&index, body.clone()).await;
            assert_eq!(implicit, explicit);
            assert_eq!(implicit["matches"][0]["id"], "a");

            body["candidate_backend"] = json!("auto");
            let error = query(
                State(Arc::clone(&index)),
                Json(serde_json::from_value(body).unwrap()),
            )
            .await
            .expect_err("explicit auto must reject legacy backend knobs");
            assert!(matches!(error.0, IndexError::Invalid(_)));
        }
    }

    #[tokio::test]
    async fn omitted_backend_accepts_queries_before_and_after_graph_build() {
        let (_directory, index) = fixture();
        for built in [false, true] {
            if built {
                index.build_fde_ann(4, 16).unwrap();
            }
            let mut body = json!({
                "vectors": [[1., 0.]], "top_k": 1, "candidates": 2,
            });
            let implicit = query_value(&index, body.clone()).await;
            body["candidate_backend"] = json!("auto");
            assert_eq!(implicit, query_value(&index, body).await);
            assert_eq!(implicit["matches"][0]["id"], "a");
        }
    }
    #[tokio::test]
    async fn explain_reports_the_executed_backend() {
        let (_directory, index) = fixture();
        for built in [false, true] {
            if built {
                index.build_fde_ann(4, 16).unwrap();
            }
            let actual = query_value(
                &index,
                json!({
                    "vectors": [[1., 0.]], "top_k": 1, "explain": true,
                }),
            )
            .await;
            assert_eq!(
                actual["stats"]["candidate_backend"],
                if built { "hnsw" } else { "muvera" }
            );
            assert!(actual["stats"]["candidate_backend_requested"].is_null());
        }
        for (knob, backend) in [("probes", "centroid"), ("rerank_candidates", "muvera")] {
            let mut body = json!({"vectors": [[1., 0.]], "top_k": 1, "explain": true});
            body[knob] = json!(2);
            let actual = query_value(&index, body).await;
            assert_eq!(actual["stats"]["candidate_backend"], backend);
            if backend == "centroid" {
                assert!(actual["matches"][0].get("fde_score").is_none());
                assert!(actual["stats"]["fde_top_score"].is_null());
                assert!(actual["stats"]["fde_maxsim_agreement"].is_null());
            }
        }
    }
}
