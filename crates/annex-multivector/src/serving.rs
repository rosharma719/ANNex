//! Global admission and separate worker pools for synchronous service work.
use axum::{
    Json,
    extract::{Request, State},
    http::{HeaderValue, Method, StatusCode, header::RETRY_AFTER},
    middleware::Next,
    response::{IntoResponse, Response},
};
use multivector::IndexError;
use rayon::{ThreadPool, ThreadPoolBuilder};
use serde_json::{Value, json};
use std::{
    io,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::Instant,
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WorkClass {
    Query,
    Ingest,
    Maintenance,
}
impl WorkClass {
    fn name(self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Ingest => "ingest",
            Self::Maintenance => "maintenance",
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct PoolConfig {
    pub threads: usize,
    pub capacity: usize,
}

struct Pool {
    workers: ThreadPool,
    permits: Arc<Semaphore>,
    capacity: usize,
    admitted: AtomicU64,
    rejected: AtomicU64,
    queued: AtomicUsize,
    running: AtomicUsize,
    completed: AtomicU64,
    cancelled_before_start: AtomicU64,
    panicked: AtomicU64,
    queue_us: AtomicU64,
    work_us: AtomicU64,
}
impl Pool {
    fn new(class: WorkClass, config: PoolConfig) -> Result<Arc<Self>, Box<dyn std::error::Error>> {
        if config.threads == 0 || config.capacity == 0 {
            return Err("worker counts and admission capacities must be positive".into());
        }
        let name = class.name();
        let workers = ThreadPoolBuilder::new()
            .num_threads(config.threads)
            .thread_name(move |i| format!("annex-{name}-{i}"))
            .build()?;
        Ok(Arc::new(Self {
            workers,
            permits: Arc::new(Semaphore::new(config.capacity)),
            capacity: config.capacity,
            admitted: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
            queued: AtomicUsize::new(0),
            running: AtomicUsize::new(0),
            completed: AtomicU64::new(0),
            cancelled_before_start: AtomicU64::new(0),
            panicked: AtomicU64::new(0),
            queue_us: AtomicU64::new(0),
            work_us: AtomicU64::new(0),
        }))
    }
    fn snapshot(&self) -> Value {
        json!({"threads": self.workers.current_num_threads(), "capacity": self.capacity,
            "in_flight": self.capacity - self.permits.available_permits(),
            "admitted": self.admitted.load(Ordering::Relaxed), "rejected": self.rejected.load(Ordering::Relaxed),
            "queued_jobs": self.queued.load(Ordering::Relaxed), "running_jobs": self.running.load(Ordering::Relaxed),
            "completed_jobs": self.completed.load(Ordering::Relaxed),
            "cancelled_before_start": self.cancelled_before_start.load(Ordering::Relaxed),
            "panicked_jobs": self.panicked.load(Ordering::Relaxed),
            "total_queue_ms": self.queue_us.load(Ordering::Relaxed) as f64 / 1000.,
            "total_work_ms": self.work_us.load(Ordering::Relaxed) as f64 / 1000.})
    }
}

pub(crate) struct Serving {
    query: Arc<Pool>,
    ingest: Arc<Pool>,
    maintenance: Arc<Pool>,
}
impl Serving {
    pub fn new(
        query: PoolConfig,
        ingest: PoolConfig,
        maintenance: PoolConfig,
    ) -> Result<Arc<Self>, Box<dyn std::error::Error>> {
        Ok(Arc::new(Self {
            query: Pool::new(WorkClass::Query, query)?,
            ingest: Pool::new(WorkClass::Ingest, ingest)?,
            maintenance: Pool::new(WorkClass::Maintenance, maintenance)?,
        }))
    }
    pub fn snapshot(&self) -> Value {
        json!({"query": self.query.snapshot(), "ingest": self.ingest.snapshot(), "maintenance": self.maintenance.snapshot()})
    }
    fn pool(&self, class: WorkClass) -> Arc<Pool> {
        Arc::clone(match class {
            WorkClass::Query => &self.query,
            WorkClass::Ingest => &self.ingest,
            WorkClass::Maintenance => &self.maintenance,
        })
    }
    pub fn admit(&self, class: WorkClass) -> Result<AdmittedWork, Box<Response>> {
        let pool = self.pool(class);
        let Ok(permit) = Arc::clone(&pool.permits).try_acquire_owned() else {
            pool.rejected.fetch_add(1, Ordering::Relaxed);
            return Err(Box::new(
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    [(RETRY_AFTER, "1")],
                    Json(json!({"error":"server busy", "class":class.name()})),
                )
                    .into_response(),
            ));
        };
        pool.admitted.fetch_add(1, Ordering::Relaxed);
        Ok(AdmittedWork {
            pool,
            _permit: Arc::new(permit),
            timings: Arc::new(Timings::default()),
        })
    }
}

#[derive(Default)]
struct Timings {
    queue_us: AtomicU64,
    work_us: AtomicU64,
}

/// The CPU closure owns a clone, so a dropped HTTP future cannot release its slot.
#[derive(Clone)]
pub(crate) struct AdmittedWork {
    pool: Arc<Pool>,
    _permit: Arc<OwnedSemaphorePermit>,
    timings: Arc<Timings>,
}
impl AdmittedWork {
    pub async fn run<T, F>(&self, job: F) -> Result<T, IndexError>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T, IndexError> + Send + 'static,
    {
        let (sender, receiver) = oneshot::channel();
        let work = self.clone();
        let queued_at = Instant::now();
        self.pool.queued.fetch_add(1, Ordering::Relaxed);
        self.pool.workers.spawn_fifo(move || {
            // Explicitly capture the permit: disjoint closure capture would
            // otherwise move only the pool/timing fields and release capacity
            // when the awaiting HTTP future is dropped.
            let _permit_guard = work._permit;
            work.pool.queued.fetch_sub(1, Ordering::Relaxed);
            if sender.is_closed() {
                work.pool
                    .cancelled_before_start
                    .fetch_add(1, Ordering::Relaxed);
                return;
            }
            let queue_us = queued_at.elapsed().as_micros() as u64;
            work.timings.queue_us.fetch_add(queue_us, Ordering::Relaxed);
            work.pool.queue_us.fetch_add(queue_us, Ordering::Relaxed);
            work.pool.running.fetch_add(1, Ordering::Relaxed);
            let start = Instant::now();
            let result = catch_unwind(AssertUnwindSafe(job)).unwrap_or_else(|_| {
                work.pool.panicked.fetch_add(1, Ordering::Relaxed);
                Err(IndexError::Io(io::Error::other("service worker panicked")))
            });
            let work_us = start.elapsed().as_micros() as u64;
            work.timings.work_us.fetch_add(work_us, Ordering::Relaxed);
            work.pool.work_us.fetch_add(work_us, Ordering::Relaxed);
            work.pool.running.fetch_sub(1, Ordering::Relaxed);
            work.pool.completed.fetch_add(1, Ordering::Relaxed);
            let _ = sender.send(result);
        });
        receiver
            .await
            .map_err(|_| IndexError::Io(io::Error::other("service worker stopped")))?
    }
}

fn work_class(path: &str, method: &Method) -> Option<WorkClass> {
    match path {
        "/v1/vectors/upsert" | "/v1/vectors/delete" | "/v1/train" => Some(WorkClass::Ingest),
        "/v1/compact" | "/v1/dense/index" | "/v1/fde/index" => Some(WorkClass::Maintenance),
        "/v1/collections" if method == Method::POST => Some(WorkClass::Ingest),
        "/v1/collections"
        | "/v1/query"
        | "/v1/retrieve"
        | "/v1/plan"
        | "/v1/stats"
        | "/v1/debug/score"
        | "/v1/debug/candidates" => Some(WorkClass::Query),
        _ => None,
    }
}

pub(crate) async fn admission(
    State(serving): State<Arc<Serving>>,
    mut request: Request,
    next: Next,
) -> Response {
    let path = match crate::request_path::index_path(&mut request).await {
        Ok(path) => path,
        Err(response) => return *response,
    };
    request.extensions_mut().insert(Arc::clone(&serving));
    let Some(class) = work_class(&path, request.method()).or_else(|| {
        request
            .uri()
            .path()
            .starts_with("/v1/collections/")
            .then_some(WorkClass::Query)
    }) else {
        return next.run(request).await;
    };
    let work = match serving.admit(class) {
        Ok(work) => work,
        Err(response) => return *response,
    };
    let timings = Arc::clone(&work.timings);
    let started = Instant::now();
    request.extensions_mut().insert(work.clone());
    let mut response = next.run(request).await;
    // Headers include synchronous work only; wall time also includes parsing,
    // routing and response construction. JSON/API payloads remain unchanged.
    for (name, value) in [
        (
            "x-annex-queue-ms",
            timings.queue_us.load(Ordering::Relaxed) as f64 / 1000.,
        ),
        (
            "x-annex-work-ms",
            timings.work_us.load(Ordering::Relaxed) as f64 / 1000.,
        ),
        (
            "x-annex-request-ms",
            started.elapsed().as_secs_f64() * 1000.,
        ),
    ] {
        response
            .headers_mut()
            .insert(name, HeaderValue::from_str(&format!("{value:.3}")).unwrap());
    }
    drop(work);
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Extension, Router,
        body::Body,
        http::Request,
        routing::{get, post},
    };
    use std::{future::Future, task::Poll};
    use tower::ServiceExt;

    fn fixture(capacity: usize) -> Arc<Serving> {
        let config = PoolConfig {
            threads: 1,
            capacity,
        };
        Serving::new(config, config, config).unwrap()
    }

    #[tokio::test]
    async fn worker_classes_are_independent_and_nested_rayon_stays_in_pool() {
        let serving = fixture(1);
        let query = serving.admit(WorkClass::Query).unwrap();
        let (started_tx, started_rx) = oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let running = tokio::spawn(async move {
            query
                .run(move || {
                    started_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                    Ok(())
                })
                .await
        });
        started_rx.await.unwrap();
        assert!(serving.admit(WorkClass::Query).is_err());
        for class in [WorkClass::Ingest, WorkClass::Maintenance] {
            let work = serving.admit(class).unwrap();
            let name = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                work.run(|| {
                    let (name, workers) = rayon::join(
                        || std::thread::current().name().unwrap().to_owned(),
                        rayon::current_num_threads,
                    );
                    assert_eq!(workers, 1);
                    Ok(name)
                }),
            )
            .await
            .unwrap()
            .unwrap();
            assert!(name.starts_with(&format!("annex-{}-", class.name())));
        }
        release_tx.send(()).unwrap();
        running.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn cancelled_waiters_keep_capacity_until_worker_finishes_or_skips() {
        let serving = fixture(2);
        let first = serving.admit(WorkClass::Query).unwrap();
        let (started_tx, started_rx) = oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let running = tokio::spawn(async move {
            first
                .run(move || {
                    started_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                    Ok(())
                })
                .await
        });
        started_rx.await.unwrap();
        let second = serving.admit(WorkClass::Query).unwrap();
        let executed = Arc::new(AtomicUsize::new(0));
        let marker = Arc::clone(&executed);
        let mut waiting = Box::pin(second.run(move || {
            marker.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }));
        let poll = std::future::poll_fn(|cx| Poll::Ready(waiting.as_mut().poll(cx))).await;
        assert!(poll.is_pending());
        drop(waiting);
        drop(second);
        assert!(
            serving.admit(WorkClass::Query).is_err(),
            "queued closure retains its admission slot"
        );
        running.abort();
        let _ = running.await;
        assert!(
            serving.admit(WorkClass::Query).is_err(),
            "running closure retains its admission slot"
        );
        release_tx.send(()).unwrap();
        let (drained_tx, drained_rx) = oneshot::channel();
        serving.query.workers.spawn_fifo(move || {
            let _ = drained_tx.send(());
        });
        tokio::time::timeout(std::time::Duration::from_secs(5), drained_rx)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(executed.load(Ordering::Relaxed), 0);
        assert_eq!(
            serving.query.cancelled_before_start.load(Ordering::Relaxed),
            1
        );
        assert!(serving.admit(WorkClass::Query).is_ok());
    }

    #[tokio::test]
    async fn worker_panics_become_errors_and_release_capacity() {
        let serving = fixture(1);
        let work = serving.admit(WorkClass::Query).unwrap();
        let result = work
            .run(|| -> Result<(), IndexError> { panic!("expected test panic") })
            .await;
        assert!(matches!(result, Err(IndexError::Io(_))));
        drop(work);
        let (drained_tx, drained_rx) = oneshot::channel();
        serving.query.workers.spawn_fifo(move || {
            let _ = drained_tx.send(());
        });
        drained_rx.await.unwrap();
        assert_eq!(serving.query.panicked.load(Ordering::Relaxed), 1);
        assert!(serving.admit(WorkClass::Query).is_ok());
    }

    async fn tiny_job(
        Extension(work): Extension<AdmittedWork>,
    ) -> Result<Json<Value>, crate::ApiError> {
        work.run(|| Ok(Json(json!({"ok":true}))))
            .await
            .map_err(crate::ApiError)
    }

    #[tokio::test]
    async fn overload_precedes_body_parsing_and_health_stays_available() {
        let serving = fixture(1);
        let slot = serving.admit(WorkClass::Query).unwrap();
        let app = Router::new()
            .route("/v1/retrieve", post(|_: Json<Value>| async { "parsed" }))
            .route("/v1/vectors/upsert", post(tiny_job))
            .route("/healthz", get(|| async { "ok" }))
            .route(
                "/v1/runtime",
                get(|Extension(s): Extension<Arc<Serving>>| async move { Json(s.snapshot()) }),
            )
            .layer(axum::middleware::from_fn_with_state(
                Arc::clone(&serving),
                admission,
            ));
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/retrieve")
                    .header("content-type", "application/json")
                    .body(Body::from("not JSON"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers()[RETRY_AFTER], "1");
        for route in ["/healthz", "/v1/runtime"] {
            let response = app
                .clone()
                .oneshot(Request::builder().uri(route).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/vectors/upsert")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        for name in ["x-annex-queue-ms", "x-annex-work-ms", "x-annex-request-ms"] {
            assert!(
                response.headers()[name]
                    .to_str()
                    .unwrap()
                    .parse::<f64>()
                    .unwrap()
                    >= 0.
            );
        }
        drop(slot);
    }

    #[tokio::test]
    async fn collection_paths_share_global_limits_after_decoding() {
        let serving = fixture(1);
        let _slot = serving.admit(WorkClass::Maintenance).unwrap();
        let app = Router::new()
            .route("/v1/compact", post(tiny_job))
            .route("/v1/collections/{name}/{*operation}", post(tiny_job))
            .layer(axum::middleware::from_fn_with_state(serving, admission));
        for path in [
            "/v1/compact",
            "/v1/collections/a/compact",
            "/v1/collections/b/%63ompact",
            "/v1/collections/b/dense%2Findex",
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(path)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                StatusCode::SERVICE_UNAVAILABLE,
                "path={path}"
            );
        }
    }
}
