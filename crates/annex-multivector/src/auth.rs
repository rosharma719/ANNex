use axum::{
    body::Body,
    http::{HeaderMap, Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use std::sync::Arc;

const WRITE_PATHS: &[&str] = &[
    "/v1/vectors/upsert",
    "/v1/vectors/delete",
    "/v1/train",
    "/v1/fde/index",
    "/v1/dense/index",
    "/v1/compact",
    "/v1/collections",
];

fn is_write_route(path: &str, method: &axum::http::Method) -> bool {
    use axum::http::Method;
    if path == "/v1/collections" {
        return method == Method::POST;
    }
    let stripped = if let Some(rest) = path.strip_prefix("/v1/collections/") {
        rest.split_once('/').map(|(_, op)| format!("/v1/{op}")).unwrap_or_default()
    } else {
        path.to_owned()
    };
    WRITE_PATHS.iter().any(|&w| stripped == w || path == w)
}

fn bearer(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
}

fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(json!({"error": "authentication required"})),
    )
        .into_response()
}

fn forbidden() -> Response {
    (
        StatusCode::FORBIDDEN,
        Json(json!({"error": "invalid key"})),
    )
        .into_response()
}

#[derive(Clone)]
pub struct AuthConfig {
    pub read_key: Option<String>,
    pub write_key: Option<String>,
}

pub async fn auth_middleware(
    axum::extract::State(config): axum::extract::State<Arc<AuthConfig>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    if config.read_key.is_none() && config.write_key.is_none() {
        return next.run(request).await;
    }

    let write = is_write_route(request.uri().path(), request.method());
    let token = bearer(request.headers());

    if write {
        match (&config.write_key, token) {
            (None, _) => return forbidden(),
            (Some(wk), Some(t)) if t == wk => {}
            (Some(_), Some(_)) => return forbidden(),
            (Some(_), None) => return unauthorized(),
        }
    } else {
        let valid_read = config
            .read_key
            .as_deref()
            .zip(token)
            .map(|(rk, t)| t == rk)
            .unwrap_or(false);
        let valid_write = config
            .write_key
            .as_deref()
            .zip(token)
            .map(|(wk, t)| t == wk)
            .unwrap_or(false);

        if !valid_read && !valid_write {
            return if token.is_some() {
                forbidden()
            } else {
                unauthorized()
            };
        }
    }

    next.run(request).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, body::Body, http::Request, middleware, routing::get};
    use tower::ServiceExt;

    async fn ok_handler() -> &'static str { "ok" }

    fn app(config: AuthConfig) -> Router {
        Router::new()
            .route("/healthz", get(ok_handler))
            .route("/v1/retrieve", get(ok_handler))
            .route("/v1/vectors/upsert", get(ok_handler))
            .layer(middleware::from_fn_with_state(Arc::new(config), auth_middleware))
    }

    #[tokio::test]
    async fn open_server_passes_all_routes() {
        let config = AuthConfig { read_key: None, write_key: None };
        let response = app(config)
            .oneshot(Request::builder().uri("/v1/vectors/upsert").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
    }

    #[tokio::test]
    async fn write_key_accepted_on_read_route() {
        let config = AuthConfig {
            read_key: None,
            write_key: Some("wk-secret".into()),
        };
        let response = app(config)
            .oneshot(
                Request::builder()
                    .uri("/v1/retrieve")
                    .header("authorization", "Bearer wk-secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
    }

    #[tokio::test]
    async fn read_key_rejected_on_write_route() {
        let config = AuthConfig {
            read_key: Some("rk-reader".into()),
            write_key: Some("wk-secret".into()),
        };
        let response = app(config)
            .oneshot(
                Request::builder()
                    .uri("/v1/vectors/upsert")
                    .header("authorization", "Bearer rk-reader")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 403);
    }

    #[tokio::test]
    async fn missing_key_returns_401() {
        let config = AuthConfig {
            read_key: Some("rk-reader".into()),
            write_key: Some("wk-secret".into()),
        };
        let response = app(config)
            .oneshot(Request::builder().uri("/v1/retrieve").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), 401);
    }

    #[tokio::test]
    async fn wrong_key_returns_403() {
        let config = AuthConfig {
            read_key: Some("rk-reader".into()),
            write_key: Some("wk-secret".into()),
        };
        let response = app(config)
            .oneshot(
                Request::builder()
                    .uri("/v1/retrieve")
                    .header("authorization", "Bearer wrong")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 403);
    }

    #[tokio::test]
    async fn write_key_accepted_on_write_route() {
        let config = AuthConfig {
            read_key: Some("rk-reader".into()),
            write_key: Some("wk-secret".into()),
        };
        let response = app(config)
            .oneshot(
                Request::builder()
                    .uri("/v1/vectors/upsert")
                    .header("authorization", "Bearer wk-secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
    }
}
