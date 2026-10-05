//! Use Axum's decoded route parameters consistently for policy decisions.
use axum::{
    RequestExt,
    extract::{Path, Request},
    response::{IntoResponse, Response},
};
use std::collections::HashMap;

pub(crate) async fn index_path(request: &mut Request) -> Result<String, Box<Response>> {
    if request.uri().path().starts_with("/v1/collections/") {
        let Path(parameters) = request
            .extract_parts::<Path<HashMap<String, String>>>()
            .await
            .map_err(|error| Box::new(error.into_response()))?;
        if let Some(operation) = parameters.get("operation") {
            return Ok(format!("/v1/{operation}"));
        }
    }
    Ok(request.uri().path().to_owned())
}
