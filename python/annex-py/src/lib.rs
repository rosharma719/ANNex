use annex::segment::Segment;
use annex::vector::hnsw::{HNSWIndex, SearchRuntimeOptions};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
    ndarray::Array2,
};
use pyo3::exceptions::{PyOverflowError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

type PySearchResult<'py> = PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<f32>>)>;
type PyBatchSearchResult<'py> = PyResult<(Bound<'py, PyArray2<u64>>, Bound<'py, PyArray2<f32>>)>;

const MAX_BATCH_OUTPUT_BYTES: usize = 1 << 30;

fn runtime_error<E: std::fmt::Display>(error: E) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

#[inline]
fn opts(
    k: usize,
    ef: usize,
    sq8_screen: bool,
    scan_cap: usize,
    patience: usize,
) -> SearchRuntimeOptions {
    SearchRuntimeOptions {
        ef_search: Some(ef.max(k)),
        neighbor_scan_cap_level0: Some(scan_cap),
        early_exit_patience: Some(patience),
        sq8_screen: Some(sq8_screen),
        ..Default::default()
    }
}

fn validate_values(values: &[f32], label: &str) -> PyResult<()> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(value_error(format!(
            "{label} contains a non-finite value at flat index {index}"
        )));
    }
    Ok(())
}

fn validate_batch_output(rows: usize, columns: usize) -> PyResult<usize> {
    if columns > isize::MAX as usize {
        return Err(PyOverflowError::new_err(
            "batch output column count exceeds NumPy's platform limit",
        ));
    }
    let elements = rows
        .checked_mul(columns)
        .ok_or_else(|| PyOverflowError::new_err("batch output shape overflows usize"))?;
    let id_bytes = elements
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or_else(|| PyOverflowError::new_err("batch ID output size overflows usize"))?;
    let score_bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| PyOverflowError::new_err("batch score output size overflows usize"))?;
    if id_bytes > isize::MAX as usize || score_bytes > isize::MAX as usize {
        return Err(PyOverflowError::new_err(
            "batch output exceeds NumPy's platform limit",
        ));
    }
    let total_bytes = id_bytes
        .checked_add(score_bytes)
        .ok_or_else(|| PyOverflowError::new_err("combined batch output size overflows usize"))?;
    if total_bytes > MAX_BATCH_OUTPUT_BYTES {
        return Err(PyOverflowError::new_err(format!(
            "batch output requires {total_bytes} bytes; limit is {MAX_BATCH_OUTPUT_BYTES}"
        )));
    }
    Ok(elements)
}

fn empty_batch<'py>(py: Python<'py>, rows: usize, columns: usize) -> PyBatchSearchResult<'py> {
    batch_arrays(py, rows, columns, Vec::new(), Vec::new())
}

fn batch_arrays<'py>(
    py: Python<'py>,
    rows: usize,
    columns: usize,
    ids: Vec<u64>,
    scores: Vec<f32>,
) -> PyBatchSearchResult<'py> {
    let ids = Array2::from_shape_vec((rows, columns), ids).map_err(runtime_error)?;
    let scores = Array2::from_shape_vec((rows, columns), scores).map_err(runtime_error)?;
    Ok((ids.into_pyarray(py), scores.into_pyarray(py)))
}

/// Thread-local reusable query buffer because ANNex currently accepts `&Vec<f32>`.
fn with_query_buf<R>(query: &[f32], f: impl FnOnce(&Vec<f32>) -> R) -> R {
    thread_local! {
        static BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    BUF.with(|cell| {
        let mut buffer = cell.borrow_mut();
        buffer.clear();
        buffer.extend_from_slice(query);
        f(&buffer)
    })
}

/// In-memory ANN index loaded from an ANNex segment snapshot.
///
/// Search calls release the GIL. Query data is copied into Rust-owned memory
/// before release so native search never aliases Python-owned mutable storage.
#[pyclass]
struct Index {
    inner: HNSWIndex,
}

#[pymethods]
impl Index {
    /// Load a segment snapshot from `path`. When `quantize` is true, build the
    /// SQ8 codes required by the screening fast path.
    #[new]
    #[pyo3(signature = (path, quantize = false))]
    fn new(path: &str, quantize: bool) -> PyResult<Self> {
        let segment = Segment::load_from_path(path).map_err(runtime_error)?;
        let mut inner = HNSWIndex::from_snapshot(segment.hnsw().to_snapshot());
        if inner.dim() == 0 {
            return Err(value_error("the loaded index has zero dimensions"));
        }
        if quantize {
            inner.quantize_all();
        }
        Ok(Self { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Search one query. Returns `(ids uint64[k], scores float32[k])`.
    ///
    /// `scores` are raw ANNex scores. `scan_cap=0` means uncapped L0 scans.
    #[pyo3(signature = (query, k = 20, ef = 128, sq8_screen = false, scan_cap = 0, patience = 2))]
    #[allow(clippy::too_many_arguments)] // Flat keyword arguments are the public Python API.
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<'py, f32>,
        k: usize,
        ef: usize,
        sq8_screen: bool,
        scan_cap: usize,
        patience: usize,
    ) -> PySearchResult<'py> {
        let query = query.as_slice()?;
        if query.len() != self.inner.dim() {
            return Err(value_error(format!(
                "query dimension mismatch: expected {}, got {}",
                self.inner.dim(),
                query.len()
            )));
        }
        validate_values(query, "query")?;

        // Own all Python-backed input before releasing the GIL.
        let query = query.to_vec();
        if k == 0 {
            return Ok((
                PyArray1::from_vec(py, Vec::new()),
                PyArray1::from_vec(py, Vec::new()),
            ));
        }

        let search_k = k.min(self.inner.len());
        let search_ef = ef.max(search_k).min(self.inner.len());
        let options = opts(search_k, search_ef, sq8_screen, scan_cap, patience);
        let results = py
            .detach(|| self.inner.search_with_options(&query, search_k, &options))
            .map_err(runtime_error)?;
        let ids = results.iter().map(|result| result.id).collect();
        let scores = results.iter().map(|result| result.raw_score).collect();
        Ok((PyArray1::from_vec(py, ids), PyArray1::from_vec(py, scores)))
    }

    /// Search contiguous float32 queries with shape `[n, dim]`.
    ///
    /// Returns `(ids uint64[n, k], scores float32[n, k])`, padding short rows
    /// with `uint64::MAX` and `NaN`. Native workers are capped to the machine's
    /// available parallelism; `threads=0` and `threads=1` run sequentially.
    #[pyo3(signature = (queries, k = 20, ef = 128, sq8_screen = false, scan_cap = 0, patience = 2, threads = 1))]
    #[allow(clippy::too_many_arguments)] // Flat keyword arguments are the public Python API.
    fn search_batch<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<'py, f32>,
        k: usize,
        ef: usize,
        sq8_screen: bool,
        scan_cap: usize,
        patience: usize,
        threads: usize,
    ) -> PyBatchSearchResult<'py> {
        let shape = queries.shape();
        let rows = shape[0];
        let dimensions = shape[1];
        if dimensions != self.inner.dim() {
            return Err(value_error(format!(
                "query dimension mismatch: expected {}, got {}",
                self.inner.dim(),
                dimensions
            )));
        }

        // NumPy's contiguous slice also accepts Fortran order. The batch loop
        // indexes rows, so accepting that layout would silently mix queries.
        if !queries.is_c_contiguous() {
            return Err(value_error("queries must be C-contiguous (row-major)"));
        }
        let queries = queries.as_slice()?;
        validate_values(queries, "queries")?;
        let output_elements = validate_batch_output(rows, k)?;
        if rows == 0 || k == 0 {
            return empty_batch(py, rows, k);
        }

        // Own all Python-backed input before releasing the GIL.
        let queries = queries.to_vec();
        let search_k = k.min(self.inner.len());
        let search_ef = ef.max(search_k).min(self.inner.len());
        let options = opts(search_k, search_ef, sq8_screen, scan_cap, patience);
        let available_threads = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        let worker_count = threads.max(1).min(rows).min(available_threads);

        let run = |range: std::ops::Range<usize>| -> Result<(Vec<u64>, Vec<f32>), String> {
            let worker_elements = range
                .len()
                .checked_mul(k)
                .ok_or_else(|| "batch worker output shape overflowed usize".to_owned())?;
            let mut ids = Vec::with_capacity(worker_elements);
            let mut scores = Vec::with_capacity(worker_elements);
            for row in range {
                let offset = row * dimensions;
                let query = &queries[offset..offset + dimensions];
                let results = with_query_buf(query, |buffer| {
                    self.inner.search_with_options(buffer, search_k, &options)
                })
                .map_err(|error| error.to_string())?;
                let result_count = results.len().min(k);
                ids.extend(results.iter().take(k).map(|result| result.id));
                scores.extend(results.iter().take(k).map(|result| result.raw_score));
                ids.resize(ids.len() + (k - result_count), u64::MAX);
                scores.resize(scores.len() + (k - result_count), f32::NAN);
            }
            Ok((ids, scores))
        };

        let (ids, scores) = py
            .detach(|| {
                if worker_count == 1 {
                    return run(0..rows);
                }
                std::thread::scope(|scope| {
                    let handles: Vec<_> = (0..worker_count)
                        .map(|worker| {
                            // Balanced boundaries are monotonic and non-empty because
                            // worker_count <= rows, including skewed n/thread ratios.
                            let start = worker * rows / worker_count;
                            let end = (worker + 1) * rows / worker_count;
                            scope.spawn(move || run(start..end))
                        })
                        .collect();

                    let mut ids = Vec::with_capacity(output_elements);
                    let mut scores = Vec::with_capacity(output_elements);
                    for handle in handles {
                        let (mut worker_ids, mut worker_scores) = handle
                            .join()
                            .map_err(|_| "ANNex batch worker panicked".to_owned())??;
                        ids.append(&mut worker_ids);
                        scores.append(&mut worker_scores);
                    }
                    Ok((ids, scores))
                })
            })
            .map_err(runtime_error)?;

        batch_arrays(py, rows, k, ids, scores)
    }
}

#[pymodule]
fn annex_py(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Index>()?;
    Ok(())
}
