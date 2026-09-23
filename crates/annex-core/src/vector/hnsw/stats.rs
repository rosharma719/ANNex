use std::cell::RefCell;

use serde::Serialize;

use super::config::unfiltered_log_chunk;

thread_local! {
    pub(crate) static FILTER_EDGE_TOTAL_KEYS: RefCell<usize> = RefCell::new(0);
}
thread_local! {
    pub(crate) static UNFILTERED_SEARCH_AGG: RefCell<UnfilteredSearchAgg> = RefCell::new(UnfilteredSearchAgg::default());
}
thread_local! {
    pub(crate) static FILTER_SEED_COUNT: RefCell<usize> = RefCell::new(0);
}
thread_local! {
    pub(crate) static FILTER_EDGE_STATS: RefCell<FilterEdgeAgg> = RefCell::new(FilterEdgeAgg::default());
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SearchStats {
    pub ef_search: usize,
    pub visited: usize,
    pub expanded: usize,
    pub best_score: f32,
    pub worst_score: f32,
    pub adjacency_reads: usize,
    pub distance_computations: usize,
    pub cap_breaks: usize,
    pub patience_breaks: usize,
    pub exact: bool,
}

#[derive(Default)]
pub(crate) struct UnfilteredSearchAgg {
    samples: Vec<UnfilteredSample>,
}

#[derive(Clone)]
pub(crate) struct UnfilteredSample {
    pub(crate) ef_search: usize,
    pub(crate) visited: usize,
    pub(crate) expanded: usize,
    pub(crate) best_score: f32,
    pub(crate) worst_score: f32,
}

#[derive(Default)]
pub(crate) struct SearchLayerStats {
    pub(crate) visited: usize,
    pub(crate) expanded: usize,
    pub(crate) adjacency_reads: usize,
    pub(crate) distance_computations: usize,
    pub(crate) cap_breaks: usize,
    pub(crate) patience_breaks: usize,
}

#[derive(Serialize)]
pub(crate) struct FilterSearchLogEntry {
    pub(crate) seq: u64,
    pub(crate) ef_search: usize,
    pub(crate) top_k: usize,
    pub(crate) filter_present: bool,
    pub(crate) patience_limit: usize,
    pub(crate) early_exit: bool,
    pub(crate) seeds_pool: usize,
    pub(crate) seeds_added: usize,
    pub(crate) seeds_accepted: usize,
    pub(crate) seeds_in_results: usize,
    pub(crate) seeds_popped: usize,
    pub(crate) filter_checked: usize,
    pub(crate) filter_passed: usize,
    pub(crate) visited: usize,
    pub(crate) expansions: usize,
    pub(crate) max_expansions: usize,
    pub(crate) results_len: usize,
    pub(crate) stop_reason: String,
    pub(crate) elapsed_ms: f64,
    pub(crate) routing_popped_total: usize,
    pub(crate) routing_popped_passing: usize,
    pub(crate) routing_popped_failing: usize,
    pub(crate) results_inserted: usize,
    pub(crate) results_pq_peek_dist: f32,
    pub(crate) best_routing_dist_at_exit: f32,
}

#[derive(Default)]
pub(crate) struct FilterEdgeAgg {
    pub(crate) count: usize,
    // Buckets: Bool, Str, Int, Float
    pub(crate) ns_by_type: [u128; 4],
    pub(crate) samples: usize,
    pub(crate) scored: usize,
    pub(crate) added: usize,
}

impl UnfilteredSearchAgg {
    pub(crate) fn record(&mut self, sample: UnfilteredSample) {
        self.samples.push(sample);
        let chunk = unfiltered_log_chunk();
        if self.samples.len() >= chunk {
            self.flush();
        }
    }

    pub(crate) fn flush(&mut self) {
        if self.samples.is_empty() {
            return;
        }

        let mut visited: Vec<f64> = self.samples.iter().map(|s| s.visited as f64).collect();
        let mut expanded: Vec<f64> = self.samples.iter().map(|s| s.expanded as f64).collect();
        let mut util_pct: Vec<f64> = self
            .samples
            .iter()
            .map(|s| (s.expanded as f64 / s.ef_search.max(1) as f64) * 100.0)
            .collect();
        let mut best: Vec<f64> = self.samples.iter().map(|s| s.best_score as f64).collect();
        let mut worst: Vec<f64> = self.samples.iter().map(|s| s.worst_score as f64).collect();
        let ef_min = self.samples.iter().map(|s| s.ef_search).min().unwrap_or(0);
        let ef_max = self.samples.iter().map(|s| s.ef_search).max().unwrap_or(0);

        let stats = |vals: &mut Vec<f64>| {
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p = |pct: f64| -> f64 {
                if vals.is_empty() {
                    return 0.0;
                }
                let idx = ((pct / 100.0) * (vals.len() as f64 - 1.0)).round() as usize;
                vals[idx]
            };
            (
                p(50.0),
                p(90.0),
                p(99.0),
                *vals.first().unwrap_or(&0.0),
                *vals.last().unwrap_or(&0.0),
            )
        };

        let (vis_p50, vis_p90, vis_p99, vis_min, vis_max) = stats(&mut visited);
        let (exp_p50, exp_p90, exp_p99, exp_min, exp_max) = stats(&mut expanded);
        let (util_p50, util_p90, util_p99, util_min, util_max) = stats(&mut util_pct);
        let (best_p50, best_p90, best_p99, best_min, best_max) = stats(&mut best);
        let (worst_p50, worst_p90, worst_p99, worst_min, worst_max) = stats(&mut worst);

        log::info!(
            target: "vector::hnsw",
            "[unfiltered_search_stats] n={} ef_search={}..{} visited(min/p50/p90/p99/max)={:.0}/{:.0}/{:.0}/{:.0}/{:.0} expanded={:.0}/{:.0}/{:.0}/{:.0}/{:.0} util%={:.1}/{:.1}/{:.1}/{:.1}/{:.1} best_score={:.4}/{:.4}/{:.4}/{:.4}/{:.4} worst_score={:.4}/{:.4}/{:.4}/{:.4}/{:.4}",
            self.samples.len(),
            ef_min,
            ef_max,
            vis_min,
            vis_p50,
            vis_p90,
            vis_p99,
            vis_max,
            exp_min,
            exp_p50,
            exp_p90,
            exp_p99,
            exp_max,
            util_min,
            util_p50,
            util_p90,
            util_p99,
            util_max,
            best_min,
            best_p50,
            best_p90,
            best_p99,
            best_max,
            worst_min,
            worst_p50,
            worst_p90,
            worst_p99,
            worst_max
        );

        self.samples.clear();
    }
}

impl Drop for UnfilteredSearchAgg {
    fn drop(&mut self) {
        self.flush();
    }
}
