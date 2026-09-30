//! Planner regret benchmark framework.
//!
//! Enumerates feasible retrieval plans for a query, scores each against an
//! exact-oracle baseline, builds the quality/latency Pareto frontier, and
//! computes regret of ANNex's chosen plan against that frontier.

use super::*;

// ── Core metrics ──────────────────────────────────────────────────────────────

/// Recall@k: fraction of oracle's top-k found in the candidate list.
pub fn recall_at_k(oracle: &[String], candidate: &[String], k: usize) -> f32 {
    let oracle_set: HashSet<&str> = oracle.iter().take(k).map(String::as_str).collect();
    let found = candidate
        .iter()
        .take(k)
        .filter(|id| oracle_set.contains(id.as_str()))
        .count();
    if oracle_set.is_empty() {
        1.0
    } else {
        found as f32 / oracle_set.len() as f32
    }
}

// ── Feasible plan result ─────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct PlanEvaluation {
    pub description: String,
    pub recall: f32,
    pub latency_ms: f64,
    pub is_oracle: bool,
}

impl PlanEvaluation {
    pub fn dominates(&self, other: &PlanEvaluation) -> bool {
        self.recall >= other.recall && self.latency_ms <= other.latency_ms
            && (self.recall > other.recall || self.latency_ms < other.latency_ms)
    }
}

// ── Pareto frontier ───────────────────────────────────────────────────────────

/// Returns only non-dominated plans (Pareto-optimal in recall × latency).
pub fn pareto_frontier(plans: &[PlanEvaluation]) -> Vec<&PlanEvaluation> {
    plans
        .iter()
        .filter(|p| !plans.iter().any(|other| other.dominates(p)))
        .collect()
}

// ── Regret report ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct RegretReport {
    /// Oracle quality (recall against exhaustive exact scan).
    pub oracle_recall: f32,
    /// ANNex planner's chosen plan quality.
    pub planner_recall: f32,
    /// oracle_recall − planner_recall (positive = planner underperforms).
    pub quality_regret: f32,
    pub planner_latency_ms: f64,
    /// Whether planner's plan is Pareto-optimal.
    pub on_pareto_frontier: bool,
    /// Within ε_quality=0.01 and ε_latency=0.05 of frontier.
    pub eps_pareto_hit: bool,
    /// Whether planner's latency exceeded the stated budget.
    pub budget_violated: bool,
}

const EPS_QUALITY: f32 = 0.01;
const EPS_LATENCY: f64 = 0.05;

pub fn compute_regret(
    oracle: &PlanEvaluation,
    planner: &PlanEvaluation,
    frontier: &[&PlanEvaluation],
    latency_budget_ms: Option<f64>,
) -> RegretReport {
    let quality_regret = (oracle.recall - planner.recall).max(0.0);
    let on_pareto_frontier = frontier.iter().any(|p| p.description == planner.description);
    let eps_pareto_hit = frontier.iter().any(|frontier_plan| {
        planner.recall >= frontier_plan.recall - EPS_QUALITY
            && planner.latency_ms <= frontier_plan.latency_ms * (1.0 + EPS_LATENCY)
    });
    let budget_violated = latency_budget_ms
        .is_some_and(|budget| planner.latency_ms > budget);
    RegretReport {
        oracle_recall: oracle.recall,
        planner_recall: planner.recall,
        quality_regret,
        planner_latency_ms: planner.latency_ms,
        on_pareto_frontier,
        eps_pareto_hit,
        budget_violated,
    }
}

// ── Plan enumerator ───────────────────────────────────────────────────────────

/// Configuration for feasible plan enumeration.
pub struct EnumerationConfig {
    pub ef_search_values: Vec<usize>,
    pub result_limit: usize,
}

impl Default for EnumerationConfig {
    fn default() -> Self {
        Self {
            ef_search_values: vec![64, 256, 1024],
            result_limit: 10,
        }
    }
}

/// Run a single plan and time it, returning a `PlanEvaluation` against the oracle.
pub fn evaluate_plan(
    index: &MultiVectorIndex,
    request: &RetrieveRequest,
    oracle_ids: &[String],
    description: impl Into<String>,
    is_oracle: bool,
) -> Result<PlanEvaluation, IndexError> {
    let started = std::time::Instant::now();
    let response = index.retrieve(request)?;
    let latency_ms = started.elapsed().as_secs_f64() * 1000.;
    let candidate_ids: Vec<String> = response.matches.iter().map(|h| h.id.clone()).collect();
    let recall = recall_at_k(oracle_ids, &candidate_ids, oracle_ids.len());
    Ok(PlanEvaluation {
        description: description.into(),
        recall,
        latency_ms,
        is_oracle,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(slice: &[&str]) -> Vec<String> {
        slice.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn recall_is_one_against_identical_list() {
        let oracle = ids(&["a", "b", "c"]);
        assert_eq!(recall_at_k(&oracle, &oracle, 3), 1.0);
    }

    #[test]
    fn recall_measures_partial_overlap() {
        let oracle = ids(&["a", "b", "c", "d"]);
        let candidate = ids(&["a", "b", "x", "y"]);
        assert!((recall_at_k(&oracle, &candidate, 4) - 0.5).abs() < 0.01);
    }

    #[test]
    fn recall_handles_empty_oracle() {
        assert_eq!(recall_at_k(&[], &ids(&["a"]), 3), 1.0);
    }

    #[test]
    fn pareto_removes_dominated_plans() {
        let plans = vec![
            PlanEvaluation { description: "a".into(), recall: 0.9, latency_ms: 5.0, is_oracle: false },
            PlanEvaluation { description: "b".into(), recall: 0.8, latency_ms: 8.0, is_oracle: false }, // dominated by a
            PlanEvaluation { description: "c".into(), recall: 0.7, latency_ms: 2.0, is_oracle: false }, // not dominated (faster)
        ];
        let frontier = pareto_frontier(&plans);
        let names: Vec<&str> = frontier.iter().map(|p| p.description.as_str()).collect();
        assert!(names.contains(&"a"), "a should be on frontier");
        assert!(names.contains(&"c"), "c should be on frontier (faster)");
        assert!(!names.contains(&"b"), "b should be dominated");
    }

    #[test]
    fn eps_pareto_hit_when_within_tolerance() {
        let oracle = PlanEvaluation {
            description: "oracle".into(), recall: 1.0, latency_ms: 10.0, is_oracle: true,
        };
        // Frontier plan: recall 0.95, latency 8ms
        let frontier_plan = PlanEvaluation {
            description: "fast".into(), recall: 0.95, latency_ms: 8.0, is_oracle: false,
        };
        let frontier = vec![&frontier_plan];
        // Planner: recall 0.945 (within 0.01 of 0.95), latency 8.3ms (within 5% of 8ms)
        let planner = PlanEvaluation {
            description: "planner".into(), recall: 0.945, latency_ms: 8.3, is_oracle: false,
        };
        let report = compute_regret(&oracle, &planner, &frontier, None);
        assert!(report.eps_pareto_hit, "should be within ε tolerance");
    }

    #[test]
    fn eps_pareto_miss_when_outside_tolerance() {
        let oracle = PlanEvaluation {
            description: "oracle".into(), recall: 1.0, latency_ms: 10.0, is_oracle: true,
        };
        let frontier_plan = PlanEvaluation {
            description: "fast".into(), recall: 0.95, latency_ms: 8.0, is_oracle: false,
        };
        let frontier = vec![&frontier_plan];
        // Planner: recall 0.90 (>0.01 below 0.95)
        let planner = PlanEvaluation {
            description: "planner".into(), recall: 0.90, latency_ms: 8.3, is_oracle: false,
        };
        let report = compute_regret(&oracle, &planner, &frontier, None);
        assert!(!report.eps_pareto_hit, "recall gap too large for ε-Pareto");
    }

    #[test]
    fn quality_regret_is_oracle_minus_planner() {
        let oracle = PlanEvaluation {
            description: "oracle".into(), recall: 1.0, latency_ms: 20.0, is_oracle: true,
        };
        let frontier_plan = PlanEvaluation {
            description: "fp".into(), recall: 1.0, latency_ms: 20.0, is_oracle: false,
        };
        let planner = PlanEvaluation {
            description: "planner".into(), recall: 0.8, latency_ms: 5.0, is_oracle: false,
        };
        let report = compute_regret(&oracle, &planner, &[&frontier_plan], None);
        assert!((report.quality_regret - 0.2).abs() < 0.001);
    }

    #[test]
    fn budget_violation_when_latency_exceeds_budget() {
        let oracle = PlanEvaluation {
            description: "o".into(), recall: 1.0, latency_ms: 5.0, is_oracle: true,
        };
        let planner = PlanEvaluation {
            description: "p".into(), recall: 0.95, latency_ms: 12.0, is_oracle: false,
        };
        let report = compute_regret(&oracle, &planner, &[], Some(10.0));
        assert!(report.budget_violated);
    }

    #[test]
    fn no_budget_violation_when_within_budget() {
        let oracle = PlanEvaluation {
            description: "o".into(), recall: 1.0, latency_ms: 5.0, is_oracle: true,
        };
        let planner = PlanEvaluation {
            description: "p".into(), recall: 0.95, latency_ms: 8.0, is_oracle: false,
        };
        let report = compute_regret(&oracle, &planner, &[], Some(10.0));
        assert!(!report.budget_violated);
    }
}
