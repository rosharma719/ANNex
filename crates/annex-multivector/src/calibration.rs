use super::*;

const MIN_OBSERVATIONS: u64 = 5;
const ALPHA: f64 = 0.2;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(tag = "kind", content = "operator", rename_all = "snake_case")]
pub enum CalibrationTarget {
    Channel(PhysicalOperator),
    Fusion(FusionOperator),
    RerankMaxsim,
    Context(ContextOperator),
}

impl CalibrationTarget {
    fn sort_key(self) -> String {
        match self {
            Self::Channel(operator) => format!("channel/{}", operator.as_str()),
            Self::Fusion(operator) => format!("fusion/{operator:?}"),
            Self::RerankMaxsim => "rerank/maxsim".into(),
            Self::Context(operator) => format!("context/{operator:?}"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
pub struct CalibrationKey {
    pub target: CalibrationTarget,
    pub dimension_bucket: usize,
    pub corpus_bucket: usize,
    pub selectivity_bucket: u8,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CalibrationEntry {
    pub key: CalibrationKey,
    pub observations: u64,
    pub mean_ms_per_cost_unit: f64,
    pub p90_ms_per_cost_unit: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct CalibrationSnapshot {
    pub entries: Vec<CalibrationEntry>,
}

#[derive(Clone, Debug, Default)]
struct RunningCalibration {
    observations: u64,
    mean: f64,
    variance: f64,
}

#[derive(Debug, Default)]
pub(super) struct CalibrationStats {
    entries: HashMap<CalibrationKey, RunningCalibration>,
}

impl CalibrationStats {
    pub(super) fn observe(&mut self, key: CalibrationKey, cost_units: f64, elapsed_ms: f64) {
        if !cost_units.is_finite()
            || cost_units <= 0.0
            || !elapsed_ms.is_finite()
            || elapsed_ms < 0.0
        {
            return;
        }
        let sample = elapsed_ms / cost_units;
        let entry = self.entries.entry(key).or_default();
        entry.observations += 1;
        if entry.observations == 1 {
            entry.mean = sample;
            return;
        }
        let delta = sample - entry.mean;
        entry.mean += ALPHA * delta;
        entry.variance = (1.0 - ALPHA) * (entry.variance + ALPHA * delta * delta);
    }

    pub(super) fn snapshot(&self) -> CalibrationSnapshot {
        let mut entries = self
            .entries
            .iter()
            .map(|(&key, value)| CalibrationEntry {
                key,
                observations: value.observations,
                mean_ms_per_cost_unit: value.mean,
                p90_ms_per_cost_unit: value.mean + 1.282 * value.variance.sqrt(),
            })
            .collect::<Vec<_>>();
        entries.sort_by_key(|entry| {
            (
                entry.key.target.sort_key(),
                entry.key.dimension_bucket,
                entry.key.corpus_bucket,
                entry.key.selectivity_bucket,
            )
        });
        CalibrationSnapshot { entries }
    }
}

impl CalibrationSnapshot {
    pub(super) fn estimate_ms(&self, key: CalibrationKey, cost_units: f64) -> Option<f64> {
        let entry = self
            .entries
            .iter()
            .find(|entry| entry.key == key && entry.observations >= MIN_OBSERVATIONS)?;
        Some(entry.p90_ms_per_cost_unit * cost_units)
    }
}

pub(super) fn key(
    target: CalibrationTarget,
    dimension: usize,
    documents: usize,
    selectivity: Option<f32>,
) -> CalibrationKey {
    CalibrationKey {
        target,
        dimension_bucket: bucket(dimension, &[128, 256, 512, 1_024, 2_048]),
        corpus_bucket: bucket(documents, &[1_000, 10_000, 100_000, 1_000_000]),
        selectivity_bucket: match selectivity.unwrap_or(1.0) {
            value if value <= 0.01 => 1,
            value if value <= 0.05 => 5,
            value if value <= 0.2 => 20,
            value if value <= 0.5 => 50,
            _ => 100,
        },
    }
}

fn bucket(value: usize, boundaries: &[usize]) -> usize {
    boundaries
        .iter()
        .copied()
        .find(|boundary| value <= *boundary)
        .unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_is_gated_and_stratified() {
        let dense = key(
            CalibrationTarget::Channel(PhysicalOperator::ExactDense),
            128,
            10_000,
            Some(0.1),
        );
        let filtered = key(
            CalibrationTarget::Channel(PhysicalOperator::ExactDense),
            128,
            10_000,
            Some(0.01),
        );
        let mut stats = CalibrationStats::default();
        for _ in 0..4 {
            stats.observe(dense, 100.0, 2.0);
        }
        assert_eq!(stats.snapshot().estimate_ms(dense, 200.0), None);
        stats.observe(dense, 100.0, 2.0);
        let snapshot = stats.snapshot();
        assert_eq!(snapshot.estimate_ms(dense, 200.0), Some(4.0));
        assert_eq!(snapshot.estimate_ms(filtered, 200.0), None);
    }

    #[test]
    fn p90_penalizes_variable_operators() {
        let key = key(
            CalibrationTarget::Channel(PhysicalOperator::HnswDense),
            768,
            1_000_000,
            None,
        );
        let mut stats = CalibrationStats::default();
        for elapsed in [1.0, 1.0, 1.0, 1.0, 5.0] {
            stats.observe(key, 1.0, elapsed);
        }
        let entry = &stats.snapshot().entries[0];
        assert!(entry.p90_ms_per_cost_unit > entry.mean_ms_per_cost_unit);
    }
}
