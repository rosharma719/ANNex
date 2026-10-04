/// CPU capability level, detected once per call (std_detect caches internally).
/// Use this to select a function pointer *before* a hot inner loop — not inside it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CpuLevel {
    NeonDotprod, // aarch64 + dotprod extension (M-series, Graviton3+)
    Neon,        // aarch64 baseline
    /// x86: avx512f + avx512bf16 (Sapphire Rapids, Zen4). Only reported when BF16 is opted
    /// into with `VECTORDB_BF16=1`; see [`bf16_enabled`].
    Avx512Bf16,
    Avx512Vnni, // x86: avx512f + avx512vnni (Ice Lake, Zen4, Sapphire Rapids)
    Avx512F,    // x86: avx512f only (Skylake-X, Cascade Lake pre-VNNI)
    Avx2Fma,    // x86: avx2 + fma (most current cloud: c5, m5, c6a)
    Avx2,       // x86: avx2 only
    Scalar,
}

/// Whether `value` (the contents of `VECTORDB_BF16`) opts into BF16 scoring.
/// Anything other than an explicit on-value leaves BF16 off.
pub(crate) fn parse_bf16_flag(value: Option<&str>) -> bool {
    matches!(
        value.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("1" | "true" | "on" | "yes" | "auto")
    )
}

/// Whether AVX-512 BF16 scoring is opted into (`VECTORDB_BF16=1`). Read once and cached.
///
/// Off by default. The BF16 kernels round each operand to bfloat16 before the fused
/// multiply-accumulate, so a dot product of unit vectors is only accurate to about 2^-7, and
/// the exact and ANN backends (BF16 vs f32) stop agreeing on a document's score. That is a
/// reasonable trade for throughput on CPUs that have the instructions (Sapphire Rapids,
/// Zen 4), but not one to make silently, so it takes an explicit opt-in.
pub fn bf16_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| parse_bf16_flag(std::env::var("VECTORDB_BF16").ok().as_deref()))
}

pub fn cpu_level() -> CpuLevel {
    #[cfg(target_arch = "aarch64")]
    {
        return if std::arch::is_aarch64_feature_detected!("dotprod") {
            CpuLevel::NeonDotprod
        } else {
            CpuLevel::Neon
        };
    }
    #[cfg(target_arch = "x86_64")]
    {
        if bf16_enabled()
            && std::arch::is_x86_feature_detected!("avx512bf16")
            && std::arch::is_x86_feature_detected!("avx512f")
        {
            return CpuLevel::Avx512Bf16;
        }
        if std::arch::is_x86_feature_detected!("avx512vnni")
            && std::arch::is_x86_feature_detected!("avx512f")
        {
            return CpuLevel::Avx512Vnni;
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            return CpuLevel::Avx512F;
        }
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return CpuLevel::Avx2Fma;
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return CpuLevel::Avx2;
        }
        CpuLevel::Scalar
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        CpuLevel::Scalar
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_level_does_not_panic() {
        let level = cpu_level();
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            assert!(matches!(
                level,
                CpuLevel::Avx2
                    | CpuLevel::Avx2Fma
                    | CpuLevel::Avx512F
                    | CpuLevel::Avx512Vnni
                    | CpuLevel::Avx512Bf16
            ));
        }
        let _ = level;
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn cpu_level_bf16_requires_both_the_cpu_feature_and_the_opt_in() {
        let has_bf16 = std::arch::is_x86_feature_detected!("avx512bf16")
            && std::arch::is_x86_feature_detected!("avx512f");
        let level = cpu_level();
        if has_bf16 && bf16_enabled() {
            assert_eq!(level, CpuLevel::Avx512Bf16, "opted in on a BF16 CPU");
        } else {
            assert_ne!(
                level,
                CpuLevel::Avx512Bf16,
                "BF16 selected without the CPU feature or without VECTORDB_BF16"
            );
        }
    }

    #[test]
    fn bf16_flag_needs_an_explicit_on_value() {
        for on in ["1", "true", "TRUE", "on", "Yes", "auto", " 1 "] {
            assert!(parse_bf16_flag(Some(on)), "{on:?} should opt in");
        }
        for off in ["", "0", "false", "off", "no", "2", "enable", "bf16"] {
            assert!(!parse_bf16_flag(Some(off)), "{off:?} should stay off");
        }
        assert!(!parse_bf16_flag(None), "unset means off");
    }
}
