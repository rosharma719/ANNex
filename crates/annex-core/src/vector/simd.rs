/// CPU capability level, detected once per call (std_detect caches internally).
/// Use this to select a function pointer *before* a hot inner loop — not inside it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CpuLevel {
    NeonDotprod, // aarch64 + dotprod extension (M-series, Graviton3+)
    Neon,        // aarch64 baseline
    Avx512Bf16,  // x86: avx512f + avx512bf16 (Sapphire Rapids, Zen4)
    Avx512Vnni,  // x86: avx512f + avx512vnni (Ice Lake, Zen4, Sapphire Rapids)
    Avx512F,     // x86: avx512f only (Skylake-X, Cascade Lake pre-VNNI)
    Avx2Fma,     // x86: avx2 + fma (most current cloud: c5, m5, c6a)
    Avx2,        // x86: avx2 only
    Scalar,
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
        if std::arch::is_x86_feature_detected!("avx512bf16")
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
    fn cpu_level_bf16_detection_is_consistent() {
        let avx512bf16 = std::arch::is_x86_feature_detected!("avx512bf16");
        let avx512f = std::arch::is_x86_feature_detected!("avx512f");
        let level = cpu_level();
        if avx512bf16 && avx512f {
            assert_eq!(
                level,
                CpuLevel::Avx512Bf16,
                "avx512bf16 present but not selected"
            );
        } else {
            assert_ne!(
                level,
                CpuLevel::Avx512Bf16,
                "avx512bf16 absent but incorrectly selected"
            );
        }
    }
}
