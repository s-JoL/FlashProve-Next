// SPDX-License-Identifier: Apache-2.0
//! Canonical `FPN2` profile bytes and the pinned registry (`specs/11-wire-format.md` §5.2).
//! Profiles carry no architecture fields: only public scale upper bounds and protocol constants.

use crate::blake2b::blake2b_256;

pub const PROFILE_MAGIC: &[u8; 8] = b"FPN2PROF";
pub const PROFILE_FORMAT: u16 = 1;
pub const PROFILE_LEN: usize = 128;
pub const HEADER_LEN: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SecurityTier {
    Reduced = 0,
    Standard = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Relation {
    HarnessSqSumcheck = 1,
    LlmInferenceV2 = 2,
    FoldedGkrLookup = 3,
    FoldedGkrAir = 4,
    FoldedVecMatmul = 5,
    FoldedVecBound = 6,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Profile {
    pub name: &'static str,
    pub security_tier: SecurityTier,
    pub relation: Relation,
    pub security_bits: u16,
    pub max_layers: u32,
    pub max_hidden: u32,
    pub max_seq_len: u32,
    pub max_vocab: u32,
    pub max_new_tokens: u32,
    pub stop_ids: &'static [u32],
    pub n_vars: u8,
    pub sc_degree: u8,
    pub n_queries: u16,
    pub c_p: u64,
}

/// Name used by the M1-1 brief and the architecture memo.
pub type Fpn2Profile = Profile;

impl Profile {
    pub fn canonical_bytes(&self) -> [u8; PROFILE_LEN] {
        let mut b = [0u8; PROFILE_LEN];
        b[0..8].copy_from_slice(PROFILE_MAGIC);
        b[8..10].copy_from_slice(&PROFILE_FORMAT.to_le_bytes());
        let nm = self.name.as_bytes();
        assert!(!nm.is_empty() && nm.len() <= 32 && nm.iter().all(|c| valid_name_char(*c)));
        b[10..10 + nm.len()].copy_from_slice(nm);
        b[42] = self.security_tier as u8;
        b[43] = self.relation as u8;
        b[44..46].copy_from_slice(&self.security_bits.to_le_bytes());
        let scale = [
            self.max_layers,
            self.max_hidden,
            self.max_seq_len,
            self.max_vocab,
            self.max_new_tokens,
        ];
        for (k, v) in scale.iter().enumerate() {
            b[46 + 4 * k..50 + 4 * k].copy_from_slice(&v.to_le_bytes());
        }
        assert!(self.stop_ids.len() <= 8);
        b[66] = self.stop_ids.len() as u8;
        for (k, v) in self.stop_ids.iter().enumerate() {
            b[67 + 4 * k..71 + 4 * k].copy_from_slice(&v.to_le_bytes());
        }
        b[99] = self.n_vars;
        b[100] = self.sc_degree;
        b[101..103].copy_from_slice(&self.n_queries.to_le_bytes());
        b[103..111].copy_from_slice(&self.c_p.to_le_bytes());
        b
    }

    pub fn profile_id(&self) -> [u8; 32] {
        blake2b_256(&self.canonical_bytes())
    }

    pub const STMT_LEN: usize = 120;
    pub const FINAL_LEN: usize = 24;

    pub fn round_len(&self) -> usize {
        (self.sc_degree as usize + 1) * 24
    }
    pub fn opening_len(&self) -> usize {
        8 + 32 * self.n_vars as usize
    }
    /// Bytes used by the body of a relation-1, -3, -4, -5 or -6 container (before zero padding).
    pub fn body_used(&self) -> Option<usize> {
        match self.relation {
            Relation::HarnessSqSumcheck => Some(
                Self::STMT_LEN
                    + self.n_vars as usize * self.round_len()
                    + Self::FINAL_LEN
                    + self.n_queries as usize * self.opening_len(),
            ),
            Relation::LlmInferenceV2 => None,
            Relation::FoldedGkrLookup => crate::folded_gkr::Layout::of(self).map(|l| l.body_used()),
            Relation::FoldedGkrAir => crate::folded_air::AirLayout::of(self).map(|l| l.body_used()),
            Relation::FoldedVecMatmul => {
                crate::folded_vec::VecLayout::of(self).map(|l| l.body_used())
            }
            Relation::FoldedVecBound => {
                crate::folded_vec_bound::BoundLayout::of(self).map(|l| l.body_used())
            }
        }
    }
}

fn valid_name_char(c: u8) -> bool {
    c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'.' || c == b'-'
}

pub static HARNESS_SQ_N4: Profile = Profile {
    name: "fpn2-harness-sq-n4",
    security_tier: SecurityTier::Reduced,
    relation: Relation::HarnessSqSumcheck,
    security_bits: 0,
    max_layers: 0,
    max_hidden: 0,
    max_seq_len: 0,
    max_vocab: 0,
    max_new_tokens: 0,
    stop_ids: &[],
    n_vars: 4,
    sc_degree: 2,
    n_queries: 4,
    c_p: 2048,
};

pub static LLM_STD_V1: Profile = Profile {
    name: "fpn2-llm-std-v1",
    security_tier: SecurityTier::Standard,
    relation: Relation::LlmInferenceV2,
    security_bits: 100,
    max_layers: 128,
    max_hidden: 16384,
    max_seq_len: 131072,
    max_vocab: 262144,
    max_new_tokens: 32768,
    stop_ids: &[151643, 151645],
    n_vars: 0,
    sc_degree: 0,
    n_queries: 0,
    c_p: 8 * 1024 * 1024,
};

/// End-to-end relation-3 profile: 128 private queries range-checked against `[0, 128)` by a
/// ZK-masked FracLogUp-GKR + fused two-round folded ZK-Basefold (n = 12, rate 1/8, Q = 72).
pub static FOLDED_GKR_V1: Profile = Profile {
    name: "fpn2-folded-gkr-v1",
    security_tier: SecurityTier::Standard,
    relation: Relation::FoldedGkrLookup,
    security_bits: 100,
    max_layers: 0,
    max_hidden: 0,
    max_seq_len: 0,
    max_vocab: 0,
    max_new_tokens: 0,
    stop_ids: &[],
    n_vars: 12,
    sc_degree: 3,
    n_queries: 72,
    c_p: 163_840,
};

/// Small `REDUCED` relation-3 test profile (n = 8, Q = 2) carrying most negative golden vectors.
pub static FOLDED_GKR_N8: Profile = Profile {
    name: "fpn2-folded-gkr-n8",
    security_tier: SecurityTier::Reduced,
    relation: Relation::FoldedGkrLookup,
    security_bits: 0,
    max_layers: 0,
    max_hidden: 0,
    max_seq_len: 0,
    max_vocab: 0,
    max_new_tokens: 0,
    stop_ids: &[],
    n_vars: 8,
    sc_degree: 3,
    n_queries: 2,
    c_p: 6144,
};

/// End-to-end relation-4 profile (`FOLDED_GKR_AIR`): Plonkish zerocheck + wiring/range FracLogUp-GKR
/// over K = 512 gate rows / variables, one batched folded ZK-Basefold opening (n = 14, Q = 72).
pub static FOLDED_AIR_V1: Profile = Profile {
    name: "fpn2-folded-air-v1",
    security_tier: SecurityTier::Standard,
    relation: Relation::FoldedGkrAir,
    security_bits: 100,
    max_layers: 0,
    max_hidden: 0,
    max_seq_len: 0,
    max_vocab: 0,
    max_new_tokens: 0,
    stop_ids: &[],
    n_vars: 14,
    sc_degree: 3,
    n_queries: 72,
    c_p: 262_144,
};

/// Small `REDUCED` relation-4 test profile (n = 12: K = 128, Q = 2).
pub static FOLDED_AIR_N12: Profile = Profile {
    name: "fpn2-folded-air-n12",
    security_tier: SecurityTier::Reduced,
    relation: Relation::FoldedGkrAir,
    security_bits: 0,
    max_layers: 0,
    max_hidden: 0,
    max_seq_len: 0,
    max_vocab: 0,
    max_new_tokens: 0,
    stop_ids: &[],
    n_vars: 12,
    sc_degree: 3,
    n_queries: 2,
    c_p: 16_384,
};

/// Production-scale relation-4 profile: the same relation at n = 20, i.e. K = 32,768 gate rows /
/// variables (64x `fpn2-folded-air-v1`), Q = 72, C_p = 512 KiB.
pub static FOLDED_AIR_N20_V1: Profile = Profile {
    name: "fpn2-folded-air-n20-v1",
    security_tier: SecurityTier::Standard,
    relation: Relation::FoldedGkrAir,
    security_bits: 100,
    max_layers: 0,
    max_hidden: 0,
    max_seq_len: 0,
    max_vocab: 0,
    max_new_tokens: 0,
    stop_ids: &[],
    n_vars: 20,
    sc_degree: 3,
    n_queries: 72,
    c_p: 524_288,
};

/// Relation-5 profile (`FOLDED_VEC_MATMUL`): ONE full-width `32 × 1024` GEMM tile (K = `max_hidden`
/// = 1024, B = 1, 32,768 multiply-adds), n = 16, Q = 72, C_p = 384 KiB.
pub static FOLDED_VEC_N16_V1: Profile = Profile {
    name: "fpn2-folded-vec-n16-v1",
    security_tier: SecurityTier::Standard,
    relation: Relation::FoldedVecMatmul,
    security_bits: 100,
    max_layers: 0,
    max_hidden: 1024,
    max_seq_len: 0,
    max_vocab: 0,
    max_new_tokens: 0,
    stop_ids: &[],
    n_vars: 16,
    sc_degree: 3,
    n_queries: 72,
    c_p: 393_216,
};

/// Production-scale relation-5 profile: 64 full-width `32 × 1024` tiles (2,097,152 multiply-adds), n = 22,
/// Q = 72, C_p = 512 KiB.
pub static FOLDED_VEC_N22_V1: Profile = Profile {
    name: "fpn2-folded-vec-n22-v1",
    security_tier: SecurityTier::Standard,
    relation: Relation::FoldedVecMatmul,
    security_bits: 100,
    max_layers: 0,
    max_hidden: 1024,
    max_seq_len: 0,
    max_vocab: 0,
    max_new_tokens: 0,
    stop_ids: &[],
    n_vars: 22,
    sc_degree: 3,
    n_queries: 72,
    c_p: 524_288,
};

/// Relation-6 profile (`FOLDED_VEC_BOUND`): relation 5's ONE full-width `32 × 1024` GEMM tile with the
/// activations `X` pre-committed (ZK-blinded) under a caller-pinned `x_root`. n = 16, Q = 72, C_p = 384 KiB.
pub static FOLDED_VEC_BOUND_N16_V1: Profile = Profile {
    name: "fpn2-folded-vec-bound-n16-v1",
    security_tier: SecurityTier::Standard,
    relation: Relation::FoldedVecBound,
    security_bits: 100,
    max_layers: 0,
    max_hidden: 1024,
    max_seq_len: 0,
    max_vocab: 0,
    max_new_tokens: 0,
    stop_ids: &[],
    n_vars: 16,
    sc_degree: 3,
    n_queries: 72,
    c_p: 393_216,
};

/// Production-scale relation-6 profile: 64 full-width `32 × 1024` tiles, n = 22, Q = 72. The body (529,120 B)
/// does not fit 512 KiB, so C_p = 1 MiB.
pub static FOLDED_VEC_BOUND_N22_V1: Profile = Profile {
    name: "fpn2-folded-vec-bound-n22-v1",
    security_tier: SecurityTier::Standard,
    relation: Relation::FoldedVecBound,
    security_bits: 100,
    max_layers: 0,
    max_hidden: 1024,
    max_seq_len: 0,
    max_vocab: 0,
    max_new_tokens: 0,
    stop_ids: &[],
    n_vars: 22,
    sc_degree: 3,
    n_queries: 72,
    c_p: 1_048_576,
};

/// Pinned registry: (profile, BLAKE2b-256 of its canonical bytes). Self-tested in `tests`.
pub static REGISTRY: [(&Profile, &str); 11] = [
    (
        &HARNESS_SQ_N4,
        "e24f718cf8c2f4d21417f9e8a3038b9993aa418ca0f7205a5cf6a8c53a3442db",
    ),
    (
        &LLM_STD_V1,
        "588659a6e46c51c021cd65dd287c8968562b637d20afaf0d68f4ad57b16db9a1",
    ),
    (
        &FOLDED_GKR_V1,
        "e238db306348de19561c2fb4896ea4a249bd8ce7bd41beb1f715b6fcb305e986",
    ),
    (
        &FOLDED_GKR_N8,
        "d7a3d169b65f8c5e328911f4a7edb3750c3244ca0202a9b259a7325d5705c147",
    ),
    (
        &FOLDED_AIR_V1,
        "f6c6b4ca3f4292e6e8917d1285f6105bde168ed7517b71aa65e5c562933d0a97",
    ),
    (
        &FOLDED_AIR_N12,
        "5f661ec4ea1ab8b67de2ed9d13a13d148b4ed639a242d30c81fdb4f2705ced7d",
    ),
    (
        &FOLDED_AIR_N20_V1,
        "966dec06ad5fb40ef7843fe91310684a12378990bbe53483ce7718ef606ba60d",
    ),
    (
        &FOLDED_VEC_N16_V1,
        "a48dfb0a1b0f17580cbab6b756678238200c1b3028a26f92898430d8da801327",
    ),
    (
        &FOLDED_VEC_N22_V1,
        "ba72066888c8fc3cf0ca31c86527a5292759eb629c0eed131d812623b3e6b3ef",
    ),
    (
        &FOLDED_VEC_BOUND_N16_V1,
        "9492aa70db8c3d79fb242001f178a43b31ba5e610114d2606c87d6350eb0472e",
    ),
    (
        &FOLDED_VEC_BOUND_N22_V1,
        "83bab0b60ea45833b70498f5c19dfc2768a5583c0e947cc8b418a66d10591745",
    ),
];

pub fn parse_hex32(s: &str) -> Option<[u8; 32]> {
    let b = s.as_bytes();
    if b.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        let hi = (b[2 * i] as char).to_digit(16)?;
        let lo = (b[2 * i + 1] as char).to_digit(16)?;
        out[i] = (hi * 16 + lo) as u8;
    }
    Some(out)
}

/// Look up a registered profile by its pinned id.
pub fn lookup(id: &[u8; 32]) -> Option<&'static Profile> {
    REGISTRY
        .iter()
        .find(|(_, pin)| parse_hex32(pin).as_ref() == Some(id))
        .map(|(p, _)| *p)
}

pub fn by_name(name: &str) -> Option<(&'static Profile, [u8; 32])> {
    REGISTRY
        .iter()
        .find(|(p, _)| p.name == name)
        .and_then(|(p, pin)| parse_hex32(pin).map(|id| (*p, id)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_ids_match_canonical_bytes() {
        for (p, pin) in REGISTRY.iter() {
            assert_eq!(Some(p.profile_id()), parse_hex32(pin), "{}", p.name);
        }
    }

    #[test]
    fn harness_layout() {
        assert_eq!(HARNESS_SQ_N4.body_used(), Some(976));
        assert!(HEADER_LEN + 976 <= HARNESS_SQ_N4.c_p as usize);
    }
}
