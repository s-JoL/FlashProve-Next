// SPDX-License-Identifier: Apache-2.0
//! `fpn-core`: zero-dependency core of FlashProve-Next.
//!
//! * [`field`] — Goldilocks `F_p` and `E = F_p[u]/(u^3 - 2)`;
//! * [`poseidon2`] — Plonky3 Goldilocks width-12 Poseidon2, sponges, compression and Merkle paths;
//! * [`blake2b`] — vendored RFC 7693 BLAKE2b (with personalization);
//! * [`transcript`] — `Fpn2Transcript` raw-byte Fiat–Shamir;
//! * [`profile`], [`wire`] — canonical profiles and the fixed-size `FPN2` container;
//! * [`folded_gkr`] — relation 3: ZK-masked FracLogUp-GKR + fused two-round folded ZK-Basefold;
//! * [`folded_air`] — relation 4: Plonkish zerocheck + wiring/range FracLogUp-GKR + batched folded ZK-Basefold;
//! * [`folded_vec`] — relation 5: full-width GEMM tiles via a Libra-masked inner-product sumcheck + the same opening;
//! * [`folded_vec_bound`] — relation 6: relation 5 with the activations pre-committed under a pinned `x_root`;
//! * [`verify`] — the normative verifier of `specs/11-wire-format.md` §6.
//!
//! The library never reads the process environment and contains no `unsafe` code.
#![forbid(unsafe_code)]
// Index loops mirror the reference formulas in the spec and field.py/poseidon2.py.
#![allow(clippy::needless_range_loop)]

pub mod blake2b;
pub mod field;
pub mod folded_air;
pub mod folded_gkr;
pub mod folded_vec;
pub mod folded_vec_bound;
pub mod multi_gkr;
pub mod multi_pcs;
pub mod poseidon2;
mod poseidon2_constants;
pub mod profile;
pub mod relation2;
pub mod transcript;
pub mod verify;
pub mod wire;

pub use profile::Fpn2Profile;
pub use transcript::Fpn2Transcript;
pub use verify::{verify_bytes, Policy, RejectCode};
pub use wire::{Fpn2Container, Fpn2Header};
