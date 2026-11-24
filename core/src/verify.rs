// SPDX-License-Identifier: Apache-2.0
//! The normative `FPN2` verifier (`specs/11-wire-format.md` §6) and reusable verification primitives
//! (sumcheck replay, Merkle anchors, Basefold fold check) for the folded ZK-Basefold backend.

use crate::field::{lagrange_eval, Fp, Fp3};
use crate::poseidon2::{leaf_digest, verify_path, Digest};
use crate::profile::HEADER_LEN;
use crate::transcript::{Fpn2Transcript, LABEL_V2};
use crate::wire::{split_sections, Body, Fpn2Container};

/// Stable reject codes (identical strings in every implementation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RejectCode {
    Length,
    Magic,
    Version,
    Flags,
    Reserved,
    ProfileId,
    ProfilePinned,
    ReducedSecurity,
    BackendUnavailable,
    Padding,
    NonCanonical,
    Sumcheck,
    Final,
    Merkle,
    GkrRoot,
    Gkr,
    Fold,
    Zerocheck,
    CircuitUnpinned,
    CircuitPinned,
    XRootUnpinned,
    XRootPinned,
    StmtUnpinned,
    StmtPinned,
}

impl RejectCode {
    pub fn as_str(self) -> &'static str {
        match self {
            RejectCode::Length => "E_LENGTH",
            RejectCode::Magic => "E_MAGIC",
            RejectCode::Version => "E_VERSION",
            RejectCode::Flags => "E_FLAGS",
            RejectCode::Reserved => "E_RESERVED",
            RejectCode::ProfileId => "E_PROFILE_ID",
            RejectCode::ProfilePinned => "E_PROFILE_PINNED",
            RejectCode::ReducedSecurity => "E_REDUCED_SECURITY",
            RejectCode::BackendUnavailable => "E_BACKEND_UNAVAILABLE",
            RejectCode::Padding => "E_PADDING",
            RejectCode::NonCanonical => "E_NONCANONICAL",
            RejectCode::Sumcheck => "E_SUMCHECK",
            RejectCode::Final => "E_FINAL",
            RejectCode::Merkle => "E_MERKLE",
            RejectCode::GkrRoot => "E_GKR_ROOT",
            RejectCode::Gkr => "E_GKR",
            RejectCode::Fold => "E_FOLD",
            RejectCode::Zerocheck => "E_ZEROCHECK",
            RejectCode::CircuitUnpinned => "E_CIRCUIT_UNPINNED",
            RejectCode::CircuitPinned => "E_CIRCUIT_PINNED",
            RejectCode::XRootUnpinned => "E_X_ROOT_UNPINNED",
            RejectCode::XRootPinned => "E_X_ROOT_PINNED",
            RejectCode::StmtUnpinned => "E_STMT_UNPINNED",
            RejectCode::StmtPinned => "E_STMT_PINNED",
        }
    }
}

impl core::fmt::Display for RejectCode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Caller policy.
#[derive(Clone, Copy, Debug, Default)]
pub struct Policy {
    /// Accept `REDUCED`-tier profiles (`--allow-reduced-security`).
    pub allow_reduced: bool,
    /// Pinned profile id (`--profile`).
    pub expected_profile_id: Option<[u8; 32]>,
    /// Pinned public commitment (`--circuit-root` / `--weight-root`): relation 2/4's circuit root or relation
    /// 5 / 6's `weight_root`. Both fail closed: absent -> `E_CIRCUIT_UNPINNED`, different -> `E_CIRCUIT_PINNED`.
    pub expected_circuit_root: Option<[u8; 32]>,
    /// Pinned pre-committed activation root (`--x-root`, relation 6 only). Fails closed: absent ->
    /// `E_X_ROOT_UNPINNED`, different -> `E_X_ROOT_PINNED`.
    pub expected_x_root: Option<[u8; 32]>,
    /// Pinned statement digest (`--stmt-digest` / `--stmt-file`, relation 2 only). Fails closed: absent ->
    /// `E_STMT_UNPINNED`, different -> `E_STMT_PINNED`.
    pub expected_stmt_digest: Option<[u8; 32]>,
}

// ----------------------------------------------------------------------------- primitives
/// Sumcheck replay: for each round absorb its raw bytes, check `g(0) + g(1) == claim`, squeeze
/// `r`, and set `claim = g(r)`. Returns `(challenges, final claim)`.
pub fn sumcheck_replay(
    tr: &mut Fpn2Transcript,
    claim0: Fp3,
    rounds: &[Vec<Fp3>],
    round_bytes: &[&[u8]],
) -> Result<(Vec<Fp3>, Fp3), RejectCode> {
    let mut claim = claim0;
    let mut rs = Vec::with_capacity(rounds.len());
    for (g, raw) in rounds.iter().zip(round_bytes.iter()) {
        tr.absorb_bytes(raw);
        if g.len() < 2 || g[0].add(g[1]) != claim {
            return Err(RejectCode::Sumcheck);
        }
        let r = tr.squeeze_fp3();
        claim = lagrange_eval(g, r);
        rs.push(r);
    }
    Ok((rs, claim))
}

/// Poseidon2 Merkle anchor check of a base-field leaf value.
pub fn check_anchor(root: &Digest, value: Fp, idx: u64, path: &[Digest]) -> bool {
    verify_path(root, &leaf_digest(value), idx, path)
}

/// One Basefold / FRI fold step: from codeword values `f(x)`, `f(-x)` and challenge `r`,
/// returns `(f(x) + f(-x))/2 + r (f(x) - f(-x))/(2x)` = `f_e(x^2) + r f_o(x^2)`.
pub fn basefold_fold(fx: Fp3, fmx: Fp3, x: Fp, r: Fp3) -> Option<Fp3> {
    let inv2 = Fp::new(2).inv()?;
    let inv2x = Fp::new(2).mul(x).inv()?;
    let even = fx.add(fmx).scale(inv2);
    let odd = fx.sub(fmx).scale(inv2x);
    Some(even.add(r.mul(odd)))
}

// ----------------------------------------------------------------------------- normative verifier
/// Checks 1–12: everything up to and including canonical decoding ([`Fpn2Container::parse`]).
pub fn parse_container(data: &[u8], policy: &Policy) -> Result<Fpn2Container, RejectCode> {
    Fpn2Container::parse(data, policy)
}

/// Full verification. `Ok(())` means `VERIFIED`.
pub fn verify_bytes(data: &[u8], policy: &Policy) -> Result<(), RejectCode> {
    let c = parse_container(data, policy)?;
    if policy.expected_stmt_digest.is_some() {
        return Err(RejectCode::StmtPinned);
    }
    let prof = c.profile;
    let body = match &c.body {
        Body::Harness(b) => {
            if policy.expected_circuit_root.is_some() {
                return Err(RejectCode::CircuitPinned);
            }
            if policy.expected_x_root.is_some() {
                return Err(RejectCode::XRootPinned);
            }
            b
        }
        Body::FoldedGkr(_) => {
            if policy.expected_circuit_root.is_some() {
                return Err(RejectCode::CircuitPinned);
            }
            if policy.expected_x_root.is_some() {
                return Err(RejectCode::XRootPinned);
            }
            let lay = crate::folded_gkr::Layout::of(prof).ok_or(RejectCode::BackendUnavailable)?;
            let used = HEADER_LEN + lay.body_used();
            return crate::folded_gkr::verify(&data[..HEADER_LEN], &data[HEADER_LEN..used], &lay);
        }
        Body::FoldedAir(_) => {
            if policy.expected_x_root.is_some() {
                return Err(RejectCode::XRootPinned);
            }
            let lay =
                crate::folded_air::AirLayout::of(prof).ok_or(RejectCode::BackendUnavailable)?;
            let used = HEADER_LEN + lay.body_used();
            return crate::folded_air::verify(
                &data[..HEADER_LEN],
                &data[HEADER_LEN..used],
                &lay,
                policy.expected_circuit_root.as_ref(),
            );
        }
        Body::FoldedVec(_) => {
            if policy.expected_x_root.is_some() {
                return Err(RejectCode::XRootPinned);
            }
            let lay =
                crate::folded_vec::VecLayout::of(prof).ok_or(RejectCode::BackendUnavailable)?;
            let used = HEADER_LEN + lay.body_used();
            return crate::folded_vec::verify(
                &data[..HEADER_LEN],
                &data[HEADER_LEN..used],
                &lay,
                policy.expected_circuit_root.as_ref(),
            );
        }
        Body::FoldedVecBound(_) => {
            let lay = crate::folded_vec_bound::BoundLayout::of(prof)
                .ok_or(RejectCode::BackendUnavailable)?;
            let used = HEADER_LEN + lay.body_used();
            return crate::folded_vec_bound::verify(
                &data[..HEADER_LEN],
                &data[HEADER_LEN..used],
                &lay,
                policy.expected_circuit_root.as_ref(),
                policy.expected_x_root.as_ref(),
            );
        }
    };
    let sec = split_sections(data, prof).ok_or(RejectCode::Length)?;
    let mut tr = Fpn2Transcript::new(LABEL_V2);
    tr.absorb_bytes(sec.header);
    tr.absorb_bytes(sec.stmt);
    let (_, claim) = sumcheck_replay(&mut tr, body.claim, &body.rounds, &sec.rounds)?; // 13
    tr.absorb_bytes(sec.final_);
    if claim != body.final_.mul(body.final_) {
        return Err(RejectCode::Final); // 14
    }
    let mask = (1u64 << prof.n_vars) - 1;
    for (v, sibs) in &body.openings {
        let raw = tr.squeeze_bytes(8);
        let mut b = [0u8; 8];
        b.copy_from_slice(&raw);
        let idx = u64::from_le_bytes(b) & mask;
        if !check_anchor(&body.r_tab, *v, idx, sibs) {
            return Err(RejectCode::Merkle); // 15
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_matches_even_odd_decomposition() {
        // f(X) = c0 + c1 X + c2 X^2 + c3 X^3 ; f_e(Y) = c0 + c2 Y, f_o(Y) = c1 + c3 Y.
        let c = [Fp::new(5), Fp::new(11), Fp::new(17), Fp::new(23)];
        let f = |x: Fp| {
            c[0].add(c[1].mul(x))
                .add(c[2].mul(x.square()))
                .add(c[3].mul(x.square().mul(x)))
        };
        let x = Fp::new(1234567);
        let r = Fp3([Fp::new(9), Fp::new(8), Fp::new(7)]);
        let y = x.square();
        let expect =
            Fp3::from_fp(c[0].add(c[2].mul(y))).add(r.mul(Fp3::from_fp(c[1].add(c[3].mul(y)))));
        let got = basefold_fold(Fp3::from_fp(f(x)), Fp3::from_fp(f(x.neg())), x, r).unwrap();
        assert_eq!(got, expect);
        assert_eq!(basefold_fold(Fp3::ZERO, Fp3::ZERO, Fp::ZERO, r), None);
    }
}
