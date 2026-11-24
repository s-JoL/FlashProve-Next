// SPDX-License-Identifier: Apache-2.0
//! `FPN2` header and fixed-size container codec (`specs/11-wire-format.md` §5).

use crate::field::{Fp, Fp3};
use crate::folded_gkr::{self, FoldedBody, Layout};
use crate::poseidon2::Digest;
use crate::profile::{lookup, Profile, Relation, SecurityTier, HEADER_LEN};
use crate::verify::{Policy, RejectCode};

pub const MAGIC: &[u8; 4] = b"FPN2";
pub const VERSION: u16 = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fpn2Header {
    pub profile_id: [u8; 32],
}

impl Fpn2Header {
    pub fn encode(&self) -> [u8; HEADER_LEN] {
        let mut b = [0u8; HEADER_LEN];
        b[0..4].copy_from_slice(MAGIC);
        b[4..6].copy_from_slice(&VERSION.to_le_bytes());
        b[8..40].copy_from_slice(&self.profile_id);
        b
    }

    /// Normative checks 1–5.
    pub fn parse(b: &[u8]) -> Result<Fpn2Header, RejectCode> {
        if b.len() < HEADER_LEN {
            return Err(RejectCode::Length);
        }
        if &b[0..4] != MAGIC {
            return Err(RejectCode::Magic);
        }
        if u16::from_le_bytes([b[4], b[5]]) != VERSION {
            return Err(RejectCode::Version);
        }
        if u16::from_le_bytes([b[6], b[7]]) != 0 {
            return Err(RejectCode::Flags);
        }
        if b[40..64].iter().any(|&x| x != 0) {
            return Err(RejectCode::Reserved);
        }
        let mut id = [0u8; 32];
        id.copy_from_slice(&b[8..40]);
        Ok(Fpn2Header { profile_id: id })
    }

    pub fn profile(&self) -> Option<&'static Profile> {
        lookup(&self.profile_id)
    }
}

/// Raw byte views of a relation-1 container, in wire order (what the transcript absorbs).
pub struct Sections<'a> {
    pub header: &'a [u8],
    pub stmt: &'a [u8],
    pub rounds: Vec<&'a [u8]>,
    pub final_: &'a [u8],
    pub openings: Vec<&'a [u8]>,
    pub pad: &'a [u8],
}

/// Splits a relation-1 container whose length is exactly `C_p` (caller-checked).
pub fn split_sections<'a>(b: &'a [u8], prof: &Profile) -> Option<Sections<'a>> {
    if prof.relation != Relation::HarnessSqSumcheck {
        return None;
    }
    let used = prof.body_used()?;
    if b.len() as u64 != prof.c_p || HEADER_LEN + used > b.len() {
        return None;
    }
    let mut off = HEADER_LEN;
    let mut take = |n: usize| {
        let s = &b[off..off + n];
        off += n;
        s
    };
    let header = &b[..HEADER_LEN];
    let stmt = take(Profile::STMT_LEN);
    let rounds = (0..prof.n_vars).map(|_| take(prof.round_len())).collect();
    let final_ = take(Profile::FINAL_LEN);
    let openings = (0..prof.n_queries)
        .map(|_| take(prof.opening_len()))
        .collect();
    let pad = &b[HEADER_LEN + used..];
    Some(Sections {
        header,
        stmt,
        rounds,
        final_,
        openings,
        pad,
    })
}

/// Decoded body of a relation-1 (`HARNESS_SQ_SUMCHECK`) container.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HarnessBody {
    pub r_tab: Digest,
    pub n_s: [u8; 32],
    pub st_new: Digest,
    pub claim: Fp3,
    pub rounds: Vec<Vec<Fp3>>,
    pub final_: Fp3,
    pub openings: Vec<(Fp, Vec<Digest>)>,
}

fn dec_digest(b: &[u8]) -> Result<Digest, RejectCode> {
    let mut d = [Fp::ZERO; 4];
    for (i, x) in d.iter_mut().enumerate() {
        *x = Fp::decode(&b[8 * i..8 * i + 8]).map_err(|_| RejectCode::NonCanonical)?;
    }
    Ok(d)
}

fn enc_digest(d: &Digest, out: &mut Vec<u8>) {
    for x in d {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

fn dec_fp3(b: &[u8]) -> Result<Fp3, RejectCode> {
    Fp3::decode(b).map_err(|_| RejectCode::NonCanonical)
}

/// Normative check 12: canonical decoding of every typed field in wire order.
pub fn decode_harness(sec: &Sections<'_>, prof: &Profile) -> Result<HarnessBody, RejectCode> {
    let s = sec.stmt;
    let r_tab = dec_digest(&s[0..32])?;
    let mut n_s = [0u8; 32];
    n_s.copy_from_slice(&s[32..64]);
    let st_new = dec_digest(&s[64..96])?;
    let claim = dec_fp3(&s[96..120])?;
    let mut rounds = Vec::with_capacity(sec.rounds.len());
    for rb in &sec.rounds {
        let mut g = Vec::with_capacity(prof.sc_degree as usize + 1);
        for j in 0..=prof.sc_degree as usize {
            g.push(dec_fp3(&rb[24 * j..24 * j + 24])?);
        }
        rounds.push(g);
    }
    let final_ = dec_fp3(sec.final_)?;
    let mut openings = Vec::with_capacity(sec.openings.len());
    for ob in &sec.openings {
        let v = Fp::decode(&ob[0..8]).map_err(|_| RejectCode::NonCanonical)?;
        let mut sibs = Vec::with_capacity(prof.n_vars as usize);
        for k in 0..prof.n_vars as usize {
            sibs.push(dec_digest(&ob[8 + 32 * k..40 + 32 * k])?);
        }
        openings.push((v, sibs));
    }
    Ok(HarnessBody {
        r_tab,
        n_s,
        st_new,
        claim,
        rounds,
        final_,
        openings,
    })
}

/// Serializer (inverse of `decode_harness`), producing exactly `C_p` bytes.
pub fn encode_harness(prof: &Profile, body: &HarnessBody) -> Option<Vec<u8>> {
    if prof.relation != Relation::HarnessSqSumcheck {
        return None;
    }
    let mut out = Fpn2Header {
        profile_id: prof.profile_id(),
    }
    .encode()
    .to_vec();
    enc_digest(&body.r_tab, &mut out);
    out.extend_from_slice(&body.n_s);
    enc_digest(&body.st_new, &mut out);
    out.extend_from_slice(&body.claim.to_le_bytes());
    for g in &body.rounds {
        for v in g {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out.extend_from_slice(&body.final_.to_le_bytes());
    for (v, sibs) in &body.openings {
        out.extend_from_slice(&v.to_le_bytes());
        for d in sibs {
            enc_digest(d, &mut out);
        }
    }
    if out.len() != HEADER_LEN + prof.body_used()? || out.len() as u64 > prof.c_p {
        return None;
    }
    out.resize(prof.c_p as usize, 0);
    Some(out)
}

/// Decoded body of a wired relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Body {
    Harness(HarnessBody),
    FoldedGkr(FoldedBody),
    /// Relation 4: the canonical body bytes (checked by `folded_air::decode`).
    FoldedAir(Vec<u8>),
    /// Relation 5: the canonical body bytes (checked by `folded_vec::decode`).
    FoldedVec(Vec<u8>),
    /// Relation 6: the canonical body bytes (checked by `folded_vec_bound::decode`).
    FoldedVecBound(Vec<u8>),
}

/// A fully parsed container (relation 1, 3, 4, 5 or 6).
pub struct Fpn2Container {
    pub header: Fpn2Header,
    pub profile: &'static Profile,
    pub body: Body,
}

impl Fpn2Container {
    /// Strict parser: normative checks 1–12 (header, profile registry and pin, security tier,
    /// exact length `C_p`, wired backend, all-zero padding, canonical field encodings).
    pub fn parse(data: &[u8], policy: &Policy) -> Result<Fpn2Container, RejectCode> {
        let header = Fpn2Header::parse(data)?; // 1-5
        let profile = header.profile().ok_or(RejectCode::ProfileId)?; // 6
        if let Some(pin) = policy.expected_profile_id {
            if pin != header.profile_id {
                return Err(RejectCode::ProfilePinned); // 7
            }
        }
        if profile.security_tier == SecurityTier::Reduced && !policy.allow_reduced {
            return Err(RejectCode::ReducedSecurity); // 8
        }
        if data.len() as u64 != profile.c_p {
            return Err(RejectCode::Length); // 9
        }
        let body = match profile.relation {
            Relation::HarnessSqSumcheck => {
                let sec = split_sections(data, profile).ok_or(RejectCode::Length)?;
                if sec.pad.iter().any(|&b| b != 0) {
                    return Err(RejectCode::Padding); // 11
                }
                Body::Harness(decode_harness(&sec, profile)?) // 12
            }
            Relation::FoldedGkrLookup => {
                let lay = Layout::of(profile).ok_or(RejectCode::BackendUnavailable)?; // 10
                let used = HEADER_LEN + lay.body_used();
                if used > data.len() {
                    return Err(RejectCode::Length);
                }
                if data[used..].iter().any(|&b| b != 0) {
                    return Err(RejectCode::Padding); // 11
                }
                Body::FoldedGkr(folded_gkr::decode(&data[HEADER_LEN..used], &lay)?)
                // 12
            }
            Relation::FoldedGkrAir => {
                let lay = crate::folded_air::AirLayout::of(profile)
                    .ok_or(RejectCode::BackendUnavailable)?; // 10
                let used = HEADER_LEN + lay.body_used();
                if used > data.len() {
                    return Err(RejectCode::Length);
                }
                if data[used..].iter().any(|&b| b != 0) {
                    return Err(RejectCode::Padding); // 11
                }
                let body = &data[HEADER_LEN..used];
                crate::folded_air::decode(body, &lay)?; // 12
                Body::FoldedAir(body.to_vec())
            }
            Relation::FoldedVecMatmul => {
                let lay = crate::folded_vec::VecLayout::of(profile)
                    .ok_or(RejectCode::BackendUnavailable)?; // 10
                let used = HEADER_LEN + lay.body_used();
                if used > data.len() {
                    return Err(RejectCode::Length);
                }
                if data[used..].iter().any(|&b| b != 0) {
                    return Err(RejectCode::Padding); // 11
                }
                let body = &data[HEADER_LEN..used];
                crate::folded_vec::decode(body, &lay)?; // 12
                Body::FoldedVec(body.to_vec())
            }
            Relation::FoldedVecBound => {
                let lay = crate::folded_vec_bound::BoundLayout::of(profile)
                    .ok_or(RejectCode::BackendUnavailable)?; // 10
                let used = HEADER_LEN + lay.body_used();
                if used > data.len() {
                    return Err(RejectCode::Length);
                }
                if data[used..].iter().any(|&b| b != 0) {
                    return Err(RejectCode::Padding); // 11
                }
                let body = &data[HEADER_LEN..used];
                crate::folded_vec_bound::decode(body, &lay)?; // 12
                Body::FoldedVecBound(body.to_vec())
            }
            Relation::LlmInferenceV2 => return Err(RejectCode::BackendUnavailable), // 10
        };
        Ok(Fpn2Container {
            header,
            profile,
            body,
        })
    }

    /// Canonical serializer: exactly `C_p` bytes (inverse of [`Fpn2Container::parse`]).
    pub fn serialize(&self) -> Option<Vec<u8>> {
        match &self.body {
            Body::Harness(b) => encode_harness(self.profile, b),
            Body::FoldedGkr(_)
            | Body::FoldedAir(_)
            | Body::FoldedVec(_)
            | Body::FoldedVecBound(_) => {
                let mut out = self.header.encode().to_vec();
                match &self.body {
                    Body::FoldedGkr(b) => out.extend_from_slice(&folded_gkr::encode(b)),
                    Body::FoldedAir(b) | Body::FoldedVec(b) | Body::FoldedVecBound(b) => {
                        out.extend_from_slice(b)
                    }
                    Body::Harness(_) => return None,
                }
                if out.len() as u64 > self.profile.c_p {
                    return None;
                }
                out.resize(self.profile.c_p as usize, 0);
                Some(out)
            }
        }
    }
}
