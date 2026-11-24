// SPDX-License-Identifier: Apache-2.0
//! `Fpn2Transcript` (`specs/11-wire-format.md` §4): BLAKE2b-512 hash chain with 1-byte domain tags.

use crate::blake2b::Blake2b;
use crate::field::{Fp, Fp3};

pub const PERSONAL_FS: &[u8; 16] = b"FPN2-FS-v2\0\0\0\0\0\0";
pub const LABEL_V2: &[u8] = b"FPN2/transcript/v2";

pub const TAG_INIT: u8 = 0x00;
pub const TAG_BYTES: u8 = 0x01;
pub const TAG_FP: u8 = 0x02;
pub const TAG_FP3: u8 = 0x03;
pub const TAG_SQ_BYTES: u8 = 0x10;
pub const TAG_SQ_FP3: u8 = 0x11;
pub const TAG_SQ_U192: u8 = 0x12;
pub const TAG_RATCHET: u8 = 0x7F;

fn h_fs(parts: &[&[u8]]) -> [u8; 64] {
    let mut h = Blake2b::new(64, PERSONAL_FS);
    for p in parts {
        h.update(p);
    }
    let mut out = [0u8; 64];
    out.copy_from_slice(&h.finalize());
    out
}

#[derive(Clone)]
pub struct Fpn2Transcript {
    state: [u8; 64],
}

impl Fpn2Transcript {
    pub fn new(label: &[u8]) -> Self {
        let st = h_fs(&[&[TAG_INIT], &(label.len() as u64).to_le_bytes(), label]);
        Fpn2Transcript { state: st }
    }
    pub fn state(&self) -> &[u8; 64] {
        &self.state
    }
    fn absorb(&mut self, tag: u8, data: &[u8]) {
        self.state = h_fs(&[
            &self.state,
            &[tag],
            &(data.len() as u64).to_le_bytes(),
            data,
        ]);
    }
    pub fn absorb_bytes(&mut self, b: &[u8]) {
        self.absorb(TAG_BYTES, b);
    }
    pub fn absorb_fp(&mut self, x: Fp) {
        self.absorb(TAG_FP, &x.to_le_bytes());
    }
    pub fn absorb_fp3(&mut self, a: Fp3) {
        self.absorb(TAG_FP3, &a.to_le_bytes());
    }
    fn squeeze(&mut self, tag: u8, n: usize) -> Vec<u8> {
        let nb = (n as u64).to_le_bytes();
        let mut out = Vec::with_capacity(n + 64);
        let mut i: u64 = 0;
        while out.len() < n {
            out.extend_from_slice(&h_fs(&[&self.state, &[tag], &nb, &i.to_le_bytes()]));
            i += 1;
        }
        out.truncate(n);
        self.state = h_fs(&[&self.state, &[TAG_RATCHET, tag], &nb]);
        out
    }
    pub fn squeeze_bytes(&mut self, n: usize) -> Vec<u8> {
        self.squeeze(TAG_SQ_BYTES, n)
    }
    pub fn squeeze_fp3(&mut self) -> Fp3 {
        let o = self.squeeze(TAG_SQ_FP3, 48);
        let limb = |k: usize| {
            let mut b = [0u8; 16];
            b.copy_from_slice(&o[16 * k..16 * k + 16]);
            Fp::from_u128(u128::from_le_bytes(b))
        };
        Fp3([limb(0), limb(1), limb(2)])
    }
    /// 192-bit challenge as three little-endian `u64` limbs.
    pub fn squeeze_u192(&mut self) -> [u64; 3] {
        let o = self.squeeze(TAG_SQ_U192, 24);
        let mut r = [0u64; 3];
        for (k, w) in r.iter_mut().enumerate() {
            let mut b = [0u8; 8];
            b.copy_from_slice(&o[8 * k..8 * k + 8]);
            *w = u64::from_le_bytes(b);
        }
        r
    }
}
