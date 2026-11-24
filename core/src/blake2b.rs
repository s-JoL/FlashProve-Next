// SPDX-License-Identifier: Apache-2.0
//! Vendored BLAKE2b (RFC 7693), unkeyed, with optional 16-byte personalization.
//! Output lengths 1..=64 bytes; `FPN2` uses 32 (profile ids) and 64 (transcript).

const IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

const SIGMA: [[usize; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

#[inline(always)]
fn g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

fn compress(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
    let mut m = [0u64; 16];
    for (i, w) in m.iter_mut().enumerate() {
        let mut b = [0u8; 8];
        b.copy_from_slice(&block[8 * i..8 * i + 8]);
        *w = u64::from_le_bytes(b);
    }
    let mut v = [0u64; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&IV);
    v[12] ^= t as u64;
    v[13] ^= (t >> 64) as u64;
    if last {
        v[14] = !v[14];
    }
    for s in SIGMA.iter() {
        g(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        g(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        g(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        g(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        g(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        g(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        g(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        g(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

/// Incremental unkeyed BLAKE2b state.
#[derive(Clone)]
pub struct Blake2b {
    h: [u64; 8],
    buf: [u8; 128],
    buf_len: usize,
    t: u128,
    nn: usize,
}

impl Blake2b {
    /// `nn` in `1..=64`; `personal` is zero-extended to 16 bytes (at most 16 bytes).
    pub fn new(nn: usize, personal: &[u8]) -> Blake2b {
        assert!(
            (1..=64).contains(&nn) && personal.len() <= 16,
            "invalid BLAKE2b parameters"
        );
        // Parameter block: digest_length, key_length = 0, fanout = 1, depth = 1, rest zero except personal.
        let mut p = [0u8; 64];
        p[0] = nn as u8;
        p[2] = 1;
        p[3] = 1;
        p[48..48 + personal.len()].copy_from_slice(personal);
        let mut h = IV;
        for i in 0..8 {
            let mut w = [0u8; 8];
            w.copy_from_slice(&p[8 * i..8 * i + 8]);
            h[i] ^= u64::from_le_bytes(w);
        }
        Blake2b {
            h,
            buf: [0u8; 128],
            buf_len: 0,
            t: 0,
            nn,
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        while !data.is_empty() {
            // Keep the last (possibly full) block buffered: it must be compressed with the final flag.
            if self.buf_len == 128 {
                self.t += 128;
                let blk = self.buf;
                compress(&mut self.h, &blk, self.t, false);
                self.buf_len = 0;
            }
            let take = core::cmp::min(128 - self.buf_len, data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
        }
    }

    pub fn finalize(mut self) -> Vec<u8> {
        self.t += self.buf_len as u128;
        for b in self.buf[self.buf_len..].iter_mut() {
            *b = 0;
        }
        let blk = self.buf;
        compress(&mut self.h, &blk, self.t, true);
        let mut out = Vec::with_capacity(64);
        for w in self.h.iter() {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out.truncate(self.nn);
        out
    }
}

/// One-shot BLAKE2b.
pub fn blake2b(nn: usize, personal: &[u8], data: &[u8]) -> Vec<u8> {
    let mut s = Blake2b::new(nn, personal);
    s.update(data);
    s.finalize()
}

/// BLAKE2b-256 without personalization (FPN2 `profile_id`).
pub fn blake2b_256(data: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    out.copy_from_slice(&blake2b(32, &[], data));
    out
}

/// BLAKE2b-512 with personalization.
pub fn blake2b_512(personal: &[u8], data: &[u8]) -> [u8; 64] {
    let mut out = [0u8; 64];
    out.copy_from_slice(&blake2b(64, personal, data));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{:02x}", x)).collect()
    }

    #[test]
    fn rfc7693_abc() {
        // RFC 7693 Appendix A.
        assert_eq!(
            hex(&blake2b(64, &[], b"abc")),
            "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d1\
             7d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923"
        );
    }

    #[test]
    fn empty_and_incremental() {
        // BLAKE2b-512("") (widely published KAT).
        assert_eq!(
            hex(&blake2b(64, &[], b"")),
            "786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419\
             d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce"
        );
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 31 + 7) as u8).collect();
        for split in [0usize, 1, 127, 128, 129, 256, 999, 1000] {
            let mut s = Blake2b::new(64, b"x");
            s.update(&data[..split]);
            s.update(&data[split..]);
            assert_eq!(s.finalize(), blake2b(64, b"x", &data));
        }
    }
}
