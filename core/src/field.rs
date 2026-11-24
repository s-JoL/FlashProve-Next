// SPDX-License-Identifier: Apache-2.0
//! Goldilocks field `F_p`, `p = 2^64 - 2^32 + 1`, and the cubic extension `E = F_p[u]/(u^3 - 2)`.
//!
//! Bit-exact with `flashprove_next/field.py` (`hmul`, `hinv`, `hpow`, `lagrange_eval`, `eq_pt`).

// Explicit, auditable method names (`a.mul(b)`) mirror field.py; operator traits are deliberately not used.
#![allow(clippy::should_implement_trait)]

/// The Goldilocks prime.
pub const P: u64 = 0xFFFF_FFFF_0000_0001;

/// Canonical element of `F_p` (always `< P`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Fp(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NonCanonical;

impl Fp {
    pub const ZERO: Fp = Fp(0);
    pub const ONE: Fp = Fp(1);

    /// Reduces any `u64` modulo `P`.
    pub const fn new(x: u64) -> Fp {
        Fp(if x >= P { x - P } else { x })
    }
    /// Accepts only canonical values `< P`.
    pub fn from_canonical(x: u64) -> Result<Fp, NonCanonical> {
        if x < P {
            Ok(Fp(x))
        } else {
            Err(NonCanonical)
        }
    }
    pub fn from_u128(x: u128) -> Fp {
        Fp((x % (P as u128)) as u64)
    }
    pub const fn value(self) -> u64 {
        self.0
    }
    pub fn add(self, o: Fp) -> Fp {
        let s = self.0 as u128 + o.0 as u128;
        Fp(if s >= P as u128 {
            (s - P as u128) as u64
        } else {
            s as u64
        })
    }
    pub fn sub(self, o: Fp) -> Fp {
        if self.0 >= o.0 {
            Fp(self.0 - o.0)
        } else {
            Fp((self.0 as u128 + P as u128 - o.0 as u128) as u64)
        }
    }
    pub fn neg(self) -> Fp {
        Fp::ZERO.sub(self)
    }
    pub fn mul(self, o: Fp) -> Fp {
        Fp(((self.0 as u128 * o.0 as u128) % P as u128) as u64)
    }
    pub fn square(self) -> Fp {
        self.mul(self)
    }
    pub fn pow(self, mut e: u64) -> Fp {
        let mut r = Fp::ONE;
        let mut b = self;
        while e != 0 {
            if e & 1 == 1 {
                r = r.mul(b);
            }
            b = b.square();
            e >>= 1;
        }
        r
    }
    /// Multiplicative inverse; `None` for zero.
    pub fn inv(self) -> Option<Fp> {
        if self.0 == 0 {
            None
        } else {
            Some(self.pow(P - 2))
        }
    }
    pub fn to_le_bytes(self) -> [u8; 8] {
        self.0.to_le_bytes()
    }
    /// Canonical 8-byte little-endian decoding (rejects `>= P`).
    pub fn decode(b: &[u8]) -> Result<Fp, NonCanonical> {
        let arr: [u8; 8] = b.try_into().map_err(|_| NonCanonical)?;
        Fp::from_canonical(u64::from_le_bytes(arr))
    }
}

/// Element `a0 + a1 u + a2 u^2` of `E = F_p[u]/(u^3 - 2)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Fp3(pub [Fp; 3]);

impl Fp3 {
    pub const ZERO: Fp3 = Fp3([Fp::ZERO; 3]);
    pub const ONE: Fp3 = Fp3([Fp::ONE, Fp::ZERO, Fp::ZERO]);

    pub fn from_fp(x: Fp) -> Fp3 {
        Fp3([x, Fp::ZERO, Fp::ZERO])
    }
    pub fn from_u64(x: u64) -> Fp3 {
        Fp3::from_fp(Fp::new(x))
    }
    pub fn add(self, o: Fp3) -> Fp3 {
        Fp3([
            self.0[0].add(o.0[0]),
            self.0[1].add(o.0[1]),
            self.0[2].add(o.0[2]),
        ])
    }
    pub fn sub(self, o: Fp3) -> Fp3 {
        Fp3([
            self.0[0].sub(o.0[0]),
            self.0[1].sub(o.0[1]),
            self.0[2].sub(o.0[2]),
        ])
    }
    pub fn neg(self) -> Fp3 {
        Fp3::ZERO.sub(self)
    }
    pub fn scale(self, c: Fp) -> Fp3 {
        Fp3([self.0[0].mul(c), self.0[1].mul(c), self.0[2].mul(c)])
    }
    pub fn mul(self, o: Fp3) -> Fp3 {
        let [a0, a1, a2] = self.0;
        let [b0, b1, b2] = o.0;
        let two = Fp::new(2);
        let c0 = a0.mul(b0).add(two.mul(a1.mul(b2).add(a2.mul(b1))));
        let c1 = a0.mul(b1).add(a1.mul(b0)).add(two.mul(a2.mul(b2)));
        let c2 = a0.mul(b2).add(a1.mul(b1)).add(a2.mul(b0));
        Fp3([c0, c1, c2])
    }
    pub fn square(self) -> Fp3 {
        self.mul(self)
    }
    pub fn pow(self, e: u64) -> Fp3 {
        self.pow_limbs(&[e])
    }
    /// Exponentiation by `Σ limbs[i] · 2^(64 i)`.
    pub fn pow_limbs(self, limbs: &[u64]) -> Fp3 {
        let mut r = Fp3::ONE;
        let mut b = self;
        for &l in limbs {
            let mut e = l;
            for _ in 0..64 {
                if e & 1 == 1 {
                    r = r.mul(b);
                }
                b = b.square();
                e >>= 1;
            }
        }
        r
    }
    /// Inverse via the norm map (`field.py::hinv`); `None` for zero.
    pub fn inv(self) -> Option<Fp3> {
        let [a0, a1, a2] = self.0;
        let two = Fp::new(2);
        let t0 = a0.square().sub(two.mul(a1.mul(a2)));
        let t1 = two.mul(a2.square()).sub(a0.mul(a1));
        let t2 = a1.square().sub(a0.mul(a2));
        let n = a0.mul(t0).add(two.mul(a2.mul(t1).add(a1.mul(t2))));
        let ni = n.inv()?;
        Some(Fp3([t0.mul(ni), t1.mul(ni), t2.mul(ni)]))
    }
    pub fn to_le_bytes(self) -> [u8; 24] {
        let mut out = [0u8; 24];
        for i in 0..3 {
            out[8 * i..8 * i + 8].copy_from_slice(&self.0[i].to_le_bytes());
        }
        out
    }
    /// Canonical 24-byte decoding (every limb `< P`).
    pub fn decode(b: &[u8]) -> Result<Fp3, NonCanonical> {
        if b.len() != 24 {
            return Err(NonCanonical);
        }
        Ok(Fp3([
            Fp::decode(&b[0..8])?,
            Fp::decode(&b[8..16])?,
            Fp::decode(&b[16..24])?,
        ]))
    }
}

/// The degree-`<= d` polynomial with `g(j) = vals[j]` (`j = 0..=d`), evaluated at `r`.
pub fn lagrange_eval(vals: &[Fp3], r: Fp3) -> Fp3 {
    let d = vals.len();
    let mut out = Fp3::ZERO;
    for i in 0..d {
        let mut num = Fp3::ONE;
        let mut den = Fp::ONE;
        for j in 0..d {
            if j != i {
                num = num.mul(r.sub(Fp3::from_u64(j as u64)));
                den = den.mul(Fp::new(i as u64).sub(Fp::new(j as u64)));
            }
        }
        // den is a product of nonzero small differences, hence invertible.
        let inv = den.inv().unwrap_or(Fp::ZERO);
        out = out.add(vals[i].mul(num).scale(inv));
    }
    out
}

/// `eq(a, b) = Π (a_i b_i + (1 - a_i)(1 - b_i))` over the common prefix.
pub fn eq_pt(a: &[Fp3], b: &[Fp3]) -> Fp3 {
    let mut out = Fp3::ONE;
    for (x, y) in a.iter().zip(b.iter()) {
        let t = x.mul(*y).add(Fp3::ONE.sub(*x).mul(Fp3::ONE.sub(*y)));
        out = out.mul(t);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edges() {
        assert_eq!(Fp::new(P - 1).add(Fp::ONE), Fp::ZERO);
        assert_eq!(Fp::ZERO.sub(Fp::ONE), Fp::new(P - 1));
        assert_eq!(Fp::new(1 << 32).square(), Fp::new((1u64 << 32) - 1)); // 2^64 = 2^32 - 1 mod p
        assert!(Fp::from_canonical(P).is_err());
        assert_eq!(Fp::ZERO.inv(), None);
        let x = Fp3([Fp::new(3), Fp::new(5), Fp::new(7)]);
        assert_eq!(x.mul(x.inv().unwrap()), Fp3::ONE);
        // u^3 = 2
        let u = Fp3([Fp::ZERO, Fp::ONE, Fp::ZERO]);
        assert_eq!(u.mul(u).mul(u), Fp3::from_u64(2));
    }
}
