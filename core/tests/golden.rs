// SPDX-License-Identifier: Apache-2.0
//! fpn-core against specs/vectors/ (the same files the Python reference suite consumes).

#![allow(clippy::needless_range_loop)]
mod common;
use common::*;

use fpn_core::blake2b::blake2b;
use fpn_core::field::{eq_pt, lagrange_eval, Fp, Fp3, P};
use fpn_core::poseidon2 as p2;
use fpn_core::profile::{parse_hex32, REGISTRY};
use fpn_core::transcript::Fpn2Transcript;
use fpn_core::{verify_bytes, Fpn2Container, Policy};

fn fp(s: &J) -> Fp {
    Fp::from_canonical(u64h(s.str())).expect("canonical")
}
fn fp3(s: &J) -> Fp3 {
    let a = s.arr();
    Fp3([fp(&a[0]), fp(&a[1]), fp(&a[2])])
}
fn digest(s: &J) -> p2::Digest {
    let a = s.arr();
    [fp(&a[0]), fp(&a[1]), fp(&a[2]), fp(&a[3])]
}

#[test]
fn field_vectors() {
    let prim = load("fpn2_primitives.json");
    assert_eq!(u64h(prim.get("p").str()), P);
    let mut n = 0;
    for v in prim.get("field").arr() {
        n += 1;
        let op = v.get("op").str();
        let out = v.get("out");
        match op {
            "fp_add" => assert_eq!(fp(v.get("a")).add(fp(v.get("b"))), fp(out)),
            "fp_sub" => assert_eq!(fp(v.get("a")).sub(fp(v.get("b"))), fp(out)),
            "fp_mul" => assert_eq!(fp(v.get("a")).mul(fp(v.get("b"))), fp(out)),
            "fp_neg" => assert_eq!(fp(v.get("a")).neg(), fp(out)),
            "fp_inv" => {
                if out.str() == "error" {
                    assert_eq!(fp(v.get("a")).inv(), None)
                } else {
                    assert_eq!(fp(v.get("a")).inv(), Some(fp(out)))
                }
            }
            "fp_pow" => assert_eq!(fp(v.get("a")).pow(u64h(v.get("e").str())), fp(out)),
            "fp_decode" => {
                let b = unhex(v.get("bytes").str());
                if out.str() == "noncanonical" {
                    assert!(Fp::decode(&b).is_err())
                } else {
                    assert_eq!(Fp::decode(&b).unwrap(), fp(out));
                    assert_eq!(fp(out).to_le_bytes().to_vec(), b);
                }
            }
            "fp3_add" => assert_eq!(fp3(v.get("a")).add(fp3(v.get("b"))), fp3(out)),
            "fp3_sub" => assert_eq!(fp3(v.get("a")).sub(fp3(v.get("b"))), fp3(out)),
            "fp3_mul" => assert_eq!(fp3(v.get("a")).mul(fp3(v.get("b"))), fp3(out)),
            "fp3_neg" => assert_eq!(fp3(v.get("a")).neg(), fp3(out)),
            "fp3_inv" => {
                if out.is_null() || matches!(out, J::Str(s) if s == "error") {
                    assert_eq!(fp3(v.get("a")).inv(), None)
                } else {
                    assert_eq!(fp3(v.get("a")).inv(), Some(fp3(out)))
                }
            }
            "fp3_pow" => assert_eq!(
                fp3(v.get("a")).pow_limbs(&limbs(v.get("e").str())),
                fp3(out)
            ),
            "fp3_lagrange_eval" => {
                let vals: Vec<Fp3> = v.get("vals").arr().iter().map(fp3).collect();
                assert_eq!(lagrange_eval(&vals, fp3(v.get("r"))), fp3(out));
            }
            "fp3_eq_pt" => {
                let a: Vec<Fp3> = v.get("a").arr().iter().map(fp3).collect();
                let b: Vec<Fp3> = v.get("b").arr().iter().map(fp3).collect();
                assert_eq!(eq_pt(&a, &b), fp3(out));
            }
            "fp3_decode" => {
                let b = unhex(v.get("bytes").str());
                if out.str_or_none() == Some("noncanonical") {
                    assert!(Fp3::decode(&b).is_err())
                } else {
                    assert_eq!(Fp3::decode(&b).unwrap(), fp3(out));
                    assert_eq!(fp3(out).to_le_bytes().to_vec(), b);
                }
            }
            _ => panic!("unknown op {op}"),
        }
    }
    assert!(n > 500);
}

trait StrOrNone {
    fn str_or_none(&self) -> Option<&str>;
}
impl StrOrNone for J {
    fn str_or_none(&self) -> Option<&str> {
        match self {
            J::Str(s) => Some(s),
            _ => None,
        }
    }
}

#[test]
fn poseidon2_vectors() {
    let prim = load("fpn2_primitives.json");
    let v = prim.get("poseidon2");
    assert_eq!(v.get("dom_leaf").num() as u64, p2::DOM_LEAF);
    for t in v.get("permute").arr() {
        let mut st = [Fp::ZERO; 12];
        let mut ex = [Fp::ZERO; 12];
        for i in 0..12 {
            st[i] = fp(&t.get("in").arr()[i]);
            ex[i] = fp(&t.get("out").arr()[i]);
        }
        assert_eq!(p2::permute(&st), ex);
    }
    for t in v.get("compress").arr() {
        assert_eq!(
            p2::compress(&digest(t.get("left")), &digest(t.get("right"))),
            digest(t.get("out"))
        );
    }
    for t in v.get("sponge_blocks").arr() {
        let blocks: Vec<[Fp; 8]> = t
            .get("blocks")
            .arr()
            .iter()
            .map(|b| {
                let mut o = [Fp::ZERO; 8];
                for i in 0..8 {
                    o[i] = fp(&b.arr()[i]);
                }
                o
            })
            .collect();
        assert_eq!(
            p2::sponge_blocks(&blocks, t.get("domain").num() as u64),
            digest(t.get("out"))
        );
    }
    for t in v.get("sponge_msg").arr() {
        let msg: Vec<Fp> = t.get("msg").arr().iter().map(fp).collect();
        assert_eq!(
            p2::sponge_msg(&msg, t.get("domain").num() as u64),
            digest(t.get("out"))
        );
    }
    for t in v.get("merkle").arr() {
        let leaves: Vec<p2::Digest> = t.get("leaves").arr().iter().map(digest).collect();
        let levels = p2::merkle_levels(&leaves).unwrap();
        let root = digest(t.get("root"));
        assert_eq!(p2::merkle_root(&leaves), Some(root));
        for p in t.get("paths").arr() {
            let idx = p.get("idx").num() as usize;
            let path: Vec<p2::Digest> = p.get("path").arr().iter().map(digest).collect();
            assert_eq!(p2::merkle_path(&levels, idx), path);
            assert!(p2::verify_path(&root, &leaves[idx], idx as u64, &path));
            if !path.is_empty() {
                assert!(!p2::verify_path(
                    &root,
                    &leaves[idx],
                    (idx ^ 1) as u64,
                    &path
                ));
            }
        }
    }
}

#[test]
fn blake2b_vectors() {
    let prim = load("fpn2_primitives.json");
    let mut n = 0;
    for t in prim.get("blake2b").arr() {
        let out = blake2b(
            t.get("nn").num() as usize,
            &unhex(t.get("person").str()),
            &unhex(t.get("msg").str()),
        );
        assert_eq!(hex(&out), t.get("out").str());
        n += 1;
    }
    assert!(n >= 70);
}

#[test]
fn transcript_vectors() {
    let prim = load("fpn2_primitives.json");
    for script in prim.get("transcript").arr() {
        let mut tr: Option<Fpn2Transcript> = None;
        for st in script.arr() {
            let op = st.get("op").str();
            if op == "init" {
                tr = Some(Fpn2Transcript::new(&unhex(st.get("label").str())));
            } else {
                let t = tr.as_mut().unwrap();
                match op {
                    "absorb_bytes" => t.absorb_bytes(&unhex(st.get("data").str())),
                    "absorb_fp" => t.absorb_fp(fp(st.get("x"))),
                    "absorb_fp3" => t.absorb_fp3(fp3(st.get("a"))),
                    "squeeze_bytes" => {
                        assert_eq!(
                            hex(&t.squeeze_bytes(st.get("n").num() as usize)),
                            st.get("out").str()
                        )
                    }
                    "squeeze_fp3" => assert_eq!(t.squeeze_fp3(), fp3(st.get("out"))),
                    "squeeze_u192" => {
                        let mut want = limbs(st.get("out").str());
                        want.resize(3, 0);
                        assert_eq!(t.squeeze_u192().to_vec(), want);
                    }
                    _ => panic!("unknown op {op}"),
                }
            }
            assert_eq!(
                hex(tr.as_ref().unwrap().state()),
                st.get("state").str(),
                "{op}"
            );
        }
    }
}

#[test]
fn profile_vectors() {
    let prim = load("fpn2_primitives.json");
    let profs = prim.get("profiles").arr();
    assert_eq!(profs.len(), REGISTRY.len());
    for t in profs {
        let (p, pin) = REGISTRY
            .iter()
            .find(|(p, _)| p.name == t.get("name").str())
            .expect("registered");
        assert_eq!(hex(&p.canonical_bytes()), t.get("canonical_hex").str());
        assert_eq!(*pin, t.get("profile_id").str());
        assert_eq!(Some(p.profile_id()), parse_hex32(pin));
        assert_eq!(p.c_p, t.get("c_p").num() as u64);
    }
}

fn policy_of(v: &J) -> Policy {
    let pin = v.get("expected_profile_id");
    Policy {
        allow_reduced: v.get("allow_reduced").bool(),
        expected_profile_id: if pin.is_null() {
            None
        } else {
            parse_hex32(pin.str())
        },
        expected_circuit_root: None,
        expected_x_root: None,
        expected_stmt_digest: None,
    }
}

fn result_str(r: Result<(), fpn_core::RejectCode>) -> &'static str {
    match r {
        Ok(()) => "VERIFIED",
        Err(c) => c.as_str(),
    }
}

#[test]
fn container_vectors() {
    let cont = load("fpn2_containers.json");
    let mut n = 0;
    for v in cont.get("vectors").arr() {
        let data = container_bytes(v);
        let got = result_str(verify_bytes(&data, &policy_of(v)));
        assert_eq!(got, v.get("expect").str(), "vector {}", v.get("name").str());
        n += 1;
    }
    assert!(n >= 30);
}

#[test]
fn valid_containers_reserialize_identically() {
    let cont = load("fpn2_containers.json");
    let pol = Policy {
        allow_reduced: true,
        expected_profile_id: None,
        expected_circuit_root: None,
        expected_x_root: None,
        expected_stmt_digest: None,
    };
    let mut n = 0;
    for v in cont.get("vectors").arr() {
        let data = container_bytes(v);
        match Fpn2Container::parse(&data, &pol) {
            Ok(c) => {
                assert_eq!(c.serialize().unwrap(), data, "{}", v.get("name").str());
                n += 1;
            }
            Err(_) => assert_ne!(v.get("expect").str(), "VERIFIED"),
        }
    }
    assert!(n >= 3);
    // Strict parser alone (without the verifier) enforces padding and the security tier.
    let v0 = cont
        .get("vectors")
        .arr()
        .iter()
        .find(|v| v.get("name").str() == "valid_harness_0")
        .unwrap();
    let mut bad = container_bytes(v0);
    let last = bad.len() - 1;
    bad[last] = 1;
    assert_eq!(
        Fpn2Container::parse(&bad, &pol).err(),
        Some(fpn_core::RejectCode::Padding)
    );
    assert_eq!(
        Fpn2Container::parse(&container_bytes(v0), &Policy::default()).err(),
        Some(fpn_core::RejectCode::ReducedSecurity)
    );
}

#[test]
fn every_bit_flip_rejects_and_no_panic_on_truncation() {
    let cont = load("fpn2_containers.json");
    let v = cont
        .get("vectors")
        .arr()
        .iter()
        .find(|v| v.get("name").str() == "valid_harness_0")
        .unwrap();
    let base = container_bytes(v);
    let pol = Policy {
        allow_reduced: true,
        expected_profile_id: None,
        expected_circuit_root: None,
        expected_x_root: None,
        expected_stmt_digest: None,
    };
    assert!(verify_bytes(&base, &pol).is_ok());
    for i in 0..base.len() {
        for bit in 0..8 {
            let mut b = base.clone();
            b[i] ^= 1 << bit;
            assert!(
                verify_bytes(&b, &pol).is_err(),
                "byte {i} bit {bit} accepted"
            );
        }
    }
    for n in 0..base.len() {
        assert_eq!(
            verify_bytes(&base[..n], &pol),
            Err(fpn_core::RejectCode::Length)
        );
    }
}
