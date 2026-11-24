// SPDX-License-Identifier: Apache-2.0
//! Relation 6 (`fpn2-folded-vec-bound-n16-v1`, `-n22-v1`) against specs/vectors/fpn2_folded_vec_bound.json (the same
//! file the detached Python reference suite consumes): pins, case-(c) forgeries, adversarial provers, mutations.

mod common;
use common::*;

use fpn_core::folded_vec_bound::BoundLayout;
use fpn_core::profile::{
    parse_hex32, FOLDED_VEC_BOUND_N16_V1, FOLDED_VEC_BOUND_N22_V1, FOLDED_VEC_N16_V1,
};
use fpn_core::wire::Body;
use fpn_core::{verify_bytes, Fpn2Container, Policy, RejectCode};

fn find<'a>(all: &'a [J], name: &str) -> &'a J {
    all.iter()
        .find(|v| v.get("name").str() == name)
        .unwrap_or_else(|| panic!("vector {name}"))
}

fn build(all: &[J], v: &J) -> Vec<u8> {
    if v.has("prefix_hex") {
        return container_bytes(v);
    }
    let mut b = build(all, find(all, v.get("base").str()));
    for p in v.get("patches").arr() {
        let off = p.arr()[0].num() as usize;
        let by = unhex(p.arr()[1].str());
        b[off..off + by.len()].copy_from_slice(&by);
    }
    let t = v.get("truncate");
    if !t.is_null() {
        b.truncate(t.num() as usize);
    }
    b.extend_from_slice(&unhex(v.get("append_hex").str()));
    b
}

fn hex_opt(j: &J) -> Option<[u8; 32]> {
    if j.is_null() {
        None
    } else {
        parse_hex32(j.str())
    }
}

fn policy_of(v: &J) -> Policy {
    Policy {
        allow_reduced: v.get("allow_reduced").bool(),
        expected_profile_id: hex_opt(v.get("expected_profile_id")),
        expected_circuit_root: hex_opt(v.get("expected_circuit_root")),
        expected_x_root: hex_opt(v.get("expected_x_root")),
        expected_stmt_digest: None,
    }
}

fn res(r: Result<(), RejectCode>) -> &'static str {
    match r {
        Ok(()) => "VERIFIED",
        Err(c) => c.as_str(),
    }
}

#[test]
fn bound_vectors_exact_codes() {
    let f = load("fpn2_folded_vec_bound.json");
    let all = f.get("vectors").arr();
    let mut codes = std::collections::BTreeSet::new();
    for v in all {
        let got = res(verify_bytes(&build(all, v), &policy_of(v)));
        assert_eq!(got, v.get("expect").str(), "vector {}", v.get("name").str());
        codes.insert(got);
    }
    assert!(all.len() >= 35, "{} vectors", all.len());
    for c in [
        "VERIFIED",
        "E_PROFILE_PINNED",
        "E_CIRCUIT_UNPINNED",
        "E_CIRCUIT_PINNED",
        "E_X_ROOT_UNPINNED",
        "E_X_ROOT_PINNED",
        "E_ZEROCHECK",
        "E_SUMCHECK",
        "E_FINAL",
        "E_MERKLE",
        "E_FOLD",
        "E_NONCANONICAL",
        "E_PADDING",
        "E_LENGTH",
    ] {
        assert!(codes.contains(c), "{c}");
    }
    for name in [
        "vec_bound_n16_valid_0",
        "vec_bound_n16_valid_other_x",
        "vec_bound_n22_valid_multi_tile",
        "vec_bound_n16_case_c_substituted_x_under_honest_x_root",
        "vec_bound_n16_case_c_substituted_x_own_x_root",
        "vec_bound_n16_case_c_forged_e_x",
        "vec_bound_n16_tamper_x_leaf",
        "vec_bound_n16_tamper_x_path",
    ] {
        find(all, name);
    }
}

#[test]
fn bound_case_c_forgeries_are_rejected() {
    let f = load("fpn2_folded_vec_bound.json");
    let all = f.get("vectors").arr();
    for (name, code) in [
        (
            "vec_bound_n16_case_c_substituted_x_under_honest_x_root",
            "E_MERKLE",
        ),
        (
            "vec_bound_n16_case_c_substituted_x_own_x_root",
            "E_X_ROOT_PINNED",
        ),
        ("vec_bound_n16_case_c_forged_e_x", "E_SUMCHECK"),
    ] {
        let v = find(all, name);
        assert_eq!(res(verify_bytes(&build(all, v), &policy_of(v))), code, "{name}");
    }
}

#[test]
fn bound_profiles_pinned_and_layout() {
    let f = load("fpn2_folded_vec_bound.json");
    let p = f.get("profiles");
    assert_eq!(
        parse_hex32(p.get("fpn2-folded-vec-bound-n16-v1").str()),
        Some(FOLDED_VEC_BOUND_N16_V1.profile_id())
    );
    assert_eq!(
        parse_hex32(p.get("fpn2-folded-vec-bound-n22-v1").str()),
        Some(FOLDED_VEC_BOUND_N22_V1.profile_id())
    );
    // relation-6 profiles are distinct from relation 5's (profile_id binds the relation byte)
    assert_ne!(
        FOLDED_VEC_BOUND_N16_V1.profile_id(),
        FOLDED_VEC_N16_V1.profile_id()
    );
    let end = f
        .get("offsets")
        .get("fpn2-folded-vec-bound-n16-v1")
        .get("end")
        .num() as usize;
    assert_eq!(64 + FOLDED_VEC_BOUND_N16_V1.body_used().unwrap(), end);
    let lay = BoundLayout::of(&FOLDED_VEC_BOUND_N22_V1).unwrap();
    assert_eq!(lay.0.tiles() * 32 * lay.0.kdim(), 2_097_152);
    assert!(64 + lay.body_used() <= FOLDED_VEC_BOUND_N22_V1.c_p as usize);
}

#[test]
fn bound_valid_reserialize_identically() {
    let f = load("fpn2_folded_vec_bound.json");
    let all = f.get("vectors").arr();
    for name in ["vec_bound_n16_valid_0", "vec_bound_n22_valid_multi_tile"] {
        let v = find(all, name);
        let data = build(all, v);
        let c = Fpn2Container::parse(&data, &policy_of(v)).expect("parse");
        assert!(matches!(c.body, Body::FoldedVecBound(_)));
        assert_eq!(c.serialize().unwrap(), data);
    }
}

/// Flip one bit of byte `i` (bit `i % 8`, or every bit when `all_bits`) and require rejection.
fn flip_sweep(base: &[u8], pol: &Policy, idx: &[usize], all_bits: bool) {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    std::thread::scope(|sc| {
        for tid in 0..threads {
            sc.spawn(move || {
                let mut b = base.to_vec();
                for &i in idx.iter().skip(tid).step_by(threads) {
                    let bits: Vec<u32> = if all_bits {
                        (0..8).collect()
                    } else {
                        vec![(i % 8) as u32]
                    };
                    for bit in bits {
                        b[i] ^= 1 << bit;
                        assert!(verify_bytes(&b, pol).is_err(), "byte {i} bit {bit} accepted");
                        b[i] ^= 1 << bit;
                    }
                }
            });
        }
    });
}

#[test]
fn bound_every_byte_flip_rejects() {
    let f = load("fpn2_folded_vec_bound.json");
    let all = f.get("vectors").arr();
    let v = find(all, "vec_bound_n16_valid_0");
    let pol = policy_of(v);
    let base = build(all, v);
    assert!(verify_bytes(&base, &pol).is_ok());
    // ALL 8 bits of every byte of the header, statement, inner-product sumcheck and PCS rounds.
    let dense = 64 + 128 + 24 + 96 * 10 + 48 + 72 * 18 + 32 * 7 + 24;
    let head: Vec<usize> = (0..dense).collect();
    flip_sweep(&base, &pol, &head, true);
    // EVERY byte of the container (queries and zero padding included), one bit per byte (bit = index mod 8).
    let every: Vec<usize> = (dense..base.len()).collect();
    flip_sweep(&base, &pol, &every, false);
    for n in (0..base.len()).step_by(4099) {
        assert_eq!(verify_bytes(&base[..n], &pol), Err(RejectCode::Length));
    }
}

#[test]
fn bound_cli_requires_and_checks_both_pins() {
    let f = load("fpn2_folded_vec_bound.json");
    let all = f.get("vectors").arr();
    let root = f.get("weights").get("tile_n16").str().to_string();
    let other = f.get("weights").get("tile_n16_tampered").str().to_string();
    let xr = f.get("x_roots").get("x_n16").str().to_string();
    let xo = f.get("x_roots").get("x_n16_other").str().to_string();
    let dir = std::env::temp_dir().join(format!("fpn-verify-bound-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("bound_n16.fpn2");
    std::fs::write(&p, build(all, find(all, "vec_bound_n16_valid_0"))).unwrap();
    let cases: Vec<(Vec<&str>, &str, i32)> = vec![
        (
            vec!["--weight-root", &root, "--x-root", &xr],
            "VERIFIED",
            0,
        ),
        (
            vec!["--circuit-root", &root, "--x-root", &xr],
            "VERIFIED",
            0,
        ),
        (vec!["--x-root", &xr], "REJECTED: E_CIRCUIT_UNPINNED", 1),
        (vec!["--weight-root", &root], "REJECTED: E_X_ROOT_UNPINNED", 1),
        (vec![], "REJECTED: E_CIRCUIT_UNPINNED", 1),
        (
            vec!["--weight-root", &other, "--x-root", &xr],
            "REJECTED: E_CIRCUIT_PINNED",
            1,
        ),
        (
            vec!["--weight-root", &root, "--x-root", &xo],
            "REJECTED: E_X_ROOT_PINNED",
            1,
        ),
        (
            vec![
                "--weight-root",
                &root,
                "--x-root",
                &xr,
                "--profile",
                "fpn2-folded-vec-bound-n16-v1",
            ],
            "VERIFIED",
            0,
        ),
    ];
    for (args, want, code) in cases {
        let out = std::process::Command::new(env!("CARGO_BIN_EXE_fpn-verify"))
            .arg(&p)
            .args(&args)
            .output()
            .unwrap();
        assert_eq!(
            String::from_utf8_lossy(&out.stdout).trim(),
            want,
            "{args:?}"
        );
        assert_eq!(out.status.code(), Some(code), "{args:?}");
    }
    for args in [vec!["--x-root", "zz"], vec!["--weight-root", "00"]] {
        let out = std::process::Command::new(env!("CARGO_BIN_EXE_fpn-verify"))
            .arg(&p)
            .args(&args)
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(2), "{args:?}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn non_relation6_proof_cannot_satisfy_x_root_or_unsupported_circuit_pin() {
    // Cross-relation pin guard: supplying --x-root MUST reject a valid Relation 5 (or 4/3/1) proof
    // instead of silently ignoring the activation pin, and --circuit-root MUST reject Relations 1/3.
    let fv5 = load("fpn2_folded_vec.json");
    let v5 = find(fv5.get("vectors").arr(), "vec_n16_valid_0");
    let b5 = build(fv5.get("vectors").arr(), v5);
    let w5 = hex_opt(v5.get("expected_circuit_root"));
    let any_x = Some([0x42u8; 32]);
    assert_eq!(
        res(verify_bytes(
            &b5,
            &Policy {
                allow_reduced: false,
                expected_profile_id: None,
                expected_circuit_root: w5,
                expected_x_root: None,
        expected_stmt_digest: None,
            }
        )),
        "VERIFIED"
    );
    assert_eq!(
        res(verify_bytes(
            &b5,
            &Policy {
                allow_reduced: false,
                expected_profile_id: None,
                expected_circuit_root: w5,
                expected_x_root: any_x,
                expected_stmt_digest: None,
            }
        )),
        "E_X_ROOT_PINNED"
    );
    // CLI check on Relation 5 with valid --weight-root plus --x-root must reject with E_X_ROOT_PINNED (exit 1)
    let dir = std::env::temp_dir().join(format!("fpn-r5-xpin-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p5 = dir.join("r5.fpn2");
    std::fs::write(&p5, &b5).unwrap();
    let w5_hex = v5.get("expected_circuit_root").str();
    let xr_hex = "42".repeat(32);
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_fpn-verify"))
        .arg(&p5)
        .args(["--weight-root", w5_hex, "--x-root", &xr_hex])
        .output()
        .unwrap();
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        "REJECTED: E_X_ROOT_PINNED"
    );
    assert_eq!(out.status.code(), Some(1));
    std::fs::remove_dir_all(&dir).ok();

    // Relation 4 with --x-root -> E_X_ROOT_PINNED
    let fv4 = load("fpn2_folded_air.json");
    let v4 = find(fv4.get("vectors").arr(), "air_v1_valid_0");
    let b4 = build(fv4.get("vectors").arr(), v4);
    assert_eq!(
        res(verify_bytes(
            &b4,
            &Policy {
                allow_reduced: false,
                expected_profile_id: None,
                expected_circuit_root: hex_opt(v4.get("expected_circuit_root")),
                expected_x_root: any_x,
                expected_stmt_digest: None,
            }
        )),
        "E_X_ROOT_PINNED"
    );

    // Relation 3 with --circuit-root -> E_CIRCUIT_PINNED; with --x-root -> E_X_ROOT_PINNED; both -> E_CIRCUIT_PINNED
    let fv3 = load("fpn2_folded_gkr.json");
    let v3 = find(fv3.get("vectors").arr(), "folded_v1_valid_0");
    let b3 = build(fv3.get("vectors").arr(), v3);
    assert_eq!(
        res(verify_bytes(
            &b3,
            &Policy {
                allow_reduced: false,
                expected_profile_id: None,
                expected_circuit_root: any_x,
                expected_x_root: None,
        expected_stmt_digest: None,
            }
        )),
        "E_CIRCUIT_PINNED"
    );
    assert_eq!(
        res(verify_bytes(
            &b3,
            &Policy {
                allow_reduced: false,
                expected_profile_id: None,
                expected_circuit_root: None,
                expected_x_root: any_x,
                expected_stmt_digest: None,
            }
        )),
        "E_X_ROOT_PINNED"
    );
    assert_eq!(
        res(verify_bytes(
            &b3,
            &Policy {
                allow_reduced: false,
                expected_profile_id: None,
                expected_circuit_root: any_x,
                expected_x_root: any_x,
                expected_stmt_digest: None,
            }
        )),
        "E_CIRCUIT_PINNED"
    );

    // Relation 1 (harness): unpinned VERIFIED; circuit pin -> E_CIRCUIT_PINNED; x pin -> E_X_ROOT_PINNED; both -> E_CIRCUIT_PINNED
    let fv1 = load("fpn2_containers.json");
    let v1 = find(fv1.get("vectors").arr(), "valid_harness_0");
    let b1 = build(fv1.get("vectors").arr(), v1);
    assert_eq!(
        res(verify_bytes(
            &b1,
            &Policy {
                allow_reduced: true,
                expected_profile_id: None,
                expected_circuit_root: None,
                expected_x_root: None,
        expected_stmt_digest: None,
            }
        )),
        "VERIFIED"
    );
    assert_eq!(
        res(verify_bytes(
            &b1,
            &Policy {
                allow_reduced: true,
                expected_profile_id: None,
                expected_circuit_root: any_x,
                expected_x_root: None,
        expected_stmt_digest: None,
            }
        )),
        "E_CIRCUIT_PINNED"
    );
    assert_eq!(
        res(verify_bytes(
            &b1,
            &Policy {
                allow_reduced: true,
                expected_profile_id: None,
                expected_circuit_root: None,
                expected_x_root: any_x,
                expected_stmt_digest: None,
            }
        )),
        "E_X_ROOT_PINNED"
    );
    assert_eq!(
        res(verify_bytes(
            &b1,
            &Policy {
                allow_reduced: true,
                expected_profile_id: None,
                expected_circuit_root: any_x,
                expected_x_root: any_x,
                expected_stmt_digest: None,
            }
        )),
        "E_CIRCUIT_PINNED"
    );
}
