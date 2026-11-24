// SPDX-License-Identifier: Apache-2.0
//! Relation 3 (`fpn2-folded-gkr-v1`, `fpn2-folded-gkr-n8`) against specs/vectors/fpn2_folded_gkr.json
//! (the same file the detached Python reference suite consumes).

mod common;
use common::*;

use fpn_core::profile::{parse_hex32, FOLDED_GKR_N8, FOLDED_GKR_V1};
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

fn res(r: Result<(), RejectCode>) -> &'static str {
    match r {
        Ok(()) => "VERIFIED",
        Err(c) => c.as_str(),
    }
}

const ALLOW: Policy = Policy {
    allow_reduced: true,
    expected_profile_id: None,
    expected_circuit_root: None,
    expected_x_root: None,
    expected_stmt_digest: None,
};

#[test]
fn folded_vectors_exact_codes() {
    let f = load("fpn2_folded_gkr.json");
    let all = f.get("vectors").arr();
    let mut codes = std::collections::BTreeSet::new();
    for v in all {
        let got = res(verify_bytes(&build(all, v), &policy_of(v)));
        assert_eq!(got, v.get("expect").str(), "vector {}", v.get("name").str());
        codes.insert(got);
    }
    assert!(all.len() >= 45);
    for c in [
        "VERIFIED",
        "E_GKR_ROOT",
        "E_GKR",
        "E_SUMCHECK",
        "E_FINAL",
        "E_MERKLE",
        "E_FOLD",
        "E_NONCANONICAL",
    ] {
        assert!(codes.contains(c), "{c}");
    }
}

#[test]
fn folded_profiles_pinned() {
    let f = load("fpn2_folded_gkr.json");
    let p = f.get("profiles");
    assert_eq!(
        parse_hex32(p.get("fpn2-folded-gkr-v1").str()),
        Some(FOLDED_GKR_V1.profile_id())
    );
    assert_eq!(
        parse_hex32(p.get("fpn2-folded-gkr-n8").str()),
        Some(FOLDED_GKR_N8.profile_id())
    );
    for (prof, name) in [
        (&FOLDED_GKR_V1, "fpn2-folded-gkr-v1"),
        (&FOLDED_GKR_N8, "fpn2-folded-gkr-n8"),
    ] {
        let end = f.get("offsets").get(name).get("end").num() as usize;
        assert_eq!(64 + prof.body_used().unwrap(), end);
    }
}

#[test]
fn folded_valid_reserialize_identically() {
    let f = load("fpn2_folded_gkr.json");
    let all = f.get("vectors").arr();
    let mut n = 0;
    for name in [
        "folded_v1_valid_0",
        "folded_n8_valid_0",
        "folded_n8_valid_1",
    ] {
        let data = build(all, find(all, name));
        let c = Fpn2Container::parse(&data, &ALLOW).expect(name);
        assert!(matches!(c.body, Body::FoldedGkr(_)));
        assert_eq!(c.serialize().unwrap(), data, "{name}");
        n += 1;
    }
    assert_eq!(n, 3);
}

#[test]
fn folded_every_byte_flip_rejects_n8() {
    let f = load("fpn2_folded_gkr.json");
    let all = f.get("vectors").arr();
    let base = build(all, find(all, "folded_n8_valid_0"));
    assert!(verify_bytes(&base, &ALLOW).is_ok());
    for i in 0..base.len() {
        for bit in [i % 8, (i + 3) % 8] {
            let mut b = base.clone();
            b[i] ^= 1 << bit;
            assert!(
                verify_bytes(&b, &ALLOW).is_err(),
                "byte {i} bit {bit} accepted"
            );
        }
    }
    for n in 0..base.len() {
        assert_eq!(verify_bytes(&base[..n], &ALLOW), Err(RejectCode::Length));
    }
}

#[test]
fn folded_sampled_flips_reject_v1() {
    let f = load("fpn2_folded_gkr.json");
    let all = f.get("vectors").arr();
    let base = build(all, find(all, "folded_v1_valid_0"));
    let pol = Policy::default();
    assert!(verify_bytes(&base, &pol).is_ok());
    let mut i = 0;
    while i < base.len() {
        let mut b = base.clone();
        b[i] ^= 1 << (i % 8);
        assert!(verify_bytes(&b, &pol).is_err(), "byte {i} accepted");
        i += if i < 8000 { 13 } else { 211 };
    }
}

#[test]
fn folded_cli_pinned_standard_profile() {
    let f = load("fpn2_folded_gkr.json");
    let all = f.get("vectors").arr();
    let dir = std::env::temp_dir().join(format!("fpn-verify-folded-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (name, want, code) in [
        ("folded_v1_valid_0", "VERIFIED", 0),
        ("folded_v1_root_den_zero", "REJECTED: E_GKR_ROOT", 1),
        (
            "folded_v1_tamper_coset_merkle_path",
            "REJECTED: E_MERKLE",
            1,
        ),
    ] {
        let p = dir.join(format!("{name}.fpn2"));
        std::fs::write(&p, build(all, find(all, name))).unwrap();
        let out = std::process::Command::new(env!("CARGO_BIN_EXE_fpn-verify"))
            .arg(&p)
            .args(["--profile", "fpn2-folded-gkr-v1"])
            .output()
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), want, "{name}");
        assert_eq!(out.status.code(), Some(code), "{name}");
    }
    std::fs::remove_dir_all(&dir).ok();
}
