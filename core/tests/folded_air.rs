// SPDX-License-Identifier: Apache-2.0
//! Relation 4 (`fpn2-folded-air-v1`, `fpn2-folded-air-n12`) against specs/vectors/fpn2_folded_air.json
//! (the same file the detached Python reference suite consumes).

mod common;
use common::*;

use fpn_core::folded_air::AirLayout;
use fpn_core::profile::{parse_hex32, FOLDED_AIR_N12, FOLDED_AIR_V1};
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

#[test]
fn air_vectors_exact_codes() {
    let f = load("fpn2_folded_air.json");
    let all = f.get("vectors").arr();
    let mut codes = std::collections::BTreeSet::new();
    for v in all {
        let got = res(verify_bytes(&build(all, v), &policy_of(v)));
        assert_eq!(got, v.get("expect").str(), "vector {}", v.get("name").str());
        codes.insert(got);
    }
    assert!(all.len() >= 40);
    for c in [
        "VERIFIED",
        "E_REDUCED_SECURITY",
        "E_PROFILE_PINNED",
        "E_CIRCUIT_UNPINNED",
        "E_CIRCUIT_PINNED",
        "E_GKR_ROOT",
        "E_GKR",
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
}

#[test]
fn air_profiles_pinned_and_layout() {
    let f = load("fpn2_folded_air.json");
    let p = f.get("profiles");
    assert_eq!(
        parse_hex32(p.get("fpn2-folded-air-v1").str()),
        Some(FOLDED_AIR_V1.profile_id())
    );
    assert_eq!(
        parse_hex32(p.get("fpn2-folded-air-n12").str()),
        Some(FOLDED_AIR_N12.profile_id())
    );
    for (prof, name) in [
        (&FOLDED_AIR_V1, "fpn2-folded-air-v1"),
        (&FOLDED_AIR_N12, "fpn2-folded-air-n12"),
    ] {
        let end = f.get("offsets").get(name).get("end").num() as usize;
        assert_eq!(64 + prof.body_used().unwrap(), end);
        assert_eq!(
            AirLayout::of(prof).unwrap().body_used(),
            prof.body_used().unwrap()
        );
    }
}

#[test]
fn air_valid_reserialize_identically() {
    let f = load("fpn2_folded_air.json");
    let all = f.get("vectors").arr();
    for name in ["air_v1_valid_0", "air_n12_valid_0"] {
        let v = find(all, name);
        let data = build(all, v);
        let c = Fpn2Container::parse(&data, &policy_of(v)).expect(name);
        assert!(matches!(c.body, Body::FoldedAir(_)));
        assert_eq!(c.serialize().unwrap(), data, "{name}");
    }
}

#[test]
fn air_every_byte_flip_rejects_n12() {
    let f = load("fpn2_folded_air.json");
    let all = f.get("vectors").arr();
    let v = find(all, "air_n12_valid_0");
    let pol = policy_of(v);
    let base = build(all, v);
    assert!(verify_bytes(&base, &pol).is_ok());
    for i in 0..base.len() {
        let mut b = base.clone();
        b[i] ^= 1 << (i % 8);
        assert!(verify_bytes(&b, &pol).is_err(), "byte {i} accepted");
    }
    for n in (0..base.len()).step_by(97) {
        assert_eq!(verify_bytes(&base[..n], &pol), Err(RejectCode::Length));
    }
}

#[test]
fn air_cli_requires_and_checks_circuit_root() {
    let f = load("fpn2_folded_air.json");
    let all = f.get("vectors").arr();
    let root = f.get("circuits").get("air_demo_program").str().to_string();
    let other = f.get("circuits").get("air_small_program").str().to_string();
    let dir = std::env::temp_dir().join(format!("fpn-verify-air-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("air_v1.fpn2");
    std::fs::write(&p, build(all, find(all, "air_v1_valid_0"))).unwrap();
    for (args, want, code) in [
        (vec!["--circuit-root", root.as_str()], "VERIFIED", 0),
        (vec![], "REJECTED: E_CIRCUIT_UNPINNED", 1),
        (
            vec!["--circuit-root", other.as_str()],
            "REJECTED: E_CIRCUIT_PINNED",
            1,
        ),
        (
            vec![
                "--circuit-root",
                root.as_str(),
                "--profile",
                "fpn2-folded-air-v1",
            ],
            "VERIFIED",
            0,
        ),
    ] {
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
    let bad = std::process::Command::new(env!("CARGO_BIN_EXE_fpn-verify"))
        .arg(&p)
        .args(["--circuit-root", "zz"])
        .output()
        .unwrap();
    assert_eq!(bad.status.code(), Some(2));
    std::fs::remove_dir_all(&dir).ok();
}
