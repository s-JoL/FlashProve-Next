// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;
use std::process::Command;

fn run(data: &[u8], args: &[&str], tag: &str) -> (i32, String) {
    let dir = std::env::temp_dir().join(format!("fpn-verify-test-{}-{}", std::process::id(), tag));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("proof.fpn2");
    std::fs::write(&p, data).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_fpn-verify"))
        .arg(&p)
        .args(args)
        .output()
        .unwrap();
    std::fs::remove_dir_all(&dir).ok();
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).trim().to_string(),
    )
}

#[test]
fn cli_exit_codes_match_vectors() {
    let cont = load("fpn2_containers.json");
    for (k, v) in cont.get("vectors").arr().iter().enumerate() {
        let data = container_bytes(v);
        let mut args: Vec<&str> = Vec::new();
        if v.get("allow_reduced").bool() {
            args.push("--allow-reduced-security");
        }
        let pin = v.get("expected_profile_id");
        let name;
        if !pin.is_null() {
            let (p, _) = fpn_core::profile::REGISTRY
                .iter()
                .find(|(_, id)| *id == pin.str())
                .expect("pinned id is registered");
            name = p.name;
            args.push("--profile");
            args.push(name);
        }
        let (code, out) = run(&data, &args, &k.to_string());
        let exp = v.get("expect").str();
        if exp == "VERIFIED" {
            assert_eq!(
                (code, out.as_str()),
                (0, "VERIFIED"),
                "{}",
                v.get("name").str()
            );
        } else {
            assert_eq!(
                (code, out),
                (1, format!("REJECTED: {exp}")),
                "{}",
                v.get("name").str()
            );
        }
    }
}

#[test]
fn cli_usage_errors() {
    let out = Command::new(env!("CARGO_BIN_EXE_fpn-verify"))
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    let out = Command::new(env!("CARGO_BIN_EXE_fpn-verify"))
        .arg("/nonexistent/x.fpn2")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    let out = Command::new(env!("CARGO_BIN_EXE_fpn-verify"))
        .args(["x", "--profile", "nope"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
}
