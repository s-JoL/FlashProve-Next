// SPDX-License-Identifier: Apache-2.0
//! `fpn-verify <file> [--allow-reduced-security] [--profile NAME] [--circuit-root|--weight-root HEX64] [--x-root HEX64] [--circuit-desc PATH] [--stmt-digest HEX64 | --stmt-file PATH]`
//!
//! Prints `VERIFIED` (exit 0) or `REJECTED: <code>` (exit 1); usage / I/O errors exit 2.
#![forbid(unsafe_code)]

use fpn_core::profile::{by_name, parse_hex32};
use fpn_core::relation2::{stmt_digest_bytes, verify_relation2};
use fpn_core::{verify_bytes, Policy};
use std::process::ExitCode;

const USAGE: &str = "usage: fpn-verify <file> [--allow-reduced-security] [--profile NAME] \
[--circuit-root|--weight-root HEX64] [--x-root HEX64] [--circuit-desc PATH] \
[--stmt-digest HEX64 | --stmt-file PATH]";

fn main() -> ExitCode {
    let mut file: Option<String> = None;
    let mut desc_path: Option<String> = None;
    let mut policy = Policy::default();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--allow-reduced-security" => policy.allow_reduced = true,
            "--profile" => match args.next().as_deref().and_then(by_name) {
                Some((_, id)) => policy.expected_profile_id = Some(id),
                None => {
                    eprintln!("error: --profile needs a registered profile name\n{USAGE}");
                    return ExitCode::from(2);
                }
            },
            "--circuit-root" | "--weight-root" => {
                match args.next().as_deref().and_then(parse_hex32) {
                    Some(root) => policy.expected_circuit_root = Some(root),
                    None => {
                        eprintln!("error: {a} needs 64 hex characters\n{USAGE}");
                        return ExitCode::from(2);
                    }
                }
            }
            "--x-root" => match args.next().as_deref().and_then(parse_hex32) {
                Some(root) => policy.expected_x_root = Some(root),
                None => {
                    eprintln!("error: --x-root needs 64 hex characters\n{USAGE}");
                    return ExitCode::from(2);
                }
            },
            "--circuit-desc" => match args.next() {
                Some(p) => desc_path = Some(p),
                None => {
                    eprintln!("error: --circuit-desc needs a file path\n{USAGE}");
                    return ExitCode::from(2);
                }
            },
            "--stmt-digest" => {
                if policy.expected_stmt_digest.is_some() {
                    eprintln!("error: --stmt-digest and --stmt-file are mutually exclusive and may only be given once\n{USAGE}");
                    return ExitCode::from(2);
                }
                match args.next().as_deref().and_then(parse_hex32) {
                    Some(dig) => policy.expected_stmt_digest = Some(dig),
                    None => {
                        eprintln!("error: --stmt-digest needs 64 hex characters\n{USAGE}");
                        return ExitCode::from(2);
                    }
                }
            }
            "--stmt-file" => {
                if policy.expected_stmt_digest.is_some() {
                    eprintln!("error: --stmt-digest and --stmt-file are mutually exclusive and may only be given once\n{USAGE}");
                    return ExitCode::from(2);
                }
                match args.next() {
                    Some(p) => match std::fs::read(&p) {
                        Ok(raw) => policy.expected_stmt_digest = Some(stmt_digest_bytes(&raw)),
                        Err(e) => {
                            eprintln!("error: cannot read {p}: {e}");
                            return ExitCode::from(2);
                        }
                    },
                    None => {
                        eprintln!("error: --stmt-file needs a file path\n{USAGE}");
                        return ExitCode::from(2);
                    }
                }
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            _ if a.starts_with('-') || file.is_some() => {
                eprintln!("error: unexpected argument {a}\n{USAGE}");
                return ExitCode::from(2);
            }
            _ => file = Some(a),
        }
    }
    let Some(path) = file else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let data = match std::fs::read(&path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: cannot read {path}: {e}");
            return ExitCode::from(2);
        }
    };
    let verdict = if let Some(dp) = desc_path {
        let desc = match std::fs::read(&dp) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("error: cannot read {dp}: {e}");
                return ExitCode::from(2);
            }
        };
        verify_relation2(&data, &desc, &policy).map(|_| ())
    } else {
        verify_bytes(&data, &policy)
    };
    match verdict {
        Ok(()) => {
            println!("VERIFIED");
            ExitCode::SUCCESS
        }
        Err(code) => {
            println!("REJECTED: {code}");
            ExitCode::from(1)
        }
    }
}
