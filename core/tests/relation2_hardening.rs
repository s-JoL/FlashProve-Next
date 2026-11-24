// SPDX-License-Identifier: Apache-2.0
//! Hardening & malformed-descriptor regression tests for `core/src/relation2.rs`
//! (closing v1.0 security audit findings #2, #3, #4, #5: query floor, panic-free decoding, tag-2 rejection, stmt pin).

use fpn_core::blake2b::blake2b;
use fpn_core::profile::{by_name, HEADER_LEN};
use fpn_core::relation2::{
    circuit_desc_root, decode_circuit_desc, stmt_digest_bytes, verify_relation2, DESC_MAGIC,
    DESC_VERSION,
};
use fpn_core::verify::{Policy, RejectCode};
use fpn_core::wire::Fpn2Header;

fn make_raw_desc_base(n_max: u8, queries: u16, group_ns: &[u8]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(DESC_MAGIC);
    b.extend_from_slice(&DESC_VERSION.to_le_bytes());
    b.push(n_max);
    b.extend_from_slice(&queries.to_le_bytes());
    b.extend_from_slice(&(group_ns.len() as u16).to_le_bytes());
    b.extend_from_slice(group_ns);
    b
}

#[test]
fn malformed_descriptors_reject_without_panic() {
    // 1. Huge u32 count with tiny buffer -> E_LENGTH (no OOM allocation)
    let mut d1 = make_raw_desc_base(10, 72, &[10]);
    d1.extend_from_slice(&1u16.to_le_bytes()); // 1 GKR batch
    d1.push(6); // s=6
    d1.extend_from_slice(&2u16.to_le_bytes()); // m_mask=2
    d1.extend_from_slice(&0u16.to_le_bytes()); // aux group 0
    d1.extend_from_slice(&0u32.to_le_bytes()); // aux off 0
    d1.push(8); // aux nbits 8 (256 >= 60+4)
    d1.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // huge n_views!
    assert_eq!(decode_circuit_desc(&d1), Err(RejectCode::Length));

    // 2. ViewAtom(0) with empty view list -> E_NONCANONICAL (no OOB panic!)
    let mut d2 = make_raw_desc_base(10, 72, &[10]);
    d2.extend_from_slice(&0u16.to_le_bytes()); // 0 GKR batches
    d2.extend_from_slice(&1u16.to_le_bytes()); // 1 sumcheck
    d2.push(4); // nv=4
    d2.push(2); // deg=2
    d2.extend_from_slice(&1u32.to_le_bytes()); // 1 constraint
    d2.extend_from_slice(&0u16.to_le_bytes()); // mask group 0
    d2.extend_from_slice(&0u32.to_le_bytes()); // mask off 0
    d2.push(4); // mask nbits 4 (16 >= 4*3)
    d2.extend_from_slice(&0u32.to_le_bytes()); // 0 views!
    d2.push(1); // AstNode::ViewAtom
    d2.extend_from_slice(&0u32.to_le_bytes()); // idx 0 out of bounds!
    assert_eq!(decode_circuit_desc(&d2), Err(RejectCode::NonCanonical));

    // 3. Unbound GKR num_kind = 2 -> E_NONCANONICAL
    let mut d3 = make_raw_desc_base(10, 72, &[10]);
    d3.extend_from_slice(&1u16.to_le_bytes()); // 1 GKR batch
    d3.push(6); // s=6
    d3.extend_from_slice(&2u16.to_le_bytes()); // m_mask=2
    d3.extend_from_slice(&0u16.to_le_bytes()); // aux group 0
    d3.extend_from_slice(&0u32.to_le_bytes()); // aux off 0
    d3.push(7); // aux nbits 7 (128 >= 64)
    d3.extend_from_slice(&0u32.to_le_bytes()); // 0 views
    d3.extend_from_slice(&1u32.to_le_bytes()); // 1 block
    d3.extend_from_slice(&0u32.to_le_bytes()); // offset 0
    d3.push(4); // nbits 4
    d3.extend_from_slice(&0u64.to_le_bytes()); // tid 0
    d3.push(0); // 0 cols
    d3.push(2); // num_kind = 2 (REJECT!)
    d3.extend_from_slice(&0u16.to_le_bytes()); // 0 sumchecks
    assert_eq!(decode_circuit_desc(&d3), Err(RejectCode::NonCanonical));

    // 4. Out-of-bounds group_idx in mask_place -> E_NONCANONICAL
    let mut d4 = make_raw_desc_base(10, 72, &[10]);
    d4.extend_from_slice(&0u16.to_le_bytes()); // 0 GKR
    d4.extend_from_slice(&1u16.to_le_bytes()); // 1 sumcheck
    d4.push(4);
    d4.push(2);
    d4.extend_from_slice(&0u32.to_le_bytes());
    d4.extend_from_slice(&99u16.to_le_bytes()); // group 99 OOB!
    d4.extend_from_slice(&0u32.to_le_bytes());
    d4.push(4);
    d4.extend_from_slice(&0u32.to_le_bytes());
    assert_eq!(decode_circuit_desc(&d4), Err(RejectCode::NonCanonical));
}

#[test]
fn pin_checked_before_decoding_garbage_descriptor() {
    let (prof, pid) = by_name("fpn2-llm-std-v1").unwrap();
    let mut raw = vec![0u8; prof.c_p as usize];
    let hdr = Fpn2Header { profile_id: pid }.encode();
    raw[..HEADER_LEN].copy_from_slice(&hdr);

    // Completely invalid descriptor bytes (not even FPN2DESC magic), with WRONG pin -> MUST reject E_CIRCUIT_PINNED before decode!
    let garbage = b"NOT_A_VALID_DESCRIPTOR";
    let pol = Policy {
        expected_circuit_root: Some([0xAA; 32]),
        expected_stmt_digest: Some(stmt_digest_bytes(b"s")),
        ..Policy::default()
    };
    assert_eq!(verify_relation2(&raw, garbage, &pol), Err(RejectCode::CircuitPinned));

    // Matching desc pin & container circuit_root, but missing stmt pin -> E_STMT_UNPINNED before decode!
    let droot = circuit_desc_root(garbage);
    raw[HEADER_LEN..HEADER_LEN + 32].copy_from_slice(&droot);
    let pol_nostmt = Policy {
        expected_circuit_root: Some(droot),
        expected_stmt_digest: None,
        ..Policy::default()
    };
    assert_eq!(
        verify_relation2(&raw, garbage, &pol_nostmt),
        Err(RejectCode::StmtUnpinned)
    );

    // Wrong stmt digest -> E_STMT_PINNED before decode!
    let pol_badstmt = Policy {
        expected_circuit_root: Some(droot),
        expected_stmt_digest: Some([0xBB; 32]),
        ..Policy::default()
    };
    assert_eq!(
        verify_relation2(&raw, garbage, &pol_badstmt),
        Err(RejectCode::StmtPinned)
    );
    let _ = blake2b(32, &[0u8; 16], b"x");
}
