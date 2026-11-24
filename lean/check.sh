#!/usr/bin/env bash
# Build and axiom-audit all 28 FlashProve-Next Lean 4 modules (Lean 4.29.1 + Mathlib), then run the
# relation-3, relation-4 and relation-5 golden vectors through the Lean verifiers
# (tests/check_rel3_vectors.py, tests/check_rel4_vectors.py, tests/check_rel5_vectors.py; needs
# python3, skip with FPN_SKIP_VECTORS=1; the relation-5 run includes the multi-tile fixture
# tests/rel5_multitile.json, beta = 2, 26 cases incl. E_FOLD). A vector mismatch fails the script, and so does any
# `sorry`, any non-standard axiom and any use of `native_decide`.
# Standard usage (standalone clone):
#   cd lean && lake exe cache get && bash check.sh
# Optional fast mode reusing an existing Mathlib Lake build:
#   MATHLIB_LAKE_DIR=/path/to/prebuilt/lean bash check.sh
set -euo pipefail
[ -d "$HOME/.elan/bin" ] && export PATH="$HOME/.elan/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT

MODULES="SpotCheck StdAlg Mask Sumcheck StdAlgOps MaskJoint SumcheckSound Grinding LogUp Gadgets BoundedLift Speculative LinearAttn ArchGeneral LifetimeSoundness RoundByRound BucketSampling ProtocolBridge ConstrainedMask Fpn2Wire Fpn2Transcript VerifyBytes Fpn2Rel3 Fpn2Rel4 Fpn2Rel5 Fpn2Soundness Fpn2Gadgets Fpn2EndToEnd"

if [ -n "${MATHLIB_LAKE_DIR:-}" ]; then
  O="${SCRIPT_DIR}/.olean"
  mkdir -p "${O}/FlashProveNext"
  LP="$(cd "${MATHLIB_LAKE_DIR}" && lake env printenv LEAN_PATH)"
  for m in $MODULES; do
    echo "== $m"
    t0=$SECONDS
    (cd "${MATHLIB_LAKE_DIR}" && lake env env LEAN_PATH="${LP}:${O}" lean --root="${SCRIPT_DIR}" -o "${O}/FlashProveNext/${m}.olean" "${SCRIPT_DIR}/FlashProveNext/${m}.lean") 2>&1 | tee -a "$LOG"
    echo "   ($m: $((SECONDS - t0)) s)"
  done
else
  cd "${SCRIPT_DIR}"
  lake build 2>&1 | tee -a "$LOG"
  LP="$(lake env printenv LEAN_PATH)"
  for m in $MODULES; do
    echo "== axiom audit: $m"
    lake env env LEAN_PATH="$LP" lean "FlashProveNext/${m}.lean" 2>&1 | tee -a "$LOG"
  done
fi

if grep -E "declaration uses .sorry.|: error[:(]" "$LOG"; then
  echo "FAIL: sorry or error"
  exit 1
fi
if grep "depends on axioms" "$LOG" | grep -vE "depends on axioms: \[(propext|Classical\.choice|Quot\.sound)(, (propext|Classical\.choice|Quot\.sound))*\]$"; then
  echo "FAIL: non-standard axiom"
  exit 1
fi

# `native_decide` trusts the compiler (axiom `Lean.ofReduceBool`, which the audit above already
# rejects on audited declarations); this also covers the test harnesses. Prose saying
# "no `native_decide`" is allowed.
if grep -n "native_decide" "${SCRIPT_DIR}"/FlashProveNext/*.lean "${SCRIPT_DIR}"/tests/*.lean | grep -v 'no `native_decide`'; then
  echo "FAIL: native_decide"
  exit 1
fi

if [ -z "${FPN_SKIP_VECTORS:-}" ]; then
  if command -v python3 >/dev/null 2>&1; then
    echo "== relation-3 golden vectors (Lean verifyBytesRel3)"
    python3 "${SCRIPT_DIR}/tests/check_rel3_vectors.py"
    echo "== relation-4 golden vectors (Lean verifyBytesRel4, verifyBytesAny)"
    python3 "${SCRIPT_DIR}/tests/check_rel4_vectors.py"
    echo "== relation-5 golden vectors (Lean verifyBytesRel5, verifyBytesAny) + multi-tile fixture"
    python3 "${SCRIPT_DIR}/tests/check_rel5_vectors.py"
  else
    echo "WARN: python3 not found, relation-3 / relation-4 / relation-5 golden vectors not run"
  fi
fi

echo "OK: all $(echo $MODULES | wc -w) modules verified, no sorry, only standard axioms"
