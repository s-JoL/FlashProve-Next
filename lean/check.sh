#!/usr/bin/env bash
# Build and axiom-audit all 19 FlashProve-Next Lean 4 modules (Lean 4.29.1 + Mathlib).
# Standard usage (standalone clone):
#   cd lean && lake exe cache get && bash check.sh
# Optional fast mode reusing an existing Mathlib Lake build:
#   MATHLIB_LAKE_DIR=/path/to/prebuilt/lean bash check.sh
set -euo pipefail
[ -d "$HOME/.elan/bin" ] && export PATH="$HOME/.elan/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT

MODULES="SpotCheck StdAlg Mask Sumcheck StdAlgOps MaskJoint SumcheckSound Grinding LogUp Gadgets BoundedLift Speculative LinearAttn ArchGeneral LifetimeSoundness RoundByRound BucketSampling ProtocolBridge ConstrainedMask"

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

echo "OK: all $(echo $MODULES | wc -w) modules verified, no sorry, only standard axioms"
