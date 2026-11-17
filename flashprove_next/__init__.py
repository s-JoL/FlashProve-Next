# SPDX-License-Identifier: Apache-2.0
"""FlashProve-Next: zero-knowledge proofs that a production LLM API reply came from the committed model.

The prover attaches to an unmodified floating-point inference engine and proves, after the reply is streamed, that a
Fiat-Shamir sample of the computation matches the committed weights within certified rounding bounds and that every
token was drawn by the committed keyed sampling rule.  The verifier needs only the public request, the reply, the
published ``model_id`` and the proof bundle (see ``flashprove_next.cli``).

Submodules are imported on demand; importing the package does not load CuPy or torch.
"""

__version__ = "0.3.0"

__all__ = ["__version__"]
