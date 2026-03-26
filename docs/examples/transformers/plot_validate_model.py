"""
.. _l-plot-validate-model:

Validate a LLM export and inspect discrepancies
================================================

:func:`validate_model <yobx.torch.validate.validate_model>` is a convenience
function that bundles the entire export-and-verify pipeline into a single call:

1. Load the model config (bundled copy preferred, then local HF cache, then
   network).
2. Optionally apply *config_overrides* to reduce the model size for fast
   testing.
3. Load the pre-trained weights — or, when *random_weights* is ``True``,
   instantiate the model from the (possibly modified) config with random
   weights so that no large checkpoint is downloaded.
4. Run ``model.generate`` with a text prompt inside an
   :class:`InputObserver <yobx.torch.InputObserver>` context to capture real
   input/output tensors.
5. Infer export arguments and dynamic shapes from the captured tensors.
6. Export the model to ONNX.
7. Run ONNX Runtime on every captured input set, compare the outputs against
   the original PyTorch outputs, and report the per-step discrepancies.

The function returns a :class:`ValidateSummary <yobx.torch.validate.ValidateSummary>`
with status flags / error messages and a :class:`ValidateData <yobx.torch.validate.ValidateData>`
with all intermediate artefacts including the raw discrepancy records.

**Command-line options**

All steps after step 2 can be exercised offline::

    python plot_validate_model.py --no-trained               # random weights, no download
    python plot_validate_model.py --num-hidden-layers 1      # 1-layer model, faster
    python plot_validate_model.py --model arnir0/Tiny-LLM    # different model
"""

# %%
# Imports
# -------

import argparse
import sys

import pandas
import torch

from yobx.torch.validate import validate_model, ValidateSummary, ValidateData

# %%
# Command-line arguments
# ----------------------

_DEFAULT_MODEL = "arnir0/Tiny-LLM"

parser = argparse.ArgumentParser(description="Validate a HuggingFace LLM export to ONNX.")
parser.add_argument(
    "--model",
    default=_DEFAULT_MODEL,
    metavar="MODEL_ID",
    help=f"HuggingFace model ID (default: {_DEFAULT_MODEL!r}).",
)
parser.add_argument(
    "--trained",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Load pre-trained weights (default). "
        "Pass --no-trained to use random weights (no large download)."
    ),
)
parser.add_argument(
    "--num-hidden-layers",
    type=int,
    default=None,
    metavar="LAYERS",
    help="Override config.num_hidden_layers (reduces model size for testing).",
)
parser.add_argument(
    "--max-new-tokens",
    type=int,
    default=3,
    metavar="N",
    help="Number of tokens generated during input capture (default: 3).",
)

# parse_known_args avoids failures when sphinx-gallery passes extra arguments.
args, _ = parser.parse_known_args(sys.argv[1:])

config_overrides = {}
if args.num_hidden_layers is not None:
    config_overrides["num_hidden_layers"] = args.num_hidden_layers

# %%
# Run validate_model
# ------------------
#
# :func:`validate_model <yobx.torch.validate.validate_model>` orchestrates the
# entire pipeline.  Setting *verbose=2* prints a one-line status for each
# captured input set during the discrepancy check (index, SUCCESS, absolute and
# relative differences).  Use *verbose=3* to additionally print tensor shapes.

summary: ValidateSummary
data: ValidateData

summary, data = validate_model(
    args.model,
    random_weights=not args.trained,
    max_new_tokens=args.max_new_tokens,
    config_overrides=config_overrides or None,
    quiet=True,
    verbose=2,
)

# %%
# Summary
# -------
#
# :class:`ValidateSummary <yobx.torch.validate.ValidateSummary>` stores
# high-level status flags and error messages.  Every field that was not reached
# (e.g. because an earlier step failed) remains ``None`` and is omitted from
# :meth:`items`.

print("-- summary --")
for k, v in sorted(summary.items()):
    print(f"  {k}: {v}")

# %%
# Discrepancies
# -------------
#
# :attr:`ValidateData.discrepancies <yobx.torch.validate.ValidateData>` is the
# raw list of dicts returned by
# :meth:`InputObserver.check_discrepancies <yobx.torch.input_observer.InputObserver.check_discrepancies>`.
# Each row corresponds to one forward call captured during ``model.generate``.
# The most important columns are:
#
# * ``SUCCESS`` — ``True`` when the absolute difference is below *atol* and the
#   relative difference is below *rtol*.
# * ``abs`` — maximum absolute element-wise difference across all outputs.
# * ``rel`` — maximum relative element-wise difference.
# * ``index`` — position in the capture sequence (0 = first forward call,
#   i.e. the prefill step).
# * ``inputs`` / ``outputs_torch`` / ``outputs_ort`` — shape strings for
#   the feeds and outputs.
#
# A :class:`pandas.DataFrame` gives a compact overview:

if data.discrepancies is not None:
    df = pandas.DataFrame(data.discrepancies)
    # Only show the most informative columns if they exist.
    cols = [c for c in ("index", "SUCCESS", "abs", "rel", "n_inputs") if c in df.columns]
    print(df[cols].to_string(index=False))
else:
    print("(no discrepancy data — export may have failed)")

# %%
# Interpreting the results
# ------------------------
#
# A typical successful run shows ``SUCCESS=True`` for every row with very small
# ``abs`` and ``rel`` values (well below ``1e-4``).
#
# When the export fails, ``summary.export`` will be ``"FAILED"`` and
# ``summary.error_export`` will contain the exception message.  The discrepancy
# check is skipped in that case.
#
# When the export succeeds but outputs diverge, ``summary.discrepancies`` will
# be ``"FAILED"`` with ``summary.discrepancies_ok < summary.discrepancies_total``.
# Increase *verbose* to 3 to print the input and output shapes for every
# failing row.
