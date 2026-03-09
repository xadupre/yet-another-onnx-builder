"""
Utility that queries the GitHub Copilot chat API to produce a first-draft
ONNX converter for a scikit-learn estimator and writes the result to the
appropriate location inside the ``yobx/sklearn/`` sub-package tree.
"""

import json
import os
import re
import textwrap
import urllib.error
import urllib.request
from typing import Optional, Type

from sklearn.base import BaseEstimator, is_classifier, is_regressor

# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------


def _get_copilot_token(github_token: str) -> str:
    """
    Exchange a GitHub personal-access token for an ephemeral Copilot API token.

    :param github_token: GitHub PAT with the ``copilot`` scope.
    :return: Short-lived Copilot API Bearer token.
    :raises urllib.error.HTTPError: when the exchange request fails.
    """
    url = "https://api.github.com/copilot_internal/v2/token"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"token {github_token}",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    return data["token"]


def _call_copilot_api(
    token: str,
    messages: list,
    *,
    model: str = "gpt-4o",
    max_tokens: int = 2048,
) -> str:
    """
    Call the GitHub Copilot chat-completions endpoint.

    :param token: Copilot API Bearer token (obtained via :func:`_get_copilot_token`).
    :param messages: list of ``{"role": ..., "content": ...}`` dicts.
    :param model: model identifier to request.
    :param max_tokens: upper bound on the completion length.
    :return: the text content of the first choice returned by the API.
    :raises urllib.error.HTTPError: when the API returns a non-2xx status.
    """
    url = "https://api.githubcopilot.com/chat/completions"
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Editor-Version": "vscode/1.85.0",
            "Editor-Plugin-Version": "copilot-chat/0.12.0",
            "Copilot-Integration-Id": "vscode-chat",
        },
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Helpers - code extraction / module inference / prompt building
# ---------------------------------------------------------------------------


def _extract_python_code(text: str) -> str:
    """
    Extracts Python source code from a Copilot markdown response.

    Handles the following formats (in order of preference):

    1. A fenced  block.
    2. A plain fenced block.
    3. Falls back to returning *text* unchanged (in case the model replied
       with raw code without a fence).

    :param text: raw text returned by the Copilot API.
    :return: extracted Python source code (stripped of leading/trailing
             whitespace).
    """
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _infer_submodule(estimator_class: type) -> str:
    """
    Infer the ``yobx/sklearn/`` sub-package name from the estimator's module.

    For example ``sklearn.linear_model._logistic.LogisticRegression`` →
    ``linear_model``.  Unrecognised modules fall back to ``"misc"``.

    :param estimator_class: a scikit-learn estimator class.
    :return: sub-package name (e.g. ``"linear_model"``).
    """
    module = estimator_class.__module__  # e.g. "sklearn.linear_model._logistic"
    parts = module.split(".")
    if len(parts) >= 2 and parts[0] == "sklearn":
        return parts[1]
    return "misc"


def _build_converter_prompt(estimator_class: type) -> str:
    """
    Construct a detailed Copilot prompt that asks for an ONNX converter
    implementation for *estimator_class*, including:

    * a canonical example (``StandardScaler`` converter),
    * a summary of the ``GraphBuilder`` API,
    * the required function signature and decorator.

    :param estimator_class: scikit-learn estimator class to target.
    :return: prompt string ready to be sent as the user message.
    """
    cls_name = estimator_class.__name__
    sklearn_module = estimator_class.__module__

    try:
        instance = estimator_class()
        if is_regressor(instance):
            kind = "regressor"
        elif is_classifier(instance):
            kind = "classifier"
        else:
            kind = "transformer"
    except Exception:
        kind = "estimator"

    example_code = textwrap.dedent("""\
        from typing import Dict, List
        from sklearn.preprocessing import StandardScaler
        from ...typing import GraphBuilderExtendedProtocol
        from ..register import register_sklearn_converter
        from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


        @register_sklearn_converter(StandardScaler)
        def sklearn_standard_scaler(
            g: GraphBuilderExtendedProtocol,
            sts: Dict,
            outputs: List[str],
            estimator: StandardScaler,
            X: str,
            name: str = "scaler",
        ) -> str:
            \"\"\"
            Converts a :class:`sklearn.preprocessing.StandardScaler` into ONNX.

            The implementation respects the ``with_mean`` and ``with_std`` flags:

            .. code-block:: text

                X  --Sub(mean)-->  centered  --Div(scale)-->  output
                     (if with_mean)               (if with_std)

            :param g: the graph builder to add nodes to
            :param sts: shapes defined by :epkg:`scikit-learn`
            :param estimator: a fitted ``StandardScaler``
            :param outputs: desired names (scaled inputs)
            :param X: inputs
            :param name: prefix name for the added nodes
            :return: output
            \"\"\"
            assert isinstance(estimator, StandardScaler), (
                f"Unexpected type {type(estimator)} for estimator."
            )
            assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

            itype = g.get_type(X)
            dtype = tensor_dtype_to_np_dtype(itype)

            if getattr(estimator, "with_mean", True):
                mean = estimator.mean_.astype(dtype)
                centered = g.op.Sub(X, mean, name=name)
            else:
                centered = X

            if getattr(estimator, "with_std", True):
                scale = estimator.scale_.astype(dtype)
                res = g.op.Div(centered, scale, name=name, outputs=outputs)
            else:
                res = g.op.Identity(centered, name=name, outputs=outputs)

            assert isinstance(res, str)
            if not sts:
                g.set_type(res, g.get_type(X))
                g.set_shape(res, g.get_shape(X))
                if g.has_device(X):
                    g.set_device(res, g.get_device(X))
            return res
        """).strip()

    return textwrap.dedent(f"""\
        I need an ONNX converter for the scikit-learn estimator `{cls_name}` from
        `{sklearn_module}`. This is a {kind} estimator.

        The converter must follow the exact pattern used in this project.
        Here is a complete example for StandardScaler:

        ```python
        {example_code}
        ```

        Available GraphBuilder API:
        - `g.op.Gemm(A, B, C, transA=0, transB=0, ...)` - matrix multiplication
        - `g.op.Add(X, Y, ...)`, `g.op.Sub(...)`, `g.op.Mul(...)`, `g.op.Div(...)`
        - `g.op.MatMul(A, B, ...)` - matrix multiplication without bias
        - `g.op.Relu(X, ...)`, `g.op.Sigmoid(X, ...)`, `g.op.Softmax(X, axis=..., ...)`
        - `g.op.ArgMax(X, axis=..., keepdims=..., ...)`, `g.op.Cast(X, to=..., ...)`
        - `g.op.Gather(data, indices, axis=..., ...)`, `g.op.Concat(*inputs, axis=..., ...)`
        - `g.op.Identity(X, ...)` - no-op / rename
        - `g.make_node(op_type, inputs, outputs=..., domain=..., ...)` - lower-level API
        - `g.has_type(name)`, `g.get_type(name)`, `g.set_type(name, dtype)`
        - `g.has_shape(name)`, `g.get_shape(name)`, `g.set_shape(name, shape)`
        - `g.has_device(name)`, `g.get_device(name)`, `g.set_device(name, device)`
        - `tensor_dtype_to_np_dtype(itype)` from `yobx.helpers.onnx_helper` - convert ONNX dtype to numpy dtype
        - `g.unique_name(prefix)` - generate a unique tensor name

        Write the complete converter function for `{cls_name}`. Requirements:
        1. Use the `@register_sklearn_converter({cls_name})` decorator.
        2. Include proper type annotations.
        3. Include a docstring with an ASCII-art diagram of the ONNX graph.
        4. Assert the estimator type and that the input tensor has a known type.
        5. Return the output tensor name(s) (a single `str` for transformers /
           regressors, a tuple `(label, probabilities)` for classifiers).

        Return ONLY the Python source code for the converter file (starting with
        the imports). Do not include any prose outside a code block.
        """).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def draft_converter_with_copilot(
    estimator_class: Type[BaseEstimator],
    *,
    token: Optional[str] = None,
    output_dir: Optional[str] = None,
    dry_run: bool = False,
    verbose: int = 0,
) -> str:
    """
    Submits a query to GitHub Copilot to draft a new ONNX converter for a
    :epkg:`scikit-learn` estimator and integrate the result into ``yobx/sklearn/``.

    The function:

    1. Exchanges *token* (or ``GITHUB_TOKEN`` / ``GH_TOKEN`` env-vars) for an
       ephemeral Copilot API token.
    2. Builds a context-rich prompt that includes existing converter examples
       and the ``GraphBuilder`` API reference.
    3. Calls the Copilot chat-completions endpoint and extracts the Python code
       from the response.
    4. Writes the generated code to
       ``yobx/sklearn/<submodule>/<estimator_snake_case>.py``, creating the
       sub-package directory and its ``__init__.py`` if they do not yet exist.

    :param estimator_class: scikit-learn estimator class to create a converter
        for (must be a fitted-able class, not an instance).
    :param token: GitHub PAT with the ``copilot`` scope.  Falls back to the
        ``GITHUB_TOKEN`` or ``GH_TOKEN`` environment variable if *None*.
    :param output_dir: directory to write the generated file.  Defaults to
        ``yobx/sklearn/<inferred_submodule>/`` relative to the ``yobx``
        package directory.
    :param dry_run: when *True* the code is printed and returned but **not**
        written to disk.
    :param verbose: verbosity level (0 = silent, 1 = info messages).
    :return: the generated Python source code for the converter.
    :raises ValueError: when no GitHub token is available.
    :raises urllib.error.HTTPError: when either the token-exchange or the
        Copilot API call fails.

    Example::

        from sklearn.linear_model import Ridge
        from yobx.helpers.copilot import draft_converter_with_copilot

        code = draft_converter_with_copilot(Ridge, dry_run=True)
        print(code)
    """
    # ----- resolve token -----
    if token is None:
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        raise ValueError(
            "No GitHub token supplied and neither GITHUB_TOKEN nor GH_TOKEN "
            "environment variables are set."
        )

    if verbose:
        print(f"[copilot_draft] drafting converter for {estimator_class.__name__!r}")

    # ----- obtain Copilot API token -----
    if verbose:
        print("[copilot_draft] obtaining Copilot API token …")
    copilot_token = _get_copilot_token(token)

    # ----- build and send prompt -----
    prompt = _build_converter_prompt(estimator_class)
    system = (
        "You are an expert Python developer who writes ONNX converters for "
        "scikit-learn estimators. You write clean, well-documented, "
        "type-annotated Python 3.10+ code and always follow the exact "
        "conventions of the existing codebase."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    if verbose:
        print("[copilot_draft] calling Copilot API …")
    response = _call_copilot_api(copilot_token, messages)
    code = _extract_python_code(response)

    if verbose:
        print("[copilot_draft] received response")

    if dry_run:
        print(code)
        return code

    # ----- determine output path -----
    submodule = _infer_submodule(estimator_class)
    cls_name = estimator_class.__name__
    # CamelCase → snake_case
    filename = re.sub(r"(?<!^)(?=[A-Z])", "_", cls_name).lower() + ".py"

    if output_dir is None:
        # This file lives at yobx/helpers/copilot/_copilot_draft.py.
        # Navigate up to yobx/ then into sklearn/<submodule>.
        here = os.path.dirname(os.path.abspath(__file__))
        sklearn_root = os.path.normpath(os.path.join(here, "..", "..", "sklearn"))
        output_dir = os.path.join(sklearn_root, submodule)

    os.makedirs(output_dir, exist_ok=True)

    # Create __init__.py for a brand-new sub-package
    init_path = os.path.join(output_dir, "__init__.py")
    if not os.path.exists(init_path):
        stem = os.path.splitext(filename)[0]
        with open(init_path, "w", encoding="utf-8") as fh:
            fh.write(f"def register():\n    from . import {stem}\n")
        if verbose:
            print(f"[copilot_draft] created {init_path!r}")

    # ----- write generated code -----
    out_path = os.path.join(output_dir, filename)
    if os.path.exists(out_path) and verbose:
        print(f"[copilot_draft] WARNING: {out_path!r} already exists - overwriting")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(code + "\n")

    if verbose:
        print(f"[copilot_draft] wrote {out_path!r}")
        print(
            f"[copilot_draft] NOTE: remember to import the new module in "
            f"{submodule}/__init__.py and, if the sub-package is new, register "
            f"it in yobx/sklearn/__init__.py"
        )

    return code
