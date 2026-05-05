.. _l-design-main-to-onnx:

============
yobx.to_onnx
============

See :func:`yobx.to_onnx` and :ref:`l-main-to-onnx` for the user-facing
documentation including examples and a description of every parameter.

This page documents implementation details of the dispatcher in
:mod:`yobx.convert`.

Implementation
==============

The dispatcher is implemented in ``yobx/convert.py`` and exported from
the top-level ``yobx`` package via::

    from .convert import DEFAULT_TARGET_OPSET, to_onnx

It resolves the backend at call time (not at import time) so that optional
dependencies such as :epkg:`torch`, :epkg:`scikit-learn`, or
:epkg:`tensorflow` do not need to be installed for unrelated backends to
work.  Each backend module is imported inside the matching ``if`` branch
using a local import.

Backend availability is checked with the helpers
``has_torch()``, ``has_sklearn()``, ``has_tensorflow()``, and
``has_litert()`` from :mod:`yobx.ext_test_case`, which perform a
lightweight ``importlib.util.find_spec`` probe without importing the
package itself.

Dispatch order
--------------

The checks run in a fixed priority order:

1. **torch** — :class:`torch.nn.Module` or :class:`torch.fx.GraphModule`
2. **sklearn** — :class:`sklearn.base.BaseEstimator`
3. **tensorflow** — :class:`tensorflow.Module`
4. **litert** — :class:`bytes` raw flatbuffer, or a path / string that
   ends with ``".tflite"`` and exists on disk
5. **sql** — any :class:`str`, :class:`os.PathLike`, or :func:`callable`
   (including ``polars.LazyFrame``, whose type is duck-typed via
   ``type(model).__module__``)

If none of the checks match, a :class:`TypeError` is raised with a
descriptive message.

Common arguments dict
---------------------

Before branching, the dispatcher assembles a ``common`` dict that
collects all shared keyword arguments::

    common = dict(
        input_names=input_names,
        dynamic_shapes=dynamic_shapes,
        target_opset=target_opset,
        verbose=verbose,
        large_model=large_model,
        external_threshold=external_threshold,
        filename=filename,
        return_optimize_report=return_optimize_report,
    )

Each backend receives ``**common, **kwargs`` so that the caller's extra
keyword arguments are always passed through without being modified by
the dispatcher.

LiteRT special case
-------------------

The LiteRT backend does not support the ``filename`` parameter.  The
dispatcher therefore builds a separate ``litert_common`` dict with
``filename`` removed before forwarding to :func:`yobx.litert.to_onnx`::

    litert_common = {k: v for k, v in common.items() if k != "filename"}

API reference
=============

.. autofunction:: yobx.to_onnx
