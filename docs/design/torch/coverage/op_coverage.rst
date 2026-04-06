.. _l-design-torch-op-coverage:

==========================================
Op-db Coverage per Op and Type
==========================================

This page shows, for every op collected from
:mod:`torch.testing._internal.common_methods_invocations` (``op_db``), which
data types are covered by the op-db export tests in
:mod:`unittests.torch.coverage.test_onnx_export_common_methods`.

Legend:

* ``✔`` — converter exists and the test passes for that dtype.
* ``⚠ xfail`` — converter exists but the test is a **known failure** for that
  dtype (incorrect numerical results, unsupported dtype mapping, etc.).
* ``✘ no converter`` — no ONNX converter has been implemented for this op yet.
* ``—`` — the op does not support that dtype at all.

The exclusion sets (:data:`~yobx.torch.coverage.op_coverage.NO_CONVERTER_OPS`,
:data:`~yobx.torch.coverage.op_coverage.XFAIL_OPS`, and the per-dtype xfail sets) are
defined in :mod:`yobx.torch.coverage.op_coverage` and are imported by the test module
so that both sources always stay in sync.

.. runpython::
    :rst:
    :warningout: UserWarning

    from yobx.torch.coverage.op_coverage import get_op_coverage_rst
    print(get_op_coverage_rst())
