"""
.. _l-plot-polars-three-joins-to-onnx:

Three Consecutive Joins to ONNX (polars-style)
===============================================

This example shows how to convert a pipeline that performs **three consecutive
inner joins** into a self-contained ONNX model using
:func:`~yobx.sql.dataframe_to_onnx`.

The API accepts a plain Python callable that receives
:class:`~yobx.xtracing.dataframe_trace.TracedDataFrame` objects and returns
one.  The callable's interface deliberately mirrors the polars DataFrame API,
so existing polars code can be adapted with minimal changes.

Scenario
--------
We model a simplified order-processing pipeline with four tables:

* **orders** — ``order_id``, ``customer_id``, ``product_id``,
  ``warehouse_id``, ``qty``
* **customers** — ``cid``, ``discount``
* **products** — ``pid``, ``unit_price``
* **warehouses** — ``wid``, ``shipping_cost``

The three consecutive joins are:

1. ``orders`` ⋈ ``customers``  on ``orders.customer_id = customers.cid``
2. result ⋈ ``products``  on ``result.product_id = products.pid``
3. result ⋈ ``warehouses``  on ``result.warehouse_id = warehouses.wid``

After the joins we compute the final order cost::

    total = qty * unit_price * (1 - discount) + shipping_cost

Equivalent polars code
----------------------

.. code-block:: python

    import polars as pl

    orders_lf = pl.LazyFrame({
        "order_id": [1, 2, 3, 4],
        "customer_id": [10, 20, 10, 30],
        "product_id": [100, 200, 300, 100],
        "warehouse_id": [1000, 2000, 1000, 2000],
        "qty": [2.0, 1.0, 3.0, 1.0],
    })
    customers_lf = pl.LazyFrame({"cid": [10, 20, 30], "discount": [0.1, 0.2, 0.0]})
    products_lf = pl.LazyFrame({"pid": [100, 200, 300], "unit_price": [50.0, 80.0, 60.0]})
    warehouses_lf = pl.LazyFrame({"wid": [1000, 2000], "shipping_cost": [5.0, 8.0]})

    j1 = orders_lf.join(customers_lf, left_on="customer_id", right_on="cid")
    j2 = j1.join(products_lf, left_on="product_id", right_on="pid")
    j3 = j2.join(warehouses_lf, left_on="warehouse_id", right_on="wid")
    result_lf = j3.select([
        pl.col("qty") * pl.col("unit_price") * (1 - pl.col("discount"))
        + pl.col("shipping_cost")
    ].alias("total"))
    print(result_lf.collect())

The :func:`~yobx.sql.dataframe_to_onnx` converter below produces an ONNX model
that is equivalent to this polars pipeline.
"""

import numpy as np
import onnxruntime
from yobx.helpers.onnx_helper import pretty_onnx
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sql import dataframe_to_onnx

# %%
# Sample data
# -----------
#
# Each column is a 1-D numpy array; the four tables are passed as separate
# arguments to the transform function.

order_id = np.array([1, 2, 3, 4], dtype=np.int64)
customer_id = np.array([10, 20, 10, 30], dtype=np.int64)
product_id = np.array([100, 200, 300, 100], dtype=np.int64)
warehouse_id = np.array([1000, 2000, 1000, 2000], dtype=np.int64)
qty = np.array([2.0, 1.0, 3.0, 1.0], dtype=np.float32)

cid = np.array([10, 20, 30], dtype=np.int64)
discount = np.array([0.1, 0.2, 0.0], dtype=np.float32)

pid = np.array([100, 200, 300], dtype=np.int64)
unit_price = np.array([50.0, 80.0, 60.0], dtype=np.float32)

wid = np.array([1000, 2000], dtype=np.int64)
shipping_cost = np.array([5.0, 8.0], dtype=np.float32)

# %%
# 1. Define the transform — three consecutive joins
# -------------------------------------------------
#
# The callable receives :class:`~yobx.xtracing.dataframe_trace.TracedDataFrame`
# objects.  Their ``.join()`` method mirrors polars'
# :meth:`polars.LazyFrame.join`, accepting ``left_key`` / ``right_key``
# arguments for the join column names on each side.


def transform(orders, customers, products, warehouses):
    """Apply three consecutive inner joins and compute total order cost."""
    j1 = orders.join(customers, left_key="customer_id", right_key="cid")
    j2 = j1.join(products, left_key="product_id", right_key="pid")
    j3 = j2.join(warehouses, left_key="warehouse_id", right_key="wid")
    return j3.select(
        [
            j3["order_id"],
            j3["customer_id"],
            j3["product_id"],
            j3["warehouse_id"],
            j3["qty"],
            j3["discount"],
            j3["unit_price"],
            j3["shipping_cost"],
            (j3["qty"] * j3["unit_price"] * (1.0 - j3["discount"]) + j3["shipping_cost"]).alias(
                "total"
            ),
        ]
    )


# %%
# 2. Convert to ONNX
# ------------------
#
# :func:`~yobx.sql.dataframe_to_onnx` traces *transform* and emits a
# self-contained ONNX model.  The ``input_dtypes`` list describes each of the
# four input tables in the same order as the function arguments.

dtypes_orders = {
    "order_id": np.int64,
    "customer_id": np.int64,
    "product_id": np.int64,
    "warehouse_id": np.int64,
    "qty": np.float32,
}
dtypes_customers = {"cid": np.int64, "discount": np.float32}
dtypes_products = {"pid": np.int64, "unit_price": np.float32}
dtypes_warehouses = {"wid": np.int64, "shipping_cost": np.float32}

artifact = dataframe_to_onnx(
    transform, [dtypes_orders, dtypes_customers, dtypes_products, dtypes_warehouses]
)

print("ONNX input names :", artifact.input_names)
print("ONNX output names:", artifact.output_names)

# %%
# 3. Run with the reference evaluator
# ------------------------------------
#
# :class:`~yobx.reference.ExtendedReferenceEvaluator` lets us verify the model
# without onnxruntime.

ref = ExtendedReferenceEvaluator(artifact)
feeds = {
    "order_id": order_id,
    "customer_id": customer_id,
    "product_id": product_id,
    "warehouse_id": warehouse_id,
    "qty": qty,
    "cid": cid,
    "discount": discount,
    "pid": pid,
    "unit_price": unit_price,
    "wid": wid,
    "shipping_cost": shipping_cost,
}
ref_outputs = ref.run(None, feeds)

# Show the result
for name, val in zip(artifact.output_names, ref_outputs):
    print(f"  {name}: {val}")

# Verify totals manually:
# order 1: qty=2, price=50, disc=0.1, ship=5  → 2*50*0.9+5 = 95
# order 2: qty=1, price=80, disc=0.2, ship=8  → 1*80*0.8+8 = 72
# order 3: qty=3, price=60, disc=0.1, ship=5  → 3*60*0.9+5 = 167
# order 4: qty=1, price=50, disc=0.0, ship=8  → 1*50*1.0+8 = 58
expected_total = np.array([95.0, 72.0, 167.0, 58.0], dtype=np.float32)
total_idx = artifact.output_names.index("total")
np.testing.assert_allclose(ref_outputs[total_idx], expected_total, rtol=1e-5)
print("Reference evaluator: totals match expected values ✓")

# %%
# 4. Run with onnxruntime
# -----------------------
#
# The same feeds work transparently with onnxruntime.

sess = onnxruntime.InferenceSession(
    artifact.SerializeToString(), providers=["CPUExecutionProvider"]
)
ort_outputs = sess.run(None, feeds)
np.testing.assert_allclose(ort_outputs[total_idx], expected_total, rtol=1e-5)
print("OnnxRuntime:        totals match expected values ✓")

# %%
# 5. Inspect the ONNX graph
# -------------------------
#
# Each join is translated to a broadcast equality check followed by
# ``ArgMax``, ``Compress`` and ``Gather`` nodes.  The ``total`` column is a
# simple chain of ``Mul``, ``Sub`` and ``Add`` nodes.

print(pretty_onnx(artifact.proto))

# %%
# 6. Node count per join step
# ---------------------------
#
# The bar chart below shows how many ONNX nodes are added by each join and
# how the final arithmetic expression fits in.

import matplotlib.pyplot as plt  # noqa: E402

op_types: dict[str, int] = {}
for node in artifact.proto.graph.node:
    op_types[node.op_type] = op_types.get(node.op_type, 0) + 1

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(list(op_types.keys()), list(op_types.values()), color="#4c72b0")
ax.set_ylabel("Number of ONNX nodes")
ax.set_title("ONNX node types — three-join order-cost pipeline")
for bar, count in zip(bars, op_types.values()):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        str(count),
        ha="center",
        va="bottom",
        fontsize=9,
    )
ax.tick_params(axis="x", labelrotation=20)
plt.tight_layout()
plt.show()

# %%
# 7. Display the ONNX graph
# -------------------------
#
# :func:`~yobx.doc.plot_dot` renders the full ONNX graph so you can trace
# how data flows from the four input tables to the final ``total`` output.

from yobx.doc import plot_dot  # noqa: E402

plot_dot(artifact.proto)
