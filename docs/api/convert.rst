yobx.convert
============

Export discovery from a Python file
----------------------------------

``yobx.guess_export`` executes a **trusted** Python file and finds an existing
model and example inputs. It returns keyword arguments for ``yobx.to_onnx``,
which selects the appropriate backend. For example, a file ``model.py`` may
contain:

.. code-block:: python

    import numpy as np
    from sklearn.linear_model import LinearRegression

    X = np.arange(12, dtype=np.float32).reshape(4, 3)
    y = X.sum(axis=1)
    reg = LinearRegression().fit(X, y)

Export it with:

.. code-block:: python

    from yobx import guess_export, to_onnx

    export_args = guess_export("model.py")
    artifact = to_onnx(**export_args, target_opset=18)

Use ``model_name`` and ``args_name`` to resolve ambiguous files or select
unconventionally named variables. Discovery does not fit models, instantiate
classes, invoke factories, or invent example inputs. Code inside an
``if __name__ == "__main__":`` block is not executed. Imports must be resolvable
in the current Python environment. Errors from the file are not suppressed.

.. automodule:: yobx.convert
    :members:
    :no-undoc-members:
    :exclude-members: to_onnx
