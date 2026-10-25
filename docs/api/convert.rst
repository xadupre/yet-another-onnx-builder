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

The file is executed outside a package context. Explicit relative imports
such as ``from .foo import bar`` are not supported; use absolute imports
from packages available in the current Python environment instead.

Exporting a pickled model
-------------------------

``to_onnx`` also accepts a path ending in ``.pkl`` or ``.pickle`` and loads
the pickled model before selecting its backend. The example inputs must still
be supplied:

.. code-block:: python

    artifact = to_onnx("model.pkl", (X,))

The pickle must be compatible with the current Python environment, including
the packages and class definitions needed to deserialize it. Pickle loading
can execute arbitrary code, so only load trusted files. Checkpoint files that
contain only model weights (rather than a complete model object) cannot be
converted without first recreating the corresponding model architecture and
loading the weights into it.

.. warning::

    ``guess_export`` is not thread-safe. Its use of ``runpy.run_path``
    temporarily modifies process-global state, including ``sys.argv[0]`` and
    ``sys.path[0]``.
    Call it sequentially, not concurrently with other executions or changes
    to import state.

.. automodule:: yobx.convert
    :members:
    :no-undoc-members:
    :exclude-members: to_onnx
