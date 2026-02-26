yet-another-onnx-builder documentation
======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/index
   design/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Older versions
==============

* `0.1.0 <../v0.1.0/index.html>`_

The documentation was updated on:

.. runpython::
    
    import datetime
    print(datetime.datetime.now())

With the following versions:

.. runpython::

    import numpy    
    import ml_dtypes
    import sklearn
    import onnx
    import onnx_ir
    import onnxruntime
    import onnxscript
    import torch
    import transformers

    for m in [
        numpy,
        ml_dtypes,
        sklearn,
        onnx,
        onnx_ir,
        onnxruntime,
        onnxscript,
        torch,
        transformers,
    ]:
        print(f"{m.__name__}: {getattr(m, '__version__', 'dev')}")

Size of the package:

.. runpython::

    import os
    import pprint
    import pandas
    from yobx import __file__
    from yobx.ext_test_case import statistics_on_folder

    df = pandas.DataFrame(statistics_on_folder(os.path.dirname(__file__), aggregation=1))
    gr = df[["dir", "ext", "lines", "chars"]].groupby(["ext", "dir"]).sum()
    print(gr)
