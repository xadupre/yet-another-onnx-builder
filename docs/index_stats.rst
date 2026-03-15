Versions
========

The documentation was updated on:

.. runpython::
    
    import datetime
    print(datetime.datetime.now())

With the following versions:

.. runpython::

    import category_encoders
    import lightgbm
    import numpy
    import ml_dtypes
    import scipy
    import sklearn
    import onnx
    import onnx_ir
    import onnxruntime
    import onnxscript
    import optree
    import tensorflow
    import torch
    import transformers
    import xgboost

    for m in [
        category_encoders,
        lightgbm,
        numpy,
        ml_dtypes,
        scipy,
        sklearn,
        optree,
        onnx,
        onnx_ir,
        onnxruntime,
        onnxscript,
        tensorflow,
        torch,
        transformers,
        xgboost,
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
