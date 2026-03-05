.. _flattened-classes:

===============
Flattening List
===============

transformers
============

.. runpython::
    :showcode:

    from yobx.torch.in_transformers.flatten_class import TRANSFORMERS_CLASSES

    for k in TRANSFORMERS_CLASSES:
        print(k)
