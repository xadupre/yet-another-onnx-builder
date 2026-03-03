yobx.xoptim
===========

API for the pattern-based optimization of ONNX graphs.

.. toctree::
    :maxdepth: 1
    :caption: modules

    graph_builder_optim
    patterns_api
    patterns
    patterns_exp
    patterns_fix
    patterns_investigation
    patterns_ml
    patterns_ort
    repeated_optim
    unfused

get_pattern
+++++++++++

.. autofunction:: yobx.xoptim.get_pattern

get_pattern_list
++++++++++++++++

.. autofunction:: yobx.xoptim.get_pattern_list

remove_constants_for_initializers
++++++++++++++++++++++++++++++++++

.. autofunction:: yobx.xoptim.remove_constants_for_initializers

GraphBuilderPatternOptimization
++++++++++++++++++++++++++++++++

.. autoclass:: yobx.xoptim.GraphBuilderPatternOptimization
    :members:
    :no-undoc-members:

PatternOptimization
+++++++++++++++++++

.. autoclass:: yobx.xoptim.PatternOptimization
    :members:
    :no-undoc-members:

EasyPatternOptimization
+++++++++++++++++++++++

.. autoclass:: yobx.xoptim.EasyPatternOptimization
    :members:
    :no-undoc-members:

MatchResult
+++++++++++

.. autoclass:: yobx.xoptim.MatchResult
    :members:
    :no-undoc-members:
