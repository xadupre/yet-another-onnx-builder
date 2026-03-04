"""
onnxscript sub-package — bridge between yobx's GraphBuilder API and
:class:`onnxscript._internal.builder.GraphBuilder`.
"""

from ._builder import OnnxScriptGraphBuilder

__all__ = ["OnnxScriptGraphBuilder"]
