import unittest

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from yobx.sklearn import ConvertOptions, SklearnOnnxLightGraphBuilder, to_onnx
from yobx.xbuilder import FunctionOptions


class TestSklearnOnnxLightGraphBuilder(unittest.TestCase):
    def test_converter_specific_names_are_unique(self):
        builder = SklearnOnnxLightGraphBuilder(18)
        self.assertEqual("value", builder.unique_node_name("value"))
        self.assertEqual("value_2", builder.unique_node_name("value"))
        self.assertEqual("function", builder.unique_function_name("function"))
        self.assertEqual("function_2", builder.unique_function_name("function"))

    def test_function_child_preserves_conversion_options(self):
        x = np.arange(20, dtype=np.float32).reshape(10, 2)
        estimator = DecisionTreeRegressor(max_depth=2).fit(x, x[:, 0])
        artifact = to_onnx(
            estimator,
            (x,),
            convert_options=ConvertOptions(decision_leaf=True),
            function_options=FunctionOptions(
                export_as_function=True, name="ignored", domain="sklearn.local"
            ),
        )
        self.assertIsInstance(artifact.builder, SklearnOnnxLightGraphBuilder)
        self.assertEqual(
            ["predictions", "decision_leaf"],
            [str(output.name) for output in artifact.proto.graph.output],
        )


if __name__ == "__main__":
    unittest.main()
