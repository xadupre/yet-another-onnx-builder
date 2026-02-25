import unittest
from yobx.ext_test_case import ExtTestCase
from yobx.xshape.evaluate_expressions import evaluate_expression


class TestEvaluateExpressions(ExtTestCase):
    def test_evaluate_expression(self):
        self.assertEqual(-1, evaluate_expression("x - y", dict(x=5, y=6)))
        self.assertEqual(-5, evaluate_expression("- x", dict(x=5)))

    def test_evaluate_expression_syntax_error(self):
        # An expression with a SyntaxError should raise SyntaxError.
        with self.assertRaises(SyntaxError):
            evaluate_expression("x +", dict(x=5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
