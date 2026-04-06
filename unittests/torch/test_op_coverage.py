"""
Unit tests for :mod:`yobx.torch.coverage.op_coverage`.
"""

import unittest

from yobx.ext_test_case import ExtTestCase, requires_torch


@requires_torch("2.0")
class TestOpCoverageData(ExtTestCase):
    """Sanity checks on the coverage frozensets."""

    @classmethod
    def setUpClass(cls):
        from yobx.torch.coverage.op_coverage import (
            NO_CONVERTER_OPS,
            NON_DETERMINISTIC_OPS,
            XFAIL_OPS,
            XFAIL_OPS_BFLOAT16,
            XFAIL_OPS_FLOAT16,
            XFAIL_OPS_INT32,
            XFAIL_OPS_INT64,
        )

        cls.NO_CONVERTER_OPS = NO_CONVERTER_OPS
        cls.NON_DETERMINISTIC_OPS = NON_DETERMINISTIC_OPS
        cls.XFAIL_OPS = XFAIL_OPS
        cls.XFAIL_OPS_FLOAT16 = XFAIL_OPS_FLOAT16
        cls.XFAIL_OPS_BFLOAT16 = XFAIL_OPS_BFLOAT16
        cls.XFAIL_OPS_INT32 = XFAIL_OPS_INT32
        cls.XFAIL_OPS_INT64 = XFAIL_OPS_INT64

    def test_no_converter_ops_is_frozenset(self):
        """NO_CONVERTER_OPS is a non-empty frozenset of strings."""
        self.assertIsInstance(self.NO_CONVERTER_OPS, frozenset)
        self.assertTrue(self.NO_CONVERTER_OPS, "NO_CONVERTER_OPS should not be empty")
        for entry in self.NO_CONVERTER_OPS:
            self.assertIsInstance(entry, str, f"entry {entry!r} should be a str")

    def test_non_deterministic_ops_is_frozenset(self):
        """NON_DETERMINISTIC_OPS is a non-empty frozenset of strings."""
        self.assertIsInstance(self.NON_DETERMINISTIC_OPS, frozenset)
        self.assertTrue(self.NON_DETERMINISTIC_OPS)

    def test_xfail_ops_is_frozenset(self):
        """XFAIL_OPS is a non-empty frozenset of strings."""
        self.assertIsInstance(self.XFAIL_OPS, frozenset)
        self.assertTrue(self.XFAIL_OPS)

    def test_xfail_float16_is_frozenset(self):
        """XFAIL_OPS_FLOAT16 is a frozenset of strings."""
        self.assertIsInstance(self.XFAIL_OPS_FLOAT16, frozenset)

    def test_xfail_bfloat16_is_frozenset(self):
        """XFAIL_OPS_BFLOAT16 is a frozenset of strings."""
        self.assertIsInstance(self.XFAIL_OPS_BFLOAT16, frozenset)

    def test_xfail_int32_is_frozenset(self):
        """XFAIL_OPS_INT32 is a frozenset of strings."""
        self.assertIsInstance(self.XFAIL_OPS_INT32, frozenset)

    def test_xfail_int64_is_frozenset(self):
        """XFAIL_OPS_INT64 is a frozenset of strings."""
        self.assertIsInstance(self.XFAIL_OPS_INT64, frozenset)

    def test_no_overlap_no_converter_and_xfail(self):
        """NO_CONVERTER_OPS and XFAIL_OPS should be disjoint.

        An op should not appear in both sets; it either has no converter or it
        has a converter with a known failure, not both.
        """
        overlap = self.NO_CONVERTER_OPS & self.XFAIL_OPS
        self.assertEqual(
            overlap, frozenset(), f"ops in both NO_CONVERTER_OPS and XFAIL_OPS: {sorted(overlap)}"
        )

    def test_no_overlap_no_converter_and_non_deterministic(self):
        """NO_CONVERTER_OPS and NON_DETERMINISTIC_OPS should be disjoint."""
        overlap = self.NO_CONVERTER_OPS & self.NON_DETERMINISTIC_OPS
        self.assertEqual(overlap, frozenset(), f"ops in both: {sorted(overlap)}")

    def test_well_known_no_converter_ops_present(self):
        """A sample of well-known missing-converter ops should be in the set."""
        for op in ("atan2", "dot", "diag", "sort"):
            self.assertIn(op, self.NO_CONVERTER_OPS, f"expected {op!r} in NO_CONVERTER_OPS")

    def test_well_known_xfail_ops_present(self):
        """A sample of well-known xfail ops should be in XFAIL_OPS."""
        for op in ("clamp", "nan_to_num", "corrcoef"):
            self.assertIn(op, self.XFAIL_OPS, f"expected {op!r} in XFAIL_OPS")

    def test_well_known_non_deterministic_present(self):
        """A sample of non-deterministic ops should be in NON_DETERMINISTIC_OPS."""
        for op in ("randn", "rand", "normal"):
            self.assertIn(op, self.NON_DETERMINISTIC_OPS)


if __name__ == "__main__":
    unittest.main()
