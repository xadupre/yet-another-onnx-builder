import unittest
from yobx.ext_test_case import ExtTestCase, requires_torch
from yobx.torch import apply_patches


@requires_torch("2.0")
class TestPatch(ExtTestCase):
    def test_apply_patches(self):
        with apply_patches(patch_torch=True, verbose=1) as details:
            self.assertIsInstance(details, list)
            self.assertGreater(len(details), 0)
            patch0 = details[0]
            diff = patch0.make_diff()
            self.assertIn("-    if not self.symbol_to_source.get(expr):", diff)


if __name__ == "__main__":
    unittest.main(verbosity=2)
