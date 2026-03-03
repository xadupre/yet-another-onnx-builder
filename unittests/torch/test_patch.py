import unittest
from yobx.ext_test_case import ExtTestCase, requires_torch, hide_stdout
from yobx.torch import apply_patches_for_model
from yobx.helpers.patch_helper import PatchDetails


@requires_torch("2.0")
class TestPatch(ExtTestCase):
    @hide_stdout()
    def test_apply_patches_for_model(self):
        with apply_patches_for_model(patch_torch=True, verbose=1) as details:
            self.assertIsInstance(details, PatchDetails)
            self.assertGreater(len(details), 1)
            patch0 = details[0]
            diff = patch0.make_diff()
            self.assertInOr(
                (
                    "-    if not self.symbol_to_source.get(expr):",
                    "-    assert self.symbol_to_source.get(expr)",
                ),
                diff,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
