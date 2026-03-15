"""
Unit tests for yobx.sklearn.feature_extraction.PatchExtractor converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestPatchExtractor(ExtTestCase):
    def _make_images(self, dtype, n_samples=4, height=8, width=8, seed=42):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n_samples, height, width)).astype(dtype)

    def test_patch_extractor_float32(self):
        from sklearn.feature_extraction.image import PatchExtractor
        from yobx.sklearn import to_onnx

        X = self._make_images(np.float32)
        pe = PatchExtractor(patch_size=(3, 3))
        pe.fit(X)

        onx = to_onnx(pe, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Slice", op_types)
        self.assertIn("Reshape", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pe.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_patch_extractor_float64(self):
        from sklearn.feature_extraction.image import PatchExtractor
        from yobx.sklearn import to_onnx

        X = self._make_images(np.float64)
        pe = PatchExtractor(patch_size=(3, 3))
        pe.fit(X)

        onx = to_onnx(pe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pe.transform(X).astype(np.float64)
        self.assertEqualArray(expected, result, atol=1e-10)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-10)

    def test_patch_extractor_small_patch(self):
        from sklearn.feature_extraction.image import PatchExtractor
        from yobx.sklearn import to_onnx

        X = self._make_images(np.float32, height=6, width=6)
        pe = PatchExtractor(patch_size=(2, 2))
        pe.fit(X)

        onx = to_onnx(pe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pe.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_patch_extractor_patch_equals_image(self):
        """When patch_size equals image size, a single patch per image is extracted."""
        from sklearn.feature_extraction.image import PatchExtractor
        from yobx.sklearn import to_onnx

        X = self._make_images(np.float32, height=4, width=4)
        pe = PatchExtractor(patch_size=(4, 4))
        pe.fit(X)

        onx = to_onnx(pe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pe.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_patch_extractor_output_shape(self):
        """Verify output shape: (n_images * n_patches_per_image, ph, pw)."""
        from sklearn.feature_extraction.image import PatchExtractor
        from yobx.sklearn import to_onnx

        n_images, h, w, ph, pw = 3, 8, 8, 3, 3
        X = self._make_images(np.float32, n_samples=n_images, height=h, width=w)
        pe = PatchExtractor(patch_size=(ph, pw))
        pe.fit(X)

        onx = to_onnx(pe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]

        n_patches_per_image = (h - ph + 1) * (w - pw + 1)
        self.assertEqual(result.shape, (n_images * n_patches_per_image, ph, pw))

    def test_patch_extractor_non_square_patch(self):
        """Test non-square patch sizes."""
        from sklearn.feature_extraction.image import PatchExtractor
        from yobx.sklearn import to_onnx

        X = self._make_images(np.float32, height=8, width=10)
        pe = PatchExtractor(patch_size=(3, 4))
        pe.fit(X)

        onx = to_onnx(pe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pe.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_patch_extractor_max_patches_raises(self):
        """max_patches != None must raise NotImplementedError."""
        from sklearn.feature_extraction.image import PatchExtractor
        from yobx.sklearn import to_onnx

        X = self._make_images(np.float32)
        pe = PatchExtractor(patch_size=(3, 3), max_patches=5)
        pe.fit(X)

        with self.assertRaises(NotImplementedError):
            to_onnx(pe, (X,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
