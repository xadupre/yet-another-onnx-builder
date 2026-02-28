import os
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from yobx.ext_test_case import (
    ExtTestCase,
    requires_matplotlib,
    skipif_if_windows,
    skipif_if_apple,
)


@skipif_if_windows("too long")
@skipif_if_apple("too long")
@requires_matplotlib()
class TestDocMatplotlib(ExtTestCase):
    @classmethod
    def setUp(cls):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cls.plt = plt

    def test_plot_legend_returns_axes(self):
        from yobx.doc import plot_legend

        ax = plot_legend("TEST")
        import matplotlib.axes

        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.plt.close("all")

    def test_plot_legend_with_text_bottom(self):
        from yobx.doc import plot_legend

        ax = plot_legend("LABEL", text_bottom="bottom text", color="blue", fontsize=12)
        import matplotlib.axes

        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.plt.close("all")

    def test_rotate_align_returns_axes(self):
        from yobx.doc import rotate_align

        _fig, ax = self.plt.subplots()
        ax.bar(["a", "b", "c"], [1, 2, 3])
        result = rotate_align(ax)
        self.assertIs(result, ax)
        self.plt.close("all")

    def test_rotate_align_custom_angle_and_align(self):
        from yobx.doc import rotate_align

        _fig, ax = self.plt.subplots()
        ax.bar(["x", "y"], [1, 2])
        result = rotate_align(ax, angle=30, align="left")
        self.assertIs(result, ax)
        for label in ax.get_xticklabels():
            self.assertEqual(label.get_rotation(), 30)
            self.assertEqual(label.get_ha(), "left")
        self.plt.close("all")

    def test_save_fig_creates_file(self):
        from yobx.doc import save_fig

        _fig, ax = self.plt.subplots()
        ax.plot([1, 2, 3])
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            fname = f.name
        try:
            result = save_fig(ax, fname)
            self.assertIs(result, ax)
            self.assertTrue(os.path.exists(fname))
            self.assertGreater(os.path.getsize(fname), 0)
        finally:
            os.unlink(fname)
        self.plt.close("all")

    def test_title_sets_title(self):
        from yobx.doc import title

        _fig, ax = self.plt.subplots()
        result = title(ax, "My Title")
        self.assertIs(result, ax)
        self.assertEqual(ax.get_title(), "My Title")
        self.plt.close("all")

    def test_plot_histogram_returns_axes(self):
        from yobx.doc import plot_histogram

        data = np.random.default_rng(0).standard_normal(100)
        ax = plot_histogram(data)
        import matplotlib.axes

        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.plt.close("all")

    def test_plot_histogram_with_axes(self):
        from yobx.doc import plot_histogram

        data = np.random.default_rng(0).standard_normal(50)
        _fig, ax = self.plt.subplots()
        result = plot_histogram(data, ax=ax, bins=20, color="blue", alpha=0.5)
        self.assertIs(result, ax)
        self.plt.close("all")


@skipif_if_windows("too long")
@skipif_if_apple("too long")
class TestDocVersionHelpers(ExtTestCase):
    def test_update_version_package_same_major(self):
        from yobx.doc import update_version_package

        with patch("yobx.doc.get_latest_pypi_version", return_value="1.2.3"):
            result = update_version_package("1.2.5")
        self.assertEqual(result, "1.2.5")

    def test_update_version_package_different_major(self):
        from yobx.doc import update_version_package

        with patch("yobx.doc.get_latest_pypi_version", return_value="1.2.3"):
            result = update_version_package("2.0.0")
        self.assertEqual(result, "2.0.dev")

    def test_update_version_package_returns_dev_suffix(self):
        from yobx.doc import update_version_package

        with patch("yobx.doc.get_latest_pypi_version", return_value="0.1.0"):
            result = update_version_package("0.2.0")
        self.assertEqual(result, "0.2.dev")


if __name__ == "__main__":
    unittest.main(verbosity=2)
