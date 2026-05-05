import os
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import onnx
from yobx.ext_test_case import (
    ExtTestCase,
    requires_ipython,
    requires_matplotlib,
    skipif_ci_windows,
    skipif_ci_apple,
)


@skipif_ci_windows("too long")
@skipif_ci_apple("too long")
@requires_matplotlib()
class TestDocMatplotlib(ExtTestCase):
    @classmethod
    def setUp(cls):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cls.plt = plt

    def test_plot_legend_returns_axes(self):
        import matplotlib.axes
        from yobx.doc import plot_legend

        self.assertEqual(sys.platform, "linux")

        ax = plot_legend("TEST")
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.plt.close("all")

    def test_plot_legend_with_text_bottom(self):
        import matplotlib.axes
        from yobx.doc import plot_legend

        ax = plot_legend("LABEL", text_bottom="bottom text", color="blue", fontsize=12)
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
        import matplotlib.axes
        from yobx.doc import plot_histogram

        data = np.random.default_rng(0).standard_normal(100)
        ax = plot_histogram(data)
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.plt.close("all")

    def test_plot_histogram_with_axes(self):
        from yobx.doc import plot_histogram

        data = np.random.default_rng(0).standard_normal(50)
        _fig, ax = self.plt.subplots()
        result = plot_histogram(data, ax=ax, bins=20, color="blue", alpha=0.5)
        self.assertIs(result, ax)
        self.plt.close("all")


@skipif_ci_windows("too long")
@skipif_ci_apple("too long")
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


@skipif_ci_windows("too long")
@skipif_ci_apple("too long")
class TestRunSubprocess(ExtTestCase):
    def test_run_subprocess_captures_stdout(self):
        from yobx.doc import _run_subprocess

        result = _run_subprocess([sys.executable, "-c", "print('hello')"])
        self.assertIn("hello", result)

    def test_run_subprocess_captures_stderr(self):
        from yobx.doc import _run_subprocess

        result = _run_subprocess(
            [sys.executable, "-c", "import sys; sys.stderr.write('err_msg\\n')"]
        )
        self.assertIn("err_msg", result)

    def test_run_subprocess_no_deadlock_with_large_stderr(self):
        # Verify that _run_subprocess does not deadlock when the subprocess
        # writes a large amount of data to stderr (more than the pipe buffer
        # ~64 KB), while writing nothing to stdout.
        from yobx.doc import _run_subprocess

        # Write 200 KB to stderr, nothing to stdout — the old line-by-line
        # stdout reader would deadlock here; communicate() handles it safely.
        script = "import sys; sys.stderr.write('x' * 200_000)"
        result = _run_subprocess([sys.executable, "-c", script])
        # The function should return without hanging; stderr ends up in the
        # return value (appended after stdout).
        self.assertGreater(len(result), 100_000)

    def test_run_subprocess_raises_on_build_error(self):
        from yobx.doc import _run_subprocess

        # A script that prints a "fatal error" to stdout AND writes to stderr
        # should cause _run_subprocess to raise RuntimeError.
        script = (
            "import sys; "
            "print('fatal error: something went wrong'); "
            "sys.stderr.write('build failed\\n')"
        )
        with self.assertRaises(RuntimeError):
            _run_subprocess([sys.executable, "-c", script])


class TestDocDemoMlpModel(ExtTestCase):
    def test_demo_mlp_model_returns_model_proto(self):
        from yobx.doc import demo_mlp_model

        model = demo_mlp_model("")
        self.assertIsInstance(model, onnx.ModelProto)

    def test_demo_mlp_model_graph_has_five_nodes(self):
        from yobx.doc import demo_mlp_model

        model = demo_mlp_model("")
        self.assertEqual(len(model.graph.node), 5)

    def test_demo_mlp_model_op_types(self):
        from yobx.doc import demo_mlp_model

        model = demo_mlp_model("")
        ops = [n.op_type for n in model.graph.node]
        self.assertIn("MatMul", ops)
        self.assertIn("Add", ops)
        self.assertIn("Relu", ops)

    def test_demo_mlp_model_input_output(self):
        from yobx.doc import demo_mlp_model

        model = demo_mlp_model("")
        self.assertEqual(len(model.graph.input), 1)
        self.assertEqual(model.graph.input[0].name, "x")
        self.assertEqual(len(model.graph.output), 1)
        self.assertEqual(model.graph.output[0].name, "output_0")

    def test_demo_mlp_model_initializers(self):
        from yobx.doc import demo_mlp_model

        model = demo_mlp_model("")
        self.assertEqual(len(model.graph.initializer), 4)


@skipif_ci_windows("too long")
@skipif_ci_apple("too long")
@requires_matplotlib()
class TestDocPlotText(ExtTestCase):
    @classmethod
    def setUp(cls):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cls.plt = plt

    def test_plot_text_returns_axes(self):
        import matplotlib.axes
        from yobx.doc import plot_text

        ax = plot_text("line one\nline two")
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.plt.close("all")

    def test_plot_text_uses_provided_axes(self):
        from yobx.doc import plot_text

        _fig, ax = self.plt.subplots()
        result = plot_text("hello", ax=ax)
        self.assertIs(result, ax)
        self.plt.close("all")

    def test_plot_text_sets_title(self):
        from yobx.doc import plot_text

        ax = plot_text("content", title="My Title")
        # plot_text calls ax.set_title(..., loc="left"), so the title text
        # lives in ax._left_title rather than the default centre title.
        self.assertEqual("My Title", ax._left_title.get_text())
        self.plt.close("all")

    def test_plot_text_with_line_color_map(self):
        import matplotlib.axes
        from yobx.doc import plot_text

        ax = plot_text("+added\n-removed\n neutral", line_color_map={"+": "green", "-": "red"})
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.plt.close("all")

    def test_plot_text_custom_figsize(self):
        import matplotlib.axes
        from yobx.doc import plot_text

        ax = plot_text("text", figsize=(8, 4))
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.plt.close("all")

    def test_plot_text_empty_string(self):
        import matplotlib.axes
        from yobx.doc import plot_text

        ax = plot_text("")
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.plt.close("all")


@skipif_ci_windows("too long")
@skipif_ci_apple("too long")
@requires_matplotlib()
class TestDocPlotDot(ExtTestCase):
    @classmethod
    def setUp(cls):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cls.plt = plt

    def _write_fake_png(self, path):
        """Write a minimal valid PNG to *path* so PIL can open it."""
        from PIL import Image

        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(path)

    def test_plot_dot_returns_axes_from_model_proto(self):
        import matplotlib.axes
        from yobx.doc import demo_mlp_model, plot_dot

        model = demo_mlp_model("")
        with patch(
            "yobx.doc.draw_graph_graphviz",
            side_effect=lambda dot, path, engine="dot": self._write_fake_png(path),
        ):
            ax = plot_dot(model)
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.plt.close("all")

    def test_plot_dot_uses_provided_axes(self):
        from yobx.doc import demo_mlp_model, plot_dot

        model = demo_mlp_model("")
        _fig, ax = self.plt.subplots()
        with patch(
            "yobx.doc.draw_graph_graphviz",
            side_effect=lambda dot, path, engine="dot": self._write_fake_png(path),
        ):
            result = plot_dot(model, ax=ax)
        self.assertIs(result, ax)
        self.plt.close("all")

    def test_plot_dot_accepts_dot_string(self):
        import matplotlib.axes
        from yobx.doc import plot_dot

        dot_str = "digraph { a -> b }"
        with patch(
            "yobx.doc.draw_graph_graphviz",
            side_effect=lambda dot, path, engine="dot": self._write_fake_png(path),
        ):
            ax = plot_dot(dot_str)
        self.assertIsInstance(ax, matplotlib.axes.Axes)
        self.plt.close("all")


@skipif_ci_windows("too long")
@skipif_ci_apple("too long")
@requires_ipython()
class TestDocPlotMermaid(ExtTestCase):
    def test_plot_mermaid_graph(self):
        from IPython.display import SVG as IPythonSVG

        from yobx.doc import draw_graph_mermaid

        # Verify IPython.display.SVG is importable (used in the gallery example).
        self.assertIsNotNone(IPythonSVG)
        # Verify draw_graph_mermaid is a callable.
        self.assertTrue(callable(draw_graph_mermaid))


if __name__ == "__main__":
    unittest.main(verbosity=2)
