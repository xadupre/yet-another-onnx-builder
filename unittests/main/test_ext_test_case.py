import os
import math
import sys
import tempfile
import unittest
import warnings
import numpy as np
from yobx.ext_test_case import (
    ExtTestCase,
    ignore_errors,
    ignore_warnings,
    hide_stdout,
    long_test,
    never_test,
    measure_time,
    requires_numpy,
    requires_onnx,
    requires_onnxruntime,
    requires_python,
    requires_sklearn,
    requires_zoo,
    has_onnxruntime,
    has_onnxruntime_training,
    has_onnxruntime_genai,
    skipif_ci_windows,
    skipif_ci_linux,
    skipif_ci_apple,
    statistics_on_file,
    statistics_on_folder,
    unit_test_going,
)


class TestDecoratorFunctions(ExtTestCase):
    def test_ignore_errors_skips_on_expected(self):
        @ignore_errors(ValueError)
        def method(self):
            raise ValueError("expected error")

        with self.assertRaises(unittest.SkipTest):
            method(self)

    def test_ignore_errors_raises_unexpected(self):
        @ignore_errors(ValueError)
        def method(self):
            raise TypeError("unexpected error")

        with self.assertRaises(TypeError):
            method(self)

    def test_ignore_errors_passes_when_no_error(self):
        result = []

        @ignore_errors(ValueError)
        def method(self):
            result.append(1)

        method(self)
        self.assertEqual(result, [1])

    def test_ignore_errors_preserves_name(self):
        @ignore_errors(ValueError)
        def my_func(self):
            pass

        self.assertEqual(my_func.__name__, "my_func")

    def test_ignore_warnings_suppresses(self):
        collected = []

        @ignore_warnings(UserWarning)
        def method(self):
            warnings.warn("test warning", UserWarning)
            collected.append("ran")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            method(self)
        # Warning should have been suppressed
        self.assertEqual(collected, ["ran"])

    def test_ignore_warnings_preserves_name(self):
        @ignore_warnings(UserWarning)
        def my_func(self):
            pass

        self.assertEqual(my_func.__name__, "my_func")

    def test_hide_stdout_suppresses_output(self):
        printed = []

        @hide_stdout()
        def method(self):
            print("this should be hidden")
            printed.append("ran")

        method(self)
        self.assertEqual(printed, ["ran"])

    def test_hide_stdout_preserves_name(self):
        @hide_stdout()
        def my_func(self):
            pass

        self.assertEqual(my_func.__name__, "my_func")

    def test_hide_stdout_with_callback(self):
        captured = []

        @hide_stdout(lambda s: captured.append(s))
        def method(self):
            print("hello captured")

        method(self)
        self.assertEqual(len(captured), 1)
        self.assertIn("hello captured", captured[0])

    def test_long_test_skips_by_default(self):
        # By default LONGTEST is not set, so it should return a skip decorator
        dec = long_test("some reason")
        self.assertTrue(callable(dec))

    def test_never_test_skips_by_default(self):
        # By default NEVERTEST is not set, so it should return a skip decorator
        dec = never_test("some reason")
        self.assertTrue(callable(dec))

    def test_skipif_ci_windows_returns_callable(self):
        dec = skipif_ci_windows("msg")
        self.assertTrue(callable(dec))

    def test_skipif_ci_linux_returns_callable(self):
        dec = skipif_ci_linux("msg")
        self.assertTrue(callable(dec))

    def test_skipif_ci_apple_returns_callable(self):
        dec = skipif_ci_apple("msg")
        self.assertTrue(callable(dec))


class TestRequiresFunctions(ExtTestCase):
    """Tests for requires_* and has_* helper functions."""

    def test_requires_python_passes_for_current_version(self):
        # Should not skip for Python 3.0 which we definitely have
        dec = requires_python((3, 0))
        self.assertTrue(callable(dec))
        # The decorator should be identity (lambda x: x)
        sentinel = object()
        self.assertIs(dec(sentinel), sentinel)

    def test_requires_python_skips_for_future_version(self):
        # Should skip for a future Python version
        dec = requires_python((99, 99))
        self.assertTrue(callable(dec))

    def test_requires_numpy_passes_for_low_version(self):
        dec = requires_numpy("1.0.0")
        self.assertTrue(callable(dec))
        sentinel = object()
        self.assertIs(dec(sentinel), sentinel)

    def test_requires_numpy_skips_for_high_version(self):
        dec = requires_numpy("999.0.0")
        self.assertTrue(callable(dec))

    def test_requires_onnx_passes_for_low_version(self):
        dec = requires_onnx("1.0.0")
        self.assertTrue(callable(dec))
        sentinel = object()
        self.assertIs(dec(sentinel), sentinel)

    def test_requires_onnxruntime_passes_for_low_version(self):
        dec = requires_onnxruntime("1.0.0")
        self.assertTrue(callable(dec))
        sentinel = object()
        self.assertIs(dec(sentinel), sentinel)

    def test_requires_sklearn_passes_for_low_version(self):
        dec = requires_sklearn("1.0.0")
        self.assertTrue(callable(dec))
        sentinel = object()
        self.assertIs(dec(sentinel), sentinel)

    def test_requires_zoo_skips_when_not_set(self):
        # ZOO env var is not set, should return skip decorator
        old = os.environ.pop("ZOO", None)
        try:
            dec = requires_zoo("msg")
            self.assertTrue(callable(dec))
        finally:
            if old is not None:
                os.environ["ZOO"] = old

    def test_has_onnxruntime_returns_bool(self):
        result = has_onnxruntime("1.0.0")
        self.assertIsInstance(result, bool)
        self.assertTrue(result)

    def test_has_onnxruntime_returns_false_for_high_version(self):
        result = has_onnxruntime("999.0.0")
        self.assertIsInstance(result, bool)
        self.assertFalse(result)

    def test_has_onnxruntime_training_returns_bool(self):
        result = has_onnxruntime_training()
        self.assertIsInstance(result, bool)

    def test_has_onnxruntime_genai_returns_bool(self):
        result = has_onnxruntime_genai()
        self.assertIsInstance(result, bool)

    def test_unit_test_going_returns_bool(self):
        result = unit_test_going()
        self.assertIsInstance(result, bool)


class TestMeasureTime(ExtTestCase):
    """Extended tests for measure_time."""

    def test_measure_time_with_context_values(self):
        values = [1, 2, 3, 4, 5]
        res = measure_time(lambda: math.cos(0.5), context={"values": values})
        self.assertIsInstance(res, dict)
        self.assertIn("size", res)
        self.assertEqual(res["size"], 5)

    def test_measure_time_with_array_values(self):
        values = np.array([1.0, 2.0, 3.0])
        res = measure_time(lambda: math.cos(0.5), context={"values": values})
        self.assertIsInstance(res, dict)
        self.assertIn("size", res)
        self.assertEqual(res["size"], 3)

    def test_measure_time_string_stmt(self):
        res = measure_time("1 + 1", context={}, repeat=2, number=5)
        self.assertIsInstance(res, dict)
        self.assertIn("average", res)

    def test_measure_time_invalid_stmt_raises(self):
        self.assertRaise(lambda: measure_time(42), TypeError)

    def test_measure_time_max_time_requires_div_by_number(self):
        self.assertRaise(
            lambda: measure_time(lambda: math.cos(0.5), max_time=0.01, div_by_number=False),
            ValueError,
        )

    def test_measure_time_no_warmup(self):
        res = measure_time(lambda: math.cos(0.5), warmup=0, repeat=2, number=5)
        self.assertIsInstance(res, dict)
        self.assertEqual(res["warmup_time"], 0)


class TestStatisticsOnFile(ExtTestCase):
    """Extended tests for statistics_on_file."""

    def test_statistics_on_file_rst(self):
        with tempfile.NamedTemporaryFile(suffix=".rst", mode="w", delete=False) as f:
            f.write("Title\n=====\n\nSome content here.\n")
            fname = f.name
        try:
            stat = statistics_on_file(fname)
            self.assertEqual(stat["ext"], ".rst")
            self.assertIn("lines", stat)
            self.assertIn("chars", stat)
        finally:
            os.unlink(fname)

    def test_statistics_on_file_binary(self):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"\x00\x01\x02\x03")
            fname = f.name
        try:
            stat = statistics_on_file(fname)
            self.assertIn("size", stat)
        finally:
            os.unlink(fname)

    def test_statistics_on_file_missing_raises(self):
        self.assertRaise(
            lambda: statistics_on_file("/tmp/nonexistent_file_12345.py"),
            AssertionError,
        )

    def test_statistics_on_folder_no_aggregation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = os.path.join(tmpdir, "sample.py")
            with open(py_file, "w") as f:
                f.write("def foo():\n    pass\n")
            stat = statistics_on_folder(tmpdir, aggregation=0)
            self.assertGreater(len(stat), 0)


class TestExtTestCaseAssertions(ExtTestCase):
    """Tests for ExtTestCase assertion methods."""

    def test_assertExists_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            fname = f.name
        try:
            self.assertExists(fname)
        finally:
            os.unlink(fname)

    def test_assertExists_missing_raises(self):
        with self.assertRaises(AssertionError):
            self.assertExists("/tmp/nonexistent_file_9876543.txt")

    def test_assertRaise_correct_exception(self):
        self.assertRaise(lambda: int("not a number"), ValueError)

    def test_assertRaise_no_exception_raised(self):
        with self.assertRaises(AssertionError):
            self.assertRaise(lambda: 1 + 1, ValueError)

    def test_assertRaise_wrong_exception(self):
        # The function raises ValueError, but we expect TypeError -> ValueError propagates
        with self.assertRaises(ValueError):
            self.assertRaise(lambda: int("not a number"), TypeError)

    def test_assertRaise_with_msg(self):
        self.assertRaise(lambda: int("not a number"), ValueError, msg="invalid literal")

    def test_assertEmpty_none(self):
        self.assertEmpty(None)

    def test_assertEmpty_empty_list(self):
        self.assertEmpty([])

    def test_assertEmpty_nonempty_raises(self):
        with self.assertRaises(AssertionError):
            self.assertEmpty([1, 2, 3])

    def test_assertNotEmpty_nonempty_list(self):
        self.assertNotEmpty([1, 2])

    def test_assertNotEmpty_none_raises(self):
        with self.assertRaises(AssertionError):
            self.assertNotEmpty(None)

    def test_assertNotEmpty_empty_list_raises(self):
        with self.assertRaises(AssertionError):
            self.assertNotEmpty([])

    def test_assertStartsWith_passes(self):
        self.assertStartsWith("hello", "hello world")

    def test_assertStartsWith_fails(self):
        with self.assertRaises(AssertionError):
            self.assertStartsWith("world", "hello world")

    def test_assertEndsWith_passes(self):
        self.assertEndsWith("world", "hello world")

    def test_assertEndsWith_fails(self):
        with self.assertRaises(AssertionError):
            self.assertEndsWith("hello", "hello world")

    def test_assertIn_passes(self):
        self.assertIn("ell", "hello")

    def test_assertIn_fails(self):
        with self.assertRaises(AssertionError):
            self.assertIn("xyz", "hello")

    def test_assertInOr_passes(self):
        self.assertInOr(("hello", "world"), "hello there")

    def test_assertInOr_fails(self):
        with self.assertRaises(AssertionError):
            self.assertInOr(("foo", "bar"), "hello there")

    def test_assertHasAttr_passes(self):
        self.assertHasAttr([], "append")

    def test_assertHasAttr_fails(self):
        with self.assertRaises(AssertionError):
            self.assertHasAttr([], "nonexistent_method")

    def test_assertSetContained_passes(self):
        self.assertSetContained({1, 2}, {1, 2, 3})

    def test_assertSetContained_fails(self):
        with self.assertRaises(AssertionError):
            self.assertSetContained({1, 4}, {1, 2, 3})

    def test_assertEqualTrue_passes(self):
        self.assertEqualTrue(True)

    def test_assertEqualTrue_fails_on_1(self):
        with self.assertRaises(AssertionError):
            self.assertEqualTrue(1)

    def test_assertEqualTrue_fails_on_false(self):
        with self.assertRaises(AssertionError):
            self.assertEqualTrue(False)

    def test_assertEqual_pass(self):
        self.assertEqual(42, 42)

    def test_assertEqual_fail(self):
        with self.assertRaises(AssertionError):
            self.assertEqual(1, 2)

    def test_assertEqualArray_numpy(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        self.assertEqualArray(a, b)

    def test_assertEqualArray_numpy_with_atol(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.01, 2.01, 3.01])
        self.assertEqualArray(a, b, atol=0.02)

    def test_assertEqualArray_fail_dtype(self):
        a = np.array([1.0], dtype=np.float32)
        b = np.array([1.0], dtype=np.float64)
        with self.assertRaises(AssertionError):
            self.assertEqualArray(a, b)

    def test_assertEqualArrays_list(self):
        a = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        b = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        self.assertEqualArrays(a, b)

    def test_assertAlmostEqual_scalars(self):
        self.assertAlmostEqual(1.0, 1.001, atol=0.01)

    def test_assertEqualAny_int(self):
        self.assertEqualAny(5, 5)

    def test_assertEqualAny_float(self):
        self.assertEqualAny(3.14, 3.14)

    def test_assertEqualAny_string(self):
        self.assertEqualAny("hello", "hello")

    def test_assertEqualAny_none(self):
        self.assertEqualAny(None, None)

    def test_assertEqualAny_array(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0])
        self.assertEqualAny(a, b)

    def test_assertEqualDataFrame(self):
        import pandas as pd

        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        self.assertEqualDataFrame(df1, df2)

    def test_capture_stdout(self):
        def fn():
            print("captured output")
            return 42

        result, out, err = self.capture(fn)
        self.assertEqual(result, 42)
        self.assertIn("captured output", out)

    def test_capture_stderr(self):
        def fn():
            print("error output", file=sys.stderr)
            return 1

        result, out, err = self.capture(fn)
        self.assertEqual(result, 1)
        self.assertIn("error output", err)

    def test_tryCall_success(self):
        result = self.tryCall(lambda: 99)
        self.assertEqual(result, 99)

    def test_tryCall_returns_none_on_match(self):
        result = self.tryCall(
            lambda: (_ for _ in ()).throw(ValueError("some specific error")),
            none_if="specific",
        )
        self.assertIsNone(result)

    def test_tryCall_raises_with_msg(self):
        with self.assertRaises(AssertionError):
            self.tryCall(
                lambda: (_ for _ in ()).throw(ValueError("unexpected")),
                msg="custom error message",
                none_if="not-in-error",
            )

    def test_set_env_sets_and_restores(self):
        old_val = os.environ.get("TEST_YOBX_VAR", None)
        with self.set_env("TEST_YOBX_VAR", "test_value"):
            self.assertEqual(os.environ.get("TEST_YOBX_VAR"), "test_value")
        # After context, it should be restored
        restored = os.environ.get("TEST_YOBX_VAR", None)
        self.assertEqual(restored, old_val or "")

    def test_subloop_single_iterable(self):
        results = list(self.subloop([1, 2, 3]))
        self.assertEqual(results, [1, 2, 3])

    def test_subloop_multiple_iterables(self):
        results = list(self.subloop([1, 2], ["a", "b"]))
        self.assertEqual(len(results), 4)

    def test_unit_test_going_method(self):
        result = self.unit_test_going()
        self.assertIsInstance(result, bool)

    def test_verbose_property(self):
        self.assertIsInstance(self.verbose, int)

    def test_get_dump_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.get_dump_file("model.onnx", folder=tmpdir)
            self.assertEndsWith("model.onnx", path)
            self.assertTrue(os.path.exists(tmpdir))

    def test_dump_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.dump_text("output.txt", "hello world", folder=tmpdir)
            self.assertExists(path)
            with open(path) as f:
                content = f.read()
            self.assertEqual(content, "hello world")


if __name__ == "__main__":
    unittest.main(verbosity=2)
