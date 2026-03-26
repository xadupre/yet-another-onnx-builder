import os
import itertools
import unittest
import pandas
import torch
from yobx.ext_test_case import ExtTestCase, requires_torch, hide_stdout, ignore_warnings
from yobx.torch.input_observer import InputCandidate, InputObserver, _infer_dynamic_dimensions
from yobx.torch import apply_patches_for_model, to_onnx


class TestInputObserver(ExtTestCase):
    def test_infer_dynamic_dimensions(self):
        self.assertEqual([2], _infer_dynamic_dimensions([(1, 2, 3), (1, 2, 4)]))
        self.assertEqual([0, 2], _infer_dynamic_dimensions([(1, 2, 3), (2, 2, 4)]))

    def test_is_empty(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x + 1

        model = Model()
        observer = InputObserver()
        self.assertTrue(observer.is_empty())
        with observer(model):
            model(torch.randn((3, 4)))
        self.assertFalse(observer.is_empty())

    def test_io_captured_args(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        inputs = [
            (torch.randn((5, 6)), torch.randn((1, 6))),
            (torch.randn((7, 7)), torch.randn((1, 7))),
            (torch.randn((7, 8)), torch.randn((1, 8))),
            (torch.randn((7, 9)), torch.randn((1, 9))),
        ]

        model = Model()
        expected = [model(*args) for args in inputs]
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(({0: cst, 1: cst}, {1: cst}), observer.infer_dynamic_shapes())
        args = observer.infer_arguments()
        self.assertIsInstance(args, tuple)
        self.assertEqual(2, len(args))

    def test_io_captured_not_forward(self):
        class Model(torch.nn.Module):
            def notforward(self, w):
                return w.abs()

            def forward(self, x, y):
                return x + self.notforward(y)

        inputs = [
            (torch.randn((5, 6)), torch.randn((1, 6))),
            (torch.randn((7, 7)), torch.randn((1, 7))),
            (torch.randn((7, 8)), torch.randn((1, 8))),
            (torch.randn((7, 9)), torch.randn((1, 9))),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model, method_name="notforward"):
            for args in inputs:
                model(*args)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(({1: cst},), observer.infer_dynamic_shapes())
        args = observer.infer_arguments()
        self.assertIsInstance(args, tuple)
        self.assertEqual(1, len(args))

    def test_io_captured_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        inputs = [
            dict(x=torch.randn((5, 6)), y=torch.randn((1, 6))),
            dict(x=torch.randn((7, 7)), y=torch.randn((1, 7))),
            dict(x=torch.randn((7, 8)), y=torch.randn((1, 8))),
            dict(x=torch.randn((7, 9)), y=torch.randn((1, 9))),
        ]

        model = Model()
        expected = [model(**kwargs) for kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(dict(x={0: cst, 1: cst}, y={1: cst}), observer.infer_dynamic_shapes())
        args = observer.infer_arguments()
        self.assertIsInstance(args, dict)
        self.assertEqual(2, len(args))

    def test_io_captured_kwargs_bool(self):
        class Model(torch.nn.Module):
            def forward(self, x, y, add=True):
                if add:
                    return x + y
                return x - y

        inputs = [
            dict(x=torch.randn((5, 6)), y=torch.randn((1, 6)), add=False),
            dict(x=torch.randn((7, 7)), y=torch.randn((1, 7)), add=False),
            dict(x=torch.randn((7, 8)), y=torch.randn((1, 8)), add=False),
            dict(x=torch.randn((7, 9)), y=torch.randn((1, 9)), add=False),
        ]

        model = Model()
        expected = [model(**kwargs) for kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            dict(x={0: cst, 1: cst}, y={1: cst}, add=None), observer.infer_dynamic_shapes()
        )
        args = observer.infer_arguments()
        self.assertIsInstance(args, dict)
        self.assertEqual(3, len(args))

    def test_io_captured_args_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x, y, z=None, w=None):
                r = x + y
                if z is not None:
                    r += z
                if w is not None:
                    r += w
                return r

        inputs = [
            (
                (torch.randn((5, 6)), torch.randn((1, 6))),
                dict(z=torch.randn((5, 6)), w=torch.randn((1, 6))),
            ),
            (
                (torch.randn((6, 7)), torch.randn((1, 7))),
                dict(z=torch.randn((6, 7)), w=torch.randn((1, 7))),
            ),
            (
                (torch.randn((7, 8)), torch.randn((1, 8))),
                dict(z=torch.randn((7, 8)), w=torch.randn((1, 8))),
            ),
            (
                (torch.randn((8, 9)), torch.randn((1, 9))),
                dict(z=torch.randn((8, 9)), w=torch.randn((1, 9))),
            ),
        ]

        model = Model()
        expected = [model(*args, **kwargs) for args, kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            dict(x={0: cst, 1: cst}, y={1: cst}, z={0: cst, 1: cst}, w={1: cst}),
            observer.infer_dynamic_shapes(),
        )
        args = observer.infer_arguments()
        self.assertIsInstance(args, dict)
        self.assertEqual(4, len(args))

    def test_io_captured_optional_args(self):
        class Model(torch.nn.Module):
            def forward(self, x, y=None):
                if y is None:
                    return x
                return x - y

        inputs = [
            (torch.randn((5, 6)),),
            (torch.randn((6, 7)), torch.randn((1, 7))),
            (torch.randn((7, 8)), torch.randn((1, 8))),
            (torch.randn((8, 9)), torch.randn((1, 9))),
        ]

        model = Model()
        expected = [model(*args) for args in inputs]
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(({0: cst, 1: cst}, {1: cst}), observer.infer_dynamic_shapes())

    def test_infer_arguments_optional(self):
        class Model(torch.nn.Module):
            def forward(self, x, y=None):
                if y is None:
                    return x
                return x - y

        inputs = [
            (torch.randn((5, 6)),),
            (torch.randn((6, 7)), torch.randn((1, 7))),
            (torch.randn((7, 8)), torch.randn((1, 8))),
            (torch.randn((8, 9)), torch.randn((1, 9))),
        ]

        model = Model()
        expected = [model(*args) for args in inputs]
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(({0: cst, 1: cst}, {1: cst}), observer.infer_dynamic_shapes())
        infer_args = observer.infer_arguments(0)
        self.assertIsInstance(infer_args, tuple)
        self.assertEqual(len(infer_args), 2)
        self.assertIsInstance(infer_args[0], torch.Tensor)
        self.assertIsInstance(infer_args[1], torch.Tensor)
        self.assertEqual(infer_args[0].shape, (5, 6))
        self.assertEqual(infer_args[1].shape, (1, 0))

    def test_io_captured_optional_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x, y=None):
                if y is None:
                    return x
                return x - y

        inputs = [
            dict(x=torch.randn((5, 6))),
            dict(x=torch.randn((6, 7)), y=torch.randn((1, 7))),
            dict(x=torch.randn((7, 8)), y=torch.randn((1, 8))),
            dict(x=torch.randn((8, 9)), y=torch.randn((1, 9))),
        ]

        model = Model()
        expected = [model(**kwargs) for kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(dict(x={0: cst, 1: cst}, y={1: cst}), observer.infer_dynamic_shapes())

    def test_io_captured_optional_args_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x, y=None, z=None, w=None):
                r = x + y if y is not None else x
                if z is not None:
                    r += z
                if w is not None:
                    r += w
                return r

        inputs = [
            ((torch.randn((5, 6)),), dict(w=torch.randn((1, 6)))),
            (
                (torch.randn((6, 7)), torch.randn((1, 7))),
                dict(z=torch.randn((6, 7)), w=torch.randn((1, 7))),
            ),
            (
                (torch.randn((7, 8)), torch.randn((1, 8))),
                dict(z=torch.randn((7, 8)), w=torch.randn((1, 8))),
            ),
            (
                (torch.randn((8, 9)), torch.randn((1, 9))),
                dict(z=torch.randn((8, 9)), w=torch.randn((1, 9))),
            ),
        ]

        model = Model()
        expected = [model(*args, **kwargs) for args, kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            dict(x={0: cst, 1: cst}, y={1: cst}, z={0: cst, 1: cst}, w={1: cst}),
            observer.infer_dynamic_shapes(),
        )

    def test_io_captured_not_supported_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x=None, y=None):
                if y is None:
                    return x
                if x is None:
                    return y
                return x - y

        inputs = [
            dict(x=torch.randn((5, 6))),
            dict(y=torch.randn((1, 7))),
            dict(y=torch.randn((1, 7))),
            dict(y=torch.randn((1, 7))),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)
        with self.assertRaisesRegex(RuntimeError, "At least one call to the observed model"):
            observer.infer_dynamic_shapes()

    def test_io_captured_incompatible_number_of_flattened_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x=None, y=None):
                if y is None:
                    return x
                if x is None:
                    return y[0]
                return x - y[0]

        inputs = [
            dict(x=torch.randn((5, 6))),
            dict(x=torch.randn((5, 7)), y=[torch.randn((1, 7))]),
            dict(x=torch.randn((5, 7)), y=[torch.randn((1, 7)), torch.randn((1, 7))]),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)
        with self.assertRaisesRegex(RuntimeError, "Named argument 'y' has"):
            observer.infer_dynamic_shapes()

    def test_io_captured_incompatible_number_of_flattened_args(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x - y[0]

        inputs = [
            (torch.randn((5, 7)), [torch.randn((1, 7))]),
            (torch.randn((5, 7)), [torch.randn((1, 7)), torch.randn((1, 7))]),
        ]

        model = Model()
        observer = InputObserver()
        with self.assertRaisesRegex(RuntimeError, "No inputs were captured."):
            observer.infer_dynamic_shapes()
        with observer(model):
            for args in inputs:
                model(*args)
        with self.assertRaisesRegex(RuntimeError, "Positional argument 1 has"):
            observer.infer_dynamic_shapes()

    def test_io_captured_args_list(self):
        class Model(torch.nn.Module):
            def forward(self, x, y_list):
                return x + y_list[0] + y_list[1]

        inputs = [
            (torch.randn((5, 6)), [torch.randn((1, 6)), torch.randn((1, 6))]),
            (torch.randn((7, 7)), [torch.randn((1, 7)), torch.randn((1, 7))]),
            (torch.randn((7, 8)), [torch.randn((1, 8)), torch.randn((1, 8))]),
            (torch.randn((7, 9)), [torch.randn((1, 9)), torch.randn((1, 9))]),
        ]

        model = Model()
        expected = [model(*args) for args in inputs]
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            ({0: cst, 1: cst}, [{1: cst}, {1: cst}]), observer.infer_dynamic_shapes()
        )

    def test_io_captured_args_list_list(self):
        class Model(torch.nn.Module):
            def forward(self, x, y_list):
                return x + y_list[0] + y_list[1][0]

        inputs = [
            (torch.randn((5, 6)), [torch.randn((1, 6)), [torch.randn((1, 6))]]),
            (torch.randn((7, 7)), [torch.randn((1, 7)), [torch.randn((1, 7))]]),
            (torch.randn((7, 8)), [torch.randn((1, 8)), [torch.randn((1, 8))]]),
            (torch.randn((7, 9)), [torch.randn((1, 9)), [torch.randn((1, 9))]]),
        ]

        model = Model()
        expected = [model(*args) for args in inputs]
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            ({0: cst, 1: cst}, [{1: cst}, [{1: cst}]]), observer.infer_dynamic_shapes()
        )

    def test_io_captured_args_dict(self):
        class Model(torch.nn.Module):
            def forward(self, x, y_dict):
                return x + y_dict["x"] + y_dict["y"]

        inputs = [
            (torch.randn((5, 6)), dict(x=torch.randn((1, 6)), y=torch.randn((1, 6)))),
            (torch.randn((7, 7)), dict(x=torch.randn((1, 7)), y=torch.randn((1, 7)))),
            (torch.randn((7, 8)), dict(x=torch.randn((1, 8)), y=torch.randn((1, 8)))),
            (torch.randn((7, 9)), dict(x=torch.randn((1, 9)), y=torch.randn((1, 9)))),
        ]

        cst = torch.export.Dim.DYNAMIC
        expected = ({0: cst, 1: cst}, dict(x={1: cst}, y={1: cst}))
        model = Model()
        torch.export.export(model, inputs[-1], dynamic_shapes=expected)

        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)

        self.assertEqual(
            ({0: cst, 1: cst}, dict(x={1: cst}, y={1: cst})), observer.infer_dynamic_shapes()
        )

    def test_io_captured_args_dict_args_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x, y_list, z_tuple=None):
                if z_tuple is None:
                    return x + y_list[0] + y_list[1]
                return x + y_list[0] + y_list[1] + z_tuple[0] + z_tuple[1]

        inputs = [
            ((torch.randn((5, 6)), [torch.randn((5, 6)), torch.randn((1, 6))]), {}),
            (
                (torch.randn((6, 7)), [torch.randn((6, 7)), torch.randn((1, 7))]),
                {"z_tuple": (torch.randn((6, 7)), torch.randn((1, 7)))},
            ),
            (
                (torch.randn((7, 8)), [torch.randn((7, 8)), torch.randn((1, 8))]),
                {"z_tuple": (torch.randn((7, 8)), torch.randn((1, 8)))},
            ),
        ]

        cst = torch.export.Dim.DYNAMIC
        expected = dict(
            x={0: cst, 1: cst},
            y_list=[{0: cst, 1: cst}, {1: cst}],
            z_tuple=({0: cst, 1: cst}, {1: cst}),
        )
        model = Model()
        torch.export.export(model, inputs[-1][0], kwargs=inputs[-1][1], dynamic_shapes=expected)

        observer = InputObserver()
        with observer(model):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
        self.assertEqual(expected, observer.infer_dynamic_shapes())

    def test_io_captured_custom_class(self):
        class TestCustomClass:
            def __init__(self, keys, values):
                self.data = list(zip(keys, values))

        def _flatten(custom):
            data = custom.data
            flat = list(itertools.chain.from_iterable(data))
            keys = list(
                itertools.chain.from_iterable(
                    (f"key_{i}", f"value_{i}") for i in range(len(data))
                )
            )
            return flat, keys

        def _flatten_with_keys(custom):
            values, context = _flatten(custom)
            return [
                (torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)
            ], context

        def _unflatten(values, context, output_type=None):
            return TestCustomClass(values[::2], values[1::2])

        torch.utils._pytree.register_pytree_node(
            TestCustomClass,
            _flatten,
            _unflatten,
            serialized_type_name="onnxtest.TestCustomClass",
            flatten_with_keys_fn=_flatten_with_keys,
        )

        class Model(torch.nn.Module):
            def forward(self, x, custom=None):
                if not custom:
                    return x
                data = custom.data
                return x + data[0][0] + data[0][1] + data[1][0] + data[1][1]

        inputs = [
            (torch.randn((5, 6)),),
            (
                torch.randn((6, 7)),
                TestCustomClass(
                    [torch.randn((6, 7)), torch.randn((1, 7))],
                    [torch.randn((1, 7)), torch.randn((6, 7))],
                ),
            ),
            (
                torch.randn((7, 8)),
                TestCustomClass(
                    [torch.randn((7, 8)), torch.randn((1, 8))],
                    [torch.randn((1, 8)), torch.randn((7, 8))],
                ),
            ),
        ]

        cst = torch.export.Dim.DYNAMIC
        expected = ({0: cst, 1: cst}, [{0: cst, 1: cst}, {1: cst}, {1: cst}, {0: cst, 1: cst}])
        flat = torch.utils._pytree.tree_flatten(inputs[-1])[0]
        self.assertEqual(len(flat), 5)

        model = Model()
        model(*inputs[-1])
        torch.export.export(model, inputs[-1], dynamic_shapes=expected)
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)
        self.assertEqual(expected, observer.infer_dynamic_shapes())

    def test_io_captured_args_kwargs_dynamic_batch(self):
        class Model(torch.nn.Module):
            def forward(self, x, y, z=None, w=None):
                r = x + y
                if z is not None:
                    r += z
                if w is not None:
                    r += w
                return r

        inputs = [
            (
                (torch.randn((5, 6)), torch.randn((1, 6))),
                dict(z=torch.randn((5, 6)), w=torch.randn((1, 6))),
            ),
            (
                (torch.randn((5, 7)), torch.randn((1, 7))),
                dict(z=torch.randn((5, 7)), w=torch.randn((1, 7))),
            ),
            (
                (torch.randn((5, 8)), torch.randn((1, 8))),
                dict(z=torch.randn((5, 8)), w=torch.randn((1, 8))),
            ),
            (
                (torch.randn((5, 9)), torch.randn((1, 9))),
                dict(z=torch.randn((5, 9)), w=torch.randn((1, 9))),
            ),
        ]

        model = Model()
        expected = [model(*args, **kwargs) for args, kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            dict(x={0: cst, 1: cst}, y={1: cst}, z={0: cst, 1: cst}, w={1: cst}),
            observer.infer_dynamic_shapes(set_batch_dimension_for={0, "z"}),
        )
        self.assertEqual(
            dict(x={0: cst, 1: cst}, y={1: cst}, z={0: cst, 1: cst}, w={1: cst}),
            observer.infer_dynamic_shapes(set_batch_dimension_for={"x", "z"}),
        )

    @requires_torch("2.10.99")
    def test_io_captured_different_order(self):
        class Model(torch.nn.Module):
            def forward(self, x, y, z=None, w=None):
                r = x + y
                if z is not None:
                    r += z
                if w is not None:
                    r += w
                return r

        inputs = [
            (
                (torch.randn((5, 6)), torch.randn((1, 6))),
                dict(w=torch.randn((1, 6)), z=torch.randn((5, 6))),
            ),
            (
                (torch.randn((5, 7)), torch.randn((1, 7))),
                dict(z=torch.randn((5, 7)), w=torch.randn((1, 7))),
            ),
            (
                (torch.randn((5, 8)), torch.randn((1, 8))),
                dict(w=torch.randn((1, 8)), z=torch.randn((5, 8))),
            ),
            (
                (torch.randn((5, 9)), torch.randn((1, 9))),
                dict(z=torch.randn((5, 9)), w=torch.randn((1, 9))),
            ),
        ]

        model = Model()
        expected = [model(*args, **kwargs) for args, kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            dict(x={0: cst, 1: cst}, y={1: cst}, z={0: cst, 1: cst}, w={1: cst}),
            observer.infer_dynamic_shapes(set_batch_dimension_for={0, "z"}),
        )
        self.assertEqual(
            dict(x={0: cst, 1: cst}, y={1: cst}, z={0: cst, 1: cst}, w={1: cst}),
            observer.infer_dynamic_shapes(set_batch_dimension_for={"x", "z"}),
        )
        proto_name = self.get_dump_file("test_io_captured_different_order.onnx")
        to_onnx(
            model,
            observer.infer_arguments(),
            dynamic_shapes=observer.infer_dynamic_shapes(set_batch_dimension_for=True),
            filename=proto_name,
        )
        if not os.path.exists(proto_name):
            self.skipTest("to_onnx not implemented yet")
        data = observer.check_discrepancies(proto_name, progress_bar=False, include_io=True)
        df = pandas.DataFrame(data)
        self.assertLess(df["abs"].max(), 1e-5)

    def test_io_check_discrepancies(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        inputs = [
            (torch.randn((5, 6)), torch.randn((1, 6))),
            (torch.randn((7, 7)), torch.randn((1, 7))),
            (torch.randn((7, 8)), torch.randn((1, 8))),
            (torch.randn((7, 9)), torch.randn((1, 9))),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)

        proto_name = self.get_dump_file("test_io_check_discrepancies.onnx")
        to_onnx(
            model,
            observer.infer_arguments(),
            dynamic_shapes=observer.infer_dynamic_shapes(set_batch_dimension_for=True),
            filename=proto_name,
        )
        if not os.path.exists(proto_name):
            self.skipTest("to_onnx not implemented yet")
        data = observer.check_discrepancies(proto_name, progress_bar=False)
        self.assertEqual(len(data), 3)
        self.assertIsInstance(data[0], dict)
        self.assertLess(max(obs["abs"] for obs in data), 1e-5)
        df = pandas.DataFrame(data)
        self.assertLess(df["abs"].max(), 1e-5)

    def test_io_infer_arguments(self):
        class Model(torch.nn.Module):
            def forward(self, x, y, z=None, w=None):
                r = x + y
                if z is not None:
                    r += z
                if w is not None:
                    r += w
                return r

        inputs = [
            (
                (torch.randn((5, 6)), torch.randn((1, 6))),
                dict(w=torch.randn((1, 6)), z=torch.randn((5, 6))),
            ),
            (
                (torch.randn((5, 7)), torch.randn((1, 7))),
                dict(z=torch.randn((5, 7)), w=torch.randn((1, 7))),
            ),
            (
                (torch.randn((5, 8)), torch.randn((1, 8))),
                dict(w=torch.randn((1, 8)), z=torch.randn((5, 8))),
            ),
            (
                (torch.randn((5, 9)), torch.randn((1, 9))),
                dict(z=torch.randn((5, 9)), w=torch.randn((1, 9))),
            ),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        iargs = observer.infer_arguments(dict(w=torch.randn((1, 6)), z=torch.randn((5, 6))))
        self.assertEqual(len(iargs), 4)
        self.assertEqual(iargs["x"].shape, (5, 0))
        self.assertEqual(iargs["y"].shape, (1, 0))
        self.assertEqual(iargs["w"].shape, (1, 6))
        self.assertEqual(iargs["z"].shape, (5, 6))

        iargs = observer.infer_arguments((torch.randn((5, 6)), torch.randn((1, 6))))
        self.assertEqual(len(iargs), 4)
        self.assertEqual(iargs["x"].shape, (5, 6))
        self.assertEqual(iargs["y"].shape, (1, 6))
        self.assertEqual(iargs["w"].shape, (1, 0))
        self.assertEqual(iargs["z"].shape, (5, 0))

    def test_io_mixed_args_kwargs_as_dict_1(self):
        class Model(torch.nn.Module):
            def forward(self, x=None, y=None):
                if y is None:
                    return x
                return x + y

        inputs = [
            ((torch.randn((5, 6)),), dict()),
            ((), dict(x=torch.randn((5, 7)), y=torch.randn((5, 7)))),
            ((torch.randn((5, 8)),), dict()),
            ((), dict(x=torch.randn((5, 9)), y=torch.randn((5, 9)))),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model, store_n_calls=4):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        self.assertEqual(len(observer.info), 4)
        observer.infer_dynamic_shapes()
        for cand in observer.info.inputs:
            cand.str_obs()
            self.assertEqual(
                len(cand.flat_list), len([t for t in cand.aligned_flat_list if t is not None])
            )

        cst = torch.export.Dim.DYNAMIC
        dynamic_shapes = observer.infer_dynamic_shapes()
        self.assertEqual({"x": {1: cst}, "y": {1: cst}}, dynamic_shapes)
        args = observer.infer_arguments()
        self.assertIsInstance(args, dict)
        self.assertEqual(2, len(args))
        self.assertEqual(len([v for v in args.values() if v is not None]), 2)

    @hide_stdout()
    @ignore_warnings(FutureWarning)
    def test_io_int_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x=None, y=None, option=1):
                if option == 1:
                    return x + y
                return x - y

        inputs = [
            dict(x=torch.randn((5, 7)), y=torch.randn((5, 7)), option=0),
            dict(x=torch.randn((5, 9)), y=torch.randn((5, 9)), option=0),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model, store_n_calls=4):
            for kwargs in inputs:
                model(**kwargs)
        kwargs = observer.infer_arguments()
        self.assertIn("option", kwargs)
        self.assertEqual(kwargs["option"], 0)
        shapes = observer.infer_dynamic_shapes()
        self.assertIn("option", shapes)
        self.assertEqual(shapes["option"], None)
        ep = torch.export.export(model, (), kwargs=kwargs, dynamic_shapes=shapes)
        self.assertEqualArray(model(**kwargs), ep.module()(**kwargs))
        epo = torch.onnx.export(model, (), kwargs=kwargs, dynamic_shapes=shapes)
        proto = epo.model_proto
        self.assertEqual(["x", "y"], [i.name for i in proto.graph.input])

    def test_io_mixed_args_kwargs_as_dict_2(self):
        class Model(torch.nn.Module):
            def forward(self, x=None, y=None):
                if x is None:
                    return y
                return x + y

        inputs = [
            ((), dict(y=torch.randn((5, 6)))),
            ((torch.randn((5, 7)), torch.randn((5, 7))), dict()),
            ((), dict(y=torch.randn((5, 8)))),
            ((torch.randn((5, 9)), torch.randn((5, 9))), dict()),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model, store_n_calls=4):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        self.assertEqual(len(observer.info), 4)
        with self.assertRaises(RuntimeError):
            observer.infer_dynamic_shapes()
        # for cand in observer.info.inputs:
        #    cand.str_obs()
        #    self.assertEqual(
        #        len(cand.flat_list), len([t for t in cand.aligned_flat_list if t is not None])
        #    )
        # cst = torch.export.Dim.DYNAMIC
        # dynamic_shapes = observer.infer_dynamic_shapes()
        # self.assertEqual({"x": {1: cst}, "y": {1: cst}}, dynamic_shapes)
        # args = observer.infer_arguments()
        # self.assertIsInstance(args, dict)
        # self.assertEqual(2, len(args))
        # self.assertEqual(len([v for v in args.values() if v is not None]), 2)

    def test_infer_dynamic_shapes_missing_kwargs(self):
        class Model(torch.nn.Module):
            def forward(
                self,
                input_ids=None,
                pixel_values=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                token_type_ids=None,
                cache_position=None,
            ):
                return input_ids

        inputs = [
            dict(
                input_ids=torch.ones((1, 28), dtype=torch.int64),
                pixel_values=torch.ones((1, 3, 112, 112), dtype=torch.int64),
                attention_mask=torch.ones((1, 28), dtype=torch.int64),
                position_ids=torch.ones((1, 28), dtype=torch.int64),
                token_type_ids=torch.ones((1, 28), dtype=torch.int64),
                cache_position=torch.ones((28,), dtype=torch.int64),
            ),
            dict(
                input_ids=torch.ones((1, 1), dtype=torch.int64),
                attention_mask=torch.ones((1, 29), dtype=torch.int64),
                position_ids=torch.ones((1, 1), dtype=torch.int64),
                past_key_values=torch.rand((1, 1, 28, 32)),
                token_type_ids=torch.ones((1, 1), dtype=torch.int64),
                cache_position=torch.ones((1,), dtype=torch.int64),
            ),
            dict(
                input_ids=torch.ones((1, 1), dtype=torch.int64),
                attention_mask=torch.ones((1, 30), dtype=torch.int64),
                position_ids=torch.ones((1, 1), dtype=torch.int64),
                past_key_values=torch.rand((1, 1, 29, 32)),
                token_type_ids=torch.ones((1, 1), dtype=torch.int64),
                cache_position=torch.ones((1,), dtype=torch.int64),
            ),
        ]

        model = Model()
        observer = InputObserver(
            value_if_missing=dict(pixel_values=torch.empty((0, 3, 112, 112)))
        )
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)

        shapes = observer.infer_dynamic_shapes(set_batch_dimension_for=True)
        cst = torch.export.Dim.DYNAMIC
        expected = {
            "input_ids": {0: cst, 1: cst},
            "pixel_values": {0: cst},
            "attention_mask": {0: cst, 1: cst},
            "position_ids": {0: cst, 1: cst},
            "past_key_values": {0: cst, 2: cst},
            "token_type_ids": {0: cst, 1: cst},
            "cache_position": {0: cst},
        }
        self.assertEqual(expected, shapes)
        kwargs = observer.infer_arguments()
        self.assertEqual(list(expected), list(kwargs))
        self.assertEqual((0, 3, 112, 112), kwargs["pixel_values"].shape)

    def test_infer_dynamic_shapes_missing_args(self):
        class Model(torch.nn.Module):
            def forward(
                self, input_ids=None, pixel_values=None, attention_mask=None, past_key_values=None
            ):
                return input_ids

        inputs = [
            (
                torch.ones((1, 28), dtype=torch.int64),
                torch.ones((1, 3, 112, 112), dtype=torch.int64),
                torch.ones((1, 28), dtype=torch.int64),
            ),
            (
                torch.ones((1, 1), dtype=torch.int64),
                None,
                torch.ones((1, 29), dtype=torch.int64),
                torch.rand((1, 1, 28, 32)),
            ),
            (
                torch.ones((1, 1), dtype=torch.int64),
                None,
                torch.ones((1, 30), dtype=torch.int64),
                torch.rand((1, 1, 29, 32)),
            ),
        ]

        model = Model()
        observer = InputObserver(
            value_if_missing={1: torch.empty((0, 3, 112, 112), dtype=torch.int64)}
        )
        with observer(model):
            for args in inputs:
                model(*args)

        shapes = observer.infer_dynamic_shapes(set_batch_dimension_for=True)
        cst = torch.export.Dim.DYNAMIC
        expected = ({0: cst, 1: cst}, {0: cst}, {0: cst, 1: cst}, {0: cst, 2: cst})
        self.assertEqual(expected, shapes)
        args = observer.infer_arguments()
        self.assertEqual(len(expected), len(args))
        self.assertEqual((0, 3, 112, 112), args[1].shape)

    def test_infer_dynamic_shapes_missing_kwargs_nested(self):
        class Model(torch.nn.Module):
            def forward(
                self,
                input_ids=None,
                pixel_values=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                token_type_ids=None,
                cache_position=None,
            ):
                return input_ids

        inputs = [
            dict(
                input_ids=torch.ones((1, 28), dtype=torch.int64),
                pixel_values=(
                    torch.ones((1, 3, 112, 112), dtype=torch.int64),
                    torch.ones((1, 3, 112, 112), dtype=torch.int64),
                ),
                attention_mask=torch.ones((1, 28), dtype=torch.int64),
                position_ids=torch.ones((1, 28), dtype=torch.int64),
                token_type_ids=torch.ones((1, 28), dtype=torch.int64),
                cache_position=torch.ones((28,), dtype=torch.int64),
            ),
            dict(
                input_ids=torch.ones((1, 1), dtype=torch.int64),
                attention_mask=torch.ones((1, 29), dtype=torch.int64),
                position_ids=torch.ones((1, 1), dtype=torch.int64),
                past_key_values=torch.rand((1, 1, 28, 32)),
                token_type_ids=torch.ones((1, 1), dtype=torch.int64),
                cache_position=torch.ones((1,), dtype=torch.int64),
            ),
            dict(
                input_ids=torch.ones((1, 1), dtype=torch.int64),
                attention_mask=torch.ones((1, 30), dtype=torch.int64),
                position_ids=torch.ones((1, 1), dtype=torch.int64),
                past_key_values=torch.rand((1, 1, 29, 32)),
                token_type_ids=torch.ones((1, 1), dtype=torch.int64),
                cache_position=torch.ones((1,), dtype=torch.int64),
            ),
        ]

        model = Model()
        observer = InputObserver(
            value_if_missing=dict(
                pixel_values=(
                    torch.empty((0, 3, 112, 112), dtype=torch.int64),
                    torch.empty((0, 3, 112, 112), dtype=torch.int64),
                )
            )
        )
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)

        shapes = observer.infer_dynamic_shapes(set_batch_dimension_for=True)
        cst = torch.export.Dim.DYNAMIC
        expected = {
            "input_ids": {0: cst, 1: cst},
            "pixel_values": ({0: cst}, {0: cst}),
            "attention_mask": {0: cst, 1: cst},
            "position_ids": {0: cst, 1: cst},
            "past_key_values": {0: cst, 2: cst},
            "token_type_ids": {0: cst, 1: cst},
            "cache_position": {0: cst},
        }
        self.assertEqual(expected, shapes)
        kwargs = observer.infer_arguments()
        self.assertEqual(list(expected), list(kwargs))
        self.assertIsInstance(kwargs["pixel_values"], tuple)
        self.assertEqual(2, len(kwargs["pixel_values"]))
        self.assertEqual((0, 3, 112, 112), kwargs["pixel_values"][0].shape)
        self.assertEqual((0, 3, 112, 112), kwargs["pixel_values"][1].shape)

    def test_remove_inputs_kwargs(self):
        """Test that remove_inputs removes a kwarg from the observer info."""

        class Model(torch.nn.Module):
            def forward(self, x, y, z=None):
                r = x + y
                if z is not None:
                    r += z
                return r

        inputs = [
            dict(x=torch.randn((5, 6)), y=torch.randn((1, 6)), z=torch.randn((5, 6))),
            dict(x=torch.randn((7, 7)), y=torch.randn((1, 7)), z=torch.randn((7, 7))),
            dict(x=torch.randn((7, 8)), y=torch.randn((1, 8)), z=torch.randn((7, 8))),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)
        self.assertEqual(len(observer.info), 3)

        cst = torch.export.Dim.DYNAMIC
        ds = observer.infer_dynamic_shapes()
        self.assertIn("z", ds)
        self.assertIn("x", ds)
        self.assertIn("y", ds)

        # Remove z input
        observer.remove_inputs(["z"])

        ds_after = observer.infer_dynamic_shapes()
        self.assertNotIn("z", ds_after)
        self.assertIn("x", ds_after)
        self.assertIn("y", ds_after)
        self.assertEqual(dict(x={0: cst, 1: cst}, y={1: cst}), ds_after)

        args_after = observer.infer_arguments()
        self.assertIsInstance(args_after, dict)
        self.assertNotIn("z", args_after)
        self.assertIn("x", args_after)
        self.assertIn("y", args_after)

    def test_remove_inputs_multiple_kwargs(self):
        """Test that remove_inputs removes multiple kwargs at once."""

        class Model(torch.nn.Module):
            def forward(self, x, y, z=None, w=None):
                r = x + y
                if z is not None:
                    r += z
                if w is not None:
                    r += w
                return r

        inputs = [
            dict(
                x=torch.randn((5, 6)),
                y=torch.randn((1, 6)),
                z=torch.randn((5, 6)),
                w=torch.randn((1, 6)),
            ),
            dict(
                x=torch.randn((6, 7)),
                y=torch.randn((1, 7)),
                z=torch.randn((6, 7)),
                w=torch.randn((1, 7)),
            ),
            dict(
                x=torch.randn((7, 8)),
                y=torch.randn((1, 8)),
                z=torch.randn((7, 8)),
                w=torch.randn((1, 8)),
            ),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)
        self.assertEqual(len(observer.info), 3)

        cst = torch.export.Dim.DYNAMIC
        ds = observer.infer_dynamic_shapes()
        self.assertIn("z", ds)
        self.assertIn("w", ds)

        # Remove z and w inputs
        observer.remove_inputs(["z", "w"])

        ds_after = observer.infer_dynamic_shapes()
        self.assertNotIn("z", ds_after)
        self.assertNotIn("w", ds_after)
        self.assertIn("x", ds_after)
        self.assertIn("y", ds_after)
        self.assertEqual(dict(x={0: cst, 1: cst}, y={1: cst}), ds_after)

        args_after = observer.infer_arguments()
        self.assertIsInstance(args_after, dict)
        self.assertNotIn("z", args_after)
        self.assertNotIn("w", args_after)
        self.assertIn("x", args_after)
        self.assertIn("y", args_after)

    def test_exception_method_not_found(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return x

        model = Model()
        observer = InputObserver()
        with self.assertRaisesRegex(ValueError, "does not have a method"):  # noqa: SIM117
            with observer(model, method_name="nonexistent"):
                pass

    def test_exception_remove_inputs_no_capture(self):
        observer = InputObserver()
        with self.assertRaisesRegex(RuntimeError, "No input was captured"):
            observer.remove_inputs(["x"])

    def test_exception_infer_arguments_unexpected_type(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        inputs = [(torch.randn((5, 6)), torch.randn((1, 6)))]
        model = Model()
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)
        with self.assertRaisesRegex(ValueError, "Unexpected type"):
            observer.infer_arguments("invalid_string")

    def test_exception_value_if_missing_unknown_string_key(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        model = Model()
        observer = InputObserver(value_if_missing=dict(nonexistent=torch.empty((0,))))
        with self.assertRaisesRegex(ValueError, "Unexpected keyword argument"):  # noqa: SIM117
            with observer(model):
                model(torch.randn((5, 6)), torch.randn((1, 6)))

    def test_exception_value_if_missing_int_key_out_of_signature(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        model = Model()
        observer = InputObserver(value_if_missing={10: torch.empty((0,))})
        with self.assertRaisesRegex(ValueError, "Unexpected keyword argument"):  # noqa: SIM117
            with observer(model):
                model(torch.randn((5, 6)), torch.randn((1, 6)))

    def test_exception_value_if_missing_int_key_beyond_args(self):
        class Model(torch.nn.Module):
            def forward(self, x, y=None):
                return x

        model = Model()
        observer = InputObserver(value_if_missing={1: torch.empty((0,))})
        with self.assertRaisesRegex(  # noqa: SIM117
            NotImplementedError, "Unexpected keyword argument"
        ):
            with observer(model):
                model(torch.randn((5, 6)))

    def test_exception_value_if_missing_invalid_key_type(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        model = Model()
        observer = InputObserver(value_if_missing={1.5: torch.empty((0,))})
        with self.assertRaisesRegex(TypeError, "Unexpected type"):  # noqa: SIM117
            with observer(model):
                model(torch.randn((5, 6)), torch.randn((1, 6)))

    def test_exception_remove_variadic_args(self):
        class Model(torch.nn.Module):
            def forward(self, a, *args):
                return a + sum(args)

        inputs = [
            (torch.randn((5, 6)), torch.randn((5, 6))),
            (torch.randn((7, 7)), torch.randn((7, 7))),
        ]
        model = Model()
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)
        with self.assertRaisesRegex(ValueError, "Cannot remove variadic"):
            observer.remove_inputs(["args"])

    def test_exception_remove_variadic_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x, **kwargs):
                return x + kwargs["y"]

        inputs = [
            dict(x=torch.randn((5, 6)), y=torch.randn((5, 6))),
            dict(x=torch.randn((7, 7)), y=torch.randn((7, 7))),
        ]
        model = Model()
        observer = InputObserver()
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)
        with self.assertRaisesRegex(ValueError, "Cannot remove variadic"):
            observer.remove_inputs(["kwargs"])

    def test_exception_different_constant_values(self):
        class Model(torch.nn.Module):
            def forward(self, x, y, add=True):
                if add:
                    return x + y
                return x - y

        inputs = [
            dict(x=torch.randn((5, 6)), y=torch.randn((5, 6)), add=True),
            dict(x=torch.randn((5, 6)), y=torch.randn((5, 6)), add=False),
        ]
        model = Model()
        observer = InputObserver()
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)
        with self.assertRaisesRegex(
            RuntimeError, "Two calls were made with different constant values"
        ):
            observer.infer_dynamic_shapes()

    def test_exception_n_aligned_tensors_not_aligned(self):
        candidate = InputCandidate(
            args=(torch.randn((2, 3)),), kwargs={}, clone=False, cst_kwargs={}
        )
        with self.assertRaisesRegex(RuntimeError, "This input was not aligned with the others"):
            _ = candidate.n_aligned_tensors

    @requires_torch("2.12")
    def test_mixed_named_and_unnamed_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, x, **kwargs):
                return x + kwargs["y"]

        kwargs = dict(x=torch.randn((5, 6)), y=torch.randn((1, 6)))
        model = Model()
        with apply_patches_for_model(patch_torch=True, model=model):
            torch.export.export(
                model,
                (),
                kwargs=kwargs,
                dynamic_shapes={
                    "x": {0: torch.export.Dim.DYNAMIC, 1: torch.export.Dim.DYNAMIC},
                    "kwargs": {"y": {1: torch.export.Dim.DYNAMIC}},
                },
            )

    def test_io_captured_kwargs_kwargs_patch(self):
        class Model(torch.nn.Module):
            def forward(self, x, **kwargs):
                return x + kwargs["y"]

        inputs = [
            dict(x=torch.randn((5, 6)), y=torch.randn((1, 6))),
            dict(x=torch.randn((7, 7)), y=torch.randn((1, 7))),
            dict(x=torch.randn((7, 8)), y=torch.randn((1, 8))),
            dict(x=torch.randn((7, 9)), y=torch.randn((1, 9))),
        ]

        model = Model()
        expected = [model(**kwargs) for kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)
        self.assertEqual(len(observer.info), 3)
        for i in range(3):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        ds = observer.infer_dynamic_shapes()
        self.assertEqual(dict(x={0: cst, 1: cst}, kwargs=dict(y={1: cst})), ds)
        args = observer.infer_arguments()
        self.assertIsInstance(args, dict)
        self.assertEqual(2, len(args))
        self.assertEqual(["x", "y"], list(args))

        dynamic_shapes = torch.export.AdditionalInputs()
        for kwargs in inputs:
            dynamic_shapes.add((), kwargs)
        dss = dynamic_shapes.dynamic_shapes(model, (), inputs[0])
        self.assertEqual({"x": (cst, cst), "kwargs": {"y": (None, cst)}}, dss)

        # _get_range_constraints does not allow this case, let's support it.
        with apply_patches_for_model(patch_torch=True):
            torch.export.export(
                model,
                (),
                kwargs=args,
                dynamic_shapes={"x": {0: cst, 1: cst}, "kwargs": {"y": {1: cst}}},
            )
            torch.export.export(model, (), kwargs=args, dynamic_shapes=ds)

    def test_io_captured_kwargs_kwargs_with_args_patch(self):
        class Model(torch.nn.Module):
            def forward(self, a, *args, **kwargs):
                return a - args[0] * args[1] + kwargs["x"] - kwargs["y"]

        inputs = [
            (
                (torch.randn((5, 6)), torch.randn((5, 6)), torch.randn((5, 6))),
                dict(x=torch.randn((5, 6)), y=torch.randn((1, 6))),
            ),
            (
                (torch.randn((7, 7)), torch.randn((7, 7)), torch.randn((7, 7))),
                dict(x=torch.randn((7, 7)), y=torch.randn((1, 7))),
            ),
        ]

        model = Model()
        expected = [model(*args, **kwargs) for args, kwargs in inputs]
        observer = InputObserver()
        with observer(model):
            for args, kwargs in inputs:
                model(*args, **kwargs)
        self.assertEqual(len(observer.info), 2)
        for i in range(2):
            self.assertEqual(len(observer.info.flat_outputs[i]), 1)
            torch.testing.assert_close(expected[i], observer.info.flat_outputs[i][0])

        cst = torch.export.Dim.DYNAMIC
        ds = observer.infer_dynamic_shapes()
        self.assertEqual(
            {
                "a": {0: cst, 1: cst},
                "args": ({0: cst, 1: cst}, {0: cst, 1: cst}),
                "kwargs": {"x": {0: cst, 1: cst}, "y": {1: cst}},
            },
            ds,
        )

        dynamic_shapes = torch.export.AdditionalInputs()
        for args, kwargs in inputs:
            dynamic_shapes.add(args, kwargs)
        dss = dynamic_shapes.dynamic_shapes(model, *inputs[0])
        self.assertEqual(
            {
                "a": (cst, cst),
                "args": ((cst, cst), (cst, cst)),
                "kwargs": {"x": (cst, cst), "y": (None, cst)},
            },
            dss,
        )

        with self.assertRaises(RuntimeError):
            observer.infer_arguments()

        args, kwargs = observer.infer_arguments(as_args_kwargs=True)
        self.assertIsInstance(kwargs, dict)
        self.assertEqual(["x", "y"], list(kwargs))
        self.assertIsInstance(args, tuple)
        self.assertEqual(len(args), 3)

        # _get_range_constraints
        with apply_patches_for_model(patch_torch=True):
            torch.export.export(
                model,
                args,
                kwargs=kwargs,
                dynamic_shapes={
                    "a": {0: cst, 1: cst},
                    "args": ({0: cst, 1: cst}, {0: cst, 1: cst}),
                    "kwargs": {"x": {0: cst, 1: cst}, "y": {1: cst}},
                },
            )
            torch.export.export(model, args, kwargs=kwargs, dynamic_shapes=ds)

    def test_dim_names_positional_args(self):
        """Test that dim_names produces string labels for dynamic dimensions (positional args)."""

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        inputs = [
            (torch.randn((5, 6)), torch.randn((1, 6))),
            (torch.randn((7, 7)), torch.randn((1, 7))),
            (torch.randn((7, 8)), torch.randn((1, 8))),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)

        cst = torch.export.Dim.DYNAMIC
        # Without dim_names: default behaviour, all DYNAMIC
        default_shapes = observer.infer_dynamic_shapes()
        self.assertEqual(({0: cst, 1: cst}, {1: cst}), default_shapes)

        # With dim_names by input name: dimension 0 of x → "batch", dim 1 → "seq"
        # y is not in dim_names → auto-generated fallback "y_dim_1"
        named_shapes = observer.infer_dynamic_shapes(dim_names={"x": {0: "batch", 1: "seq"}})
        self.assertEqual(({0: "batch", 1: "seq"}, {1: "y_dim_1"}), named_shapes)

        # With dim_names by position index 0 → x
        # y is not in dim_names → auto-generated fallback "y_dim_1"
        named_shapes_by_pos = observer.infer_dynamic_shapes(dim_names={0: {0: "batch", 1: "seq"}})
        self.assertEqual(({0: "batch", 1: "seq"}, {1: "y_dim_1"}), named_shapes_by_pos)

    def test_dim_names_kwargs(self):
        """Test that dim_names produces string labels for dynamic dimensions (keyword args)."""

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        inputs = [
            dict(x=torch.randn((5, 6)), y=torch.randn((1, 6))),
            dict(x=torch.randn((7, 7)), y=torch.randn((1, 7))),
            dict(x=torch.randn((7, 8)), y=torch.randn((1, 8))),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)

        cst = torch.export.Dim.DYNAMIC
        # Without dim_names: default behaviour
        default_shapes = observer.infer_dynamic_shapes()
        self.assertEqual(dict(x={0: cst, 1: cst}, y={1: cst}), default_shapes)

        # With dim_names specifying names for both x and y
        named_shapes = observer.infer_dynamic_shapes(
            dim_names={"x": {0: "batch", 1: "seq"}, "y": {1: "seq"}}
        )
        self.assertEqual(dict(x={0: "batch", 1: "seq"}, y={1: "seq"}), named_shapes)

    def test_dim_names_partial(self):
        """Test that dim_names generates unique fallback strings for unspecified inputs/dims."""

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        inputs = [
            (torch.randn((5, 6)), torch.randn((5, 6))),
            (torch.randn((7, 7)), torch.randn((7, 7))),
            (torch.randn((8, 8)), torch.randn((8, 8))),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)

        # Name only dimension 0 of x; unspecified dims get unique auto-generated strings.
        named_shapes = observer.infer_dynamic_shapes(dim_names={"x": {0: "batch"}})
        # x dim 0 → "batch" (explicit), x dim 1 → "x_dim_1" (auto), y both dims → auto
        self.assertEqual(({0: "batch", 1: "x_dim_1"}, {0: "y_dim_0", 1: "y_dim_1"}), named_shapes)

    def test_dim_names_mixed_args_kwargs(self):
        """Test dim_names with a mix of positional and keyword arguments."""

        class Model(torch.nn.Module):
            def forward(self, x, y, z=None):
                r = x + y
                if z is not None:
                    r += z
                return r

        inputs = [
            ((torch.randn((5, 6)), torch.randn((1, 6))), dict(z=torch.randn((5, 6)))),
            ((torch.randn((6, 7)), torch.randn((1, 7))), dict(z=torch.randn((6, 7)))),
            ((torch.randn((7, 8)), torch.randn((1, 8))), dict(z=torch.randn((7, 8)))),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model):
            for args, kwargs in inputs:
                model(*args, **kwargs)

        named_shapes = observer.infer_dynamic_shapes(
            dim_names={"x": {0: "batch"}, "z": {0: "batch", 1: "seq"}}
        )
        # x dim 0 → "batch" (explicit), x dim 1 → "x_dim_1" (auto)
        # y dim 1 → "y_dim_1" (y not in dim_names → auto)
        # z dim 0 → "batch", z dim 1 → "seq"
        self.assertEqual(
            dict(x={0: "batch", 1: "x_dim_1"}, y={1: "y_dim_1"}, z={0: "batch", 1: "seq"}),
            named_shapes,
        )

    def test_dim_names_auto_basic(self):
        """dim_names=True auto-assigns names; shared value sequences share a label."""

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        # x and y have the same dynamic dim-1 values → they share the same label.
        # x dim-0 values differ from y dim-0 (y dim-0 is always 1 → static).
        inputs = [
            (torch.randn((5, 6)), torch.randn((1, 6))),
            (torch.randn((7, 7)), torch.randn((1, 7))),
            (torch.randn((7, 8)), torch.randn((1, 8))),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)

        shapes = observer.infer_dynamic_shapes(dim_names=True)
        # Result is a tuple (positional args).
        self.assertIsInstance(shapes, tuple)
        self.assertEqual(len(shapes), 2)
        x_shape, y_shape = shapes

        # All returned dim values must be strings (no Dim.DYNAMIC).
        for dim_val in x_shape.values():
            self.assertIsInstance(dim_val, str)
        for dim_val in y_shape.values():
            self.assertIsInstance(dim_val, str)

        # x dim 1 and y dim 1 have identical observed sequences → same label.
        # (Both take values (6, 7, 8) across the three observations.)
        self.assertEqual(x_shape[1], y_shape[1])

        # x dim 0 is distinct → different label from x dim 1.
        self.assertNotEqual(x_shape[0], x_shape[1])

    def test_dim_names_auto_known_llm_names(self):
        """dim_names=True uses precise labels for well-known LLM parameter names."""

        class LLMModel(torch.nn.Module):
            def forward(self, input_ids, attention_mask):
                return torch.cat([input_ids, attention_mask.to(input_ids.dtype)], dim=1)

        inputs = [
            (torch.randint(0, 100, (2, 6), dtype=torch.long), torch.ones(2, 7, dtype=torch.long)),
            (torch.randint(0, 100, (3, 8), dtype=torch.long), torch.ones(3, 9, dtype=torch.long)),
        ]

        model = LLMModel()
        observer = InputObserver()
        with observer(model):
            for args in inputs:
                model(*args)

        shapes = observer.infer_dynamic_shapes(set_batch_dimension_for=True, dim_names=True)
        # Both are positional → tuple result.
        self.assertIsInstance(shapes, tuple)
        ids_shape, mask_shape = shapes

        # input_ids: dim 0 → "batch_size", dim 1 → "sequence_length"
        self.assertEqual(ids_shape[0], "batch_size")
        self.assertEqual(ids_shape[1], "sequence_length")

        # attention_mask: dim 0 → "batch_size" (shared with input_ids),
        # dim 1 → "total_sequence_length" (from _KNOWN_DIM_NAMES)
        self.assertEqual(mask_shape[0], "batch_size")
        self.assertEqual(mask_shape[1], "total_sequence_length")

        # Batch dimension is the same label for both.
        self.assertEqual(ids_shape[0], mask_shape[0])

    def test_remove_inputs_not_in_any_candidate_but_in_signature(self):
        """Removing an input that is in the signature but was never passed should succeed.

        Regression test for: ValueError: No input in all candidates was removed from
        input_names=['cache_position'] when cache_position is absent from all captured
        forward calls (e.g. attention module in newer transformers versions).
        """

        class Model(torch.nn.Module):
            def forward(self, x, y, cache_position=None):
                # cache_position is never supplied by callers in this scenario
                return x + y

        inputs = [
            dict(x=torch.randn((5, 6)), y=torch.randn((1, 6))),
            dict(x=torch.randn((7, 7)), y=torch.randn((1, 7))),
        ]

        model = Model()
        observer = InputObserver()
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)

        # cache_position is in the signature but was never passed — removal must not raise.
        observer.remove_inputs(["cache_position"])

        ds_after = observer.infer_dynamic_shapes()
        self.assertNotIn("cache_position", ds_after)
        self.assertIn("x", ds_after)
        self.assertIn("y", ds_after)

        args_after = observer.infer_arguments()
        self.assertNotIn("cache_position", args_after)

    def test_remove_inputs_not_in_signature_raises(self):
        """Removing an input that is neither in candidates nor in the signature returns 0."""

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        inputs = [dict(x=torch.randn((5, 6)), y=torch.randn((1, 6)))]

        model = Model()
        observer = InputObserver()
        with observer(model):
            for kwargs in inputs:
                model(**kwargs)

        # Silently succeeds (nothing to remove) and returns 0.
        n = observer.remove_inputs(["totally_unknown"])
        self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
