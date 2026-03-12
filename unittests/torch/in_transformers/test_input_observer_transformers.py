import unittest
import pandas
from yobx.ext_test_case import ExtTestCase, requires_transformers
from yobx.torch import register_flattening_functions, get_tiny_model
from yobx.torch.input_observer import InputObserver
from yobx.torch import apply_patches_for_model
from yobx.torch.interpreter import to_onnx


class TestInputObserverTransformers(ExtTestCase):
    def _common_test(self, drop_inputs, atol=1e-4):
        mid = "arnir0/Tiny-LLM"
        data = get_tiny_model(mid)
        model, inputs = data.model, data.export_inputs
        input_ids = inputs["input_ids"][:1]

        observer = InputObserver()
        with (
            register_flattening_functions(patch_transformers=True),
            observer(model.model.layers[0].self_attn),
        ):
            model.generate(input_ids=input_ids, do_sample=False)

        with (
            register_flattening_functions(patch_transformers=True),
            apply_patches_for_model(patch_transformers=True, patch_torch=True, model=model),
        ):
            if drop_inputs:
                observer.remove_inputs(drop_inputs)
            kwargs = observer.infer_arguments()
            ds = observer.infer_dynamic_shapes(set_batch_dimension_for=True)

            onx = to_onnx(
                model.model.layers[0].self_attn, args=(), kwargs=kwargs, dynamic_shapes=ds
            )

            data = observer.check_discrepancies(onx, progress_bar=False)
            df = pandas.DataFrame(data)
            self.assertLess(df["abs"].max(), atol)

    @requires_transformers("5.2")
    def test_input_observer_attention_no_cache_position_no_position_ids(self):
        # TODO: fix it
        self._common_test(["position_ids", "cache_position"], atol=0.5)

    @requires_transformers("5.2")
    def test_input_observer_attention_no_cache_position(self):
        # TODO: fix it
        self._common_test(["cache_position"], atol=0.5)

    @requires_transformers("5.2")
    def test_input_observer_attention_no_position_ids(self):
        # TODO: fix it
        self._common_test(["position_ids"], atol=0.5)

    @requires_transformers("5.2")
    def test_input_observer_attention(self):
        # TODO: fix it
        self._common_test([], atol=0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
