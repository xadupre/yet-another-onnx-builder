import unittest
import pandas
import torch
from yobx.ext_test_case import ExtTestCase, requires_transformers
from yobx.torch import register_flattening_functions, get_tiny_model
from yobx.torch.input_observer import InputObserver
from yobx.torch.in_transformers.cache_helper import make_dynamic_cache, make_encoder_decoder_cache
from yobx.torch import apply_patches_for_model
from yobx.torch.interpreter import to_onnx
from yobx.helpers.rt_helper import onnx_generate


class TestInputObserverTransformers(ExtTestCase):
    @requires_transformers("5.2")
    def test_input_observer_onnx_generate_tiny_llm(self):
        mid = "arnir0/Tiny-LLM"
        data = get_tiny_model(mid)
        model, inputs, _ds = data.model, data.export_inputs, data.dynamic_shapes
        input_ids = inputs["input_ids"][:1]

        observer = InputObserver()
        with (
            register_flattening_functions(patch_transformers=True),
            observer(model),
        ):
            outputs = model.generate(input_ids=input_ids, do_sample=False)

        filenamec = self.get_dump_file("test_input_observer_onnx_generate_tiny_llm.onnx")

        with (
            register_flattening_functions(patch_transformers=True),
            apply_patches_for_model(patch_transformers=True, patch_torch=True, model=model),
        ):
            kwargs = observer.infer_arguments()
            ds = observer.infer_dynamic_shapes(set_batch_dimension_for=True)
            to_onnx(model, (), kwargs=kwargs, dynamic_shapes=ds, filename=filenamec)

        data = observer.check_discrepancies(filenamec, progress_bar=False)
        df = pandas.DataFrame(data)
        self.assertLess(df["abs"].max(), 1e-4)

        onnx_tokens = onnx_generate(
            filenamec,
            input_ids=input_ids,
            attention_mask=torch.ones(input_ids.shape, dtype=torch.int64),
            eos_token_id=model.config.eos_token_id,
            max_new_tokens=20,
        )
        if onnx_tokens is not None:
            self.assertEqualArray(outputs, onnx_tokens)

    @requires_transformers("4.55")
    def test_encoder_decoder_cache_args(self):
        class Model(torch.nn.Module):
            def forward(self, cache):
                return cache

        cache1 = make_encoder_decoder_cache(
            make_dynamic_cache(
                [
                    (torch.ones((1, 6, 4, 64)), torch.ones((1, 6, 4, 64))),
                    (torch.ones((1, 6, 4, 64)), torch.ones((1, 6, 4, 64))),
                    (torch.ones((1, 6, 4, 64)), torch.ones((1, 6, 4, 64))),
                    (torch.ones((1, 6, 4, 64)), torch.ones((1, 6, 4, 64))),
                ],
            ),
            make_dynamic_cache(
                [
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                ],
            ),
        )
        cache2 = make_encoder_decoder_cache(
            make_dynamic_cache(
                [
                    (torch.ones((1, 6, 5, 64)), torch.ones((1, 6, 5, 64))),
                    (torch.ones((1, 6, 5, 64)), torch.ones((1, 6, 5, 64))),
                    (torch.ones((1, 6, 5, 64)), torch.ones((1, 6, 5, 64))),
                    (torch.ones((1, 6, 5, 64)), torch.ones((1, 6, 5, 64))),
                ],
            ),
            make_dynamic_cache(
                [
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                ],
            ),
        )

        model = Model()
        observer = InputObserver()
        with (
            register_flattening_functions(patch_transformers=True),
            observer(model),
        ):
            model(None)
            model(cache1)
            model(cache2)
            dyn_shapes = observer.infer_dynamic_shapes()
            args = observer.infer_arguments(0)

        with register_flattening_functions(patch_transformers=True):
            dyn_shapes_out = observer.infer_dynamic_shapes()
            args0 = observer.infer_arguments(0)

        self.assertEqual(dyn_shapes, dyn_shapes_out)
        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            (
                [
                    [
                        {2: cst},
                        {2: cst},
                        {2: cst},
                        {2: cst},
                        {2: cst},
                        {2: cst},
                        {2: cst},
                        {2: cst},
                    ],
                    [{}, {}, {}, {}, {}, {}, {}, {}],
                ],
            ),
            dyn_shapes_out,
        )
        self.assertEqualAny(args, args0)
        self.assertEqual(args[0].self_attention_cache.layers[0].keys.shape, (1, 6, 0, 64))
        self.assertEqual(args[0].cross_attention_cache.layers[0].keys.shape, (1, 6, 1500, 64))

    @requires_transformers("4.55")
    def test_encoder_decoder_cache_kwargs(self):
        class Model(torch.nn.Module):
            def forward(self, cache):
                return cache

        cache1 = make_encoder_decoder_cache(
            make_dynamic_cache(
                [
                    (torch.ones((1, 6, 4, 64)), torch.ones((1, 6, 4, 64))),
                    (torch.ones((1, 6, 4, 64)), torch.ones((1, 6, 4, 64))),
                    (torch.ones((1, 6, 4, 64)), torch.ones((1, 6, 4, 64))),
                    (torch.ones((1, 6, 4, 64)), torch.ones((1, 6, 4, 64))),
                ],
            ),
            make_dynamic_cache(
                [
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                ],
            ),
        )
        cache2 = make_encoder_decoder_cache(
            make_dynamic_cache(
                [
                    (torch.ones((1, 6, 5, 64)), torch.ones((1, 6, 5, 64))),
                    (torch.ones((1, 6, 5, 64)), torch.ones((1, 6, 5, 64))),
                    (torch.ones((1, 6, 5, 64)), torch.ones((1, 6, 5, 64))),
                    (torch.ones((1, 6, 5, 64)), torch.ones((1, 6, 5, 64))),
                ],
            ),
            make_dynamic_cache(
                [
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                    (torch.ones((1, 6, 1500, 64)), torch.ones((1, 6, 1500, 64))),
                ],
            ),
        )

        model = Model()
        observer = InputObserver()
        with (
            register_flattening_functions(patch_transformers=True),
            observer(model),
        ):
            model(cache=None)
            model(cache=cache1)
            model(cache=cache2)
            dyn_shapes = observer.infer_dynamic_shapes()
            args = observer.infer_arguments(0)

        with register_flattening_functions(patch_transformers=True):
            dyn_shapes_out = observer.infer_dynamic_shapes()
            args0 = observer.infer_arguments(0)

        self.assertEqual(dyn_shapes, dyn_shapes_out)
        cst = torch.export.Dim.DYNAMIC
        self.assertEqual(
            dict(
                cache=[
                    [
                        {2: cst},
                        {2: cst},
                        {2: cst},
                        {2: cst},
                        {2: cst},
                        {2: cst},
                        {2: cst},
                        {2: cst},
                    ],
                    [{}, {}, {}, {}, {}, {}, {}, {}],
                ],
            ),
            dyn_shapes_out,
        )
        self.assertEqualAny(args, args0)
        self.assertEqual(args["cache"].self_attention_cache.layers[0].keys.shape, (1, 6, 0, 64))
        self.assertEqual(
            args["cache"].cross_attention_cache.layers[0].keys.shape, (1, 6, 1500, 64)
        )

    @requires_transformers("4.57")
    def test_infer_dynamic_shapes_missing_pixels(self):
        import transformers

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
                input_ids=torch.ones((1, 282), dtype=torch.int64),
                pixel_values=torch.ones((1, 3, 896, 896), dtype=torch.int64),
                attention_mask=torch.ones((1, 282), dtype=torch.int64),
                position_ids=torch.ones((1, 282), dtype=torch.int64),
                token_type_ids=torch.ones((1, 282), dtype=torch.int64),
                cache_position=torch.ones((282,), dtype=torch.int64),
            ),
            dict(
                input_ids=torch.ones((1, 1), dtype=torch.int64),
                attention_mask=torch.ones((1, 283), dtype=torch.int64),
                position_ids=torch.ones((1, 1), dtype=torch.int64),
                past_key_values=make_dynamic_cache(
                    [
                        (torch.rand((1, 1, 282, 32)), torch.rand((1, 1, 282, 32))),
                        (torch.rand((1, 1, 282, 32)), torch.rand((1, 1, 282, 32))),
                    ],
                    cls_layers=[
                        transformers.cache_utils.DynamicSlidingWindowLayer,
                        transformers.cache_utils.DynamicLayer,
                    ],
                ),
                token_type_ids=torch.ones((1, 1), dtype=torch.int64),
                cache_position=torch.ones((1,), dtype=torch.int64),
            ),
            dict(
                input_ids=torch.ones((1, 1), dtype=torch.int64),
                attention_mask=torch.ones((1, 284), dtype=torch.int64),
                position_ids=torch.ones((1, 1), dtype=torch.int64),
                past_key_values=make_dynamic_cache(
                    [
                        (torch.rand((1, 1, 283, 32)), torch.rand((1, 1, 283, 32))),
                        (torch.rand((1, 1, 283, 32)), torch.rand((1, 1, 283, 32))),
                    ],
                    cls_layers=[
                        transformers.cache_utils.DynamicSlidingWindowLayer,
                        transformers.cache_utils.DynamicLayer,
                    ],
                ),
                token_type_ids=torch.ones((1, 1), dtype=torch.int64),
                cache_position=torch.ones((1,), dtype=torch.int64),
            ),
        ]

        model = Model()
        observer = InputObserver(
            value_if_missing=dict(pixel_values=torch.empty((0, 3, 896, 896), dtype=torch.int64))
        )
        with (
            register_flattening_functions(patch_transformers=True),
            observer(model),
        ):
            for kwargs in inputs:
                model(**kwargs)

        with register_flattening_functions(patch_transformers=True):
            shapes = observer.infer_dynamic_shapes(set_batch_dimension_for=True)

        cst = torch.export.Dim.DYNAMIC
        expected = {
            "input_ids": {0: cst, 1: cst},
            "pixel_values": {0: cst},
            "attention_mask": {0: cst, 1: cst},
            "position_ids": {0: cst, 1: cst},
            "past_key_values": [
                {0: cst, 2: cst},
                {0: cst, 2: cst},
                {0: cst, 2: cst},
                {0: cst, 2: cst},
            ],
            "token_type_ids": {0: cst, 1: cst},
            "cache_position": {0: cst},
        }
        self.assertEqual(expected, shapes)

    def _run_llama_attention_export(self, inputs1, inputs2, tag):
        """Helper: observe two forward passes, infer shapes, export with opset 22 and 24."""
        import os
        from yobx.torch.torch_helper import torch_deepcopy

        model = inputs1.pop("_model")
        inputs2.pop("_model")

        observer = InputObserver()
        with (
            register_flattening_functions(patch_transformers=True),
            observer(model),
        ):
            model(**torch_deepcopy(inputs1))
            model(**torch_deepcopy(inputs2))

        with register_flattening_functions(patch_transformers=True):
            kwargs = observer.infer_arguments()
            ds = observer.infer_dynamic_shapes(set_batch_dimension_for=True)

        self.assertIsNotNone(kwargs)
        self.assertIsNotNone(ds)

        for opset in (22, 24):
            filenamec = self.get_dump_file(f"{tag}_{opset}.onnx")
            with (
                register_flattening_functions(patch_transformers=True),
                apply_patches_for_model(patch_transformers=True, patch_torch=True, model=model),
            ):
                to_onnx(
                    model,
                    (),
                    kwargs=kwargs,
                    dynamic_shapes=ds,
                    filename=filenamec,
                    target_opset=opset,
                )
            self.assertTrue(os.path.exists(filenamec))

    def _make_llama_attention_inputs(self, config, seq1, seq2, use_position_ids, use_cache_position):
        """Return (inputs1, inputs2) dicts for LlamaAttentionWrapper forward calls."""
        from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding

        class LlamaAttentionWrapper(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.rotary_emb = LlamaRotaryEmbedding(config=config)
                self.self_attn = LlamaAttention(config=config, layer_idx=0)

            def forward(
                self,
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                cache_position=None,
            ):
                position_embeddings = self.rotary_emb(hidden_states, position_ids)
                attn_output, _, past_key_value = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=False,
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
                return attn_output, past_key_value

        bsize = 1
        n_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        initial_cache_len = 4
        model = LlamaAttentionWrapper(config)

        def _make(seq, cache_len):
            return dict(
                _model=model,
                hidden_states=torch.randn(bsize, seq, config.hidden_size),
                attention_mask=None,
                position_ids=(
                    torch.arange(seq, dtype=torch.int64).unsqueeze(0).expand(bsize, -1)
                    if use_position_ids
                    else None
                ),
                past_key_value=make_dynamic_cache(
                    [
                        (
                            torch.randn(bsize, n_kv_heads, cache_len, head_dim),
                            torch.randn(bsize, n_kv_heads, cache_len, head_dim),
                        )
                    ]
                ),
                cache_position=(
                    torch.arange(cache_len, cache_len + seq, dtype=torch.int64)
                    if use_cache_position
                    else None
                ),
            )

        return _make(seq1, initial_cache_len), _make(seq2, initial_cache_len + 2)

    @requires_transformers("4.45")
    def test_input_observer_llama_attention(self):
        from yobx.torch.in_transformers.models import get_cached_configuration

        config = get_cached_configuration("arnir0/Tiny-LLM")
        inputs1, inputs2 = self._make_llama_attention_inputs(
            config, seq1=3, seq2=5, use_position_ids=True, use_cache_position=True
        )
        self._run_llama_attention_export(
            inputs1, inputs2, tag="test_input_observer_llama_attention"
        )

    @requires_transformers("4.45")
    def test_input_observer_llama_attention_no_position_ids(self):
        from yobx.torch.in_transformers.models import get_cached_configuration

        config = get_cached_configuration("arnir0/Tiny-LLM")
        inputs1, inputs2 = self._make_llama_attention_inputs(
            config, seq1=3, seq2=5, use_position_ids=False, use_cache_position=True
        )
        self._run_llama_attention_export(
            inputs1, inputs2, tag="test_input_observer_llama_attention_no_position_ids"
        )

    @requires_transformers("4.45")
    def test_input_observer_llama_attention_no_cache_position(self):
        from yobx.torch.in_transformers.models import get_cached_configuration

        config = get_cached_configuration("arnir0/Tiny-LLM")
        inputs1, inputs2 = self._make_llama_attention_inputs(
            config, seq1=3, seq2=5, use_position_ids=True, use_cache_position=False
        )
        self._run_llama_attention_export(
            inputs1, inputs2, tag="test_input_observer_llama_attention_no_cache_position"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
