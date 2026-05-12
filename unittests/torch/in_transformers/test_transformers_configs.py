import os
import unittest
from yobx.ext_test_case import ExtTestCase, requires_transformers
from yobx.torch.in_transformers.models._configs import _retrieve_cached_configurations
from yobx.torch.in_transformers.models.configs import get_cached_configuration


class TestCachedConfigs(ExtTestCase):
    @requires_transformers("")
    def test_retrieve_cached_configurations(self):
        configs = _retrieve_cached_configurations()
        self.assertIsInstance(configs, dict)
        self.assertGreater(len(configs), 0)
        for key, val in configs.items():
            self.assertIsInstance(key, str)
            self.assertTrue(callable(val))

    @requires_transformers("")
    def test_get_cached_configuration_found(self):
        configs = _retrieve_cached_configurations()
        name = next(iter(configs))
        conf = get_cached_configuration(name)
        self.assertIsNotNone(conf)

    @requires_transformers("")
    def test_get_cached_configuration_not_found_returns_none(self):
        conf = get_cached_configuration("nonexistent/model")
        self.assertIsNone(conf)

    @requires_transformers("")
    def test_get_cached_configuration_exc_raises(self):
        with self.assertRaises(AssertionError):
            get_cached_configuration("nonexistent/model", exc=True)

    @requires_transformers("")
    def test_get_cached_configuration_nohttp_raises(self):
        old = os.environ.get("NOHTTP", "")
        try:
            os.environ["NOHTTP"] = "1"
            with self.assertRaises(AssertionError):
                get_cached_configuration("nonexistent/model")
        finally:
            if old:
                os.environ["NOHTTP"] = old
            else:
                del os.environ["NOHTTP"]

    @requires_transformers("")
    def test_get_cached_configuration_arnir0_tiny_llm(self):
        conf = get_cached_configuration("arnir0/Tiny-LLM")
        self.assertIsNotNone(conf)
        self.assertEqual(conf.hidden_size, 192)
        self.assertEqual(conf.num_hidden_layers, 1)
        self.assertEqual(conf.num_attention_heads, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
