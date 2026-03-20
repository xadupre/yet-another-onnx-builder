import unittest
from contextlib import redirect_stdout
from io import StringIO
from yobx.ext_test_case import ExtTestCase
from yobx._command_lines_parser import (
    get_main_parser,
    get_parser_agg,
    get_parser_dot,
    get_parser_find,
    get_parser_partition,
    get_parser_print,
    get_parser_render_gallery,
    get_parser_run_doc_examples,
    process_outputname,
)


class TestCommandLines(ExtTestCase):
    def test_process_outputname(self):
        self.assertEqual("ggg.g", process_outputname("ggg.g", "hhh.h"))
        self.assertEqual("hhh.ggg.h", process_outputname("+.ggg", "hhh.h"))

    def test_main_parser(self):
        st = StringIO()
        with redirect_stdout(st):
            get_main_parser().print_help()
        text = st.getvalue()
        self.assertIn("agg", text)
        self.assertIn("dot", text)

    def test_parser_print(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_print().print_help()
        text = st.getvalue()
        self.assertIn("pretty", text)
        self.assertIn("onnx-compact", text)

    def test_parser_find(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_find().print_help()
        text = st.getvalue()
        self.assertIn("names", text)

    def test_parser_agg(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_agg().print_help()
        text = st.getvalue()
        self.assertIn("--recent", text)

    def test_parser_dot(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_dot().print_help()
        text = st.getvalue()
        self.assertIn("--run", text)

    def test_parser_partition(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_partition().print_help()
        text = st.getvalue()
        self.assertIn("regex", text)

    def test_parser_agg_cmd(self):
        parser = get_parser_agg()
        args = parser.parse_args(
            [
                "o.xlsx",
                "*.zip",
                "--sbs",
                "dynamo:exporter=onnx-dynamo,opt=ir,attn_impl=eager",
                "--sbs",
                "custom:exporter=custom,opt=default,attn_impl=eager",
            ]
        )
        self.assertEqual(
            args.sbs,
            {
                "custom": {"attn_impl": "eager", "exporter": "custom", "opt": "default"},
                "dynamo": {"attn_impl": "eager", "exporter": "onnx-dynamo", "opt": "ir"},
            },
        )

    def test_parser_run_doc_examples(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_run_doc_examples().print_help()
        text = st.getvalue()
        self.assertIn("--timeout", text)
        self.assertIn("--ext", text)
        self.assertIn("inputs", text)

    def test_main_parser_has_run_doc_examples(self):
        st = StringIO()
        with redirect_stdout(st):
            get_main_parser().print_help()
        text = st.getvalue()
        self.assertIn("run-doc-examples", text)

    def test_parser_render_gallery(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_render_gallery().print_help()
        text = st.getvalue()
        self.assertNotIn("--output", text)
        self.assertIn("inputs", text)

    def test_parser_render_gallery_args(self):
        parser = get_parser_render_gallery()
        args = parser.parse_args(["docs/examples/core/plot_dot_graph.py", "-v", "1"])
        self.assertEqual(args.inputs, ["docs/examples/core/plot_dot_graph.py"])
        self.assertFalse(hasattr(args, "output"))
        self.assertEqual(args.verbose, 1)

    def test_main_parser_has_render_gallery(self):
        st = StringIO()
        with redirect_stdout(st):
            get_main_parser().print_help()
        text = st.getvalue()
        self.assertIn("render-gallery", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
