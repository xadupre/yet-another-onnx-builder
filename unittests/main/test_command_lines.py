import unittest
from contextlib import redirect_stdout
from io import StringIO
from yobx.ext_test_case import ExtTestCase
from yobx._command_lines_parser import (
    get_main_parser,
    get_parser_agg,
    get_parser_copilot_draft,
    get_parser_dot,
    get_parser_find,
    get_parser_partition,
    get_parser_print,
    get_parser_run_doc_examples,
    get_parser_stats,
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
        self.assertIn("copilot-draft", text)

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

    def test_parser_copilot_draft(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_copilot_draft().print_help()
        text = st.getvalue()
        self.assertIn("--dry-run", text)
        self.assertIn("--token", text)
        self.assertIn("--output-dir", text)

    def test_parser_copilot_draft_args(self):
        parser = get_parser_copilot_draft()
        args = parser.parse_args(
            ["sklearn.linear_model.Ridge", "--dry-run", "--token", "ghp_test", "-v", "1"]
        )
        self.assertEqual(args.estimator, "sklearn.linear_model.Ridge")
        self.assertTrue(args.dry_run)
        self.assertEqual(args.token, "ghp_test")
        self.assertEqual(args.verbose, 1)
        self.assertEqual(args.output_dir, "")

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

    def test_parser_stats(self):
        st = StringIO()
        with redirect_stdout(st):
            get_parser_stats().print_help()
        text = st.getvalue()
        self.assertIn("--output", text)
        self.assertIn("--verbose", text)

    def test_main_parser_has_stats(self):
        st = StringIO()
        with redirect_stdout(st):
            get_main_parser().print_help()
        text = st.getvalue()
        self.assertIn("stats", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
