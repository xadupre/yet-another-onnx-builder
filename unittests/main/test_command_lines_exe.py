import os
import unittest
from contextlib import redirect_stdout
from io import StringIO
from yobx.ext_test_case import ExtTestCase, ignore_warnings
from yobx._command_lines_parser import main
from yobx.helpers.cube_helper import enumerate_csv_files


class TestCommandLines(ExtTestCase):
    @property
    def dummy_path(self):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "two_nodes.onnx"))

    def test_a_parser_print(self):
        for fmt in ["raw", "pretty", "printer", "shape", "dot"]:
            with self.subTest(format=fmt):
                st = StringIO()
                with redirect_stdout(st):
                    main(["print", fmt, self.dummy_path])
                text = st.getvalue()
                self.assertIn("Add", text)

    def test_b_parser_find(self):
        st = StringIO()
        with redirect_stdout(st):
            main(["find", "-i", self.dummy_path, "-n", "node_Add_188"])
        text = st.getvalue()
        self.assertIsInstance(text, str)

    def test_c_parser_find_v2(self):
        st = StringIO()
        with redirect_stdout(st):
            main(["find", "-i", self.dummy_path, "-n", "node_Add_188", "--v2"])
        text = st.getvalue()
        self.assertIsInstance(text, str)

    @ignore_warnings(UserWarning)
    def test_d_parser_agg(self):
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "helpers", "data"))
        assert list(enumerate_csv_files([f"{path}/*.zip"]))
        output = self.get_dump_file("test_parser_agg.xlsx")
        st = StringIO()
        with redirect_stdout(st):
            main(["agg", output, f"{path}/*.zip", "--filter", ".*.csv", "-v", "1"])
        text = st.getvalue()
        self.assertIn("[CubeLogs.to_excel] plots 1 plots", text)
        self.assertExists(output)

    @ignore_warnings(UserWarning)
    def test_e_parser_dot(self):
        output = self.get_dump_file("test_i_parser_dot.dot")
        args = ["dot", self.dummy_path, "-v", "1", "-o", output]
        if not self.unit_test_going():
            args.extend(["--run", "svg"])

        st = StringIO()
        with redirect_stdout(st):
            main(args)
        text = st.getvalue()
        if text:
            # text is empty is dot is not installed,
            # dot may be missing
            self.assertInOr(("converts into dot", "No such file or directory"), text)

    def test_f_parser_partition(self):
        output = self.get_dump_file("test_parser_partition.onnx")
        st = StringIO()
        with redirect_stdout(st):
            main(["partition", self.dummy_path, output, "-v", "1"])
        text = st.getvalue()
        self.assertIn("-- done", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
