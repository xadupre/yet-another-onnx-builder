"""
Unit tests for :mod:`yobx.sql.coverage` — programmatic coverage tables.
"""

import unittest

from yobx.sql.coverage import (
    DATAFRAME_COVERAGE,
    POLARS_COVERAGE,
    SQL_COVERAGE,
    _rst_table,
    get_sql_coverage,
)


class TestCoverageData(unittest.TestCase):
    """Sanity checks on the raw coverage-data lists."""

    def _check_rows(self, rows, name):
        self.assertIsInstance(rows, list, f"{name} should be a list")
        self.assertGreater(len(rows), 0, f"{name} should be non-empty")
        valid_statuses = {"✔ supported", "⚠ partial", "✘ not supported"}
        for i, row in enumerate(rows):
            self.assertIsInstance(row, tuple, f"{name}[{i}] should be a tuple")
            self.assertEqual(len(row), 3, f"{name}[{i}] should have 3 elements")
            construct, status, notes = row
            self.assertIsInstance(construct, str, f"{name}[{i}] construct should be str")
            self.assertIsInstance(status, str, f"{name}[{i}] status should be str")
            self.assertIsInstance(notes, str, f"{name}[{i}] notes should be str")
            self.assertIn(
                status,
                valid_statuses,
                f"{name}[{i}] status {status!r} is not one of {valid_statuses}",
            )

    def test_sql_coverage_rows(self):
        self._check_rows(SQL_COVERAGE, "SQL_COVERAGE")

    def test_dataframe_coverage_rows(self):
        self._check_rows(DATAFRAME_COVERAGE, "DATAFRAME_COVERAGE")

    def test_polars_coverage_rows(self):
        self._check_rows(POLARS_COVERAGE, "POLARS_COVERAGE")


class TestRstTable(unittest.TestCase):
    """Tests for the :func:`_rst_table` helper."""

    def test_header_present(self):
        rows = [("``SELECT col``", "✔ supported", "column pass-through")]
        output = _rst_table(rows, "SQL construct")
        self.assertIn(".. list-table::", output)
        self.assertIn(":header-rows: 1", output)
        self.assertIn("SQL construct", output)
        self.assertIn("Status", output)
        self.assertIn("Notes", output)

    def test_row_content_present(self):
        rows = [("op_a", "✔ supported", "some note")]
        output = _rst_table(rows, "Operation")
        self.assertIn("op_a", output)
        self.assertIn("✔ supported", output)
        self.assertIn("some note", output)

    def test_multiple_rows(self):
        rows = [
            ("op_a", "✔ supported", "note a"),
            ("op_b", "✘ not supported", "note b"),
        ]
        output = _rst_table(rows, "Operation")
        self.assertIn("op_a", output)
        self.assertIn("op_b", output)
        self.assertIn("note a", output)
        self.assertIn("note b", output)

    def test_ends_with_newline_blank_line(self):
        rows = [("x", "✔ supported", "y")]
        output = _rst_table(rows, "H")
        # The last join produces a trailing "\n" (empty element at end of lines)
        self.assertTrue(output.endswith("\n"), repr(output[-10:]))


class TestGetSqlCoverage(unittest.TestCase):
    """Tests for the public :func:`get_sql_coverage` function."""

    def _assert_valid_rst_table(self, text: str):
        """Assert that *text* looks like a RST list-table."""
        self.assertIn(".. list-table::", text)
        self.assertIn(":header-rows: 1", text)
        self.assertIn("    * - ", text)

    def test_sql_section(self):
        output = get_sql_coverage("sql")
        self._assert_valid_rst_table(output)
        self.assertIn("SQL construct", output)
        # Spot-check one known row
        self.assertIn("SELECT col", output)

    def test_dataframe_section(self):
        output = get_sql_coverage("dataframe")
        self._assert_valid_rst_table(output)
        self.assertIn("Operation", output)
        self.assertIn("df.filter", output)

    def test_polars_section(self):
        output = get_sql_coverage("polars")
        self._assert_valid_rst_table(output)
        self.assertIn("Polars operation", output)
        self.assertIn("lf.select", output)

    def test_default_section_is_sql(self):
        self.assertEqual(get_sql_coverage(), get_sql_coverage("sql"))

    def test_unknown_section_raises(self):
        with self.assertRaises(ValueError) as ctx:
            get_sql_coverage("unknown")
        self.assertIn("unknown", str(ctx.exception))

    def test_all_sections_non_empty(self):
        for section in ("sql", "dataframe", "polars"):
            with self.subTest(section=section):
                output = get_sql_coverage(section)
                self.assertGreater(len(output), 100, f"section {section!r} output is too short")


if __name__ == "__main__":
    unittest.main()
