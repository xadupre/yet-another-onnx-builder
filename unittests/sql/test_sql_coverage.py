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
    not_implemented_error,
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
        rows = [("op_a", "✔ supported", "note a"), ("op_b", "✘ not supported", "note b")]
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


class TestNotImplementedError(unittest.TestCase):
    """Tests for :func:`not_implemented_error`."""

    def test_returns_not_implemented_error(self):
        err = not_implemented_error("sql", "SELECT DISTINCT")
        self.assertIsInstance(err, NotImplementedError)

    def test_sql_select_distinct_includes_notes(self):
        err = not_implemented_error("sql", "SELECT DISTINCT")
        msg = str(err)
        self.assertIn("SELECT DISTINCT", msg)
        self.assertIn("not supported", msg.lower())
        # Notes from SQL_COVERAGE should be included
        self.assertIn("NotImplementedError", msg)

    def test_polars_join_includes_notes(self):
        err = not_implemented_error("polars", "lf.join")
        msg = str(err)
        self.assertIn("lf.join", msg)
        self.assertIn("not supported", msg.lower())

    def test_polars_sort_includes_notes(self):
        err = not_implemented_error("polars", "lf.sort")
        msg = str(err)
        self.assertIn("lf.sort", msg)
        self.assertIn("not supported", msg.lower())

    def test_polars_unique_includes_notes(self):
        err = not_implemented_error("polars", "lf.unique")
        msg = str(err)
        self.assertIn("lf.unique", msg)
        self.assertIn("not supported", msg.lower())

    def test_polars_limit_includes_notes(self):
        err = not_implemented_error("polars", "lf.limit")
        msg = str(err)
        self.assertIn("lf.limit", msg)
        self.assertIn("not supported", msg.lower())

    def test_unknown_section_returns_generic_message(self):
        err = not_implemented_error("nosuchsection", "foo")
        msg = str(err)
        self.assertIn("foo", msg)
        self.assertIn("not supported", msg.lower())

    def test_unknown_construct_returns_generic_message(self):
        err = not_implemented_error("sql", "something_completely_unknown_xyz")
        msg = str(err)
        self.assertIn("something_completely_unknown_xyz", msg)
        self.assertIn("not supported", msg.lower())

    def test_all_unsupported_sql_constructs_have_notes(self):
        """Every _UNSUPPORTED SQL row should produce a message with notes."""
        from yobx.sql.coverage import _UNSUPPORTED

        for construct, status, notes in SQL_COVERAGE:
            if status != _UNSUPPORTED:
                continue
            with self.subTest(construct=construct):
                err = not_implemented_error("sql", construct)
                msg = str(err)
                self.assertIn("not supported", msg.lower())
                # The notes from the coverage table should appear in the message
                self.assertIn(notes, msg)

    def test_all_unsupported_polars_constructs_have_notes(self):
        """Every _UNSUPPORTED polars row should produce a message with notes."""
        from yobx.sql.coverage import _UNSUPPORTED

        for construct, status, notes in POLARS_COVERAGE:
            if status != _UNSUPPORTED:
                continue
            with self.subTest(construct=construct):
                err = not_implemented_error("polars", construct)
                msg = str(err)
                self.assertIn("not supported", msg.lower())
                self.assertIn(notes, msg)


class TestCoverageReflectsImplementation(unittest.TestCase):
    """Spot-checks that SQL_COVERAGE and DATAFRAME_COVERAGE reflect the actual
    implementation state of the converters."""

    def _find_row(self, rows, keyword):
        """Return the first row whose construct contains *keyword* (case-insensitive)."""
        keyword_lower = keyword.lower()
        for row in rows:
            if keyword_lower in row[0].lower():
                return row
        return None

    def test_subqueries_marked_supported(self):
        """Subqueries are processed by sql_convert._populate_graph and must be
        marked as supported in SQL_COVERAGE."""
        from yobx.sql.coverage import _SUPPORTED

        row = self._find_row(SQL_COVERAGE, "subqueries")
        self.assertIsNotNone(row, "No 'Subqueries' row found in SQL_COVERAGE")
        _construct, status, _notes = row
        self.assertEqual(
            status, _SUPPORTED, f"Subqueries should be {_SUPPORTED!r} but got {status!r}"
        )

    def test_dataframe_join_marked_supported(self):
        """TracedDataFrame.join() is implemented and must appear in DATAFRAME_COVERAGE
        as supported."""
        from yobx.sql.coverage import _SUPPORTED

        row = self._find_row(DATAFRAME_COVERAGE, "df.join")
        self.assertIsNotNone(row, "No 'df.join' row found in DATAFRAME_COVERAGE")
        _construct, status, _notes = row
        self.assertEqual(
            status, _SUPPORTED, f"df.join should be {_SUPPORTED!r} but got {status!r}"
        )

    def test_dataframe_pivot_table_marked_supported(self):
        """TracedDataFrame.pivot_table() is implemented and must appear in
        DATAFRAME_COVERAGE as supported."""
        from yobx.sql.coverage import _SUPPORTED

        row = self._find_row(DATAFRAME_COVERAGE, "pivot_table")
        self.assertIsNotNone(row, "No 'pivot_table' row found in DATAFRAME_COVERAGE")
        _construct, status, _notes = row
        self.assertEqual(
            status, _SUPPORTED, f"df.pivot_table should be {_SUPPORTED!r} but got {status!r}"
        )


if __name__ == "__main__":
    unittest.main()
