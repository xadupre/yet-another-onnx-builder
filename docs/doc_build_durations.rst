.. _l-doc-build-durations:

=====================================
5 Longest Pages to Generate
=====================================

This page lists the five documentation pages that took the longest to build
in the most recent Sphinx run.  Build times are collected automatically during
the documentation build and written to
``docs/_static/doc_build_durations.json``.

.. note::

   Timings cover the *reading / parsing* phase for each page (source-read →
   doctree-read).  Pages that run ``.. runpython::`` or ``.. plot::``
   directives are naturally slower because they execute Python code.  The
   JSON file is updated each time the documentation is built, so this page
   always reflects the most recent build.

.. runpython::
    :rst:

    import datetime
    print(
        f"*Page generated on "
        f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.*"
    )

.. runpython::
    :rst:

    import json
    import os

    _json_path = os.environ.get("YOBX_DOC_BUILD_DURATIONS_JSON", "")

    if not _json_path or not os.path.exists(_json_path):
        print(
            ".. note::\n\n"
            "   No duration data found yet.  Build the documentation once to "
            "   populate this page."
        )
    else:
        with open(_json_path, encoding="utf-8") as fh:
            rows = json.load(fh)

        lines = [
            ".. list-table:: 5 Slowest Pages",
            "   :header-rows: 1",
            "   :widths: 70 30",
            "",
            "   * - Page",
            "     - Duration (s)",
        ]
        for rank, entry in enumerate(rows, start=1):
            docname = entry["docname"]
            dur = entry["duration_s"]
            lines.append(f"   * - ``{docname}``")
            lines.append(f"     - {dur:.3f}")
        print("\n".join(lines))
