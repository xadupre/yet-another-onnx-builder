.. _l-ci-durations:

==================================
CI Job Durations (Past Two Months)
==================================

This page shows the duration of CI workflow runs over the past two months for
the `yet-another-onnx-builder <https://github.com/xadupre/yet-another-onnx-builder>`_
repository.  Durations are fetched from the GitHub REST API and plotted as
time-series charts — one chart per CI workflow.

.. note::

   This page queries the public GitHub REST API.  No authentication token is
   required for a public repository, but anonymous requests are subject to
   rate-limiting (60 requests/hour per IP).  When the API cannot be reached
   (offline build, rate-limit exceeded, …) the chart will be empty and a
   warning is printed to the console.

.. runpython::
    :rst:

    import datetime
    import os

    print(
        f"*Page generated on {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.*"
    )


.. plot::
    :include-source: false

    """Query the GitHub API and plot CI workflow run durations."""

    import datetime
    import urllib.request
    import json
    import math

    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np

    _OWNER = "xadupre"
    _REPO = "yet-another-onnx-builder"
    _API_BASE = f"https://api.github.com/repos/{_OWNER}/{_REPO}"
    _HEADERS = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}

    # Workflows that are NOT CI (skip documentation / style / spelling workflows)
    _SKIP_PATTERNS = ("docs", "style", "spelling", "pyrefly", "mypy", "doc_")


    def _gh_get(path, params=""):
        """Performs a GET request to the GitHub REST API and returns parsed JSON."""
        url = f"{_API_BASE}/{path}"
        if params:
            url = f"{url}?{params}"
        req = urllib.request.Request(url, headers=_HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return None


    def _fetch_workflow_runs(workflow_id, since_iso):
        """Fetches all workflow runs for *workflow_id* created on or after *since_iso*."""
        runs = []
        page = 1
        while True:
            data = _gh_get(
                f"actions/workflows/{workflow_id}/runs",
                f"per_page=100&page={page}&created=>={since_iso}&branch=main",
            )
            if data is None or not data.get("workflow_runs"):
                break
            runs.extend(data["workflow_runs"])
            if len(data["workflow_runs"]) < 100:
                break
            page += 1
        return runs


    def _run_duration_minutes(run):
        """Returns the wall-clock duration of a completed run in minutes, or None."""
        if run.get("status") != "completed":
            return None
        created = run.get("created_at")
        updated = run.get("updated_at")
        if not created or not updated:
            return None
        try:
            fmt = "%Y-%m-%dT%H:%M:%SZ"
            t0 = datetime.datetime.strptime(created, fmt)
            t1 = datetime.datetime.strptime(updated, fmt)
            minutes = (t1 - t0).total_seconds() / 60.0
            return round(minutes, 2) if minutes >= 0 else None
        except ValueError:
            return None


    def _collect_data():
        """Returns a dict mapping workflow_name -> list of (datetime, duration_min)."""
        cutoff = (
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=62)
        ).strftime("%Y-%m-%d")

        workflows_data = _gh_get("actions/workflows")
        if not workflows_data:
            return {}

        result = {}
        for wf in workflows_data.get("workflows", []):
            name = wf.get("name", "")
            wf_id = wf.get("id")
            path = wf.get("path", "")
            # Skip non-CI workflows
            filename = path.split("/")[-1].lower() if "/" in path else path.lower()
            if any(filename.startswith(p) for p in _SKIP_PATTERNS):
                continue

            runs = _fetch_workflow_runs(wf_id, cutoff)
            points = []
            for run in runs:
                dur = _run_duration_minutes(run)
                if dur is None:
                    continue
                try:
                    fmt = "%Y-%m-%dT%H:%M:%SZ"
                    dt = datetime.datetime.strptime(run["created_at"], fmt)
                    points.append((dt, dur))
                except (KeyError, ValueError):
                    continue
            if points:
                points.sort(key=lambda x: x[0])
                result[name] = points

        return result


    data = _collect_data()

    if not data:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(
            0.5, 0.5,
            "No data available (GitHub API not reachable or no completed runs found).",
            ha="center", va="center", transform=ax.transAxes, fontsize=11,
        )
        ax.axis("off")
    else:
        n = len(data)
        ncols = min(2, n)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False
        )

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for idx, (wf_name, points) in enumerate(sorted(data.items())):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            dates = [p[0] for p in points]
            durations = [p[1] for p in points]

            color = colors[idx % len(colors)]
            ax.plot(dates, durations, "o-", color=color, linewidth=1.2, markersize=4,
                    label=wf_name)

            # Rolling average (window = 5)
            if len(durations) >= 5:
                kernel = np.ones(5) / 5
                avg = np.convolve(durations, kernel, mode="valid")
                avg_dates = dates[2: 2 + len(avg)]
                ax.plot(avg_dates, avg, "--", color=color, linewidth=1.5, alpha=0.7,
                        label="5-run avg")

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            ax.set_title(wf_name, fontsize=10)
            ax.set_ylabel("Duration (min)", fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.4)
            if len(durations) >= 5:
                ax.legend(fontsize=7)

        # Hide unused subplots
        for idx in range(len(data), nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        fig.suptitle("CI Workflow Run Durations — Past Two Months", fontsize=13, y=1.01)
        fig.autofmt_xdate()

    plt.tight_layout()
