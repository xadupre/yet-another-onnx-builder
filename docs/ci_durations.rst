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
   warning is printed to the console.  Retrieved runs are cached per workflow
   for two weeks in ``.cache/ci_durations_workflows.json``.

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
    import json
    import os
    import urllib.request

    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np

    _OWNER = "xadupre"
    _REPO = "yet-another-onnx-builder"
    _API_BASE = f"https://api.github.com/repos/{_OWNER}/{_REPO}"
    _HEADERS = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    _CACHE_MAX_AGE_DAYS = 14
    _CACHE_PATH = os.path.join(".cache", "ci_durations_workflows.json")

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


    def _load_cache():
        """Loads cached workflow data.

        Returns:
            dict: Cached workflow payload or an empty dict when unavailable.
        """
        if not os.path.exists(_CACHE_PATH):
            return {}
        with open(_CACHE_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content or not content.startswith("{") or not content.endswith("}"):
            return {}
        data = json.loads(content)
        if isinstance(data, dict):
            return data
        return {}


    def _save_cache(cache):
        """Saves workflow cache data to disk."""
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, sort_keys=True)


    def _cache_is_recent(cache_entry, now):
        """Checks whether a cache entry is newer than the cache max age.

        Returns:
            bool: True when the cache entry is recent enough.
        """
        fetched_at = cache_entry.get("fetched_at")
        if not fetched_at:
            return False
        try:
            fetched_dt = datetime.datetime.strptime(fetched_at, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=datetime.timezone.utc
            )
        except ValueError:
            return False
        age = now - fetched_dt
        return age <= datetime.timedelta(days=_CACHE_MAX_AGE_DAYS)


    def _run_duration_minutes(run):
        """Computes the wall-clock duration of a successfully completed run.

        Returns:
            float | None: Duration in minutes, or None when unavailable.
        """
        if run.get("status") != "completed":
            return None
        if run.get("conclusion") != "success":
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
        """Collects workflow duration data from cache and/or GitHub API.

        Returns:
            dict: Mapping ``workflow_name -> list[(datetime, duration_min)]``.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        cutoff = (
            now - datetime.timedelta(days=62)
        ).strftime("%Y-%m-%d")
        now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        workflows_data = _gh_get("actions/workflows")
        if not workflows_data:
            return {}

        cache = _load_cache()
        cache_changed = False
        result = {}
        for wf in workflows_data.get("workflows", []):
            name = wf.get("name", "")
            wf_id = wf.get("id")
            path = wf.get("path", "")
            # Skip non-CI workflows
            filename = path.split("/")[-1].lower() if "/" in path else path.lower()
            if any(filename.startswith(p) for p in _SKIP_PATTERNS):
                continue

            cache_key = str(wf_id)
            cached = cache.get(cache_key, {})
            if _cache_is_recent(cached, now):
                runs = cached.get("runs", [])
            else:
                fetched = _fetch_workflow_runs(wf_id, cutoff)
                if fetched:
                    runs = fetched
                    cache[cache_key] = {"fetched_at": now_iso, "runs": runs}
                    cache_changed = True
                else:
                    runs = cached.get("runs", [])
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

        if cache_changed:
            _save_cache(cache)

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
        plt.tight_layout()
    else:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for idx, (wf_name, points) in enumerate(sorted(data.items())):
            fig, ax = plt.subplots(figsize=(10, 4))
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
            ax.set_title(wf_name, fontsize=12)
            ax.set_ylabel("Duration (min)", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.4)
            if len(durations) >= 5:
                ax.legend(fontsize=9)
            fig.autofmt_xdate()
            plt.tight_layout()
