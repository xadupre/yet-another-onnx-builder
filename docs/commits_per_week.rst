.. _l-commits-per-week:

===================
Commits per Week
===================

This page shows the number of commits per week for the
`yet-another-onnx-builder <https://github.com/xadupre/yet-another-onnx-builder>`_
repository.  Commit counts are fetched from the GitHub REST API and plotted as
a bar chart.

.. note::

   This page queries the public GitHub REST API.  No authentication token is
   required for a public repository, but anonymous requests are subject to
   rate-limiting (60 requests/hour per IP).  When the API cannot be reached
   (offline build, rate-limit exceeded, …) the chart will be empty and a
   warning is printed to the console.  Retrieved data is cached as a CSV file
   and refreshed when the cache is older than two weeks.

.. runpython::
    :rst:

    import datetime
    import os

    print(
        f"*Page generated on {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.*"
    )


.. plot::
    :include-source: false

    """Query the GitHub API and plot the number of commits per week."""

    import csv
    import datetime
    import json
    import os
    import urllib.request

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    _OWNER = "xadupre"
    _REPO = "yet-another-onnx-builder"
    _API_BASE = f"https://api.github.com/repos/{_OWNER}/{_REPO}"
    _HEADERS = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    _CACHE_MAX_AGE_DAYS = 14
    _USER_CACHE_DIR = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    _CACHE_DIR = os.path.join(
        _USER_CACHE_DIR, "yet-another-onnx-builder", "commits_per_week"
    )
    _CACHE_FILE = os.path.join(_CACHE_DIR, "commits_per_week.csv")


    def _gh_get(path, params=""):
        """Performs a GET request to the GitHub REST API.

        Returns:
            dict | None: Parsed JSON response, or ``None`` on any error.
        """
        url = f"{_API_BASE}/{path}"
        if params:
            url = f"{url}?{params}"
        req = urllib.request.Request(url, headers=_HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except Exception:
            return None


    def _cache_is_recent(now):
        """Checks whether the cache file is newer than the cache max age.

        Returns:
            bool: True when the cache entry is recent enough.
        """
        if not os.path.exists(_CACHE_FILE):
            return False
        try:
            fetched_ts = os.path.getmtime(_CACHE_FILE)
            fetched_dt = datetime.datetime.fromtimestamp(fetched_ts, tz=datetime.timezone.utc)
        except OSError:
            return False
        age = now - fetched_dt
        return age <= datetime.timedelta(days=_CACHE_MAX_AGE_DAYS)


    def _load_cached_data():
        """Loads cached commit counts from the CSV file.

        Returns:
            list[dict]: List of rows with keys ``week_start`` and ``commit_count``.
        """
        if not os.path.exists(_CACHE_FILE):
            return []
        rows = []
        with open(_CACHE_FILE, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    {
                        "week_start": row.get("week_start", ""),
                        "commit_count": int(row.get("commit_count", 0)),
                    }
                )
        return rows


    def _save_cached_data(rows):
        """Saves commit count rows to the CSV cache file.

        Args:
            rows: List of dicts with keys ``week_start`` and ``commit_count``.
        """
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(_CACHE_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["week_start", "commit_count"])
            writer.writeheader()
            writer.writerows(rows)


    def _fetch_commits_per_week():
        """Fetches weekly commit activity from the GitHub API.

        Returns:
            list[dict]: List of rows with keys ``week_start`` and ``commit_count``,
            or an empty list when the API is unreachable.
        """
        # The participation endpoint returns 52-week totals; use the
        # commit_activity endpoint for per-week timestamps.
        data = _gh_get("stats/commit_activity")
        if not data:
            return []
        rows = []
        for entry in data:
            week_ts = entry.get("week")
            total = entry.get("total", 0)
            if week_ts is None:
                continue
            week_start = datetime.datetime.fromtimestamp(
                week_ts, tz=datetime.timezone.utc
            ).strftime("%Y-%m-%d")
            rows.append({"week_start": week_start, "commit_count": total})
        return rows


    def _collect_data():
        """Collects commit-per-week data from the cache or the GitHub API.

        Returns:
            list[dict]: Rows with keys ``week_start`` and ``commit_count``.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        if _cache_is_recent(now):
            return _load_cached_data()
        rows = _fetch_commits_per_week()
        if rows:
            _save_cached_data(rows)
            return rows
        # Fall back to cached data when the API is unreachable.
        return _load_cached_data()


    rows = _collect_data()

    if not rows:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(
            0.5, 0.5,
            "No data available (GitHub API not reachable or no commits found).",
            ha="center", va="center", transform=ax.transAxes, fontsize=11,
        )
        ax.axis("off")
        plt.tight_layout()
    else:
        dates = [
            datetime.datetime.strptime(r["week_start"], "%Y-%m-%d") for r in rows
        ]
        counts = [r["commit_count"] for r in rows]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(dates, counts, width=6, align="edge", color="steelblue", edgecolor="white")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.set_title("Commits per Week", fontsize=13)
        ax.set_ylabel("Number of commits", fontsize=10)
        ax.set_xlabel("Week starting", fontsize=10)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        fig.autofmt_xdate()
        plt.tight_layout()
