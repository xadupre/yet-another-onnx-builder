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
   rate-limiting (60 requests/hour per IP).  Retrieved data is cached as a CSV
   file and refreshed when the cache is older than two weeks.  If an API
   request fails, cached data is displayed when available.  The chart also
   indicates when the data was last fetched.

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


    def _cache_last_fetched():
        """Returns the cache fetch timestamp.

        Returns:
            datetime.datetime | None: Cache file modification timestamp in UTC.
        """
        if not os.path.exists(_CACHE_FILE):
            return None
        try:
            fetched_ts = os.path.getmtime(_CACHE_FILE)
        except OSError:
            return None
        return datetime.datetime.fromtimestamp(fetched_ts, tz=datetime.timezone.utc)


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
                try:
                    commit_count = int(row.get("commit_count", 0))
                except (TypeError, ValueError):
                    continue
                rows.append(
                    {
                        "week_start": row.get("week_start", ""),
                        "commit_count": commit_count,
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
        if not data or not isinstance(data, list):
            return []
        rows = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
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
            tuple: ``(rows, source, fetched_at)`` with rows keys
            ``week_start`` and ``commit_count``.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        cached_rows = _load_cached_data()
        cached_fetched_at = _cache_last_fetched()
        if _cache_is_recent(now):
            return cached_rows, "cache", cached_fetched_at
        rows = _fetch_commits_per_week()
        if rows:
            _save_cached_data(rows)
            return rows, "api", _cache_last_fetched()
        # Fall back to cached data when the API is unreachable.
        return cached_rows, "cache_fallback", cached_fetched_at


    rows, source, fetched_at = _collect_data()

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
        if fetched_at is not None:
            source_name = "GitHub API" if source == "api" else "cache"
            ax.text(
                0.99, 1.02,
                f"Source: {source_name} | Last fetched: {fetched_at.strftime('%Y-%m-%d %H:%M UTC')}",
                ha="right", va="bottom", transform=ax.transAxes, fontsize=8, color="dimgray",
            )
        fig.autofmt_xdate()
        plt.tight_layout()
