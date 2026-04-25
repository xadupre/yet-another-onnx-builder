.. _l-weekly-progress:

=================================
Weekly Progress Since the Start
=================================

This page summarises the development activity of
`yet-another-onnx-builder <https://github.com/xadupre/yet-another-onnx-builder>`_
on a week-by-week basis since the repository was created.
Commit counts, lines added, and lines deleted are fetched from the
GitHub REST API and displayed as bar charts.

.. note::

   This page queries the public GitHub REST API.  No authentication token is
   required for a public repository, but anonymous requests are subject to
   rate-limiting (60 requests/hour per IP).  When the API cannot be reached
   (offline build, rate-limit exceeded, …) the charts will be empty and a
   warning is printed to the console.  Retrieved statistics are cached for two
   weeks in the user cache directory.

.. runpython::
    :rst:

    import datetime
    print(
        f"*Page generated on "
        f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.*"
    )

Commits per Week
================

.. plot::
    :include-source: false

    """Fetch weekly commit counts from the GitHub stats API and plot them."""

    import datetime
    import json
    import os
    import time
    import urllib.request

    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    _OWNER = "xadupre"
    _REPO = "yet-another-onnx-builder"
    _API_BASE = f"https://api.github.com/repos/{_OWNER}/{_REPO}"
    _HEADERS = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    _USER_CACHE_DIR = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    _CACHE_DIR = os.path.join(
        _USER_CACHE_DIR, "yet-another-onnx-builder", "weekly_progress"
    )
    _CACHE_MAX_AGE_DAYS = 14


    def _cache_is_recent(path):
        """Checks whether a cache file is newer than the cache max age.

        Returns:
            bool: True when the cache entry is recent enough to be used.
        """
        if not os.path.exists(path):
            return False
        try:
            age = datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromtimestamp(
                os.path.getmtime(path), tz=datetime.timezone.utc
            )
            return age <= datetime.timedelta(days=_CACHE_MAX_AGE_DAYS)
        except OSError:
            return False


    def _gh_get_stats(endpoint):
        """Fetches a GitHub stats endpoint, retrying once after 202 responses.

        Returns:
            list | None: Parsed JSON list on success, None on failure.
        """
        url = f"{_API_BASE}/stats/{endpoint}"
        req = urllib.request.Request(url, headers=_HEADERS)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    if resp.status == 200:
                        return json.loads(resp.read().decode())
            except (urllib.error.URLError, urllib.error.HTTPError):
                pass
            # GitHub may return 202 (computing) — wait and retry
            time.sleep(2)
        return None


    def _load_or_fetch(cache_name, endpoint):
        """Returns stats from cache if fresh, otherwise fetches from GitHub API.

        Returns:
            list | None: Stats list, or None when unavailable.
        """
        os.makedirs(_CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(_CACHE_DIR, cache_name)
        if _cache_is_recent(cache_path):
            with open(cache_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        data = _gh_get_stats(endpoint)
        if data:
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
        return data


    # ── Commit activity: list of {week (unix ts), days, total} ──────────────────
    commit_data = _load_or_fetch("commit_activity.json", "commit_activity")

    if not commit_data:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(
            0.5,
            0.5,
            "No data available (GitHub API not reachable or still computing).",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.axis("off")
        plt.tight_layout()
    else:
        weeks = []
        counts = []
        for entry in commit_data:
            ts = entry.get("week", 0)
            total = entry.get("total", 0)
            if total == 0:
                continue
            weeks.append(datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc))
            counts.append(total)

        if not weeks:
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.text(
                0.5,
                0.5,
                "No commit data found for the past 52 weeks.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
            )
            ax.axis("off")
            plt.tight_layout()
        else:
            fig, ax = plt.subplots(figsize=(12, 4))
            bar_width = datetime.timedelta(days=5)
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            ax.bar(weeks, counts, width=bar_width, color=colors[0], alpha=0.8, label="commits / week")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%Y"))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
            ax.tick_params(axis="x", labelsize=8)
            ax.set_ylabel("Commits", fontsize=10)
            ax.set_title("Commits per Week", fontsize=13)
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)
            ax.legend(fontsize=9)
            fig.autofmt_xdate()
            plt.tight_layout()

Lines of Code Added and Deleted per Week
=========================================

.. plot::
    :include-source: false

    """Fetch weekly code-frequency stats from the GitHub stats API and plot them."""

    import datetime
    import json
    import os
    import time
    import urllib.request

    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    _OWNER = "xadupre"
    _REPO = "yet-another-onnx-builder"
    _API_BASE = f"https://api.github.com/repos/{_OWNER}/{_REPO}"
    _HEADERS = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    _USER_CACHE_DIR = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    _CACHE_DIR = os.path.join(
        _USER_CACHE_DIR, "yet-another-onnx-builder", "weekly_progress"
    )
    _CACHE_MAX_AGE_DAYS = 14


    def _cache_is_recent(path):
        """Checks whether a cache file is newer than the cache max age.

        Returns:
            bool: True when the cache entry is recent enough to be used.
        """
        if not os.path.exists(path):
            return False
        try:
            age = datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromtimestamp(
                os.path.getmtime(path), tz=datetime.timezone.utc
            )
            return age <= datetime.timedelta(days=_CACHE_MAX_AGE_DAYS)
        except OSError:
            return False


    def _gh_get_stats(endpoint):
        """Fetches a GitHub stats endpoint, retrying once after 202 responses.

        Returns:
            list | None: Parsed JSON list on success, None on failure.
        """
        url = f"{_API_BASE}/stats/{endpoint}"
        req = urllib.request.Request(url, headers=_HEADERS)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    if resp.status == 200:
                        return json.loads(resp.read().decode())
            except (urllib.error.URLError, urllib.error.HTTPError):
                pass
            time.sleep(2)
        return None


    def _load_or_fetch(cache_name, endpoint):
        """Returns stats from cache if fresh, otherwise fetches from GitHub API.

        Returns:
            list | None: Stats list, or None when unavailable.
        """
        os.makedirs(_CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(_CACHE_DIR, cache_name)
        if _cache_is_recent(cache_path):
            with open(cache_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        data = _gh_get_stats(endpoint)
        if data:
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
        return data


    # ── Code frequency: list of [unix_ts, additions, deletions] ─────────────────
    freq_data = _load_or_fetch("code_frequency.json", "code_frequency")

    if not freq_data:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(
            0.5,
            0.5,
            "No data available (GitHub API not reachable or still computing).",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.axis("off")
        plt.tight_layout()
    else:
        weeks = []
        additions = []
        deletions = []
        for entry in freq_data:
            ts, adds, dels = entry[0], entry[1], entry[2]
            if adds == 0 and dels == 0:
                continue
            weeks.append(datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc))
            additions.append(adds)
            deletions.append(abs(dels))

        if not weeks:
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.text(
                0.5,
                0.5,
                "No code-frequency data found.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
            )
            ax.axis("off")
            plt.tight_layout()
        else:
            fig, ax = plt.subplots(figsize=(12, 4))
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

            ax.bar(
                weeks,
                additions,
                width=datetime.timedelta(days=3),
                color=colors[0],
                alpha=0.8,
                label="Lines added",
            )
            ax.bar(
                weeks,
                [-d for d in deletions],
                width=datetime.timedelta(days=3),
                color=colors[3],
                alpha=0.8,
                label="Lines deleted",
            )

            ax.axhline(0, color="black", linewidth=0.8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%Y"))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
            ax.tick_params(axis="x", labelsize=8)
            ax.set_ylabel("Lines", fontsize=10)
            ax.set_title("Lines of Code Added / Deleted per Week", fontsize=13)
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)
            ax.legend(fontsize=9)
            fig.autofmt_xdate()
            plt.tight_layout()

Cumulative Commit Count
=======================

.. plot::
    :include-source: false

    """Plot the cumulative commit count over time."""

    import datetime
    import json
    import os
    import time
    import urllib.request

    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np

    _OWNER = "xadupre"
    _REPO = "yet-another-onnx-builder"
    _API_BASE = f"https://api.github.com/repos/{_OWNER}/{_REPO}"
    _HEADERS = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    _USER_CACHE_DIR = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    _CACHE_DIR = os.path.join(
        _USER_CACHE_DIR, "yet-another-onnx-builder", "weekly_progress"
    )
    _CACHE_MAX_AGE_DAYS = 14


    def _cache_is_recent(path):
        """Checks whether a cache file is newer than the cache max age.

        Returns:
            bool: True when the cache entry is recent enough to be used.
        """
        if not os.path.exists(path):
            return False
        try:
            age = datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromtimestamp(
                os.path.getmtime(path), tz=datetime.timezone.utc
            )
            return age <= datetime.timedelta(days=_CACHE_MAX_AGE_DAYS)
        except OSError:
            return False


    def _gh_get_stats(endpoint):
        """Fetches a GitHub stats endpoint, retrying once after 202 responses.

        Returns:
            list | None: Parsed JSON list on success, None on failure.
        """
        url = f"{_API_BASE}/stats/{endpoint}"
        req = urllib.request.Request(url, headers=_HEADERS)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    if resp.status == 200:
                        return json.loads(resp.read().decode())
            except (urllib.error.URLError, urllib.error.HTTPError):
                pass
            time.sleep(2)
        return None


    def _load_or_fetch(cache_name, endpoint):
        """Returns stats from cache if fresh, otherwise fetches from GitHub API.

        Returns:
            list | None: Stats list, or None when unavailable.
        """
        os.makedirs(_CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(_CACHE_DIR, cache_name)
        if _cache_is_recent(cache_path):
            with open(cache_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        data = _gh_get_stats(endpoint)
        if data:
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
        return data


    commit_data = _load_or_fetch("commit_activity.json", "commit_activity")

    if not commit_data:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(
            0.5,
            0.5,
            "No data available (GitHub API not reachable or still computing).",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.axis("off")
        plt.tight_layout()
    else:
        all_weeks = []
        all_counts = []
        for entry in commit_data:
            ts = entry.get("week", 0)
            total = entry.get("total", 0)
            all_weeks.append(datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc))
            all_counts.append(total)

        cumulative = list(np.cumsum(all_counts))

        # Keep only up to current date
        now = datetime.datetime.now(datetime.timezone.utc)
        pairs = [(w, c) for w, c in zip(all_weeks, cumulative) if w <= now]

        if not pairs or pairs[-1][1] == 0:
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.text(
                0.5,
                0.5,
                "No commit data found.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
            )
            ax.axis("off")
            plt.tight_layout()
        else:
            weeks = [p[0] for p in pairs]
            cumulative = [p[1] for p in pairs]
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.fill_between(weeks, cumulative, color=colors[1], alpha=0.3)
            ax.plot(weeks, cumulative, "-", color=colors[1], linewidth=1.8, label="total commits")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%Y"))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
            ax.tick_params(axis="x", labelsize=8)
            ax.set_ylabel("Cumulative commits", fontsize=10)
            ax.set_title("Cumulative Commit Count", fontsize=13)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(fontsize=9)
            fig.autofmt_xdate()
            plt.tight_layout()
