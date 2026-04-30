.. _l-pypi-downloads:

=====================================
PyPI Download Statistics (ONNX Stack)
=====================================

This page shows the monthly download counts for ten key packages in the ONNX
ecosystem, fetched from the `pypistats.org <https://pypistats.org>`_ public API.

The packages tracked are:

* **onnx** — the ONNX standard library
* **onnxruntime** — ONNX Runtime inference engine
* **skl2onnx** — scikit-learn → ONNX converter
* **tf2onnx** — TensorFlow → ONNX converter
* **onnxmltools** — ML frameworks → ONNX converter
* **jax2onnx** — JAX → ONNX converter
* **onnx-diagnostic** — ONNX model diagnostic tools
* **onnxruntime-genai** — ONNX Runtime generative AI extensions
* **onnxscript** — ONNX Script authoring library
* **onnx-ir** — ONNX IR Python library

.. note::

   This page queries the `pypistats.org <https://pypistats.org/api/>`_ public
   REST API.  No authentication token is required.  When the API cannot be
   reached (offline build, service unavailable, …) the chart will display
   whatever data was last cached.  Retrieved data is cached as a JSON file and
   refreshed when the cache is older than two weeks.

.. runpython::
    :rst:

    import datetime
    import os

    print(
        f"*Page generated on {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.*"
    )


.. plot::
    :include-source: false

    """Fetch PyPI download statistics for key ONNX packages and display as a bar chart."""

    import datetime
    import json
    import os
    import urllib.request

    import matplotlib.pyplot as plt
    import numpy as np

    _PACKAGES = [
        "onnx",
        "onnxruntime",
        "skl2onnx",
        "tf2onnx",
        "onnxmltools",
        "jax2onnx",
        "onnx-diagnostic",
        "onnxruntime-genai",
        "onnxscript",
        "onnx-ir",
    ]
    _CACHE_MAX_AGE_DAYS = 14
    _USER_CACHE_DIR = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    _CACHE_DIR = os.path.join(
        _USER_CACHE_DIR, "yet-another-onnx-builder", "pypi_downloads"
    )
    _CACHE_FILE = os.path.join(_CACHE_DIR, "pypi_downloads.json")


    def _fetch_recent_downloads(package):
        """Fetches recent download counts for *package* from pypistats.org.

        Returns:
            dict | None: Parsed JSON ``data`` object, or ``None`` on any error.
        """
        url = f"https://pypistats.org/api/packages/{package}/recent"
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "yet-another-onnx-builder-docs/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read().decode())
                return payload.get("data")
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
        """Loads cached download stats from the JSON cache file.

        Returns:
            dict[str, dict]: Mapping ``package_name -> {last_day, last_week, last_month}``.
        """
        if not os.path.exists(_CACHE_FILE):
            return {}
        try:
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}


    def _save_cached_data(data):
        """Saves download stats to the JSON cache file.

        Args:
            data: Mapping ``package_name -> {last_day, last_week, last_month}``.
        """
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


    def _collect_data():
        """Collects PyPI download stats from the cache or the pypistats API.

        Returns:
            dict[str, dict]: Mapping ``package_name -> {last_day, last_week, last_month}``.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        if _cache_is_recent(now):
            return _load_cached_data()
        cached = _load_cached_data()
        fetched = {}
        for pkg in _PACKAGES:
            data = _fetch_recent_downloads(pkg)
            if data is not None:
                fetched[pkg] = {
                    "last_day": data.get("last_day", 0),
                    "last_week": data.get("last_week", 0),
                    "last_month": data.get("last_month", 0),
                }
            elif pkg in cached:
                fetched[pkg] = cached[pkg]
        if fetched:
            _save_cached_data(fetched)
        return fetched


    data = _collect_data()

    if not data:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(
            0.5, 0.5,
            "No data available (pypistats.org not reachable or no cached data found).",
            ha="center", va="center", transform=ax.transAxes, fontsize=11,
        )
        ax.axis("off")
        plt.tight_layout()
    else:
        packages = [p for p in _PACKAGES if p in data]
        last_month = [data[p]["last_month"] for p in packages]
        last_week = [data[p]["last_week"] for p in packages]

        x = np.arange(len(packages))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 6))
        bars_month = ax.bar(x - width / 2, last_month, width, label="Last month", color="steelblue")
        bars_week = ax.bar(x + width / 2, last_week, width, label="Last week", color="darkorange")

        ax.set_xticks(x)
        ax.set_xticklabels(packages, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Downloads", fontsize=10)
        ax.set_title("PyPI Downloads — ONNX Ecosystem Packages", fontsize=13)
        ax.legend(fontsize=10)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{int(v):,}")
        )
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
