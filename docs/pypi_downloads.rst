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

    packages = [
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
    cache_max_age_days = 14
    user_cache_dir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    cache_dir = os.path.join(user_cache_dir, "yet-another-onnx-builder", "pypi_downloads")
    cache_file = os.path.join(cache_dir, "pypi_downloads.json")

    # Determine whether the on-disk cache is still fresh.
    cache_is_recent = False
    if os.path.exists(cache_file):
        try:
            fetched_ts = os.path.getmtime(cache_file)
            fetched_dt = datetime.datetime.fromtimestamp(fetched_ts, tz=datetime.timezone.utc)
            age = datetime.datetime.now(datetime.timezone.utc) - fetched_dt
            cache_is_recent = age <= datetime.timedelta(days=cache_max_age_days)
        except OSError:
            pass

    # Load whatever is already on disk (used as fallback when the API is down).
    cached = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass

    # Fetch fresh data from the pypistats API when the cache is stale.
    data = cached if cache_is_recent else {}
    if not cache_is_recent:
        for pkg in packages:
            url = f"https://pypistats.org/api/packages/{pkg}/recent"
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "yet-another-onnx-builder-docs/1.0",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    payload = json.loads(resp.read().decode())
                    pkg_data = payload.get("data")
                if pkg_data is not None:
                    data[pkg] = {
                        "last_day": pkg_data.get("last_day", 0),
                        "last_week": pkg_data.get("last_week", 0),
                        "last_month": pkg_data.get("last_month", 0),
                    }
                elif pkg in cached:
                    data[pkg] = cached[pkg]
            except Exception:
                if pkg in cached:
                    data[pkg] = cached[pkg]
        if data:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

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
        present = [p for p in packages if p in data]
        last_month = [data[p]["last_month"] for p in present]
        last_week = [data[p]["last_week"] for p in present]

        x = np.arange(len(present))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - width / 2, last_month, width, label="Last month", color="steelblue")
        ax.bar(x + width / 2, last_week, width, label="Last week", color="darkorange")

        ax.set_xticks(x)
        ax.set_xticklabels(present, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Downloads", fontsize=10)
        ax.set_title("PyPI Downloads — ONNX Ecosystem Packages", fontsize=13)
        ax.legend(fontsize=10)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{int(v):,}")
        )
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
