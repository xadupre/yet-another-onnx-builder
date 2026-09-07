"""Installs a matching full onnx-light release wheel without a source build."""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import parse_wheel_filename


def select_wheel(assets, supported_tags):
    """Returns the highest-priority compatible full wheel URL."""
    ranks = {tag: rank for rank, tag in enumerate(supported_tags)}
    candidates = []
    for asset in assets:
        filename = asset["name"]
        if not filename.endswith(".whl") or not filename.startswith("onnx_light-"):
            continue
        _, _, build, tags = parse_wheel_filename(filename)
        if build:
            continue
        priorities = [ranks[tag] for tag in tags if tag in ranks]
        if priorities:
            candidates.append((min(priorities), asset["browser_download_url"]))
    if not candidates:
        raise RuntimeError("This onnx-light release has no full wheel for this interpreter.")
    return min(candidates)[1]


def main():
    """Selects and installs a published binary wheel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="0.1.24")
    parser.add_argument("--print-url", action="store_true")
    args = parser.parse_args()
    headers = {"Accept": "application/vnd.github+json"}
    if os.environ.get("GH_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GH_TOKEN']}"
    request = urllib.request.Request(
        f"https://api.github.com/repos/xadupre/onnx-light/releases/tags/{args.version}",
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        release = json.load(response)
    url = select_wheel(release["assets"], list(sys_tags()))
    if args.print_url:
        print(url)
    else:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--only-binary=:all:", url], check=True
        )


if __name__ == "__main__":
    main()
