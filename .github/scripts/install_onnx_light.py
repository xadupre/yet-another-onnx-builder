"""Installs a matching full onnx-light release wheel without a source build."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
import urllib.request
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename
from pip._vendor.packaging.version import Version


def read_requirement(project_path=None):
    """Returns the onnx-light requirement from the project metadata."""
    if project_path is None:
        project_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    project = tomllib.loads(project_path.read_text(encoding="utf-8"))
    requirements = [
        requirement
        for dependency in project["project"]["dependencies"]
        if canonicalize_name((requirement := Requirement(dependency)).name) == "onnx-light"
    ]
    if len(requirements) != 1:
        raise ValueError("Expected exactly one onnx-light dependency in pyproject.toml.")
    return requirements[0]


def release_version(requirement):
    """Returns the lowest explicit release satisfying the project requirement."""
    versions = [
        Version(specifier.version)
        for specifier in requirement.specifier
        if specifier.operator in {">=", "==", "~="} and "*" not in specifier.version
    ]
    candidates = [version for version in versions if version in requirement.specifier]
    if not candidates:
        raise ValueError("The onnx-light requirement must specify an inclusive minimum release.")
    return str(min(candidates))


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
    parser.add_argument("--version", help="Overrides the release selected from pyproject.toml.")
    parser.add_argument("--print-url", action="store_true")
    parser.add_argument("--check-installed", action="store_true")
    args = parser.parse_args()
    requirement = read_requirement()
    if args.check_installed:
        installed = importlib.metadata.version("onnx-light")
        if installed not in requirement.specifier:
            raise RuntimeError(
                f"Installed onnx-light {installed} does not satisfy {requirement}."
            )
        return
    version = args.version or release_version(requirement)
    headers = {"Accept": "application/vnd.github+json"}
    if os.environ.get("GH_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GH_TOKEN']}"
    request = urllib.request.Request(
        f"https://api.github.com/repos/xadupre/onnx-light/releases/tags/{version}",
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
