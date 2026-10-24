# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.2.0] - Unreleased

### New Features

- Uses `onnx-light==0.2.0` for native graph building, pattern optimization,
  and symbolic shape inference. The reference `onnx` package is no longer
  a required dependency.

### Improvements

- Requires Python 3.12 or newer and NumPy 2.2 or newer.
- Updates optional dependencies to scikit-learn 1.9.1 or newer, pandas 3.1.0
  or newer, and polars 2.0.0 or newer.

### Documentation & CI

- Bumped the release version to `0.2.0`.
- Builds a universal Python wheel and source distribution on tag pushes or
  manual runs, and creates or updates the corresponding GitHub Release assets.
