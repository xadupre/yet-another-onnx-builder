"""Measures peak resident memory while registering large native initializers.

Runs each backend in a fresh process, for example:
PYTHONPATH=. python unittests/builder/onnxlight/benchmark_initializer_memory.py numpy
PYTHONPATH=. python unittests/builder/onnxlight/benchmark_initializer_memory.py torch
"""

import argparse
import resource

import numpy

from yobx.builder.onnxlight import OnnxLightGraphBuilder


def main():
    """Measures the peak RSS increase above the allocated source weight."""
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("numpy", "torch"))
    parser.add_argument("--megabytes", type=int, default=256)
    args = parser.parse_args()
    if args.backend == "torch":
        import torch

        weight = torch.ones(args.megabytes * 1024 * 1024 // 4, dtype=torch.float32)
        pointer = weight.data_ptr()
    else:
        weight = numpy.ones(args.megabytes * 1024 * 1024 // 4, dtype=numpy.float32)
        pointer = weight.ctypes.data
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    builder = OnnxLightGraphBuilder(18)
    builder.make_initializer("weight", weight)
    registered = numpy.from_dlpack(builder.initializers_dict["weight"]).ctypes.data
    assert registered == pointer
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    delta_mib = (peak - baseline) / 1024
    print(f"{args.backend}: peak RSS increase during registration: {delta_mib:.1f} MiB")


if __name__ == "__main__":
    main()
