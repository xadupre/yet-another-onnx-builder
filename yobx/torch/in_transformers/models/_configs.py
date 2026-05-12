import functools
from typing import Dict
import transformers
from . import _cached_configs


@functools.cache
def _retrieve_cached_configurations() -> Dict[str, transformers.PretrainedConfig]:
    res = {}
    for k, v in _cached_configs.__dict__.items():
        if k.startswith("_ccached_"):
            doc = v.__doc__
            res[doc] = v
    return res
