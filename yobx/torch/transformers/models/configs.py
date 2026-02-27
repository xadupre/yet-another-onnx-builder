import os
import functools
import pprint
from typing import Dict, Optional
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


def get_cached_configuration(
    name: str, exc: bool = False, **kwargs
) -> Optional[transformers.PretrainedConfig]:
    """
    Returns cached configuration to avoid having to many accesses to internet.
    It returns None if not Cache. The list of cached models follows.
    If *exc* is True or if environment variable ``NOHTTP`` is defined,
    the function raises an exception if *name* is not found.

    .. runpython::

        import pprint
        from yobx.transformers.configs import _retrieve_cached_configurations

        configs = _retrieve_cached_configurations()
        pprint.pprint(sorted(configs))
    """
    cached = _retrieve_cached_configurations()
    assert cached, "no cached configuration, which is weird"
    if name in cached:
        conf = cached[name]()
        return conf
    assert not exc and not os.environ.get("NOHTTP", ""), (
        f"Unable to find {name!r} (exc={exc}, "
        f"NOHTTP={os.environ.get('NOHTTP', '')!r}) "
        f"in {pprint.pformat(sorted(cached))}"
    )
    return None
