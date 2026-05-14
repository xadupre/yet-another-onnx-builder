import os
import pprint
from typing import Optional
import transformers


def get_cached_configuration(
    name: str, exc: bool = False, **kwargs
) -> Optional[transformers.PretrainedConfig]:
    """
    Returns cached configuration to avoid having to many accesses to internet.
    It returns None if not Cache. The list of cached models follows.
    If *exc* is True or if environment variable ``NOHTTP`` is defined,
    the function raises an exception if *name* is not found.

    .. runpython::
        :showcode:
        :process:

        import pprint
        from yobx.torch.in_transformers.models.configs import _retrieve_cached_configurations

        configs = _retrieve_cached_configurations()
        pprint.pprint(sorted(configs))
    """
    from ._configs import _retrieve_cached_configurations

    cached = _retrieve_cached_configurations()
    assert cached, "no cached configuration, which is weird"
    if name in cached:
        conf = cached[name]()  # type: ignore[operator]
        return conf
    assert not exc and not os.environ.get("NOHTTP", ""), (
        f"Unable to find {name!r} (exc={exc}, "
        f"NOHTTP={os.environ.get('NOHTTP', '')!r}) "
        f"in {pprint.pformat(sorted(cached))}"
    )
    return None
