yobx.torch.in_transformers.models.configs
=========================================

Function `yobx.torch.in_transformers.models.configs.get_cached_configuration`
lets the user retrieve configurations coming from :epkg:`transformers`.
Their purpose is to be able to write unit tests without downloading a configuration.
Currently, the module contains the following cached configuration:

.. runpython::
    :showcode:
    :process:

    import pprint
    from yobx.torch.in_transformers.models._configs import _retrieve_cached_configurations

    configs = _retrieve_cached_configurations()
    pprint.pprint(sorted(configs))

To get them, ``yobx.torch.in_transformers.models.configs.get_cached_configuration("<name>", **kwargs)``
where `kwargs` can be used to overwrite some values.
