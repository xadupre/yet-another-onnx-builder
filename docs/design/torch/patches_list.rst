.. _patches-torch:

============
Patches List
============

torch
=====

.. runpython::
    :rst:
    :showcode:
    :process:

    import textwrap
    from yobx.torch import apply_patches_for_model
    from yobx.torch.in_torch.patches import get_patches

    with apply_patches_for_model(patch_torch=True):
        link = []
        rows = []
        for i, patch in enumerate(get_patches()):
            name = f"patch-torch-{i+1}"
            link.append(f"* :ref:`{name}`")
            rows.extend([
                "",
                f".. _{name}:",
                "",
                patch.patch.__qualname__,
                "-" * len(patch.patch.__qualname__),
                "",
                "::",
                "",
                textwrap.indent(patch.format_diff(), "    "),
                "",
            ])
        print("\n".join([*link, "", *rows]))
    print()

transformers
============

.. runpython::
    :rst:
    :showcode:
    :process:

    import textwrap
    from yobx.torch import apply_patches_for_model

    with apply_patches_for_model(patch_transformers=True) as details:
        link = []
        rows = []
        for i, patch in enumerate(details):
            name = f"patch-transformers-{i+1}"
            link.append(f"* :ref:`{name}`")
            rows.extend([
                "",
                f".. _{name}:",
                "",
                patch.patch.__qualname__,
                "-" * len(patch.patch.__qualname__),
                "",
                "::",
                "",
                textwrap.indent(patch.format_diff(), "    "),
                "",
            ])
        print("\n".join([*link, "", *rows]))
