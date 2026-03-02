from ..helpers import string_sig


class FunctionOptions:
    """
    Defines how local functions must behave.

    :param name: function name
    :param domain: function domain
    :param export_as_function: export the onnx as functions or keep local function
    :param external_threshold: whether or not keep initializer as input for the function
        or move them as constant of the function
    :param move_initializer_to_constant: move initializers as constant first before
        creating the function proto, that depends on the size defined by
        external_threshold
    :param return_initializer: return the remaining initializer and add them as input
        to the function
    :param inline: inline functions
    :param rename_allowed: allow to rename the function if a duplicate is detected
    :param merge_allowed: allow to merge a function in case the same code is detected
    """

    empty_names = (None, "", "*")

    def __init__(
        self,
        export_as_function: bool = False,
        name: str = "",
        domain: str = "",
        external_threshold: int = 2**25,
        move_initializer_to_constant: bool = False,
        return_initializer: bool = False,
        inline: bool = False,
        merge_allowed: bool = False,
        rename_allowed: bool = False,
    ):
        if name:
            export_as_function = True
        assert not export_as_function or name, (
            f"to be removed, helps tracking bugs, name={name!r}, domain={domain!r}, "
            f"export_as_function={export_as_function!r}"
        )
        assert export_as_function or (not name and not domain), (
            f"to be removed help track bugs, name={name!r}, domain={domain!r}, "
            f"export_as_function={export_as_function!r}"
        )
        assert isinstance(
            external_threshold, int
        ), f"Unexpected type {type(external_threshold)} for external_threshold"
        self.export_as_function = export_as_function
        self.external_threshold = external_threshold
        self.move_initializer_to_constant = move_initializer_to_constant
        self.name = name
        self.domain = domain
        self.return_initializer = return_initializer
        self.inline = inline
        self.rename_allowed = rename_allowed
        self.merge_allowed = merge_allowed

    def __repr__(self) -> str:
        "usual"
        return string_sig(self)
