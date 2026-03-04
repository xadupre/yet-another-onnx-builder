from __future__ import annotations
from typing import Optional
from onnx import AttributeProto


def make_ref_attribute(
    key: str, attr_type: int, ref_attr_name: Optional[str] = None
) -> AttributeProto:
    """
    Creates an attribute.

    :param key: attribute name
    :param attr_type: attribute type
    :param ref_attr_name: if not None, link this attribute
        to a function attribute
    :return: attribute
    """
    att = AttributeProto()
    att.name = key
    att.type = attr_type
    if ref_attr_name is not None:
        att.ref_attr_name = ref_attr_name
    return att
