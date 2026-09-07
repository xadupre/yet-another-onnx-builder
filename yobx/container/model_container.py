"""Stores native ONNX models with separately owned large tensor data."""

import os
import time
from typing import Any, Optional

import numpy
from onnx_light import onnx
from onnx_light.onnx import helper, numpy_helper
from onnx_light.onnx.external_data_helper import load_external_data_for_tensor, uses_external_data
from onnx_light.onnx.inliner import inline_local_functions

from .build_stats import BuildStats

STORAGE_TYPE = {onnx.TensorProto.FLOAT16: numpy.int16, onnx.TensorProto.BFLOAT16: numpy.int16}


def _get_all_tensors(model):
    """Iterates over native tensors in graphs, attributes and local functions."""

    def node_tensors(nodes):
        for node in nodes:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.TENSOR:
                    yield attribute.t
                elif attribute.type == onnx.AttributeProto.TENSORS:
                    yield from attribute.tensors
                elif attribute.type == onnx.AttributeProto.GRAPH:
                    yield from graph_tensors(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for graph in attribute.graphs:
                        yield from graph_tensors(graph)
                elif attribute.type == onnx.AttributeProto.SPARSE_TENSOR:
                    yield attribute.sparse_tensor.values
                    yield attribute.sparse_tensor.indices
                elif attribute.type == onnx.AttributeProto.SPARSE_TENSORS:
                    for tensor in attribute.sparse_tensors:
                        yield tensor.values
                        yield tensor.indices

    def graph_tensors(graph):
        yield from graph.initializer
        for tensor in graph.sparse_initializer:
            yield tensor.values
            yield tensor.indices
        yield from node_tensors(graph.node)

    yield from graph_tensors(model.graph)
    for function in model.functions:
        yield from node_tensors(function.node)


def _set_external_data(
    tensor: onnx.TensorProto,
    location: str,
    offset: Optional[int] = None,
    length: Optional[int] = None,
    checksum: Optional[str] = None,
    basepath: Optional[str] = None,
) -> None:
    """Sets external-data metadata without requiring an existing raw buffer."""
    tensor.ClearField("external_data")
    tensor.data_location = onnx.TensorProto.EXTERNAL
    for key, value in {
        "location": location,
        "offset": offset,
        "length": length,
        "checksum": checksum,
        "basepath": basepath,
    }.items():
        if value is not None:
            tensor.external_data.add(key=key, value=str(value))


def make_large_tensor_proto(location, tensor_name, tensor_type, shape):
    """Creates a native tensor placeholder backed by separately owned data."""
    if tensor_type == onnx.TensorProto.STRING:
        raise NotImplementedError("String tensors must remain inline, not in raw external data.")
    tensor = onnx.TensorProto()
    tensor.name = tensor_name
    tensor.data_type = tensor_type
    tensor.dims.extend(shape)
    _set_external_data(tensor, location)
    return tensor


def _get_type(elem_type: Any) -> int:
    """Returns an ONNX element type for NumPy or framework dtypes."""
    if isinstance(elem_type, int):
        return elem_type
    if elem_type is None:
        return onnx.TensorProto.UNDEFINED
    if type(elem_type).__module__.startswith(("torch", "tensorflow")):
        from ..helpers.onnx_helper import dtype_to_tensor_dtype

        return dtype_to_tensor_dtype(elem_type)
    return helper.np_dtype_to_tensor_dtype(numpy.dtype(elem_type))


class ExtendedModelContainer:
    """Owns a native model and external initializers without reference ONNX.

    External tensor locations serve as keys into :attr:`large_initializers`.
    Saving writes a copy with disk locations, leaving the in-memory model and
    its ownership keys unchanged. Loading assigns distinct keys even when
    multiple tensors share the same external file.
    """

    def __init__(self, model_proto=None, large_initializers=None):
        self.model_proto_ = None
        self.large_initializers = {}
        self._stats = BuildStats()
        self.inline = False
        if model_proto is not None:
            self.model_proto = model_proto
        if large_initializers is not None:
            self.set_large_initializers(large_initializers)

    @property
    def model_proto(self):
        """Returns the owned native model."""
        if self.model_proto_ is None:
            raise ValueError("The container has no model.")
        return self.model_proto_

    @model_proto.setter
    def model_proto(self, value):
        if not isinstance(value, onnx.ModelProto):
            raise TypeError("The container requires an onnx_light.onnx.ModelProto.")
        self.model_proto_ = value

    def set_large_initializers(self, values):
        """Registers externally owned initializers by their location keys."""
        self.large_initializers = dict(values)

    def check_large_initializers(self):
        """Verifies that every external tensor has an owned value."""
        for tensor in _get_all_tensors(self.model_proto):
            if uses_external_data(tensor):
                self.get_prop(tensor)

    def externalize_initializers(self, threshold=1024):
        """Moves numeric tensors of at least threshold elements into owned storage."""
        if not isinstance(threshold, int) or threshold < 0:
            raise ValueError("The external-data threshold must be a nonnegative integer.")
        for index, tensor in enumerate(_get_all_tensors(self.model_proto)):
            if uses_external_data(tensor):
                raise ValueError("External data must be loaded before externalizing a model.")
            if tensor.data_type == onnx.TensorProto.STRING or numpy.prod(tensor.dims) < threshold:
                continue
            value = onnx.TensorProto()
            value.ParseFromString(tensor.SerializeToString())
            key = f"#{tensor.name}_{index}"
            self.large_initializers[key] = value
            for field in (
                "raw_data",
                "float_data",
                "int32_data",
                "string_data",
                "int64_data",
                "double_data",
                "uint64_data",
            ):
                tensor.ClearField(field)
            _set_external_data(tensor, key)

    def get_prop(self, tensor):
        """Returns the location entry identifying an owned external initializer."""
        for entry in tensor.external_data:
            if str(entry.key) == "location":
                if str(entry.value) not in self.large_initializers:
                    raise RuntimeError(
                        f"Unable to find loaded external data for tensor {str(tensor.name)!r} "
                        f"at location {str(entry.value)!r}."
                    )
                return entry
        raise RuntimeError(f"No location found for tensor {str(tensor.name)!r}.")

    def get_raw_data(self, tensor):
        """Returns little-endian tensor bytes from native, NumPy or framework values."""
        begin = time.perf_counter()
        if isinstance(tensor, onnx.TensorProto):
            if tensor.raw_data:
                return bytes(tensor.raw_data)
            tensor = numpy_helper.to_array(tensor)
        elif hasattr(tensor, "detach"):
            from ..helpers.mini_onnx_builder import proto_from_array

            return bytes(proto_from_array(tensor.detach(), name="external").raw_data)
        elif hasattr(tensor, "numpy"):
            tensor = tensor.numpy()
        if not isinstance(tensor, numpy.ndarray):
            raise TypeError(f"Unsupported external initializer type {type(tensor)!r}.")
        if tensor.dtype.kind in "OUS":
            raise NotImplementedError("String and object tensors must remain inline.")
        tensor = tensor.astype(tensor.dtype.newbyteorder("<"), copy=False)
        result = tensor.tobytes()
        self._stats["time_export_tobytes"] += time.perf_counter() - begin
        return result

    def load(self, file_path, load_large_initializers=True):
        """Loads a native model and optionally owns its external tensor values."""
        self.model_proto = onnx.load(file_path, load_external_data=False)
        self.large_initializers = {}
        if load_large_initializers:
            self._load_large_initializers(file_path)
        return self

    def _load_large_initializers(self, file_path):
        """Loads external values using native readers and distinct ownership keys."""
        folder = os.path.dirname(os.path.abspath(file_path))
        for index, tensor in enumerate(_get_all_tensors(self.model_proto)):
            if not uses_external_data(tensor):
                continue
            copy = onnx.TensorProto()
            copy.ParseFromString(tensor.SerializeToString())
            load_external_data_for_tensor(copy, folder)
            copy.ClearField("external_data")
            copy.data_location = onnx.TensorProto.DEFAULT
            value = numpy_helper.to_array(copy).copy()
            key = f"#{tensor.name}"
            if key in self.large_initializers:
                key = f"{key}_{index}"
                while key in self.large_initializers:
                    key += "_"
            self.large_initializers[key] = value
            _set_external_data(tensor, key)

    def save(self, file_path, all_tensors_to_one_file=True):
        """Saves a native model and writes external weights alongside it."""
        return self._save_external(os.fspath(file_path), all_tensors_to_one_file)

    def _save_external(self, file_path, all_tensors_to_one_file):
        """Writes external buffers without modifying the owned model."""
        self.check_large_initializers()
        copy = onnx.ModelProto()
        copy.ParseFromString(self.model_proto.SerializeToString())
        folder = os.path.dirname(file_path)
        if folder and not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder {folder!r} does not exist.")
        tensors = [tensor for tensor in _get_all_tensors(copy) if uses_external_data(tensor)]
        weight_name = os.path.basename(file_path) + ".data"
        if all_tensors_to_one_file and tensors:
            with open(file_path + ".data", "wb"):
                pass
        names = set()
        offset = 0
        for tensor in tensors:
            location = str(self.get_prop(tensor).value)
            data = self.get_raw_data(self.large_initializers[location])
            begin = time.perf_counter()
            if all_tensors_to_one_file:
                _set_external_data(tensor, weight_name, offset=offset, length=len(data))
                with open(file_path + ".data", "ab") as stream:
                    stream.write(data)
                offset += len(data)
            else:
                safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in location)
                base = f"{os.path.splitext(os.path.basename(file_path))[0]}-{safe}"
                name = base + ".weight"
                index = 2
                while name in names:
                    name = f"{base}_{index}.weight"
                    index += 1
                names.add(name)
                _set_external_data(tensor, name, length=len(data))
                with open(os.path.join(folder, name), "wb") as stream:
                    stream.write(data)
            self._stats["time_export_write_tensor_bytes"] += time.perf_counter() - begin
        if self.inline:
            copy = inline_local_functions(copy)
        begin = time.perf_counter()
        with open(file_path, "wb") as stream:
            stream.write(copy.SerializeToString())
        self._stats["time_export_write_model"] += time.perf_counter() - begin
        return copy

    def get_model_with_data(self):
        """Returns a native model with all external tensors embedded recursively."""
        copy = onnx.ModelProto()
        copy.ParseFromString(self.model_proto.SerializeToString())
        for tensor in _get_all_tensors(copy):
            if uses_external_data(tensor):
                location = str(self.get_prop(tensor).value)
                tensor.raw_data = self.get_raw_data(self.large_initializers[location])
                tensor.ClearField("external_data")
                tensor.data_location = onnx.TensorProto.DEFAULT
        return copy

    def to_ir(self):
        """Rejects the removed reference-ONNX-dependent onnx-ir conversion."""
        raise NotImplementedError("Use get_model_with_data() to obtain a native ModelProto.")


ModelContainer = ExtendedModelContainer
