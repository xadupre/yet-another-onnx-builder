import os
import sys
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import onnx
from onnx.model_container import ModelContainer
from onnx.external_data_helper import _get_all_tensors, uses_external_data
from onnx.inliner import inline_local_functions
from ..helpers.mini_onnx_builder import proto_from_array
from ..helpers.onnx_helper import dtype_to_tensor_dtype, tensor_dtype_to_np_dtype
from .build_stats import BuildStats

STORAGE_TYPE = {onnx.TensorProto.FLOAT16: np.int16, onnx.TensorProto.BFLOAT16: np.int16}


def _set_external_data(
    tensor: onnx.TensorProto,
    location: str,
    offset: Optional[int] = None,
    length: Optional[int] = None,
    checksum: Optional[str] = None,
    basepath: Optional[str] = None,
) -> None:
    del tensor.external_data[:]
    tensor.data_location = onnx.TensorProto.EXTERNAL
    for k, v in {
        "location": location,
        "offset": offset,
        "length": length,
        "checksum": checksum,
        "basepath": basepath,
    }.items():
        if v is not None:
            entry = tensor.external_data.add()
            entry.key = k
            entry.value = str(v)


def _get_type(elem_type: Any) -> int:
    if isinstance(elem_type, int):
        return elem_type
    if elem_type is None:
        return onnx.TensorProto.UNDEFINED
    return dtype_to_tensor_dtype(elem_type)


class ExtendedModelContainer(ModelContainer):
    """
    Overwrites :class:`onnx.model_container.ModelContainer`
    to support torch tensors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stats = BuildStats()
        self.inline = False

    def save(self, file_path: str, all_tensors_to_one_file: bool = True) -> onnx.ModelProto:
        """
        Saves the large model.
        The function returns a ModelProto,
        the current one if the model did not need any modification,
        a modified copy of it if it required changes such as giving file names
        to every external tensor.

        :param file_path: model file
        :param all_tensors_to_one_file: saves all large tensors in one file or
                one file per large tensor
        :return: the saved ModelProto
        """
        return self._save_external(file_path, all_tensors_to_one_file=all_tensors_to_one_file)

    def load(
        self, file_path: str, load_large_initializers: bool = True
    ) -> "ExtendedModelContainer":
        """
        Loads the large model.

        :param file_path: model file
        :param load_large_initializers: loads the large initializers,
                if not done, the model is incomplete but it can be used to
                look into the model without executing it and method
                ``onnx.model_container.ModelContainer._load_large_initializers``
                can be used to load them later
        :return: self
        """
        self.model_proto_ = onnx.load(file_path, load_external_data=False)
        if load_large_initializers:
            self._load_large_initializers(file_path)
        return self

    def get_raw_data(
        self, np_tensor: Union[np.ndarray, "torch.Tensor", "tensorflow.Tensor"]  # noqa: F821
    ) -> bytes:
        """Returns the raw data of a specific tensor whatever it is."""
        if sys.byteorder == "big":
            # Convert endian from little to big
            begin = time.perf_counter()
            tensor_bytes = np_tensor.byteswap().tobytes()
            self._stats["time_export_byteswap_tobytes"] += time.perf_counter() - begin
        elif isinstance(np_tensor, np.ndarray):
            begin = time.perf_counter()
            tensor_bytes = np_tensor.tobytes()
            self._stats["time_export_tobytes"] += time.perf_counter() - begin
        elif isinstance(np_tensor, onnx.TensorProto):
            tensor_bytes = np_tensor.raw_data
            assert len(tensor_bytes) > 0, f"One tensor is null, np_tensor={np_tensor}."
        elif hasattr(np_tensor, "ref"):
            # It should be a tensorflow tensor.
            begin = time.perf_counter()
            tensor_bytes = np_tensor.numpy().tobytes()
            self._stats["time_export_tobytes"] += time.perf_counter() - begin
        elif hasattr(np_tensor, "detach"):
            # It should be a torch.Tensor.
            import torch

            if isinstance(np_tensor, torch.nn.Parameter):
                pt = np_tensor.data
            elif isinstance(np_tensor, torch.Tensor):
                pt = np_tensor
            else:
                raise NotImplementedError(
                    f"Handling of type {type(np_tensor)} as large initializer "
                    f"is not implemented yet."
                )

            begin = time.perf_counter()
            proto = proto_from_array(pt, name="dummy")
            self._stats["time_export_proto_from_array"] += time.perf_counter() - begin
            tensor_bytes = proto.raw_data
            assert pt.dtype != torch.float32 or len(tensor_bytes) == np.prod(pt.shape) * 4, (
                f"Unexpected size mismatch, buffer size is {len(tensor_bytes)}, "
                f"but tensor size={np.prod(pt.shape) * 4}, "
                f"shape={pt.shape}, dtype={pt.dtype}"
            )
        else:
            raise TypeError(f"Unsupported type {type(np_tensor)}.")
        return tensor_bytes

    def get_prop(self, tensor: onnx.TensorProto) -> onnx.StringStringEntryProto:
        """Returns the location for a tensor stored in external data."""
        prop: Optional[onnx.StringStringEntryProto] = None
        for ext in tensor.external_data:  # type: ignore[assignment]
            if ext.key == "location":  # type: ignore[attr-defined]
                prop = ext  # type: ignore[assignment]
        if prop is None:
            raise RuntimeError(f"No location found for tensor name {tensor.name!r}.")
        if prop.value not in self.large_initializers:
            raise RuntimeError(
                f"Unable to find large tensor named {tensor.name!r} "
                f"with location {prop.value!r} in "
                f"{sorted(self.large_initializers)}."
            )
        return prop

    def _save_external(self, file_path: str, all_tensors_to_one_file: bool) -> onnx.ModelProto:
        """
        Saves the large model into a main onnx file and one file
        per tensor. Follows the same format as :func:`write_external_data_tensors
        <onnx.external_data_helper.write_external_data_tensors>`.
        The main model needs to be modified to update the file location,
        the function returns this modified copy.

        Arguments:
            file_path: model file
            all_tensors_to_one_file: all tensors in one file

        Returns:
            modified main model proto
        """

        def _clean_name(prefix: str, name: str, unique_names: dict[str, int]) -> str:
            """Removes special characters from a name and ensures uniqueness
            by appending a numeric suffix when duplicates are encountered."""
            if prefix:
                name = f"{prefix}-{name}"
            for c in ":/\\;,!#":
                name = name.replace(c, "")
            base_name = name
            if name in unique_names:
                i = unique_names[name] + 1
                unique_names[name] = i
                return f"{base_name}_{i}"
            unique_names[name] = 1
            return name

        unique_names: dict[str, int] = {}
        folder = os.path.dirname(file_path)
        if folder and not os.path.exists(folder):
            raise FileNotFoundError(f"Folder {folder!r} does not exist.")
        proto = self.model_proto.SerializeToString()
        copy = onnx.ModelProto()
        copy.ParseFromString(proto)
        prefix = os.path.splitext(os.path.split(file_path)[-1])[0]

        if all_tensors_to_one_file:
            file_weight = f"{os.path.split(file_path)[1]}.data"
            full_file_weight = f"{file_path}.data"
            offset = 0
            with open(full_file_weight, "wb") as f:
                pass

        for tensor in _get_all_tensors(copy):
            if not uses_external_data(tensor):
                continue
            prop = self.get_prop(tensor)
            np_tensor = self.large_initializers[prop.value]
            tensor_bytes = self.get_raw_data(np_tensor)

            begin = time.perf_counter()
            if all_tensors_to_one_file:
                _set_external_data(
                    tensor, location=file_weight, offset=offset, length=len(tensor_bytes)
                )
                offset += len(tensor_bytes)
                with open(full_file_weight, "ab") as f:
                    f.write(tensor_bytes)
            else:
                name = f"{_clean_name(prefix, prop.value, unique_names)}.weight"
                _set_external_data(tensor, location=name)
                full_name = os.path.join(folder, name)
                prop.value = name
                with open(full_name, "wb") as f:
                    f.write(tensor_bytes)
            self._stats["time_export_write_tensor_bytes"] += time.perf_counter() - begin

        if self.inline:
            begin = time.perf_counter()
            copy = inline_local_functions(copy)
            self._stats["time_export_inline_model"] += time.perf_counter() - begin

        begin = time.perf_counter()
        with open(file_path, "wb") as f:
            f.write(copy.SerializeToString())

        self._stats["time_export_write_model"] += time.perf_counter() - begin
        return copy

    def get_model_with_data(self):
        """
        Returns a copy of the model with data included in it.
        """
        proto = self.model_proto.SerializeToString()
        copy = onnx.ModelProto()
        copy.ParseFromString(proto)
        assert (
            not copy.graph.sparse_initializer
        ), "Not implemented when the model contains sparse initializers."
        new_inits = []
        for tensor in copy.graph.initializer:
            prop = self.get_prop(tensor)
            np_tensor = self.large_initializers[prop.value]
            tensor_bytes = self.get_raw_data(np_tensor)
            new_tensor = onnx.TensorProto()
            new_tensor.name = tensor.name
            new_tensor.raw_data = tensor_bytes
            new_tensor.data_type = tensor.data_type
            new_tensor.dims = tensor.dims
            new_tensor.doc_string = tensor.doc_string
            new_tensor.metadata_props.extend(tensor.metadata_props)
            new_inits.append(new_tensor)
        del copy.graph.initializer[:]
        copy.graph.initializer.extend(new_inits)
        return copy

    def _deserialize_graph(
        self,
        proto: onnx.GraphProto,
        scoped_values: List[Dict[str, "onnx_ir.Value"]],  # type: ignore # noqa: F821
    ) -> "onnx_ir.Graph":  # type: ignore # noqa: F821
        """See :epkg:`onnxscript`."""
        import onnx.numpy_helper as onh
        import onnx_ir as oir
        import onnx_ir.serde as oirs

        quantization_annotations = {
            annotation.tensor_name for annotation in proto.quantization_annotation
        }

        initializer_tensors = []
        for tensor in proto.initializer:
            if uses_external_data(tensor):
                prop = None
                for ext in tensor.external_data:  # type: ignore[assignment]
                    if ext.key == "location":  # type: ignore[attr-defined]
                        prop = ext  # type: ignore[assignment]
                assert prop is not None, f"No location found for tensor name {tensor.name!r}."
                assert prop.value in self.large_initializers, (
                    f"Unable to find large tensor named {tensor.name!r} "
                    f"with location {prop.value!r} in "
                    f"{sorted(self.large_initializers)}."
                )
                np_tensor = self.large_initializers[prop.value]
                if isinstance(np_tensor, np.ndarray):
                    t = oir.Tensor(
                        np_tensor,
                        name=tensor.name,
                        doc_string=tensor.doc_string,
                        metadata_props=oirs.deserialize_metadata_props(tensor.metadata_props),
                    )
                elif hasattr(np_tensor, "shape"):
                    from ..torch.torch_helper import torch_dtype_to_onnx_dtype

                    t = oir.Tensor(
                        np_tensor.detach(),
                        name=tensor.name,
                        dtype=oir.DataType.from_numpy(
                            tensor_dtype_to_np_dtype(torch_dtype_to_onnx_dtype(np_tensor.dtype))
                        ),
                        doc_string=tensor.doc_string,
                        metadata_props=oirs.deserialize_metadata_props(tensor.metadata_props),
                    )
                else:
                    t = oir.Tensor(
                        onh.to_array(np_tensor),
                        name=tensor.name,
                        doc_string=tensor.doc_string,
                        metadata_props=oirs.deserialize_metadata_props(tensor.metadata_props),
                    )
            else:
                t = oirs.deserialize_tensor(tensor)  # type: ignore
            initializer_tensors.append(t)
        inputs = [oir.Input(info.name) for info in proto.input]  # pyrefly: ignore[deprecated]
        for info, value in zip(proto.input, inputs):
            oirs.deserialize_value_info_proto(info, value)
            if value.name in quantization_annotations:
                oirs._deserialize_quantization_annotation(
                    quantization_annotations[value.name], value  # type: ignore
                )

        values = {v.name: v for v in inputs}
        scoped_values.append(values)  # type: ignore

        initializer_values = []
        for i, tensor in enumerate(initializer_tensors):
            initializer_name = tensor.name
            assert initializer_name, f"Initializer {i} has no name, it should not be there."
            assert initializer_name not in values, f"Duplicated name {initializer_name!r}"
            initializer_value = oir.Value(
                None,
                index=None,
                name=initializer_name,
                type=oir.TensorType(tensor.dtype),
                shape=tensor.shape,
                const_value=tensor,
            )
            if initializer_value.name in quantization_annotations:
                oirs._deserialize_quantization_annotation(
                    quantization_annotations[initializer_value.name], initializer_value  # type: ignore
                )
            values[initializer_name] = initializer_value
            initializer_values.append(initializer_value)

        value_info = {info.name: info for info in proto.value_info}
        for node in proto.node:
            oirs._declare_node_outputs(
                node,
                scoped_values[-1],
                value_info=value_info,
                quantization_annotations=quantization_annotations,  # type: ignore
            )

        nodes = []
        for node in proto.node:
            nodes.append(
                oirs._deserialize_node(node, scoped_values, value_info, quantization_annotations)  # type: ignore
            )

        outputs = []
        for info in proto.output:
            # Fill in values for graph outputs
            output_name = info.name
            assert output_name in values, f"Missing output_name={output_name!r} in {values}"
            value = values[output_name]
            oirs.deserialize_value_info_proto(info, value)
            outputs.append(value)

        # Exit the graph scope by popping the values for this scope from the stack
        scoped_values.pop()

        return oir.Graph(
            inputs,
            outputs,
            nodes=nodes,
            initializers=initializer_values,
            doc_string=self._get_field(proto, "doc_string"),
            name=self._get_field(proto, "name"),
            metadata_props=oirs.deserialize_metadata_props(proto.metadata_props),
        )

    @classmethod
    def _get_field(cls, proto: Any, field: str) -> Any:
        if proto.HasField(field):
            return getattr(proto, field)
        return None

    def to_ir(self) -> "onnx_ir.Model":  # type: ignore # noqa: F821
        """Conversion to :class:`onnx_ir.Model`."""
        import onnx_ir as oir
        import onnx_ir.serde as oirs

        proto = self.model_proto
        graph = self._deserialize_graph(proto.graph, [])
        graph.opset_imports.update(oirs.deserialize_opset_import(proto.opset_import))

        functions = []
        for func in proto.functions:
            functions.append(oirs.deserialize_function(func))

        model = oir.Model(
            graph,
            ir_version=proto.ir_version,
            producer_name=self._get_field(proto, "producer_name"),
            producer_version=self._get_field(proto, "producer_version"),
            domain=self._get_field(proto, "domain"),
            model_version=self._get_field(proto, "model_version"),
            doc_string=self._get_field(proto, "doc_string"),
            functions=functions,
            metadata_props=oirs.deserialize_metadata_props(proto.metadata_props),
        )
        return model
