from typing import Optional, Tuple
import onnx
import torch
from . import OpRunKernel, OpRunTensor


class AveragePool_11(OpRunKernel):
    "AveragePool"

    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None, verbose: int = 0):
        super().__init__(node, version, verbose=verbose)
        self.auto_pad = self.get_attribute_string(node, "auto_pad", "NOTSET")
        self.ceil_mode = bool(self.get_attribute_int(node, "ceil_mode", 0))
        self.count_include_pad = bool(self.get_attribute_int(node, "count_include_pad", 0))
        self.dilations = self.get_attribute_ints(node, "dilations", None)
        self.kernel_shape: Tuple[int, ...] = (
            self.get_attribute_ints(node, "kernel_shape") or tuple()
        )
        self.pads = self.get_attribute_ints(node, "pads", None)
        self.strides = self.get_attribute_ints(node, "strides", None)

    def run(self, x):
        kernel_shape = self.kernel_shape
        dilations = self.dilations or [1 for _ in x.shape[2:]]
        strides = self.strides or [1 for _ in x.shape[2:]]
        pads = self.pads or ([0 for _ in x.shape[2:]] * 2)
        assert self.auto_pad == "NOTSET", f"conv not implemented for auto_pad={self.auto_pad!r}"
        assert len(set(pads)) == 1, f"conv not implemented for pads={pads}"
        assert set(dilations) == {1}, f"conv not implemented for dilations={dilations}"
        avg_pool = getattr(torch.nn.functional, f"avg_pool{len(kernel_shape)}d")
        return OpRunTensor(
            avg_pool(
                x.tensor,
                kernel_size=tuple(kernel_shape),
                stride=tuple(strides),
                padding=pads[0],
                ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
                # dilation=tuple(dilations),
            )
        )


class Conv_11(OpRunKernel):
    "Conv"

    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None, verbose: int = 0):
        super().__init__(node, version, verbose=verbose)
        self.auto_pad = self.get_attribute_string(node, "auto_pad", "NOTSET")
        self.dilations = self.get_attribute_ints(node, "dilations", None)
        self.group = self.get_attribute_int(node, "group", 1)
        self.kernel_shape: Tuple[int, ...] = (
            self.get_attribute_ints(node, "kernel_shape") or tuple()
        )
        self.pads = self.get_attribute_ints(node, "pads", None)
        self.strides = self.get_attribute_ints(node, "strides", None)

    def run(self, x, w, b=None):
        kernel_shape = self.kernel_shape or w.shape[2:]
        assert (
            tuple(kernel_shape) == w.shape[-len(kernel_shape) :]
        ), f"conv not implemented for kernel_shape={kernel_shape} and w.shape={w.shape}"
        dilations = self.dilations or [1 for _ in x.shape[2:]]
        strides = self.strides or [1 for _ in x.shape[2:]]

        if self.auto_pad in {"SAME_LOWER", "SAME_UPPER"}:
            head = []
            tail = []
            for i in range(len(x.shape) - 2):
                d = x.shape[i + 2]
                target_size = (d + strides[i] - 1) // strides[i]
                pad_needed = (target_size - 1) * strides[i] + kernel_shape[i] - d
                pad_head = (
                    (pad_needed + 1) // 2 if self.auto_pad == "SAME_LOWER" else pad_needed // 2
                )
                pad_tail = pad_needed - pad_head
                head.append(pad_head)
                tail.append(pad_tail)
            pads = head + tail
        else:
            pads = self.pads or ([0 for _ in x.shape[2:]] * 2)

        assert len(set(pads)) == 1, (
            f"conv not implemented for pads={pads}, "
            f"auto_pad={self.auto_pad!r}, strides={strides}, "
            f"x.shape={x.shape}, kernel_shape={kernel_shape}"
        )

        if b is None:
            bias = None
        else:
            bias = b.tensor.squeeze()
            if not bias.shape:
                bias = bias.unsqueeze(0)
        return OpRunTensor(
            torch.nn.functional.conv2d(
                x.tensor,
                w.tensor,
                bias=bias,
                stride=tuple(strides),
                padding=pads[0],
                dilation=tuple(dilations),
                groups=self.group,
            )
        )


class LayerNormalization_17(OpRunKernel):
    "LayerNormalization"

    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None, verbose: int = 0):
        super().__init__(node, version, verbose=verbose)
        self.axis = self.get_attribute_int(node, "axis", -1)
        self.epsilon = self.get_attribute_float(node, "epsilon", 1e-5)
        from ...torch.torch_helper import onnx_dtype_to_torch_dtype

        self.stash_type = onnx_dtype_to_torch_dtype(
            self.get_attribute_int(node, "stash_type", onnx.TensorProto.FLOAT)  # type: ignore[arg-type]
        )
        self.compute_std = len(node.output) > 1

    def run(self, x, scale, bias=None):
        original_dtype = x.dtype
        if self.stash_type == torch.float32 and x.tensor.dtype != torch.float64:
            xt = x.tensor
            res = torch.nn.functional.layer_norm(
                xt,
                xt.shape[self.axis :],
                weight=scale.tensor,
                bias=None if bias is None else bias.tensor,
                eps=self.epsilon,
            )
        else:
            xt = x.tensor.to(self.stash_type)
            res = torch.nn.functional.layer_norm(
                xt,
                xt.shape[self.axis :],
                weight=scale.tensor.to(self.stash_type),
                bias=None if bias is None else bias.tensor.to(self.stash_type),
                eps=self.epsilon,
            )
        if not self.compute_std:
            return OpRunTensor(res.to(original_dtype))
        axes = tuple(range(len(xt.shape)))[self.axis :]
        var, mean = torch.var_mean(xt, dim=axes, keepdim=False)
        x_inv_std_dev = torch.reciprocal(torch.sqrt(var + self.epsilon))
        return (
            OpRunTensor(res.to(original_dtype)),
            OpRunTensor(mean),
            OpRunTensor(x_inv_std_dev),
        )


class Resize_18(OpRunKernel):
    "Resize (opset 18, with antialias / axes support)."

    _MODE_MAP = {1: "linear", 2: "bilinear", 3: "trilinear"}

    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None, verbose: int = 0):
        super().__init__(node, version, verbose=verbose)
        self.antialias = bool(self.get_attribute_int(node, "antialias", 0))
        self.axes = self.get_attribute_ints(node, "axes", None)
        self.coordinate_transformation_mode = self.get_attribute_string(
            node, "coordinate_transformation_mode", "half_pixel"
        )
        self.cubic_coeff_a = self.get_attribute_float(node, "cubic_coeff_a", -0.75)
        self.exclude_outside = bool(self.get_attribute_int(node, "exclude_outside", 0))
        self.extrapolation_value = self.get_attribute_float(node, "extrapolation_value", 0.0)
        self.keep_aspect_ratio_policy = self.get_attribute_string(
            node, "keep_aspect_ratio_policy", "stretch"
        )
        self.mode = self.get_attribute_string(node, "mode", "nearest")
        self.nearest_mode = self.get_attribute_string(node, "nearest_mode", "round_prefer_floor")
        assert self.keep_aspect_ratio_policy == "stretch", (
            f"Resize not implemented for "
            f"keep_aspect_ratio_policy={self.keep_aspect_ratio_policy!r}"
        )
        assert not self.exclude_outside, "Resize not implemented for exclude_outside=1"
        assert self.mode in {
            "nearest",
            "linear",
            "cubic",
        }, f"Resize unsupported mode={self.mode!r}"
        if self.mode == "nearest":
            # torch.nn.functional.interpolate(mode='nearest') uses
            # asymmetric coordinates with floor rounding.
            assert self.nearest_mode == "floor" and (
                self.coordinate_transformation_mode == "asymmetric"
            ), (
                f"Resize nearest only implemented for nearest_mode='floor' and "
                f"coordinate_transformation_mode='asymmetric', got "
                f"nearest_mode={self.nearest_mode!r}, "
                f"coordinate_transformation_mode="
                f"{self.coordinate_transformation_mode!r}"
            )
        else:
            assert self.coordinate_transformation_mode in {"half_pixel", "align_corners"}, (
                f"Resize {self.mode!r} only implemented for "
                f"coordinate_transformation_mode in (half_pixel, align_corners), "
                f"got {self.coordinate_transformation_mode!r}"
            )

    def run(self, x, roi=None, scales=None, sizes=None):  # type: ignore[override]
        rank = x.tensor.dim()
        has_scales = (
            scales is not None and scales.tensor is not None and scales.tensor.numel() > 0
        )
        has_sizes = sizes is not None and sizes.tensor is not None and sizes.tensor.numel() > 0
        if has_scales == has_sizes:
            raise NotImplementedError("Resize requires exactly one of scales or sizes.")

        # Determine which axes are being resized.
        if self.axes is None:
            axes: Tuple[int, ...] = tuple(range(rank))
        else:
            axes = tuple(a if a >= 0 else a + rank for a in self.axes)

        if has_sizes:
            provided = [int(v) for v in sizes.tensor.tolist()]
            if self.axes is None:
                full_sizes = provided
            else:
                full_sizes = list(x.tensor.shape)
                for a, s in zip(axes, provided):
                    full_sizes[a] = s
            assert len(full_sizes) == rank
            spatial_axes = [a for a in range(rank) if full_sizes[a] != x.tensor.shape[a]]
            non_spatial = [a for a in range(rank) if a not in spatial_axes]
            for a in non_spatial:
                if full_sizes[a] != x.tensor.shape[a]:
                    raise NotImplementedError(
                        f"Resize only supports identity on non-resized dims, "
                        f"got sizes={full_sizes} for shape={tuple(x.tensor.shape)}"
                    )
            if not spatial_axes:
                # Nothing to resize.
                return OpRunTensor(x.tensor.clone())
            if spatial_axes != list(range(spatial_axes[0], spatial_axes[0] + len(spatial_axes))):
                raise NotImplementedError(
                    f"Resize only supports contiguous spatial axes, got {spatial_axes}"
                )
            if not (spatial_axes[-1] == rank - 1 and spatial_axes[0] >= 2):
                raise NotImplementedError(
                    f"Resize only supports trailing spatial axes (NCHW-like layout), "
                    f"got {spatial_axes} for shape={tuple(x.tensor.shape)}"
                )
            target = [full_sizes[a] for a in spatial_axes]
            scale_arg = None
        else:
            provided_f = [float(v) for v in scales.tensor.tolist()]
            if self.axes is None:
                full_scales = provided_f
            else:
                full_scales = [1.0] * rank
                for a, s in zip(axes, provided_f):
                    full_scales[a] = s
            assert len(full_scales) == rank
            spatial_axes = [a for a in range(rank) if full_scales[a] != 1.0]
            for a in range(rank):
                if a not in spatial_axes and full_scales[a] != 1.0:
                    raise NotImplementedError(
                        f"Resize only supports identity on non-resized dims, "
                        f"got scales={full_scales}"
                    )
            if not spatial_axes:
                return OpRunTensor(x.tensor.clone())
            if spatial_axes != list(range(spatial_axes[0], spatial_axes[0] + len(spatial_axes))):
                raise NotImplementedError(
                    f"Resize only supports contiguous spatial axes, got {spatial_axes}"
                )
            if not (spatial_axes[-1] == rank - 1 and spatial_axes[0] >= 2):
                raise NotImplementedError(
                    f"Resize only supports trailing spatial axes (NCHW-like layout), "
                    f"got {spatial_axes} for shape={tuple(x.tensor.shape)}"
                )
            target = None
            scale_arg = [full_scales[a] for a in spatial_axes]

        spatial_dims = len(spatial_axes)

        if self.mode == "nearest":
            torch_mode = "nearest"
            align_corners: Optional[bool] = None
        elif self.mode == "linear":
            torch_mode = self._MODE_MAP.get(spatial_dims)
            if torch_mode is None:
                raise NotImplementedError(
                    f"Resize linear not implemented for {spatial_dims} spatial dims"
                )
            align_corners = self.coordinate_transformation_mode == "align_corners"
        else:  # cubic
            if spatial_dims != 2:
                raise NotImplementedError(
                    f"Resize cubic not implemented for {spatial_dims} spatial dims"
                )
            torch_mode = "bicubic"
            align_corners = self.coordinate_transformation_mode == "align_corners"

        kwargs: dict = {"mode": torch_mode}
        if target is not None:
            kwargs["size"] = target
        else:
            kwargs["scale_factor"] = scale_arg
            kwargs["recompute_scale_factor"] = False
        if align_corners is not None:
            kwargs["align_corners"] = align_corners
        if self.antialias:
            if self.mode not in {"linear", "cubic"}:
                raise NotImplementedError(
                    f"Resize antialias requires mode in (linear, cubic), got {self.mode!r}"
                )
            kwargs["antialias"] = True
        return OpRunTensor(torch.nn.functional.interpolate(x.tensor, **kwargs))


class Softmax_13(OpRunKernel):
    "Softmax"

    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None, verbose: int = 0):
        super().__init__(node, version, verbose=verbose)
        self.axis = self.get_attribute_int(node, "axis", -1)
        assert isinstance(self.axis, int), f"Unexpected value for attribute axis={self.axis!r}"
        # this is out of spec
        stash_type = self.get_attribute_int(node, "stash_type", None)
        from ...torch.torch_helper import onnx_dtype_to_torch_dtype

        self.stash_type = None if stash_type is None else onnx_dtype_to_torch_dtype(stash_type)

    def run(self, data: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(
            torch.nn.functional.softmax(data.tensor, dim=self.axis, dtype=self.stash_type)
        )


class Tanh_6(OpRunKernel):
    "Tanh"

    def run(self, data: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(torch.nn.functional.tanh(data.tensor))
