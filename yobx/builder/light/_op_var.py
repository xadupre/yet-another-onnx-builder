from typing import List, Optional, Union


class OpsVar:
    """
    Mixin that provides ONNX operator methods for a single-input variable.

    Each method calls ``self.make_node`` which is defined in :class:`BaseVar`.
    """

    def ArgMax(self, axis: int = 0, keepdims: int = 1, select_last_index: int = 0) -> "Var":
        return self.make_node(
            "ArgMax",
            self,
            axis=axis,
            keepdims=keepdims,
            select_last_index=select_last_index,
        )

    def ArgMin(self, axis: int = 0, keepdims: int = 1, select_last_index: int = 0) -> "Var":
        return self.make_node(
            "ArgMin",
            self,
            axis=axis,
            keepdims=keepdims,
            select_last_index=select_last_index,
        )

    def AveragePool(
        self,
        auto_pad: str = "NOTSET",
        ceil_mode: int = 0,
        count_include_pad: int = 0,
        dilations: Optional[List[int]] = None,
        kernel_shape: Optional[List[int]] = None,
        pads: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
    ) -> "Var":
        kwargs = {
            "auto_pad": auto_pad,
            "ceil_mode": ceil_mode,
            "count_include_pad": count_include_pad,
        }
        if dilations is not None:
            kwargs["dilations"] = dilations
        if kernel_shape is not None:
            kwargs["kernel_shape"] = kernel_shape
        if pads is not None:
            kwargs["pads"] = pads
        if strides is not None:
            kwargs["strides"] = strides
        return self.make_node("AveragePool", self, **kwargs)

    def Bernoulli(self, dtype: int = 0, seed: float = 0.0) -> "Var":
        return self.make_node("Bernoulli", self, dtype=dtype, seed=seed)

    def BlackmanWindow(self, output_datatype: int = 1, periodic: int = 1) -> "Var":
        return self.make_node(
            "BlackmanWindow", self, output_datatype=output_datatype, periodic=periodic
        )

    def Cast(self, saturate: int = 1, to: int = 0) -> "Var":
        return self.make_node("Cast", self, saturate=saturate, to=to)

    def Celu(self, alpha: float = 1.0) -> "Var":
        return self.make_node("Celu", self, alpha=alpha)

    def ConstantOfShape(self, value=None) -> "Var":
        if value is None:
            return self.make_node("ConstantOfShape", self)
        import numpy as np
        from onnx.numpy_helper import from_array

        return self.make_node("ConstantOfShape", self, value=from_array(np.array([value])))

    def DepthToSpace(self, blocksize: int = 0, mode: str = "DCR") -> "Var":
        return self.make_node("DepthToSpace", self, blocksize=blocksize, mode=mode)

    def DynamicQuantizeLinear(self) -> "Vars":
        return self.make_node("DynamicQuantizeLinear", self, n_outputs=3)

    def Elu(self, alpha: float = 1.0) -> "Var":
        return self.make_node("Elu", self, alpha=alpha)

    def EyeLike(self, dtype: int = 0, k: int = 0) -> "Var":
        return self.make_node("EyeLike", self, dtype=dtype, k=k)

    def Flatten(self, axis: int = 1) -> "Var":
        return self.make_node("Flatten", self, axis=axis)

    def GlobalLpPool(self, p: int = 2) -> "Var":
        return self.make_node("GlobalLpPool", self, p=p)

    def HammingWindow(self, output_datatype: int = 1, periodic: int = 1) -> "Var":
        return self.make_node(
            "HammingWindow", self, output_datatype=output_datatype, periodic=periodic
        )

    def HannWindow(self, output_datatype: int = 1, periodic: int = 1) -> "Var":
        return self.make_node(
            "HannWindow", self, output_datatype=output_datatype, periodic=periodic
        )

    def HardSigmoid(self, alpha: float = 0.20000000298023224, beta: float = 0.5) -> "Var":
        return self.make_node("HardSigmoid", self, alpha=alpha, beta=beta)

    def Hardmax(self, axis: int = -1) -> "Var":
        return self.make_node("Hardmax", self, axis=axis)

    def IsInf(self, detect_negative: int = 1, detect_positive: int = 1) -> "Var":
        return self.make_node(
            "IsInf",
            self,
            detect_negative=detect_negative,
            detect_positive=detect_positive,
        )

    def LRN(
        self,
        alpha: float = 9.999999747378752e-05,
        beta: float = 0.75,
        bias: float = 1.0,
        size: int = 0,
    ) -> "Var":
        return self.make_node("LRN", self, alpha=alpha, beta=beta, bias=bias, size=size)

    def LeakyRelu(self, alpha: float = 0.009999999776482582) -> "Var":
        return self.make_node("LeakyRelu", self, alpha=alpha)

    def LogSoftmax(self, axis: int = -1) -> "Var":
        return self.make_node("LogSoftmax", self, axis=axis)

    def LpNormalization(self, axis: int = -1, p: int = 2) -> "Var":
        return self.make_node("LpNormalization", self, axis=axis, p=p)

    def LpPool(
        self,
        auto_pad: str = "NOTSET",
        ceil_mode: int = 0,
        dilations: Optional[List[int]] = None,
        kernel_shape: Optional[List[int]] = None,
        p: int = 2,
        pads: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
    ) -> "Var":
        attrs = {
            "auto_pad": auto_pad,
            "ceil_mode": ceil_mode,
            "kernel_shape": kernel_shape or [],
            "p": p,
        }
        if dilations is not None:
            attrs["dilations"] = dilations
        if pads is not None:
            attrs["pads"] = pads
        if strides is not None:
            attrs["strides"] = strides
        return self.make_node("LpPool", self, **attrs)

    def MeanVarianceNormalization(self, axes: Optional[List[int]] = None) -> "Var":
        return self.make_node("MeanVarianceNormalization", self, axes=axes or [0, 2, 3])

    def Multinomial(self, dtype: int = 6, sample_size: int = 1, seed: float = 0.0) -> "Var":
        return self.make_node(
            "Multinomial", self, dtype=dtype, sample_size=sample_size, seed=seed
        )

    def RandomNormalLike(
        self, dtype: int = 0, mean: float = 0.0, scale: float = 1.0, seed: float = 0.0
    ) -> "Var":
        return self.make_node(
            "RandomNormalLike", self, dtype=dtype, mean=mean, scale=scale, seed=seed
        )

    def RandomUniformLike(
        self, dtype: int = 0, high: float = 1.0, low: float = 0.0, seed: float = 0.0
    ) -> "Var":
        return self.make_node(
            "RandomUniformLike", self, dtype=dtype, high=high, low=low, seed=seed
        )

    def ReduceL1(self, keepdims: int = 1, noop_with_empty_axes: int = 0) -> "Var":
        return self.make_node(
            "ReduceL1", self, keepdims=keepdims, noop_with_empty_axes=noop_with_empty_axes
        )

    def ReduceL2(self, keepdims: int = 1, noop_with_empty_axes: int = 0) -> "Var":
        return self.make_node(
            "ReduceL2", self, keepdims=keepdims, noop_with_empty_axes=noop_with_empty_axes
        )

    def ReduceLogSum(self, keepdims: int = 1, noop_with_empty_axes: int = 0) -> "Var":
        return self.make_node(
            "ReduceLogSum",
            self,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )

    def ReduceLogSumExp(self, keepdims: int = 1, noop_with_empty_axes: int = 0) -> "Var":
        return self.make_node(
            "ReduceLogSumExp",
            self,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )

    def ReduceMax(self, keepdims: int = 1, noop_with_empty_axes: int = 0) -> "Var":
        return self.make_node(
            "ReduceMax", self, keepdims=keepdims, noop_with_empty_axes=noop_with_empty_axes
        )

    def ReduceMean(self, keepdims: int = 1, noop_with_empty_axes: int = 0) -> "Var":
        return self.make_node(
            "ReduceMean", self, keepdims=keepdims, noop_with_empty_axes=noop_with_empty_axes
        )

    def ReduceMin(self, keepdims: int = 1, noop_with_empty_axes: int = 0) -> "Var":
        return self.make_node(
            "ReduceMin", self, keepdims=keepdims, noop_with_empty_axes=noop_with_empty_axes
        )

    def ReduceProd(self, keepdims: int = 1, noop_with_empty_axes: int = 0) -> "Var":
        return self.make_node(
            "ReduceProd", self, keepdims=keepdims, noop_with_empty_axes=noop_with_empty_axes
        )

    def ReduceSum(self, keepdims: int = 1, noop_with_empty_axes: int = 0) -> "Var":
        return self.make_node(
            "ReduceSum", self, keepdims=keepdims, noop_with_empty_axes=noop_with_empty_axes
        )

    def ReduceSumSquare(self, keepdims: int = 1, noop_with_empty_axes: int = 0) -> "Var":
        return self.make_node(
            "ReduceSumSquare",
            self,
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )

    def Selu(self, alpha: float = 1.6732631921768188, gamma: float = 1.0507010221481323) -> "Var":
        return self.make_node("Selu", self, alpha=alpha, gamma=gamma)

    def Shrink(self, bias: float = 0.0, lambd: float = 0.5) -> "Var":
        return self.make_node("Shrink", self, bias=bias, lambd=lambd)

    def Slice(
        self,
        starts: "Var",
        ends: "Var",
        axes: Optional["Var"] = None,
        steps: Optional["Var"] = None,
    ) -> "Var":
        extra = [v for v in [axes, steps] if v is not None]
        return self.make_node("Slice", self, starts, ends, *extra)

    def Softmax(self, axis: int = -1) -> "Var":
        return self.make_node("Softmax", self, axis=axis)

    def SpaceToDepth(self, blocksize: int = 0) -> "Var":
        return self.make_node("SpaceToDepth", self, blocksize=blocksize)

    def ThresholdedRelu(self, alpha: float = 1.0) -> "Var":
        return self.make_node("ThresholdedRelu", self, alpha=alpha)

    def Transpose(self, perm: Optional[List[int]] = None) -> "Var":
        kwargs = {}
        if perm:
            kwargs["perm"] = perm
        return self.make_node("Transpose", self, **kwargs)

    def LSTM(
        self,
        direction: str = "forward",
        hidden_size: int = 1,
        activations: Optional[List[str]] = None,
        clip: Optional[float] = None,
        input_forget: int = 0,
        layout: int = 0,
    ) -> "Vars":
        kwargs: dict = {
            "direction": direction,
            "hidden_size": hidden_size,
            "input_forget": input_forget,
            "layout": layout,
        }
        if activations is not None:
            kwargs["activations"] = activations
        if clip is not None:
            kwargs["clip"] = clip
        return self.make_node("LSTM", self, n_outputs=3, **kwargs)

    def GRU(
        self,
        direction: str = "forward",
        hidden_size: int = 1,
        activations: Optional[List[str]] = None,
        clip: Optional[float] = None,
        layout: int = 0,
        linear_before_reset: int = 0,
    ) -> "Vars":
        kwargs: dict = {
            "direction": direction,
            "hidden_size": hidden_size,
            "layout": layout,
            "linear_before_reset": linear_before_reset,
        }
        if activations is not None:
            kwargs["activations"] = activations
        if clip is not None:
            kwargs["clip"] = clip
        return self.make_node("GRU", self, n_outputs=2, **kwargs)

    def RNN(
        self,
        direction: str = "forward",
        hidden_size: int = 1,
        activations: Optional[List[str]] = None,
        clip: Optional[float] = None,
        layout: int = 0,
    ) -> "Vars":
        kwargs: dict = {
            "direction": direction,
            "hidden_size": hidden_size,
            "layout": layout,
        }
        if activations is not None:
            kwargs["activations"] = activations
        if clip is not None:
            kwargs["clip"] = clip
        return self.make_node("RNN", self, n_outputs=2, **kwargs)

    def MaxPool(
        self,
        auto_pad: str = "NOTSET",
        ceil_mode: int = 0,
        dilations: Optional[List[int]] = None,
        kernel_shape: Optional[List[int]] = None,
        pads: Optional[List[int]] = None,
        storage_order: int = 0,
        strides: Optional[List[int]] = None,
    ) -> Union["Var", "Vars"]:
        kwargs: dict = {
            "auto_pad": auto_pad,
            "ceil_mode": ceil_mode,
            "storage_order": storage_order,
        }
        if dilations is not None:
            kwargs["dilations"] = dilations
        if kernel_shape is not None:
            kwargs["kernel_shape"] = kernel_shape
        if pads is not None:
            kwargs["pads"] = pads
        if strides is not None:
            kwargs["strides"] = strides
        return self.make_node("MaxPool", self, **kwargs)

    def NegativeLogLikelihoodLoss(
        self,
        ignore_index: int = 0,
        reduction: str = "mean",
    ) -> "Var":
        return self.make_node(
            "NegativeLogLikelihoodLoss",
            self,
            ignore_index=ignore_index,
            reduction=reduction,
        )

    def Unique(self, axis: Optional[int] = None, sorted: int = 1) -> "Vars":
        kwargs: dict = {"sorted": sorted}
        if axis is not None:
            kwargs["axis"] = axis
        return self.make_node("Unique", self, n_outputs=4, **kwargs)


def _complete_ops_var() -> None:
    simple_ops = [
        "Abs",
        "Acos",
        "Acosh",
        "Asin",
        "Asinh",
        "Atan",
        "Atanh",
        "BitwiseNot",
        "Ceil",
        "Cos",
        "Cosh",
        "Det",
        "Erf",
        "Exp",
        "Floor",
        "GlobalAveragePool",
        "GlobalMaxPool",
        "HardSwish",
        "Identity",
        "IsNaN",
        "Log",
        "Mish",
        "Neg",
        "NonZero",
        "Not",
        "Reciprocal",
        "Relu",
        "Round",
        "Shape",
        "Sigmoid",
        "Sign",
        "Sin",
        "Sinh",
        "Size",
        "Softplus",
        "Softsign",
        "Sqrt",
        "Tan",
        "Tanh",
    ]
    for name in simple_ops:
        if hasattr(OpsVar, name):
            continue
        setattr(OpsVar, name, lambda self, _op=name: self.make_node(_op, self))


_complete_ops_var()
