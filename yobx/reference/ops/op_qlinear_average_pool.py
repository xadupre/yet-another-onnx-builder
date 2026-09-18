from ._native_op import NativeOpKernel
from ._native_op import evaluate_native_operator


class QLinearAveragePool(NativeOpKernel):
    op_domain = "com.microsoft"

    def _run(
        self,
        x,
        x_scale,
        x_zero_point,
        y_scale,
        y_zero_point,
        auto_pad=None,
        ceil_mode=None,
        channels_last=None,
        count_include_pad=None,
        kernel_shape=None,
        pads=None,
        strides=None,
    ):
        assert channels_last in (
            None,
            0,
        ), f"QLinearAveragePool not implemented if channels_last={channels_last}"
        (dqx,) = evaluate_native_operator("DequantizeLinear", x, x_scale, x_zero_point)
        (y,) = evaluate_native_operator(
            "AveragePool",
            dqx,
            auto_pad=auto_pad,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
        )
        (qy,) = evaluate_native_operator("QuantizeLinear", y, y_scale, y_zero_point)
        return (qy,)
