from ._native_op import NativeOpKernel, evaluate_native_operator


class QLinearConv(NativeOpKernel):
    """Adapts the channels-last contrib variant to the native standard kernel."""

    op_domain = "com.microsoft"

    def _run(
        self,
        x,
        x_scale,
        x_zero_point,
        w,
        w_scale,
        w_zero_point,
        y_scale,
        y_zero_point,
        B=None,
        auto_pad=None,
        channels_last=None,
        dilations=None,
        group=None,
        kernel_shape=None,
        pads=None,
        strides=None,
    ):
        if channels_last:
            x = x.transpose(0, x.ndim - 1, *range(1, x.ndim - 1))
        (y,) = evaluate_native_operator(
            "QLinearConv",
            x,
            x_scale,
            x_zero_point,
            w,
            w_scale,
            w_zero_point,
            y_scale,
            y_zero_point,
            B,
            auto_pad=auto_pad,
            dilations=dilations,
            group=group,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
        )
        if channels_last:
            y = y.transpose(0, *range(2, y.ndim), 1)
        return (y,)
