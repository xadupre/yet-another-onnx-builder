def register():
    from . import (
        binarizer,
        dataframe_function_transformer,
        function_transformer,
        kbins_discretizer,
        kernel_centerer,
        max_abs_scaler,
        min_max_scaler,
        normalizer,
        one_hot_encoder,
        ordinal_encoder,
        polynomial_features,
        power_transformer,
        quantile_transformer,
        robust_scaler,
        spline_transformer,
        standard_scaler,
    )


def __getattr__(name: str):
    if name == "DataFrameTransformer":
        from ._dataframe_transformer import DataFrameTransformer

        return DataFrameTransformer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
