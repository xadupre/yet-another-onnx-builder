def register():
    try:
        import imblearn  # noqa: F401
        from imblearn.pipeline import Pipeline as ImbPipeline

        from . import smote  # noqa: F401

        # imblearn.pipeline.Pipeline is a subclass of sklearn.pipeline.Pipeline
        # and automatically filters sampler steps during transform/predict.
        # Register it with the same converter so that pipelines containing SMOTE
        # can be exported to ONNX transparently.
        from ..pipeline.pipeline import sklearn_pipeline
        from ..register import SKLEARN_CONVERTERS

        if ImbPipeline not in SKLEARN_CONVERTERS:
            SKLEARN_CONVERTERS[ImbPipeline] = sklearn_pipeline
    except ImportError:
        # imblearn not installed – skip silently.
        pass
