from ..builder.onnxlight import OnnxLightGraphBuilder as GraphBuilder
from ..builder.onnxlight import OnnxLightOptimizationOptions as OptimizationOptions
from .function_options import FunctionOptions
from .infer_shapes_options import InferShapesOptions
from .order_optim import OrderAlgorithm
from ..typing import GraphBuilderTorchProtocol

TEMPLATE_TYPE = 999
