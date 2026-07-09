import math
from typing import Optional
import onnx
import torch
from . import OpRunKernel, OpRunTensor


class Bernoulli_1(OpRunKernel):
    """Bernoulli"""

    def __init__(self, node: onnx.NodeProto, version: Optional[int] = None, verbose: int = 0):
        super().__init__(node, version, verbose=verbose)
        self.dtype = self.get_attribute_int(node, "dtype", 0)
        self.seed = self.get_attribute_float(node, "seed", None)

    def run(self, x: OpRunTensor) -> OpRunTensor:
        """Samples from a Bernoulli distribution with probabilities given by the input tensor."""
        # torch.bernoulli requires float32 or float64 input; cast others (e.g. float16) to float32
        prob = x.tensor if x.tensor.dtype in (torch.float32, torch.float64) else x.tensor.float()
        if self.seed is not None and not math.isnan(self.seed):
            generator = torch.Generator(device=prob.device)
            generator.manual_seed(int(self.seed))
            result = torch.bernoulli(prob, generator=generator)
        else:
            result = torch.bernoulli(prob)
        if self.dtype and self.dtype != 0:
            from ...torch.torch_helper import onnx_dtype_to_torch_dtype

            return OpRunTensor(result.to(onnx_dtype_to_torch_dtype(self.dtype)))
        return OpRunTensor(result.to(x.tensor.dtype))


class Abs_1(OpRunKernel):
    """Abs"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(torch.abs(x.tensor))


class Cos_1(OpRunKernel):
    """Cos"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor.cos())


class Erf_9(OpRunKernel):
    """Erf"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor.erf())


class Exp_1(OpRunKernel):
    """Exp"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor.exp())


class Identity_1(OpRunKernel):
    "Identity"

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor)


class IsNaN_9(OpRunKernel):
    """IsNaN"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor.isnan())


class Log_1(OpRunKernel):
    """Log"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor.log())


class Neg_1(OpRunKernel):
    """Neg"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(-x.tensor)


class Not_1(OpRunKernel):
    """Not"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(~x.tensor)


class Reciprocal_1(OpRunKernel):
    """Reciprocal"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(1 / x.tensor)


class Sigmoid_6(OpRunKernel):
    """Sigmoid"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(torch.sigmoid(x.tensor))


class Sin_1(OpRunKernel):
    """Sin"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor.sin())


class Sqrt_1(OpRunKernel):
    """Sqrt"""

    def run(self, x: OpRunTensor) -> OpRunTensor:
        return OpRunTensor(x.tensor.sqrt())
