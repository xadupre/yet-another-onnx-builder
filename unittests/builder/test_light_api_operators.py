"""
Per-operator unit tests for yobx.builder.light.

Each test verifies that the fluent builder correctly emits an ONNX node with the
expected ``op_type``.  Full-graph validation and execution are covered separately
in ``test_light_api.py``; these tests focus solely on node-creation coverage.
"""

import unittest

import numpy as np

from yobx.builder.light import start, Var, Vars
from yobx.ext_test_case import ExtTestCase


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _x():
    """Return a (OnnxGraph, Var) pair with a single float input ``X``."""
    gr = start()
    return gr, gr.vin("X")


def _xy():
    """Return a (OnnxGraph, Var, Var) triple with float inputs ``X`` and ``Y``."""
    gr = start()
    return gr, gr.vin("X"), gr.vin("Y")


# ---------------------------------------------------------------------------
# OpsVar – single-input operators
# ---------------------------------------------------------------------------


class TestOpsVarOperators(ExtTestCase):
    """One test per single-input operator method defined in OpsVar."""

    def _check(self, gr, op):
        self.assertEqual(gr.nodes[-1].op_type, op)

    # ---- auto-generated simple unary ops (_complete_ops_var) ----

    def test_Abs(self):
        gr, x = _x()
        x.Abs()
        self._check(gr, "Abs")

    def test_Acos(self):
        gr, x = _x()
        x.Acos()
        self._check(gr, "Acos")

    def test_Acosh(self):
        gr, x = _x()
        x.Acosh()
        self._check(gr, "Acosh")

    def test_Asin(self):
        gr, x = _x()
        x.Asin()
        self._check(gr, "Asin")

    def test_Asinh(self):
        gr, x = _x()
        x.Asinh()
        self._check(gr, "Asinh")

    def test_Atan(self):
        gr, x = _x()
        x.Atan()
        self._check(gr, "Atan")

    def test_Atanh(self):
        gr, x = _x()
        x.Atanh()
        self._check(gr, "Atanh")

    def test_BitwiseNot(self):
        gr, x = _x()
        x.BitwiseNot()
        self._check(gr, "BitwiseNot")

    def test_Ceil(self):
        gr, x = _x()
        x.Ceil()
        self._check(gr, "Ceil")

    def test_Cos(self):
        gr, x = _x()
        x.Cos()
        self._check(gr, "Cos")

    def test_Cosh(self):
        gr, x = _x()
        x.Cosh()
        self._check(gr, "Cosh")

    def test_Det(self):
        gr, x = _x()
        x.Det()
        self._check(gr, "Det")

    def test_Erf(self):
        gr, x = _x()
        x.Erf()
        self._check(gr, "Erf")

    def test_Exp(self):
        gr, x = _x()
        x.Exp()
        self._check(gr, "Exp")

    def test_Floor(self):
        gr, x = _x()
        x.Floor()
        self._check(gr, "Floor")

    def test_GlobalAveragePool(self):
        gr, x = _x()
        x.GlobalAveragePool()
        self._check(gr, "GlobalAveragePool")

    def test_GlobalMaxPool(self):
        gr, x = _x()
        x.GlobalMaxPool()
        self._check(gr, "GlobalMaxPool")

    def test_HardSwish(self):
        gr, x = _x()
        x.HardSwish()
        self._check(gr, "HardSwish")

    def test_Identity(self):
        gr, x = _x()
        x.Identity()
        self._check(gr, "Identity")

    def test_IsNaN(self):
        gr, x = _x()
        x.IsNaN()
        self._check(gr, "IsNaN")

    def test_Log(self):
        gr, x = _x()
        x.Log()
        self._check(gr, "Log")

    def test_Mish(self):
        gr, x = _x()
        x.Mish()
        self._check(gr, "Mish")

    def test_Neg(self):
        gr, x = _x()
        x.Neg()
        self._check(gr, "Neg")

    def test_NonZero(self):
        gr, x = _x()
        x.NonZero()
        self._check(gr, "NonZero")

    def test_Not(self):
        gr, x = _x()
        x.Not()
        self._check(gr, "Not")

    def test_Reciprocal(self):
        gr, x = _x()
        x.Reciprocal()
        self._check(gr, "Reciprocal")

    def test_Relu(self):
        gr, x = _x()
        x.Relu()
        self._check(gr, "Relu")

    def test_Round(self):
        gr, x = _x()
        x.Round()
        self._check(gr, "Round")

    def test_Shape(self):
        gr, x = _x()
        x.Shape()
        self._check(gr, "Shape")

    def test_Sigmoid(self):
        gr, x = _x()
        x.Sigmoid()
        self._check(gr, "Sigmoid")

    def test_Sign(self):
        gr, x = _x()
        x.Sign()
        self._check(gr, "Sign")

    def test_Sin(self):
        gr, x = _x()
        x.Sin()
        self._check(gr, "Sin")

    def test_Sinh(self):
        gr, x = _x()
        x.Sinh()
        self._check(gr, "Sinh")

    def test_Size(self):
        gr, x = _x()
        x.Size()
        self._check(gr, "Size")

    def test_Softplus(self):
        gr, x = _x()
        x.Softplus()
        self._check(gr, "Softplus")

    def test_Softsign(self):
        gr, x = _x()
        x.Softsign()
        self._check(gr, "Softsign")

    def test_Sqrt(self):
        gr, x = _x()
        x.Sqrt()
        self._check(gr, "Sqrt")

    def test_Tan(self):
        gr, x = _x()
        x.Tan()
        self._check(gr, "Tan")

    def test_Tanh(self):
        gr, x = _x()
        x.Tanh()
        self._check(gr, "Tanh")

    # ---- explicit parameterized ops ----

    def test_ArgMax(self):
        gr, x = _x()
        x.ArgMax(axis=0)
        self._check(gr, "ArgMax")

    def test_ArgMin(self):
        gr, x = _x()
        x.ArgMin(axis=0)
        self._check(gr, "ArgMin")

    def test_AveragePool(self):
        gr, x = _x()
        x.AveragePool(kernel_shape=[3, 3])
        self._check(gr, "AveragePool")

    def test_Bernoulli(self):
        gr, x = _x()
        x.Bernoulli()
        self._check(gr, "Bernoulli")

    def test_BlackmanWindow(self):
        gr, x = _x()
        x.BlackmanWindow()
        self._check(gr, "BlackmanWindow")

    def test_Cast(self):
        from onnx import TensorProto

        gr, x = _x()
        x.Cast(to=TensorProto.FLOAT16)
        self._check(gr, "Cast")

    def test_Celu(self):
        gr, x = _x()
        x.Celu()
        self._check(gr, "Celu")

    def test_ConstantOfShape(self):
        gr, x = _x()
        x.ConstantOfShape()
        self._check(gr, "ConstantOfShape")

    def test_DepthToSpace(self):
        gr, x = _x()
        x.DepthToSpace(blocksize=2)
        self._check(gr, "DepthToSpace")

    def test_DynamicQuantizeLinear(self):
        gr, x = _x()
        result = x.DynamicQuantizeLinear()
        self.assertIsInstance(result, Vars)
        self.assertEqual(len(result), 3)
        self._check(gr, "DynamicQuantizeLinear")

    def test_Elu(self):
        gr, x = _x()
        x.Elu()
        self._check(gr, "Elu")

    def test_EyeLike(self):
        gr, x = _x()
        x.EyeLike()
        self._check(gr, "EyeLike")

    def test_Flatten(self):
        gr, x = _x()
        x.Flatten(axis=1)
        self._check(gr, "Flatten")

    def test_GlobalLpPool(self):
        gr, x = _x()
        x.GlobalLpPool(p=2)
        self._check(gr, "GlobalLpPool")

    def test_GRU(self):
        gr, x = _x()
        result = x.GRU(hidden_size=4)
        self.assertIsInstance(result, Vars)
        self.assertEqual(len(result), 2)
        self._check(gr, "GRU")

    def test_HammingWindow(self):
        gr, x = _x()
        x.HammingWindow()
        self._check(gr, "HammingWindow")

    def test_HannWindow(self):
        gr, x = _x()
        x.HannWindow()
        self._check(gr, "HannWindow")

    def test_HardSigmoid(self):
        gr, x = _x()
        x.HardSigmoid()
        self._check(gr, "HardSigmoid")

    def test_Hardmax(self):
        gr, x = _x()
        x.Hardmax()
        self._check(gr, "Hardmax")

    def test_IsInf(self):
        gr, x = _x()
        x.IsInf()
        self._check(gr, "IsInf")

    def test_LRN(self):
        gr, x = _x()
        x.LRN(size=3)
        self._check(gr, "LRN")

    def test_LeakyRelu(self):
        gr, x = _x()
        x.LeakyRelu()
        self._check(gr, "LeakyRelu")

    def test_LogSoftmax(self):
        gr, x = _x()
        x.LogSoftmax()
        self._check(gr, "LogSoftmax")

    def test_LpNormalization(self):
        gr, x = _x()
        x.LpNormalization()
        self._check(gr, "LpNormalization")

    def test_LpPool(self):
        gr, x = _x()
        x.LpPool(kernel_shape=[3, 3])
        self._check(gr, "LpPool")

    def test_LSTM(self):
        gr, x = _x()
        result = x.LSTM(hidden_size=4)
        self.assertIsInstance(result, Vars)
        self.assertEqual(len(result), 3)
        self._check(gr, "LSTM")

    def test_MaxPool(self):
        gr, x = _x()
        x.MaxPool(kernel_shape=[3, 3])
        self._check(gr, "MaxPool")

    def test_MeanVarianceNormalization(self):
        gr, x = _x()
        x.MeanVarianceNormalization()
        self._check(gr, "MeanVarianceNormalization")

    def test_Multinomial(self):
        gr, x = _x()
        x.Multinomial()
        self._check(gr, "Multinomial")

    def test_NegativeLogLikelihoodLoss_var(self):
        gr, x = _x()
        x.NegativeLogLikelihoodLoss()
        self._check(gr, "NegativeLogLikelihoodLoss")

    def test_RandomNormalLike(self):
        gr, x = _x()
        x.RandomNormalLike()
        self._check(gr, "RandomNormalLike")

    def test_RandomUniformLike(self):
        gr, x = _x()
        x.RandomUniformLike()
        self._check(gr, "RandomUniformLike")

    def test_ReduceL1(self):
        gr, x = _x()
        x.ReduceL1(keepdims=0)
        self._check(gr, "ReduceL1")

    def test_ReduceL2(self):
        gr, x = _x()
        x.ReduceL2(keepdims=0)
        self._check(gr, "ReduceL2")

    def test_ReduceLogSum(self):
        gr, x = _x()
        x.ReduceLogSum(keepdims=0)
        self._check(gr, "ReduceLogSum")

    def test_ReduceLogSumExp(self):
        gr, x = _x()
        x.ReduceLogSumExp(keepdims=0)
        self._check(gr, "ReduceLogSumExp")

    def test_ReduceMax(self):
        gr, x = _x()
        x.ReduceMax(keepdims=0)
        self._check(gr, "ReduceMax")

    def test_ReduceMean(self):
        gr, x = _x()
        x.ReduceMean(keepdims=0)
        self._check(gr, "ReduceMean")

    def test_ReduceMin(self):
        gr, x = _x()
        x.ReduceMin(keepdims=0)
        self._check(gr, "ReduceMin")

    def test_ReduceProd(self):
        gr, x = _x()
        x.ReduceProd(keepdims=0)
        self._check(gr, "ReduceProd")

    def test_ReduceSum(self):
        gr, x = _x()
        x.ReduceSum(keepdims=0)
        self._check(gr, "ReduceSum")

    def test_ReduceSumSquare(self):
        gr, x = _x()
        x.ReduceSumSquare(keepdims=0)
        self._check(gr, "ReduceSumSquare")

    def test_RNN(self):
        gr, x = _x()
        result = x.RNN(hidden_size=4)
        self.assertIsInstance(result, Vars)
        self.assertEqual(len(result), 2)
        self._check(gr, "RNN")

    def test_Selu(self):
        gr, x = _x()
        x.Selu()
        self._check(gr, "Selu")

    def test_Shrink(self):
        gr, x = _x()
        x.Shrink()
        self._check(gr, "Shrink")

    def test_Slice_var(self):
        gr = start()
        x = gr.vin("X")
        starts = gr.cst(np.array([0], dtype=np.int64), "starts")
        ends = gr.cst(np.array([4], dtype=np.int64), "ends")
        x.Slice(starts, ends)
        self._check(gr, "Slice")

    def test_Softmax(self):
        gr, x = _x()
        x.Softmax()
        self._check(gr, "Softmax")

    def test_SpaceToDepth(self):
        gr, x = _x()
        x.SpaceToDepth(blocksize=2)
        self._check(gr, "SpaceToDepth")

    def test_ThresholdedRelu(self):
        gr, x = _x()
        x.ThresholdedRelu()
        self._check(gr, "ThresholdedRelu")

    def test_Transpose(self):
        gr, x = _x()
        x.Transpose(perm=[1, 0])
        self._check(gr, "Transpose")

    def test_Unique(self):
        gr, x = _x()
        result = x.Unique()
        self.assertIsInstance(result, Vars)
        self.assertEqual(len(result), 4)
        self._check(gr, "Unique")


# ---------------------------------------------------------------------------
# OpsVars – multi-input operators
# ---------------------------------------------------------------------------


class TestOpsVarsOperators(ExtTestCase):
    """One test per multi-input operator method defined in OpsVars."""

    def _check(self, gr, op):
        self.assertEqual(gr.nodes[-1].op_type, op)

    # ---- auto-generated binary ops (_complete_ops_vars) ----

    def test_Add(self):
        gr, x, y = _xy()
        Vars(gr, x, y).Add()
        self._check(gr, "Add")

    def test_And(self):
        gr, x, y = _xy()
        Vars(gr, x, y).And()
        self._check(gr, "And")

    def test_BitwiseAnd(self):
        gr, x, y = _xy()
        Vars(gr, x, y).BitwiseAnd()
        self._check(gr, "BitwiseAnd")

    def test_BitwiseOr(self):
        gr, x, y = _xy()
        Vars(gr, x, y).BitwiseOr()
        self._check(gr, "BitwiseOr")

    def test_BitwiseXor(self):
        gr, x, y = _xy()
        Vars(gr, x, y).BitwiseXor()
        self._check(gr, "BitwiseXor")

    def test_CastLike(self):
        gr, x, y = _xy()
        Vars(gr, x, y).CastLike()
        self._check(gr, "CastLike")

    def test_Div(self):
        gr, x, y = _xy()
        Vars(gr, x, y).Div()
        self._check(gr, "Div")

    def test_Equal(self):
        gr, x, y = _xy()
        Vars(gr, x, y).Equal()
        self._check(gr, "Equal")

    def test_Expand(self):
        gr, x, y = _xy()
        Vars(gr, x, y).Expand()
        self._check(gr, "Expand")

    def test_GatherND(self):
        gr, x, y = _xy()
        Vars(gr, x, y).GatherND()
        self._check(gr, "GatherND")

    def test_Greater(self):
        gr, x, y = _xy()
        Vars(gr, x, y).Greater()
        self._check(gr, "Greater")

    def test_GreaterOrEqual(self):
        gr, x, y = _xy()
        Vars(gr, x, y).GreaterOrEqual()
        self._check(gr, "GreaterOrEqual")

    def test_Less(self):
        gr, x, y = _xy()
        Vars(gr, x, y).Less()
        self._check(gr, "Less")

    def test_LessOrEqual(self):
        gr, x, y = _xy()
        Vars(gr, x, y).LessOrEqual()
        self._check(gr, "LessOrEqual")

    def test_MatMul(self):
        gr, x, y = _xy()
        Vars(gr, x, y).MatMul()
        self._check(gr, "MatMul")

    def test_Mul(self):
        gr, x, y = _xy()
        Vars(gr, x, y).Mul()
        self._check(gr, "Mul")

    def test_Or(self):
        gr, x, y = _xy()
        Vars(gr, x, y).Or()
        self._check(gr, "Or")

    def test_PRelu(self):
        gr, x, y = _xy()
        Vars(gr, x, y).PRelu()
        self._check(gr, "PRelu")

    def test_Pow(self):
        gr, x, y = _xy()
        Vars(gr, x, y).Pow()
        self._check(gr, "Pow")

    def test_Reshape(self):
        gr, x, y = _xy()
        Vars(gr, x, y).Reshape()
        self._check(gr, "Reshape")

    def test_StringConcat(self):
        gr, x, y = _xy()
        Vars(gr, x, y).StringConcat()
        self._check(gr, "StringConcat")

    def test_Sub(self):
        gr, x, y = _xy()
        Vars(gr, x, y).Sub()
        self._check(gr, "Sub")

    def test_Tile(self):
        gr, x, y = _xy()
        Vars(gr, x, y).Tile()
        self._check(gr, "Tile")

    def test_Unsqueeze(self):
        gr, x, y = _xy()
        Vars(gr, x, y).Unsqueeze()
        self._check(gr, "Unsqueeze")

    def test_Xor(self):
        gr, x, y = _xy()
        Vars(gr, x, y).Xor()
        self._check(gr, "Xor")

    # ---- flexible ops ----

    def test_Squeeze(self):
        gr, x, y = _xy()
        Vars(gr, x, y).Squeeze()
        self._check(gr, "Squeeze")

    # ---- explicit parameterized multi-input ops ----

    def test_BitShift(self):
        gr, x, y = _xy()
        Vars(gr, x, y).BitShift(direction="LEFT")
        self._check(gr, "BitShift")

    def test_CenterCropPad(self):
        gr, x, y = _xy()
        Vars(gr, x, y).CenterCropPad()
        self._check(gr, "CenterCropPad")

    def test_Clip(self):
        gr, x = _xy()[:2]
        Vars(gr, x).Clip()
        self._check(gr, "Clip")

    def test_Col2Im(self):
        gr = start()
        x = gr.vin("X")
        image_shape = gr.vin("IS")
        block_shape = gr.vin("BS")
        Vars(gr, x, image_shape, block_shape).Col2Im()
        self._check(gr, "Col2Im")

    def test_Compress(self):
        gr, x, y = _xy()
        Vars(gr, x, y).Compress()
        self._check(gr, "Compress")

    def test_Concat(self):
        gr, x, y = _xy()
        Vars(gr, x, y).Concat(axis=0)
        self._check(gr, "Concat")

    def test_Conv(self):
        gr, x, y = _xy()
        Vars(gr, x, y).Conv()
        self._check(gr, "Conv")

    def test_ConvInteger(self):
        gr, x, y = _xy()
        Vars(gr, x, y).ConvInteger()
        self._check(gr, "ConvInteger")

    def test_ConvTranspose(self):
        gr, x, y = _xy()
        Vars(gr, x, y).ConvTranspose()
        self._check(gr, "ConvTranspose")

    def test_CumSum(self):
        gr, x, y = _xy()
        Vars(gr, x, y).CumSum()
        self._check(gr, "CumSum")

    def test_DFT(self):
        gr = start()
        x = gr.vin("X")
        dft_length = gr.vin("L")
        Vars(gr, x, dft_length).DFT()
        self._check(gr, "DFT")

    def test_DeformConv(self):
        gr = start()
        x = gr.vin("X")
        w = gr.vin("W")
        offset = gr.vin("O")
        Vars(gr, x, w, offset).DeformConv()
        self._check(gr, "DeformConv")

    def test_DequantizeLinear(self):
        gr, x, y = _xy()
        Vars(gr, x, y).DequantizeLinear()
        self._check(gr, "DequantizeLinear")

    def test_Einsum(self):
        gr, x, y = _xy()
        Vars(gr, x, y).Einsum(equation="ij,jk->ik")
        self._check(gr, "Einsum")

    def test_Gather(self):
        gr, x, y = _xy()
        Vars(gr, x, y).Gather()
        self._check(gr, "Gather")

    def test_GatherElements(self):
        gr, x, y = _xy()
        Vars(gr, x, y).GatherElements()
        self._check(gr, "GatherElements")

    def test_Gemm(self):
        gr, x, y = _xy()
        Vars(gr, x, y).Gemm()
        self._check(gr, "Gemm")

    def test_GridSample(self):
        gr, x, y = _xy()
        Vars(gr, x, y).GridSample()
        self._check(gr, "GridSample")

    def test_GroupNormalization(self):
        gr = start()
        x = gr.vin("X")
        scale = gr.vin("S")
        bias = gr.vin("B")
        Vars(gr, x, scale, bias).GroupNormalization(num_groups=2)
        self._check(gr, "GroupNormalization")

    def test_InstanceNormalization(self):
        gr = start()
        x = gr.vin("X")
        scale = gr.vin("S")
        bias = gr.vin("B")
        Vars(gr, x, scale, bias).InstanceNormalization()
        self._check(gr, "InstanceNormalization")

    def test_MatMulInteger(self):
        gr, x, y = _xy()
        Vars(gr, x, y).MatMulInteger()
        self._check(gr, "MatMulInteger")

    def test_MaxRoiPool(self):
        gr, x, y = _xy()
        Vars(gr, x, y).MaxRoiPool(pooled_shape=[7, 7])
        self._check(gr, "MaxRoiPool")

    def test_MaxUnpool(self):
        gr = start()
        x = gr.vin("X")
        indices = gr.vin("I")
        Vars(gr, x, indices).MaxUnpool(kernel_shape=[2, 2])
        self._check(gr, "MaxUnpool")

    def test_MelWeightMatrix(self):
        gr = start()
        for name in ["NMB", "NFF", "SR", "LF", "UF"]:
            gr.vin(name)
        inputs = [gr.get_var(n) for n in ["NMB", "NFF", "SR", "LF", "UF"]]
        Vars(gr, *inputs).MelWeightMatrix()
        self._check(gr, "MelWeightMatrix")

    def test_Mod(self):
        gr, x, y = _xy()
        Vars(gr, x, y).Mod()
        self._check(gr, "Mod")

    def test_NegativeLogLikelihoodLoss(self):
        gr, x, y = _xy()
        Vars(gr, x, y).NegativeLogLikelihoodLoss()
        self._check(gr, "NegativeLogLikelihoodLoss")

    def test_NonMaxSuppression(self):
        gr = start()
        boxes = gr.vin("boxes")
        scores = gr.vin("scores")
        Vars(gr, boxes, scores).NonMaxSuppression()
        self._check(gr, "NonMaxSuppression")

    def test_OneHot(self):
        gr = start()
        x = gr.vin("X")
        depth = gr.vin("D")
        values = gr.vin("V")
        Vars(gr, x, depth, values).OneHot()
        self._check(gr, "OneHot")

    def test_Pad(self):
        gr, x, y = _xy()
        Vars(gr, x, y).Pad()
        self._check(gr, "Pad")

    def test_QLinearConv(self):
        gr = start()
        for name in ["X", "XS", "XZP", "W", "WS", "WZP", "YS", "YZP"]:
            gr.vin(name)
        inputs = [gr.get_var(n) for n in ["X", "XS", "XZP", "W", "WS", "WZP", "YS", "YZP"]]
        Vars(gr, *inputs).QLinearConv()
        self._check(gr, "QLinearConv")

    def test_QLinearMatMul(self):
        gr = start()
        for name in ["A", "AS", "AZP", "B", "BS", "BZP", "YS", "YZP"]:
            gr.vin(name)
        inputs = [gr.get_var(n) for n in ["A", "AS", "AZP", "B", "BS", "BZP", "YS", "YZP"]]
        Vars(gr, *inputs).QLinearMatMul()
        self._check(gr, "QLinearMatMul")

    def test_QuantizeLinear(self):
        gr = start()
        x = gr.vin("X")
        scale = gr.vin("S")
        Vars(gr, x, scale).QuantizeLinear()
        self._check(gr, "QuantizeLinear")

    def test_Range(self):
        gr = start()
        for name in ["start", "limit", "delta"]:
            gr.vin(name)
        inputs = [gr.get_var(n) for n in ["start", "limit", "delta"]]
        Vars(gr, *inputs).Range()
        self._check(gr, "Range")

    def test_ReduceL1_vars(self):
        gr, x, y = _xy()
        Vars(gr, x, y).ReduceL1(keepdims=0)
        self._check(gr, "ReduceL1")

    def test_ReduceL2_vars(self):
        gr, x, y = _xy()
        Vars(gr, x, y).ReduceL2(keepdims=0)
        self._check(gr, "ReduceL2")

    def test_ReduceLogSum_vars(self):
        gr, x, y = _xy()
        Vars(gr, x, y).ReduceLogSum(keepdims=0)
        self._check(gr, "ReduceLogSum")

    def test_ReduceLogSumExp_vars(self):
        gr, x, y = _xy()
        Vars(gr, x, y).ReduceLogSumExp(keepdims=0)
        self._check(gr, "ReduceLogSumExp")

    def test_ReduceMax_vars(self):
        gr, x, y = _xy()
        Vars(gr, x, y).ReduceMax(keepdims=0)
        self._check(gr, "ReduceMax")

    def test_ReduceMean_vars(self):
        gr, x, y = _xy()
        Vars(gr, x, y).ReduceMean(keepdims=0)
        self._check(gr, "ReduceMean")

    def test_ReduceMin_vars(self):
        gr, x, y = _xy()
        Vars(gr, x, y).ReduceMin(keepdims=0)
        self._check(gr, "ReduceMin")

    def test_ReduceProd_vars(self):
        gr, x, y = _xy()
        Vars(gr, x, y).ReduceProd(keepdims=0)
        self._check(gr, "ReduceProd")

    def test_ReduceSum_vars(self):
        gr, x, y = _xy()
        Vars(gr, x, y).ReduceSum(keepdims=0)
        self._check(gr, "ReduceSum")

    def test_ReduceSumSquare_vars(self):
        gr, x, y = _xy()
        Vars(gr, x, y).ReduceSumSquare(keepdims=0)
        self._check(gr, "ReduceSumSquare")

    def test_Resize(self):
        gr = start()
        x = gr.vin("X")
        roi = gr.vin("roi")
        scales = gr.vin("scales")
        Vars(gr, x, roi, scales).Resize()
        self._check(gr, "Resize")

    def test_RoiAlign(self):
        gr = start()
        x = gr.vin("X")
        rois = gr.vin("rois")
        batch_idx = gr.vin("batch_idx")
        Vars(gr, x, rois, batch_idx).RoiAlign()
        self._check(gr, "RoiAlign")

    def test_STFT(self):
        gr = start()
        x = gr.vin("X")
        step = gr.vin("step")
        Vars(gr, x, step).STFT()
        self._check(gr, "STFT")

    def test_Scatter(self):
        gr = start()
        x = gr.vin("X")
        indices = gr.vin("I")
        updates = gr.vin("U")
        Vars(gr, x, indices, updates).Scatter()
        self._check(gr, "Scatter")

    def test_ScatterElements(self):
        gr = start()
        x = gr.vin("X")
        indices = gr.vin("I")
        updates = gr.vin("U")
        Vars(gr, x, indices, updates).ScatterElements()
        self._check(gr, "ScatterElements")

    def test_ScatterND(self):
        gr = start()
        x = gr.vin("X")
        indices = gr.vin("I")
        updates = gr.vin("U")
        Vars(gr, x, indices, updates).ScatterND()
        self._check(gr, "ScatterND")

    def test_Slice_vars(self):
        gr = start()
        x = gr.vin("X")
        starts = gr.vin("starts")
        ends = gr.vin("ends")
        Vars(gr, x, starts, ends).Slice()
        self._check(gr, "Slice")

    def test_TopK(self):
        gr, x, y = _xy()
        result = Vars(gr, x, y).TopK()
        self.assertIsInstance(result, Vars)
        self.assertEqual(len(result), 2)
        self._check(gr, "TopK")

    def test_Trilu(self):
        gr, x = _xy()[:2]
        Vars(gr, x).Trilu()
        self._check(gr, "Trilu")

    def test_Upsample(self):
        gr, x, y = _xy()
        Vars(gr, x, y).Upsample()
        self._check(gr, "Upsample")

    def test_Where(self):
        gr = start()
        condition = gr.vin("C")
        x = gr.vin("X")
        y = gr.vin("Y")
        Vars(gr, condition, x, y).Where()
        self._check(gr, "Where")


if __name__ == "__main__":
    unittest.main(verbosity=2)
