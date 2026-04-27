from typing import List


def get_onnxruntime_patterns(verbose: int = 0) -> List["PatternOptimization"]:  # noqa: F821
    """
    Returns a default list of optimization patterns for :epkg:`onnxruntime`.
    It is equal to the following list.

    .. runpython::
        :showcode:
        :rst:

        from yobx.xoptim.patterns_api import pattern_table_doc
        from yobx.xoptim.patterns_ort import get_onnxruntime_patterns

        print(pattern_table_doc(get_onnxruntime_patterns(), as_rst=True))
        print()
    """
    from .activation import (
        BiasGeluPattern,
        BiasSplitGeluPattern,
        BiasSoftmaxPattern,
        FastGeluPattern,
        GeluOrtPattern,
        GeluErfPattern,
        GemmFastGeluPattern,
        QuickGeluPattern,
    )
    from .causal_conv import CausalConvWithStatePattern
    from .complex_mul import ComplexMulPattern, ComplexMulConjPattern
    from .fused_conv import FusedConvPattern
    from .greedy_search import GreedySearchPattern
    from .fused_matmul import (
        FusedMatMulDivPattern,
        FusedMatMulPattern,
        FusedMatMulx2Pattern,
        FusedMatMulTransposePattern,
        FusedMatMulActivationPattern,
        ReshapeGemmPattern,
        ReshapeGemmReshapePattern,
        TransposeFusedMatMulBPattern,
    )
    from .llm_optim import (
        Attention3DPattern,
        ContribGemmaRotaryEmbeddingPattern,
        ContribRotaryEmbeddingPattern,
        ContribRotaryEmbedding3DPattern,
        GroupQueryAttention3DPattern,
        MultiHeadAttention3DPattern,
    )
    from .missing_kernels import (
        MissingCosSinPattern,
        MissingRangePattern,
        MissingReduceMaxPattern,
        MissingTopKPattern,
    )

    from .embed_layer_normalization import EmbedLayerNormalizationPattern
    from .relative_position_bias import (
        RelativePositionBiasPattern,
        GatedRelativePositionBiasPattern,
    )
    from .simplified_layer_normalization import (
        SimplifiedLayerNormalizationPattern,
        SimplifiedLayerNormalizationMulPattern,
        SkipLayerNormalizationPattern,
        SkipSimplifiedLayerNormalizationPattern,
        SkipSimplifiedLayerNormalizationMulPattern,
    )

    return [
        Attention3DPattern(verbose=verbose),
        BiasGeluPattern(verbose=verbose),
        BiasSoftmaxPattern(verbose=verbose),
        BiasSplitGeluPattern(verbose=verbose),
        CausalConvWithStatePattern(verbose=verbose),
        ComplexMulPattern(verbose=verbose),
        ComplexMulConjPattern(verbose=verbose),
        ContribRotaryEmbeddingPattern(verbose=verbose),
        ContribRotaryEmbedding3DPattern(verbose=verbose),
        ContribGemmaRotaryEmbeddingPattern(verbose=verbose),
        EmbedLayerNormalizationPattern(verbose=verbose),
        GeluOrtPattern(verbose=verbose),
        GeluErfPattern(verbose=verbose),
        GroupQueryAttention3DPattern(verbose=verbose),
        FusedConvPattern(verbose=verbose),
        FastGeluPattern(verbose=verbose),
        FusedMatMulPattern(verbose=verbose),
        FusedMatMulActivationPattern(verbose=verbose),
        FusedMatMulx2Pattern(verbose=verbose),
        FusedMatMulDivPattern(verbose=verbose),
        FusedMatMulTransposePattern(verbose=verbose),
        GemmFastGeluPattern(verbose=verbose),
        GatedRelativePositionBiasPattern(verbose=verbose),
        GreedySearchPattern(verbose=verbose),
        MissingCosSinPattern(verbose=verbose),
        MissingRangePattern(verbose=verbose),
        MissingReduceMaxPattern(verbose=verbose),
        MissingTopKPattern(verbose=verbose),
        MultiHeadAttention3DPattern(verbose=verbose),
        QuickGeluPattern(verbose=verbose),
        ReshapeGemmPattern(verbose=verbose),
        ReshapeGemmReshapePattern(verbose=verbose),
        RelativePositionBiasPattern(verbose=verbose),
        SimplifiedLayerNormalizationPattern(verbose=verbose),
        SimplifiedLayerNormalizationMulPattern(verbose=verbose),
        SkipLayerNormalizationPattern(verbose=verbose),
        SkipSimplifiedLayerNormalizationPattern(verbose=verbose),
        SkipSimplifiedLayerNormalizationMulPattern(verbose=verbose),
        TransposeFusedMatMulBPattern(verbose=verbose),
    ]
