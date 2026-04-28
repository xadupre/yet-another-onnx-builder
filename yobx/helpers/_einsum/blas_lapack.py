import numpy


def gemm_dot(
    A: numpy.ndarray, B: numpy.ndarray, transA: bool = False, transB: bool = False
) -> numpy.ndarray:
    """
    Implements a dot product using numpy matmul.

    :param A: first matrix (2-D)
    :param B: second matrix (2-D)
    :param transA: transposes *A* before the multiplication
    :param transB: transposes *B* before the multiplication
    :return: output matrix
    """
    assert (
        A.dtype == B.dtype
    ), f"Matrices A and B must have the same dtype not {A.dtype!r}, {B.dtype!r}."
    assert len(A.shape) == 2, f"Matrix A does not have 2 dimensions but {len(A.shape)}."
    assert len(B.shape) == 2, f"Matrix B does not have 2 dimensions but {len(B.shape)}."

    if transA:
        A = A.T
    if transB:
        B = B.T
    return A @ B
