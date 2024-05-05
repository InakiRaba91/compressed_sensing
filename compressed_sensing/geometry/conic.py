from typing import Optional

import numpy as np


def check_symmetric_and_non_degenerate(mat: np.ndarray, tol: float, ndim: Optional[int] = None) -> bool:
    """
    Helper method to check whether a matrix is:
    - symmetric i.e. elements above and below the diagonal are equal
    - non degenerate i.e. the matrix determinant is non zero

    Args:
        mat (np.ndarray): Matrix to check.
        tol (float): Tolerance within which to check zero equivalence.
        ndim (Optional[int]): Dimensionality to check for in square matrix. Defaults to None.

    Returns:
        bool: Result if matrix is non degenerate and symmetric.
    """
    is_square = mat.shape[0] == mat.shape[1]

    if ndim is not None:
        is_square = is_square and (mat.shape[0] == ndim)

    if not is_square:
        return False

    symmetric_matrix = np.isclose(mat, mat.T).all()
    non_degenerate_matrix = np.abs(np.linalg.det(mat)) > tol

    return is_square and symmetric_matrix and non_degenerate_matrix  # type: ignore
