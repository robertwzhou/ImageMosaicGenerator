"""Similarity metrics between two images of the same size."""
import numpy as np
from skimage.metrics import structural_similarity


def _check_same_shape(a, b):
    """Raise if two images differ in shape.

    Args:
        a (np.ndarray): First image.
        b (np.ndarray): Second image.

    Raises:
        ValueError: If the shapes differ.
    """
    if a.shape != b.shape:
        raise ValueError(f"Images must have the same shape, got {a.shape} and {b.shape}.")


def mse(a, b):
    """Mean squared error between two images. 0 means identical.

    Args:
        a (np.ndarray): First image, ``(H, W, 3)``.
        b (np.ndarray): Second image, same shape.

    Returns:
        float: Mean of the squared pixel differences.

    Raises:
        ValueError: If the shapes differ.

    Example:
        >>> import numpy as np
        >>> mse(np.zeros((2, 2, 3), np.uint8), np.full((2, 2, 3), 2, np.uint8))
        4.0
    """
    _check_same_shape(a, b)
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def ssim(a, b):
    """Structural similarity (SSIM) between two RGB ``uint8`` images. 1.0 means identical.

    Args:
        a (np.ndarray): First image, ``(H, W, 3)``, dtype ``uint8``.
        b (np.ndarray): Second image, same shape.

    Returns:
        float: SSIM, from -1 to 1.

    Raises:
        ValueError: If the shapes differ.
    """
    _check_same_shape(a, b)
    return float(structural_similarity(a, b, channel_axis=-1, data_range=255))
