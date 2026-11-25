"""Image loading, resizing and grid division."""
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .utils import as_grid


def load_image(source):
    """Load an image as an RGB ``uint8`` array.

    Args:
        source (str | pathlib.Path | PIL.Image.Image | np.ndarray): A file
            path, a PIL image, or a ``uint8`` array that is grayscale
            ``(H, W)``, RGB ``(H, W, 3)`` or RGBA ``(H, W, 4)``.

    Returns:
        np.ndarray: Image of shape ``(H, W, 3)``, dtype ``uint8``. An input
        that is already an RGB ``uint8`` array is returned as is.

    Raises:
        ValueError: If there is no image, or it can't be read or converted.
        FileNotFoundError: If ``source`` is a path that doesn't exist.

    Example:
        >>> load_image("test_images/astronaut_512.png").shape
        (512, 512, 3)
    """
    if source is None:
        raise ValueError("No image provided. Upload an image first.")
    if isinstance(source, (str, Path)):
        try:
            with Image.open(source) as img:
                return np.array(img.convert("RGB"))
        except UnidentifiedImageError:
            raise ValueError(f"Can't read {source} as an image.") from None
    if isinstance(source, Image.Image):
        return np.array(source.convert("RGB"))
    array = np.asarray(source)
    if array.dtype != np.uint8:
        raise ValueError(f"Image arrays must be uint8, got {array.dtype}.")
    if array.ndim == 3 and array.shape[2] == 3:
        return array
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim == 2 or (array.ndim == 3 and array.shape[2] == 4):
        return np.array(Image.fromarray(array).convert("RGB"))
    raise ValueError(f"Expected a grayscale, RGB or RGBA image, got an array of shape {array.shape}.")


def validate_grid(image, grid_size):
    """Check that a grid fits an image, with every cell at least one pixel.

    Args:
        image (np.ndarray): Image of shape ``(H, W, 3)``.
        grid_size (int | tuple[int, int]): Cells down and across.

    Returns:
        tuple[int, int]: The grid as ``(rows, cols)``.

    Raises:
        ValueError: If the grid size is invalid or larger than the image.
    """
    rows, cols = as_grid(grid_size)
    height, width = image.shape[:2]
    if rows > height or cols > width:
        raise ValueError(
            f"A {rows}x{cols} grid needs an image at least {cols}x{rows} pixels; this one is {width}x{height}."
        )
    return rows, cols


def crop_to_grid(image, grid_size):
    """Crop an image so its height and width divide evenly into the grid.

    Args:
        image (np.ndarray): Image of shape ``(H, W, 3)``.
        grid_size (int | tuple[int, int]): Cells down and across.

    Returns:
        np.ndarray: The top-left part of ``image`` (a view, not a copy) whose
        size is an exact multiple of the cell size.

    Raises:
        ValueError: If the grid doesn't fit the image.
    """
    rows, cols = validate_grid(image, grid_size)
    cell_h, cell_w = image.shape[0] // rows, image.shape[1] // cols
    return image[:cell_h * rows, :cell_w * cols]


def cell_averages(image, grid_size):
    """Average color of every grid cell, using one box-filter downscale.

    Args:
        image (np.ndarray): Image of shape ``(H, W, 3)``, dtype ``uint8``.
        grid_size (int | tuple[int, int]): Cells down and across.

    Returns:
        tuple[np.ndarray, np.ndarray]: The image cropped to the grid, and the
        cell colors as a ``(rows, cols, 3)`` ``uint8`` array (rounded to whole
        numbers).

    Raises:
        ValueError: If the grid doesn't fit the image.

    Example:
        >>> image = load_image("test_images/astronaut_512.png")
        >>> cell_averages(image, 32)[1].shape
        (32, 32, 3)
    """
    cropped = crop_to_grid(image, grid_size)
    rows, cols = as_grid(grid_size)
    cell_h, cell_w = cropped.shape[0] // rows, cropped.shape[1] // cols
    colors = np.asarray(Image.fromarray(cropped).reduce((cell_w, cell_h)))
    return cropped, colors


def resize(image, size, resample=Image.BICUBIC):
    """Resize an image.

    Args:
        image (np.ndarray): Image of shape ``(H, W, 3)``, dtype ``uint8``.
        size (tuple[int, int]): Target ``(width, height)``.
        resample (int): PIL resampling filter, e.g. ``Image.NEAREST``.

    Returns:
        np.ndarray: Resized image of shape ``(height, width, 3)``.
    """
    return np.asarray(Image.fromarray(image).resize(size, resample))


def blend(mosaic, original, alpha):
    """Blend the original image into a mosaic for display.

    Args:
        mosaic (np.ndarray): Mosaic of shape ``(H, W, 3)``, dtype ``uint8``.
        original (np.ndarray): Image the mosaic was built from; it is resized
            to the mosaic's size.
        alpha (float): Share of the original in the result, from 0 to 1.

    Returns:
        np.ndarray: Blended image with the mosaic's shape.

    Raises:
        ValueError: If ``alpha`` is outside 0 to 1.
    """
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}.")
    original = resize(original, (mosaic.shape[1], mosaic.shape[0]))
    return np.asarray(Image.blend(Image.fromarray(mosaic), Image.fromarray(original), alpha))
