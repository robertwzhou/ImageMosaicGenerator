"""Mosaic construction: match every grid cell to its nearest tile and assemble the result."""
import numpy as np

from . import config, image_processor, metrics
from .utils import as_grid


def match_tiles(cell_colors, tile_colors):
    """Index of the nearest tile color for every cell.

    Args:
        cell_colors (np.ndarray): ``(rows, cols, 3)`` average color of each cell.
        tile_colors (np.ndarray): ``(N, 3)`` representative color of each tile.

    Returns:
        np.ndarray: ``(rows, cols)`` array of tile indices.
    """
    # |c - t|^2 = |c|^2 - 2 c.t + |t|^2. |c|^2 is the same for every tile,
    # so the nearest tile is the one that minimizes |t|^2 - 2 c.t.
    c = cell_colors.reshape(-1, 3).astype(np.float32)
    t = tile_colors.astype(np.float32)
    best = ((t ** 2).sum(axis=1) - 2 * c @ t.T).argmin(axis=1)
    return best.reshape(cell_colors.shape[:2])


def assemble(best, tiles):
    """Lay out the chosen tile for every cell as one image.

    Args:
        best (np.ndarray): ``(rows, cols)`` tile index for each cell.
        tiles (np.ndarray): ``(N, T, T, 3)`` tile images.

    Returns:
        np.ndarray: Mosaic of shape ``(rows * T, cols * T, 3)``, dtype ``uint8``.
    """
    rows, cols = best.shape
    size = tiles.shape[1]
    return tiles[best].transpose(0, 2, 1, 3, 4).reshape(rows * size, cols * size, 3)


class MosaicBuilder:
    """Builds mosaics from a TileManager's tiles.

    Args:
        tile_manager (TileManager): Loaded tiles and their colors.
        grid_size (int | tuple[int, int]): Cells down and across.
        num_tiles (int | None): Use only the first ``num_tiles`` tiles; None uses all.

    Raises:
        ValueError: If ``grid_size`` or ``num_tiles`` is invalid.

    Example:
        >>> builder = MosaicBuilder(TileManager("tiles/"), grid_size=(32, 32))
        >>> mosaic = builder.create_mosaic("test_images/astronaut_512.png")
        >>> mosaic.shape
        (640, 640, 3)
    """

    def __init__(self, tile_manager, grid_size=config.DEFAULT_GRID_SIZE, num_tiles=None):
        """Set the grid and pick the tiles to use; arguments are described in the class docstring."""
        self.grid_size = as_grid(grid_size)
        available = len(tile_manager)
        count = available if num_tiles is None else num_tiles
        if not isinstance(count, (int, np.integer)) or not 1 <= count <= available:
            raise ValueError(f"num_tiles must be a whole number from 1 to {available}, got {num_tiles!r}.")
        self.tiles = tile_manager.tiles[:count]
        self.tile_colors = tile_manager.colors[:count]

    def create_mosaic(self, image):
        """Build a mosaic of an image.

        Args:
            image (str | pathlib.Path | PIL.Image.Image | np.ndarray): The image
                (see ``image_processor.load_image``).

        Returns:
            np.ndarray: Mosaic of shape ``(rows * T, cols * T, 3)``, dtype ``uint8``.

        Raises:
            ValueError: If the image can't be read or the grid doesn't fit it.
        """
        image = image_processor.load_image(image)
        _, cell_colors = image_processor.cell_averages(image, self.grid_size)
        return assemble(match_tiles(cell_colors, self.tile_colors), self.tiles)

    def compute_similarity(self, image, mosaic):
        """Compare a mosaic with the image it was built from.

        The image is cropped to the grid and resized to the mosaic's size first.

        Args:
            image (str | pathlib.Path | PIL.Image.Image | np.ndarray): The original image.
            mosaic (np.ndarray): A mosaic (or blended mosaic) built from it.

        Returns:
            dict[str, float]: ``{"mse": ..., "ssim": ...}``.

        Raises:
            ValueError: If either image can't be read or the grid doesn't fit.
        """
        image = image_processor.load_image(image)
        mosaic = image_processor.load_image(mosaic)
        cropped = image_processor.crop_to_grid(image, self.grid_size)
        reference = image_processor.resize(cropped, (mosaic.shape[1], mosaic.shape[0]))
        return {"mse": metrics.mse(reference, mosaic), "ssim": metrics.ssim(reference, mosaic)}
