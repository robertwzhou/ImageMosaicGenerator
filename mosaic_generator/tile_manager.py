"""Tile loading, caching and feature extraction."""
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from . import config

logger = logging.getLogger(__name__)


class TileManager:
    """Loads every tile in a folder once and keeps the tiles and their colors in memory.

    Tiles are read in filename order, so ``tiles[:n]`` are always the same n
    tiles. Files that can't be read are skipped with a warning and listed in
    ``skipped``.

    Args:
        tile_directory (str | pathlib.Path): Folder of tile images.
        tile_size (int): Edge length in pixels; tiles are resized to
            ``tile_size`` x ``tile_size``.

    Attributes:
        tiles (np.ndarray): ``(N, tile_size, tile_size, 3)`` ``uint8`` tile images.
        colors (np.ndarray): ``(N, 3)`` ``float32`` representative color of each tile.
        skipped (list[str]): Names of files that couldn't be read.

    Raises:
        FileNotFoundError: If ``tile_directory`` doesn't exist.
        ValueError: If it contains no readable tile images.

    Example:
        >>> tile_manager = TileManager("tiles/")
        >>> len(tile_manager), tile_manager.tiles.shape
        (32, (32, 20, 20, 3))
    """

    def __init__(self, tile_directory=config.TILE_DIRECTORY, tile_size=config.TILE_SIZE):
        """Load every tile in ``tile_directory``; arguments are described in the class docstring."""
        directory = Path(tile_directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Tile folder not found: {directory}")
        tiles, self.skipped = [], []
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in config.TILE_EXTENSIONS:
                continue
            try:
                with Image.open(path) as img:
                    tiles.append(np.array(img.convert("RGB").resize((tile_size, tile_size))))
            except OSError as err:
                logger.warning("Skipping unreadable tile %s: %s", path.name, err)
                self.skipped.append(path.name)
        if not tiles:
            raise ValueError(f"No readable tile images ({', '.join(config.TILE_EXTENSIONS)}) in {directory}")
        self.tiles = np.stack(tiles)
        self.colors = self.tile_colors(self.tiles)

    def __len__(self):
        """Number of loaded tiles.

        Returns:
            int: Number of tiles.
        """
        return len(self.tiles)

    @staticmethod
    def tile_colors(tiles):
        """Representative color of each tile: its per-channel median.

        The median ignores small details such as a tile's border, so a
        solid-color tile with a dark edge is represented by its fill color.

        Args:
            tiles (np.ndarray): ``(N, H, W, 3)`` tile images.

        Returns:
            np.ndarray: ``(N, 3)`` ``float32`` colors.
        """
        return np.median(tiles, axis=(1, 2)).astype(np.float32)
