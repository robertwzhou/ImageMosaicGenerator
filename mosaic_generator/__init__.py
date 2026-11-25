"""Photomosaic generator: rebuild an image out of tiles from a folder.

Example:
    >>> from mosaic_generator import MosaicBuilder, TileManager
    >>> tile_manager = TileManager(tile_directory="tiles/")
    >>> builder = MosaicBuilder(tile_manager, grid_size=(32, 32))
    >>> mosaic = builder.create_mosaic("test_images/astronaut_512.png")
    >>> similarity = builder.compute_similarity("test_images/astronaut_512.png", mosaic)
"""
from .mosaic_builder import MosaicBuilder
from .tile_manager import TileManager

__all__ = ["MosaicBuilder", "TileManager"]
