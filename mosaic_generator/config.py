"""Configuration constants for the mosaic generator."""

# Tiles
TILE_DIRECTORY = "tiles"
TILE_SIZE = 20  # every tile is resized to TILE_SIZE x TILE_SIZE pixels
TILE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")

# Grid size (cells across and down)
DEFAULT_GRID_SIZE = 32
GRID_SIZE_MIN, GRID_SIZE_MAX, GRID_SIZE_STEP = 16, 64, 16  # web app slider

# Palette size (how many of the tiles to use)
DEFAULT_NUM_TILES = 8
NUM_TILES_MIN, NUM_TILES_MAX, NUM_TILES_STEP = 2, 32, 2  # web app slider

# Display
BLEND_ALPHA = 0.25  # share of the original image blended into the displayed mosaic
