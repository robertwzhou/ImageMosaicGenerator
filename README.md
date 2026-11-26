# Image Mosaic Generator

Rebuilds a photo out of small tiles. The image is split into a grid, each cell is matched to the tile with the closest color, and the chosen tiles are assembled into a mosaic. For Lab 8, the Lab 1 version was profiled, made 93× faster at 512×512 with a 32×32 grid, and split into the `mosaic_generator` package.

**Live demo:** https://imagemosaicgenerator.onrender.com (free hosting: the first visit after a quiet spell can take about 30 seconds while the app wakes up)

## Installation

Requires Python 3.12.

```sh
git clone https://github.com/robertwzhou/ImageMosaicGenerator.git
cd ImageMosaicGenerator
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt       # the app
pip install -r requirements-dev.txt   # also Jupyter and line_profiler, for the notebook
```

With [uv](https://docs.astral.sh/uv/): `uv venv && uv pip install -r requirements-dev.txt`.

## Usage

### Web app

```sh
python app.py
```

Open http://localhost:7860, upload an image (or click an example), and choose a grid size and palette size. The app shows each cell's average color, the mosaic blended with 25% of the original, the MSE and SSIM against the original, and the mosaic processing time.

### As a library

```python
from mosaic_generator import MosaicBuilder, TileManager

tile_manager = TileManager(tile_directory="tiles/")  # loads and caches the tiles once
builder = MosaicBuilder(tile_manager, grid_size=(32, 32))

mosaic = builder.create_mosaic("test_images/astronaut_512.png")  # (640, 640, 3) uint8 array
similarity = builder.compute_similarity("test_images/astronaut_512.png", mosaic)  # {"mse": ..., "ssim": ...}
```

`create_mosaic` accepts a file path, a PIL image or a NumPy array. Pass `num_tiles=8` to `MosaicBuilder` to use only the first 8 tiles. Invalid input, such as no image, a grid larger than the image or a missing tile folder, raises `ValueError` or `FileNotFoundError` with a message saying what's wrong. Unreadable tile files are skipped with a warning.

## Performance

The baseline is Lab 1's loop-based `loop_based_mosaic`, copied unchanged into `baseline/`. Times are on an Apple M1 with 32 tiles: the mean of 5 runs for the baseline and 50 for the optimized version. Details are in [profiling_analysis.ipynb](profiling_analysis.ipynb).

| Image size | Grid size | Baseline time (s) | Lab 8 time (s) | Speedup |
|---|---|---|---|---|
| 256×256 | 16×16 | 0.0183 | 0.00023 | 80× |
| 512×512 | 32×32 | 0.0733 | 0.00079 | 93× |
| 1024×1024 | 64×64 | 0.2995 | 0.00321 | 93× |

The optimized mosaics use the same tile as the baseline in 98–99% of cells.

What changed:
- **Cell averages:** one box-filter downscale with PIL, instead of a `np.mean` call per cell.
- **Tile matching:** one matrix product for all cells, instead of a `np.linalg.norm` call per cell per tile.
- **Assembly:** one index-and-reshape, instead of pasting tiles one at a time.
- **Caching:** tiles are loaded once at startup, and the app no longer runs the slow version on every request.

## Project structure

```
mosaic_generator/
├── __init__.py            # exports MosaicBuilder, TileManager
├── image_processor.py     # image loading, resizing, grid division, cell averages
├── tile_manager.py        # TileManager: tile loading, caching, colors
├── mosaic_builder.py      # MosaicBuilder: tile matching and assembly
├── metrics.py             # MSE and SSIM
├── config.py              # constants: tile size, grid and palette ranges
└── utils.py               # grid-size parsing, timer
app.py                     # Gradio web app
baseline/                  # Lab 1 code, frozen as the performance baseline
tiles/                     # the 32 tiles (Lab 1's palette, saved as PNGs)
test_images/               # test and example images (scikit-image's astronaut photo)
results/                   # timing data, scaling plot, profiler output
profiling_analysis.ipynb   # profiling (Part 1) and optimization results (Part 2)
```
