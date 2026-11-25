import gradio as gr
import numpy as np
from PIL import Image
from functools import lru_cache
import os
import time

# ==========================================
# 1. TILE SET (built once per palette size, then cached)
# ==========================================
@lru_cache(maxsize=None)
def build_tiles(num_tiles, tile_size=20):
    """Lab 1's synthetic tile set. Returns (tile_array, base_colors)."""
    np.random.seed(42)
    base_colors = np.random.randint(0, 255, size=(num_tiles, 3), dtype=np.uint8)
    tile_array = np.zeros((num_tiles, tile_size, tile_size, 3), dtype=np.uint8)
    for i in range(num_tiles):
        tile_array[i, :, :] = base_colors[i]
        tile_array[i, 0, :] = 20
        tile_array[i, :, 0] = 20
    return tile_array, base_colors

# ==========================================
# 2. OPTIMIZED ALGORITHM (no Python loops)
# ==========================================
def cell_averages(img_array, grid_size):
    """Crop to a multiple of the grid, then average each cell with one box-filter downscale."""
    height, width, _ = img_array.shape
    cell_h, cell_w = height // grid_size, width // grid_size
    cropped = img_array[:cell_h * grid_size, :cell_w * grid_size]
    avg_colors = np.asarray(Image.fromarray(cropped).reduce((cell_w, cell_h)))
    return cropped, avg_colors


def match_tiles(avg_colors, base_colors):
    """Index of the nearest tile color for every cell, as a (grid, grid) array."""
    # |a - t|^2 = |a|^2 - 2 a.t + |t|^2. |a|^2 is the same for every tile,
    # so the nearest tile is the one that minimizes |t|^2 - 2 a.t.
    a = avg_colors.reshape(-1, 3).astype(np.float32)
    t = base_colors.astype(np.float32)
    best = ((t ** 2).sum(axis=1) - 2 * a @ t.T).argmin(axis=1)
    return best.reshape(avg_colors.shape[:2])


def assemble(best, tile_array):
    """Gather the chosen tile for every cell and lay them out as one image."""
    rows, cols = best.shape
    tile_size = tile_array.shape[1]
    return tile_array[best].transpose(0, 2, 1, 3, 4).reshape(rows * tile_size, cols * tile_size, 3)


def optimized_mosaic(img_array, grid_size, tile_array, base_colors):
    cropped, avg_colors = cell_averages(img_array, grid_size)
    best = match_tiles(avg_colors, base_colors)
    return assemble(best, tile_array), cropped, avg_colors

# ==========================================
# 3. PIPELINE & GRADIO INTERFACE
# ==========================================
def process_image(input_img, grid_size, num_tiles):
    img_array = np.array(input_img, dtype=np.uint8)
    grid_size = int(grid_size)
    tile_array, base_colors = build_tiles(int(num_tiles))
    tile_size = tile_array.shape[1]

    start = time.perf_counter()
    mosaic, cropped_orig, avg_colors = optimized_mosaic(img_array, grid_size, tile_array, base_colors)
    elapsed = time.perf_counter() - start

    # Format Outputs
    canvas_size = grid_size * tile_size
    segmented_preview = Image.fromarray(avg_colors, 'RGB').resize((canvas_size, canvas_size), Image.NEAREST)

    mosaic_img = Image.fromarray(mosaic)
    original_resized = Image.fromarray(cropped_orig).resize((canvas_size, canvas_size))
    final_artistic_mosaic = Image.blend(mosaic_img, original_resized, alpha=0.25)

    # Calculate MSE Metric
    orig_resized_arr = np.array(original_resized, dtype=np.float32)
    result_arr = np.array(final_artistic_mosaic, dtype=np.float32)
    mse_value = np.mean((orig_resized_arr - result_arr) ** 2)

    report = (
        f"--- Performance Metric ---\n"
        f"Mean Squared Error (MSE): {mse_value:.2f}\n\n"
        f"--- Computational Performance ---\n"
        f"Grid Size: {grid_size}x{grid_size}\n"
        f"Mosaic Processing Time: {elapsed * 1000:.2f} ms"
    )

    return segmented_preview, final_artistic_mosaic, report

demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="numpy", label="Upload Test Image"),
        gr.Slider(minimum=16, maximum=64, step=16, value=32, label="Grid Size (NxN)"),
        gr.Slider(minimum=2, maximum=32, step=2, value=8, label="Tile Set Palette Size")
    ],
    outputs=[
        gr.Image(label="1. Segmented Grid Preview"),
        gr.Image(label="2. Final Blended Mosaic"),
        gr.Textbox(label="Assignment Report & Metrics", lines=9)
    ],
    title="Interactive Image Mosaic Generator",
    description="Optimized mosaic generator with processing time and MSE metrics."
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
