"""Gradio web interface for the mosaic generator."""
import os

import gradio as gr
from PIL import Image

from mosaic_generator import MosaicBuilder, TileManager
from mosaic_generator import config, image_processor
from mosaic_generator.utils import timer

TILE_MANAGER = TileManager(config.TILE_DIRECTORY)  # tiles are loaded once, at startup


def process_image(input_img, grid_size, num_tiles):
    """Build a mosaic from an uploaded image and report its metrics.

    Args:
        input_img (np.ndarray | None): The uploaded image.
        grid_size (int): Cells across and down.
        num_tiles (int): Palette size: how many of the tiles to use.

    Returns:
        tuple[np.ndarray, np.ndarray, str]: The cell-color preview, the
        blended mosaic, and the metrics report.

    Raises:
        gradio.Error: With a readable message if the input is invalid.
    """
    try:
        image = image_processor.load_image(input_img)
        builder = MosaicBuilder(TILE_MANAGER, grid_size=int(grid_size), num_tiles=int(num_tiles))
        with timer() as elapsed:
            mosaic = builder.create_mosaic(image)
        cropped, cell_colors = image_processor.cell_averages(image, builder.grid_size)
        preview = image_processor.resize(cell_colors, (mosaic.shape[1], mosaic.shape[0]), Image.NEAREST)
        blended = image_processor.blend(mosaic, cropped, config.BLEND_ALPHA)
        similarity = builder.compute_similarity(image, blended)
    except (ValueError, FileNotFoundError) as err:
        raise gr.Error(str(err))

    rows, cols = builder.grid_size
    report = (
        f"--- Performance Metric ---\n"
        f"Mean Squared Error (MSE): {similarity['mse']:.2f}\n"
        f"Structural Similarity (SSIM): {similarity['ssim']:.3f}\n\n"
        f"--- Computational Performance ---\n"
        f"Grid Size: {rows}x{cols}\n"
        f"Mosaic Processing Time: {elapsed['seconds'] * 1000:.2f} ms"
    )
    return preview, blended, report


demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="numpy", label="Upload Test Image"),
        gr.Slider(minimum=config.GRID_SIZE_MIN, maximum=config.GRID_SIZE_MAX, step=config.GRID_SIZE_STEP,
                  value=config.DEFAULT_GRID_SIZE, label="Grid Size (NxN)"),
        gr.Slider(minimum=config.NUM_TILES_MIN, maximum=config.NUM_TILES_MAX, step=config.NUM_TILES_STEP,
                  value=config.DEFAULT_NUM_TILES, label="Tile Set Palette Size"),
    ],
    outputs=[
        gr.Image(label="1. Segmented Grid Preview"),
        gr.Image(label="2. Final Blended Mosaic"),
        gr.Textbox(label="Assignment Report & Metrics", lines=9),
    ],
    examples=[
        ["test_images/astronaut_512.png", 32, 8],
        ["test_images/astronaut_1024.png", 64, 32],
    ],
    title="Interactive Image Mosaic Generator",
    description="Optimized mosaic generator with processing time and similarity metrics.",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
