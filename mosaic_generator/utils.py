"""Helper functions shared by the other modules."""
import time
from contextlib import contextmanager


def as_grid(grid_size):
    """Normalize a grid size to a ``(rows, cols)`` tuple.

    Args:
        grid_size (int | tuple[int, int]): Cells down and across. A single
            number means a square grid.

    Returns:
        tuple[int, int]: ``(rows, cols)``.

    Raises:
        ValueError: If a value isn't a positive whole number.

    Example:
        >>> as_grid(32)
        (32, 32)
    """
    values = (grid_size, grid_size) if not isinstance(grid_size, (tuple, list)) else tuple(grid_size)
    if len(values) != 2:
        raise ValueError(f"Grid size must be a number or a (rows, cols) pair, got {grid_size!r}.")
    try:
        rows, cols = (int(v) for v in values)
    except (TypeError, ValueError):
        raise ValueError(f"Grid size must be whole numbers, got {grid_size!r}.") from None
    if (rows, cols) != values or rows < 1 or cols < 1:
        raise ValueError(f"Grid size must be positive whole numbers, got {grid_size!r}.")
    return rows, cols


@contextmanager
def timer():
    """Measure the wall-clock time of a ``with`` block.

    Yields:
        dict: Empty while the block runs; holds ``{"seconds": float}`` once it exits.

    Example:
        >>> with timer() as t:
        ...     total = sum(range(1000))
        >>> t["seconds"] >= 0
        True
    """
    result = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start
