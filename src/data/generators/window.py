# src/data/generators/window.py
import math

def half_window(window: int, **kwargs) -> int:
    """Compute floor(window/2)."""
    return window // 2

def sqrt_window(window: int, **kwargs) -> int:
    """Compute floor(sqrt(window))."""
    return int(math.sqrt(window))
