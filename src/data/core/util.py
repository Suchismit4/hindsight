"""
Legacy compatibility re-exports for former `data.core.util` API.

Implementations now live in focused modules:
- `src.data.core.types`
- `src.data.core.jit`
- `src.data.core.rolling`
- `src.data.loaders.table`
"""

from src.data.core.jit import prepare_for_jit, restore_from_jit
from src.data.core.rolling import Rolling
from src.data.core.types import FrequencyType, TimeSeriesIndex
from src.data.loaders.table import Loader

# Keep this shim so old imports continue to work while internals stay split.
__all__ = [
    "FrequencyType",
    "TimeSeriesIndex",
    "Loader",
    "Rolling",
    "prepare_for_jit",
    "restore_from_jit",
]
