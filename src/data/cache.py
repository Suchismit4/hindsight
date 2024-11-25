from pathlib import Path
from typing import Optional
import shutil
import os

class CacheManager:
    def __init__(self, cache_root: Optional[str] = None):
        self.cache_root = Path(cache_root or get_default_cache_root())
        
    def clear_cache(self, registry_path: Optional[str] = None):
        """Clear specific or all cache entries."""
        if registry_path:
            cache_path = self.cache_root / registry_path.lstrip('/')
            if cache_path.exists():
                if cache_path.is_file():
                    cache_path.unlink()
                else:
                    shutil.rmtree(cache_path)
        else:
            shutil.rmtree(self.cache_root)
            self.cache_root.mkdir(parents=True)
            
    def get_cache_size(self) -> int:
        """Return total cache size in bytes."""
        total = 0
        for dirpath, _, filenames in os.walk(self.cache_root):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
        return total 