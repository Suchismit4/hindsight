"""
Cache metadata management for tracking cache entries and dependencies.

This module provides JSON-based metadata tracking for cached artifacts,
enabling dependency resolution, cache invalidation, and usage statistics.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import glob

from .stages import CacheStage


@dataclass
class CacheMetadata:
    """
    Metadata for a single cache entry.
    
    Attributes:
        cache_key: Unique content-addressable key for this cache entry
        stage: Cache stage (L1, L2, L3, or L4)
        config: Configuration dictionary used to generate this cache
        parent_keys: List of cache keys this entry depends on
        size_bytes: Size of the cached artifact in bytes
        created_at: ISO format timestamp of creation
        last_accessed: ISO format timestamp of last access
        access_count: Number of times this cache has been accessed
    """
    
    cache_key: str
    stage: str
    config: Dict[str, Any]
    parent_keys: List[str] = field(default_factory=list)
    size_bytes: int = 0
    created_at: str = ""
    last_accessed: str = ""
    access_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheMetadata':
        """Create metadata from dictionary loaded from JSON."""
        return cls(**data)


class MetadataManager:
    """
    Manager for cache metadata operations.
    
    Handles saving, loading, and querying metadata for cached artifacts.
    Each cached artifact has a corresponding .meta.json file with its metadata.
    
    Attributes:
        cache_root: Root directory for all cache storage
    """
    
    def __init__(self, cache_root: Path):
        """
        Initialize the metadata manager.
        
        Args:
            cache_root: Root directory for cache storage
        """
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
    
    def _get_metadata_path(self, cache_key: str, stage: CacheStage) -> Path:
        """
        Get the path to the metadata file for a cache entry.
        
        Args:
            cache_key: Cache key
            stage: Cache stage
            
        Returns:
            Path to the .meta.json file
        """
        stage_dir = self.cache_root / stage.value
        stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir / f"{cache_key}.meta.json"
    
    def save_metadata(
        self, 
        cache_key: str, 
        stage: CacheStage,
        config: Dict[str, Any],
        parent_keys: List[str],
        size_bytes: int
    ) -> None:
        """
        Save metadata for a cache entry.
        
        Args:
            cache_key: Unique cache key
            stage: Cache stage
            config: Configuration used to generate this cache
            parent_keys: List of parent cache keys
            size_bytes: Size of cached artifact in bytes
        """
        now = datetime.now().isoformat()
        
        metadata = CacheMetadata(
            cache_key=cache_key,
            stage=stage.value,
            config=config,
            parent_keys=parent_keys,
            size_bytes=size_bytes,
            created_at=now,
            last_accessed=now,
            access_count=1
        )
        
        metadata_path = self._get_metadata_path(cache_key, stage)
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
    
    def load_metadata(self, cache_key: str, stage: CacheStage) -> Optional[CacheMetadata]:
        """
        Load metadata for a cache entry.
        
        Args:
            cache_key: Cache key to load
            stage: Cache stage
            
        Returns:
            CacheMetadata if found, None otherwise
        """
        metadata_path = self._get_metadata_path(cache_key, stage)
        
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                data = json.load(f)
            return CacheMetadata.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load metadata from {metadata_path}: {e}")
            return None
    
    def update_access(self, cache_key: str, stage: CacheStage) -> None:
        """
        Update access timestamp and count for a cache entry.
        
        Args:
            cache_key: Cache key
            stage: Cache stage
        """
        metadata = self.load_metadata(cache_key, stage)
        if metadata is None:
            return
        
        metadata.last_accessed = datetime.now().isoformat()
        metadata.access_count += 1
        
        metadata_path = self._get_metadata_path(cache_key, stage)
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
    
    def find_dependents(self, cache_key: str) -> List[tuple[str, CacheStage]]:
        """
        Find all cache entries that depend on the given cache key.
        
        This searches through all metadata files to find entries that list
        the given key as a parent.
        
        Args:
            cache_key: Cache key to find dependents for
            
        Returns:
            List of (dependent_cache_key, stage) tuples
        """
        dependents = []
        
        # Search through all stages
        for stage in CacheStage:
            stage_dir = self.cache_root / stage.value
            if not stage_dir.exists():
                continue
            
            # Find all metadata files in this stage
            for metadata_path in stage_dir.glob("*.meta.json"):
                try:
                    with open(metadata_path, 'r') as f:
                        data = json.load(f)
                    
                    # Check if this entry depends on the given key
                    parent_keys = data.get('parent_keys', [])
                    if cache_key in parent_keys:
                        dependents.append((data['cache_key'], stage))
                        
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Failed to read metadata from {metadata_path}: {e}")
                    continue
        
        return dependents
    
    def list_all_entries(self, stage: Optional[CacheStage] = None) -> List[CacheMetadata]:
        """
        List all cache entries, optionally filtered by stage.
        
        Args:
            stage: Optional stage to filter by. If None, returns all entries.
            
        Returns:
            List of CacheMetadata objects
        """
        entries = []
        
        stages_to_search = [stage] if stage else list(CacheStage)
        
        for search_stage in stages_to_search:
            stage_dir = self.cache_root / search_stage.value
            if not stage_dir.exists():
                continue
            
            for metadata_path in stage_dir.glob("*.meta.json"):
                try:
                    with open(metadata_path, 'r') as f:
                        data = json.load(f)
                    entries.append(CacheMetadata.from_dict(data))
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Failed to read metadata from {metadata_path}: {e}")
                    continue
        
        return entries
    
    def delete_metadata(self, cache_key: str, stage: CacheStage) -> bool:
        """
        Delete metadata file for a cache entry.
        
        Args:
            cache_key: Cache key
            stage: Cache stage
            
        Returns:
            True if deleted, False if not found
        """
        metadata_path = self._get_metadata_path(cache_key, stage)
        
        if metadata_path.exists():
            metadata_path.unlink()
            return True
        
        return False

