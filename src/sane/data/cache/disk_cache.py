"""
Disk-based cache implementation for PyTorch tensors.

This module provides a disk-based caching system that can store and retrieve
PyTorch tensors efficiently, with memory management features.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Union
import torch

logger = logging.getLogger(__name__)


class DiskCache:
    """
    A disk-based cache for PyTorch tensors.

    This cache stores tensors to disk to reduce memory usage. All data is read
    from and written to disk on each access. Writes happen only during preload
    (main thread), so no locking is required.

    Features:
    - Cache integrity validation

    Args:
        cache_dir: Directory to store cache files
        create_dirs: Whether to create cache directories if they don't exist
    """

    def __init__(self, cache_dir: Union[str, Path], create_dirs: bool = True):
        self.cache_dir = Path(cache_dir).expanduser().resolve()

        # Create cache directory if it doesn't exist
        if create_dirs:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except (PermissionError, OSError) as exc:
                raise RuntimeError(
                    f"Cannot create the token cache at {self.cache_dir}: {exc}. "
                    "Set SANE_CACHE_DIR to a writable directory, or set "
                    "data.tokens_dataset.cache_dir in the config. This usually "
                    "means a checkpoint was saved with an absolute cache path "
                    "from another machine."
                ) from exc

        # Cache statistics
        self._stats = {'hits': 0, 'misses': 0, 'writes': 0}

        logger.info(f"DiskCache initialized at {self.cache_dir}")

    def _get_cache_key(self, key: Union[str, int, Tuple[Any, ...]]) -> str:
        """Generate a cache key hash from the input key."""
        key_str = str(key)

        return f"cache_{key_str}.pt"

    def _get_cache_path(self, key: Union[str, int, Tuple[Any, ...]]) -> Path:
        """Get the full path for a cache file."""
        cache_key = self._get_cache_key(key)
        return self.cache_dir / cache_key

    def _save_to_disk(self, cache_path: Path, data: Any):
        """Save data to disk. Skips if file already exists (idempotent across runs)."""
        if cache_path.exists():
            return
        torch.save(data, cache_path)
        self._stats['writes'] += 1
        logger.debug(f"Saved cache to {cache_path}")

    def _load_from_disk(self, cache_path: Path) -> Any:
        """Load data from disk with validation."""
        if not cache_path.exists():
            logger.debug(f"Cache file not found: {cache_path}")
            return None

        try:
            data = torch.load(cache_path, map_location='cpu')
            # logger.debug(f"Loaded cache from {cache_path}") # comment this out to limit log spam
            return data
        except Exception as e:
            logger.error(f"Failed to load cache from {cache_path}: {e}")
            # Remove corrupted cache file
            try:
                cache_path.unlink()
                logger.info(f"Removed corrupted cache file: {cache_path}")
            except:
                pass
            raise e

    def get(self, key: Union[str, int, Tuple[Any, ...]], default=None) -> Any:
        """
        Get item from cache.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        cache_path = self._get_cache_path(key)
        try:
            value = self._load_from_disk(cache_path)
            if value is None:
                self._stats['misses'] += 1
                return default
            self._stats['hits'] += 1
            return value
        except Exception:
            self._stats['misses'] += 1
            return default

    def put(self, key: Union[str, int, Tuple[Any, ...]], value: Any):
        """
        Put item in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        cache_path = self._get_cache_path(key)
        self._save_to_disk(cache_path, value)

    def contains(self, key: Union[str, int, Tuple[Any, ...]]) -> bool:
        """Check if key exists in cache."""
        cache_path = self._get_cache_path(key)
        return cache_path.exists()

    def remove(self, key: Union[str, int, Tuple[Any, ...]]):
        """Remove item from cache."""
        cache_path = self._get_cache_path(key)
        try:
            if cache_path.exists():
                cache_path.unlink()
                logger.debug(f"Removed cache file: {cache_path}")
        except OSError as e:
            logger.warning(f"Could not remove cache file {cache_path}: {e}")

    def clear(self):
        """Clear all cache entries."""
        # Clear disk cache
        try:
            for file_path in self.cache_dir.glob("cache_*.pt"):
                if file_path.is_file():
                    file_path.unlink()
            logger.info("Cleared all cache entries")
        except OSError as e:
            logger.error(f"Error clearing disk cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self._stats.copy()

        # Calculate disk cache stats
        disk_files = 0
        disk_size = 0
        try:
            for file_path in self.cache_dir.glob("cache_*.pt"):
                if file_path.is_file():
                    disk_files += 1
                    disk_size += file_path.stat().st_size
        except:
            pass

        stats['disk_cache_files'] = disk_files
        stats['disk_cache_size_bytes'] = disk_size
        stats['disk_cache_size_mb'] = disk_size / 1e6

        # Calculate hit rate
        total_requests = stats['hits'] + stats['misses']
        if total_requests > 0:
            stats['hit_rate'] = stats['hits'] / total_requests
        else:
            stats['hit_rate'] = 0.0

        return stats

    def get_cache_info(self) -> str:
        """Get human-readable cache information."""
        stats = self.get_stats()
        return (
            f"DiskCache Stats:\n"
            f"  Cache Directory: {self.cache_dir}\n"
            f"  Disk Cache: {stats['disk_cache_files']} files ({stats['disk_cache_size_mb']:.1f} MB)\n"
            f"  Hits/Misses: {stats['hits']}/{stats['misses']} (hit rate: {stats['hit_rate']:.1%})\n"
            f"  Writes: {stats['writes']}"
        )
