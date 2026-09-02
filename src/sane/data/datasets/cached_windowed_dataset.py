import hashlib
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Set, Tuple, Union

import torch
import yaml

from sane.data.cache import DiskCache
from sane.data.datasets.windowed_dataset import WindowedDataset

logger = logging.getLogger(__name__)


def resolve_cache_dir(configured: Optional[Union[str, Path]] = None) -> Path:
    """Portable token-cache location.

    Resolution order:
      1. $SANE_CACHE_DIR          -- overrides everything, including config
      2. the configured cache_dir
      3. $XDG_CACHE_HOME/sane/tokens
      4. ~/.sane/cache/tokens     -- the default the configs already document

    The env var deliberately outranks the config. A published checkpoint
    carries its training machine's absolute cache path inside checkpoint.pt,
    so a value that only applied when cache_dir was null could never rescue
    one. This matches TORCH_HOME / HF_HOME conventions.

    Replaces a hardcoded "/local/cache/tokens" fallback, which contradicted
    the config comments and is unwritable outside the machine it came from.
    """
    env = os.environ.get("SANE_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    if configured:
        return Path(configured).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "sane" / "tokens"
    return Path.home() / ".sane" / "cache" / "tokens"


class CachedWindowedDataset(WindowedDataset):
    """Windowed dataset with disk caching. Extends WindowedDataset with two caching strategies:

    - ``"per_model"``: All windows for a checkpoint are stored together as a single cache entry.
      Simple and fast for small-to-medium models; reading one window deserializes the full list.

    - ``"per_window"``: Each window is stored as its own cache entry. Reading one window only
      deserializes that window — ~N× less I/O for checkpoints with N windows.

    The entire cache is preloaded on initialization to avoid lazy-generation races in
    multiprocessing DataLoader workers.

    Args:
        checkpoints_dataset: The underlying dataset of checkpoints.
        tokenizer: Tokenizer used to convert state dicts to token sequences.
        window_size: Number of tokens per window.
        num_windows_per_model: Windows to sample per checkpoint. "auto" = full coverage.
        num_windows_per_layer: Windows per layer (takes precedence over num_windows_per_model).
        pad_windows: Pad short windows to exactly window_size.
        pad_to_max_length: Whether to pad emitted tokens, mask, and position to the
            inferred maximum sequence length of this windowed dataset.
        allow_overlapping_windows: Allow overlapping window placement.
        window_distribution_strategy: "consecutive" or "distributed".
        cache_mode: ``"per_model"`` or ``"per_window"``.
        cache_dir: Root directory for cache files.
        split_name: Split identifier (e.g. "train"). Used as a subdirectory to avoid
            key collisions across splits.
        dataset_name: Dataset identifier (e.g. "cifar10_resnet18"). Used as a subdirectory
            prefix to avoid collisions across datasets.
        data_config_for_hash: Full data config dict. When provided the entire dict is hashed
            so any config change auto-invalidates the cache.
    """

    def __init__(
        self,
        checkpoints_dataset,
        tokenizer,
        window_size: int,
        num_windows_per_model: int | Literal["auto"] = 1,
        num_windows_per_layer: int | None = None,
        pad_windows: bool = False,
        pad_to_max_length: bool = False,
        window_distribution_strategy: Literal["consecutive", "distributed"] = "consecutive",
        cache_mode: Literal["per_model", "per_window"] = "per_model",
        cache_dir: Optional[Union[str, Path]] = "/local/cache/tokens",
        global_cache: bool = True,
        read_only: bool = False,
        split_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        data_config_for_hash: Optional[dict] = None,
        allow_overlapping_windows: bool = False,
        **kwargs,
    ):
        super().__init__(
            checkpoints_dataset=checkpoints_dataset,
            tokenizer=tokenizer,
            window_size=window_size,
            num_windows_per_model=num_windows_per_model,
            num_windows_per_layer=num_windows_per_layer,
            pad_windows=pad_windows,
            pad_to_max_length=pad_to_max_length,
            allow_overlapping_windows=allow_overlapping_windows,
            window_distribution_strategy=window_distribution_strategy,
            **kwargs,
        )

        self._cache_mode = cache_mode
        self.split_name = split_name
        self.dataset_name = dataset_name
        self._data_config_for_hash = data_config_for_hash

        self._cache_key_suffix = self._generate_cache_key_suffix()

        self._global_cache = global_cache
        self._read_only = read_only
        self._base_cache_dir = resolve_cache_dir(cache_dir)
        dataset_part = self.dataset_name or "default"
        split_part = self.split_name or "default"
        subdir = "checkpoint" if cache_mode == "per_model" else "window"
        config_dir = self._base_cache_dir / f"{dataset_part}_{self._cache_key_suffix}"
        if not global_cache:
            instance_id = uuid.uuid4().hex[:12]
            config_dir = config_dir / f"instance_{instance_id}"
        effective_cache_dir = config_dir / subdir / split_part

        self._disk_cache = DiskCache(cache_dir=effective_cache_dir, create_dirs=True)
        logger.info(f"Cache ({cache_mode}): {effective_cache_dir}")

        if self._data_config_for_hash is not None:
            config_file = (self._base_cache_dir / f"{dataset_part}_{self._cache_key_suffix}" / "data_config.yaml")
            if not self._read_only and not config_file.exists():
                config_file.write_text(yaml.dump(self._data_config_for_hash, default_flow_style=False))

        self._cached_checkpoints: Set[int] = set()
        self._load_cache_metadata()
        if not self._read_only:
            self.preload_cache()

    def _get_window(self, checkpoint_idx: int, window_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._cache_mode == "per_model":
            windows = self._disk_cache.get(self._get_cache_key(checkpoint_idx))
            if windows is None:
                if self._read_only:
                    raise RuntimeError(f"Cache miss for checkpoint {checkpoint_idx} in read-only mode")
                raise RuntimeError(f"Cache miss for checkpoint {checkpoint_idx} after preload")
            return windows[window_idx]
        else:
            window = self._disk_cache.get(self._get_window_cache_key(checkpoint_idx, window_idx))
            if window is None:
                if self._read_only:
                    raise RuntimeError(f"Cache miss for checkpoint {checkpoint_idx} window {window_idx} in read-only mode")
                raise RuntimeError(f"Cache miss for checkpoint {checkpoint_idx} window {window_idx} after preload")
            return window

    def _get_all_windows(self, checkpoint_idx: int) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self._cache_mode == "per_model":
            windows = self._disk_cache.get(self._get_cache_key(checkpoint_idx))
            if windows is None:
                if self._read_only:
                    raise RuntimeError(f"Cache miss for checkpoint {checkpoint_idx} in read-only mode")
                raise RuntimeError(f"Cache miss for checkpoint {checkpoint_idx} after preload")
            return windows
        else:
            windows = [self._disk_cache.get(self._get_window_cache_key(checkpoint_idx, w)) for w in range(self._windows_per_checkpoint)]
            if any(w is None for w in windows):
                if self._read_only:
                    raise RuntimeError(f"Incomplete cache for checkpoint {checkpoint_idx} in read-only mode")
                raise RuntimeError(f"Incomplete cache for checkpoint {checkpoint_idx} after preload")
            return windows

    def _generate_cache_key_suffix(self) -> str:
        config_for_metadata = {
            "tokenizer_mode": self.tokenizer.mode,
            "tokenizer_size": self.tokenizer.tokensize,
            "tokenizer_padding": self.tokenizer.padding,
            "window_size": self.window_size,
            "num_windows_per_model": self.num_windows_per_model,
            "num_windows_per_layer": self.num_windows_per_layer,
            "pad_windows": self.pad_windows,
            "allow_overlapping": self.allow_overlapping_windows,
            "window_strategy": self.window_distribution_strategy,
        }
        self._config_params = {**config_for_metadata, "split_name": self.split_name or "none", "dataset_name": self.dataset_name or "none"}

        if self._data_config_for_hash is not None:
            config_str = json.dumps(self._data_config_for_hash, sort_keys=True, default=str)
        else:
            config_str = "_".join(f"{k}_{v}" for k, v in config_for_metadata.items())

        self._config_string = config_str
        logger.info(f"Cache key config: {config_str}")
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]

    def _get_cache_key(self, checkpoint_idx: int) -> str:
        return f"checkpoint_{checkpoint_idx}"

    def _get_window_cache_key(self, checkpoint_idx: int, window_idx: int) -> str:
        return f"checkpoint_{checkpoint_idx}_window_{window_idx}"

    def _get_metadata_path(self) -> Path:
        return self._disk_cache.cache_dir / "cache_metadata.json"

    def _build_empty_metadata(self) -> dict:
        return {"config": {k: str(v) for k, v in self._config_params.items()}, "checkpoints": {}}

    def _read_cache_metadata(self, metadata_path: Path) -> Optional[dict]:
        if not metadata_path.exists():
            return None

        content = metadata_path.read_text().strip()
        if not content:
            return None

        try:
            metadata = json.loads(content)
        except json.JSONDecodeError as exc:
            metadata, end = json.JSONDecoder().raw_decode(content)
            trailing = content[end:].strip()
            if trailing:
                logger.warning("Recovered cache metadata from %s despite trailing data.", metadata_path)
            else:
                raise exc

        if not isinstance(metadata, dict):
            raise ValueError("Cache metadata root must be a JSON object.")

        return metadata

    def _load_cache_metadata(self):
        metadata_path = self._get_metadata_path()
        try:
            metadata = self._read_cache_metadata(metadata_path)
            if metadata is None:
                return
            self._cached_checkpoints = set(int(idx) for idx in metadata.get("checkpoints", {}).keys())
            logger.info(f"Loaded metadata: {len(self._cached_checkpoints)} checkpoints cached")
        except Exception as e:
            logger.warning(f"Failed to load cache metadata: {e}. Starting fresh.")
            self._cached_checkpoints = set()

    def _save_cache_metadata(self):
        if self._read_only:
            return

        metadata_path = self._get_metadata_path()
        try:
            try:
                metadata = self._read_cache_metadata(metadata_path)
            except Exception as exc:
                logger.warning("Resetting invalid cache metadata at %s: %s", metadata_path, exc)
                metadata = None

            if metadata is None:
                metadata = self._build_empty_metadata()

            metadata.setdefault("config", {k: str(v) for k, v in self._config_params.items()})
            metadata.setdefault("checkpoints", {})

            for idx in self._cached_checkpoints:
                if str(idx) not in metadata["checkpoints"]:
                    metadata["checkpoints"][str(idx)] = {
                        "num_windows": self._windows_per_checkpoint,
                        "cached_at": datetime.now().isoformat(),
                    }

            tmp = metadata_path.parent / f"{metadata_path.name}.tmp"
            with open(tmp, "w") as f:
                json.dump(metadata, f, indent=2)
            tmp.replace(metadata_path)
        except Exception as e:
            pass
            #logger.error(f"Failed to save cache metadata: {e}")

    def _ensure_checkpoint_cached(self, checkpoint_idx: int):
        """Cache all windows for a checkpoint if not already done."""
        if checkpoint_idx in self._cached_checkpoints:
            return

        if self._read_only:
            raise RuntimeError(f"Cannot populate cache for checkpoint {checkpoint_idx} in read-only mode")

        windows = self._generate_windows(checkpoint_idx)
        if not windows or not isinstance(windows, list):
            raise ValueError(f"Generated invalid windows for checkpoint {checkpoint_idx}")

        if self._cache_mode == "per_model":
            self._disk_cache.put(self._get_cache_key(checkpoint_idx), windows)
        else:
            for window_idx, window in enumerate(windows):
                self._disk_cache.put(self._get_window_cache_key(checkpoint_idx, window_idx), window)

        self._cached_checkpoints.add(checkpoint_idx)

        if self._cache_mode == "per_window":
            self._save_cache_metadata()

        logger.debug(f"Cached {len(windows)} windows for checkpoint {checkpoint_idx}")

    def clear_cache(self):
        """Remove all cached entries for this dataset."""
        if self._cache_mode == "per_model":
            for idx in range(len(self.checkpoints_dataset)):
                self._disk_cache.remove(self._get_cache_key(idx))
        else:
            for idx in range(len(self.checkpoints_dataset)):
                for w in range(self._windows_per_checkpoint):
                    self._disk_cache.remove(self._get_window_cache_key(idx, w))
            meta = self._get_metadata_path()
            if meta.exists():
                try:
                    meta.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete metadata file: {e}")

        self._cached_checkpoints.clear()
        logger.info("Cache cleared")

    def preload_cache(self, checkpoint_indices: Optional[List[int]] = None):
        """Preload cache for the given checkpoints (all if None)."""
        if checkpoint_indices is None:
            checkpoint_indices = list(range(len(self.checkpoints_dataset)))

        pending = [idx for idx in checkpoint_indices if idx not in self._cached_checkpoints]
        if not pending:
            logger.info("Cache already complete, skipping preload")
            return

        logger.info(f"Preloading cache ({self._cache_mode}) for {len(pending)} checkpoints...")
        logger.info(f"Preloading cache ({self._cache_mode}) for {len(pending)} checkpoints...")
        for i, idx in enumerate(pending):
            self._ensure_checkpoint_cached(idx)
            if (i + 1) % 400 == 0:
                logger.info(f"Preloaded {i + 1}/{len(pending)} checkpoints")

        logger.info("Cache preload complete")
