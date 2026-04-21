"""Abstract base class for classification datasets yielding (image, label, key)."""
from __future__ import annotations

from abc import abstractmethod

from PIL import Image
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    """Common interface for image-classification datasets.

    Each item is ``(image, label_idx, key)`` where ``key`` is a dataset-specific
    stable identifier (synset ID for ImageNet). Subclasses populate
    ``_class_names`` and ``_synset_to_idx`` during init.
    """

    _class_names: list[str] = []
    _synset_to_idx: dict[str, int] = {}

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[Image.Image, int, str]: ...

    @property
    def class_names(self) -> list[str]:
        return self._class_names

    @property
    def synset_to_idx(self) -> dict[str, int]:
        return self._synset_to_idx
