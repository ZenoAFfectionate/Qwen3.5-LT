"""ImageNet dataset loader (standard ImageFolder layout)."""
from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from dataset.base import ClassificationDataset
from prompts.templates import load_class_names


_logger = logging.getLogger("qwen35_lt.dataset")


_ALLOWED_EXTS = {".jpeg", ".jpg", ".png"}


class ImageNetDataset(ClassificationDataset):
    """Iterate an ImageNet-style directory.

    Expected layout:  ``<root>/<split>/<synset_id>/<image>.JPEG``
    ``split`` may be ``"train"``, ``"val"``, or ``""`` (use ``<root>`` directly).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "val",
        classes_file: str | Path = "prompts/imagenet_classes.txt",
    ) -> None:
        self.root = Path(root)
        self.split_dir = self.root / split if split else self.root
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"ImageNet split dir not found: {self.split_dir}")

        synsets = sorted(
            d.name for d in self.split_dir.iterdir() if d.is_dir()
        )
        self._synset_to_idx = {s: i for i, s in enumerate(synsets)}

        self._samples: list[tuple[Path, int, str]] = []
        for synset in synsets:
            label = self._synset_to_idx[synset]
            cls_dir = self.split_dir / synset
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() not in _ALLOWED_EXTS:
                    continue
                if p.stat().st_size == 0:
                    _logger.warning("skipping empty image file %s", p)
                    continue
                self._samples.append((p, label, synset))

        raw_names = load_class_names(classes_file)
        if len(raw_names) >= len(synsets):
            self._class_names = raw_names[: len(synsets)]
        else:
            self._class_names = raw_names + synsets[len(raw_names):]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int, str]:
        path, label, synset = self._samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise RuntimeError(f"Failed to read image {path}: {e}") from e
        return image, label, synset
