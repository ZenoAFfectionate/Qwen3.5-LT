"""Placeholder ImageNet-LT dataset. Stage-2 work.

Planned behavior: inherit from ImageNetDataset and subsample per a long-tail
index file (e.g. ImageNet-LT official split). For now this just re-exports
the balanced class so the config pathway can be wired end-to-end while the
LT sampler is designed.
"""
from __future__ import annotations

from dataset.imagenet import ImageNetDataset


class ImageNetLTDataset(ImageNetDataset):
    def __init__(self, *args, split_file: str | None = None, **kwargs) -> None:
        if split_file is not None:
            raise NotImplementedError(
                "ImageNet-LT sampling not implemented yet (Stage 2)."
            )
        super().__init__(*args, **kwargs)
