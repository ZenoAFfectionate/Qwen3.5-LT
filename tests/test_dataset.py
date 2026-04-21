from __future__ import annotations

from pathlib import Path

from PIL import Image

from dataset.imagenet import ImageNetDataset


def test_dataset_length(tiny_imagefolder: Path, tiny_class_names_file: Path) -> None:
    ds = ImageNetDataset(
        root=str(tiny_imagefolder),
        split="",
        classes_file=tiny_class_names_file,
    )
    assert len(ds) == 6


def test_dataset_getitem_returns_image_label_synset(
    tiny_imagefolder: Path, tiny_class_names_file: Path
) -> None:
    ds = ImageNetDataset(
        root=str(tiny_imagefolder),
        split="",
        classes_file=tiny_class_names_file,
    )
    image, label, synset = ds[0]
    assert isinstance(image, Image.Image)
    assert isinstance(label, int)
    assert synset.startswith("n0000000")


def test_dataset_class_names_alignment(
    tiny_imagefolder: Path, tiny_class_names_file: Path
) -> None:
    ds = ImageNetDataset(
        root=str(tiny_imagefolder),
        split="",
        classes_file=tiny_class_names_file,
    )
    assert ds.synset_to_idx == {"n00000001": 0, "n00000002": 1}
    assert ds.class_names == ["first_class", "second_class"]


def test_dataset_corrupt_image_skipped(
    tiny_imagefolder: Path, tiny_class_names_file: Path
) -> None:
    bad = next((tiny_imagefolder / "n00000001").glob("*.jpg"))
    bad.write_bytes(b"")
    ds = ImageNetDataset(
        root=str(tiny_imagefolder),
        split="",
        classes_file=tiny_class_names_file,
    )
    assert len(ds) == 5
