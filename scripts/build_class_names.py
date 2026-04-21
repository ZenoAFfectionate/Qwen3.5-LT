"""One-off: derive sorted 1000-class names from ImageNet devkit/meta.mat.

Writes one class name per line to prompts/imagenet_classes.txt, ordered by
synset ID alphabetically — which matches torchvision.ImageFolder ordering on
/opt/ImageNet/train (and val).

Each line is the canonical first phrase from WordNet, e.g. "tench" instead of
"tench, Tinca tinca".
"""
from __future__ import annotations

import argparse
from pathlib import Path

import scipy.io


def load_synset_to_name(meta_mat_path: Path) -> dict[str, str]:
    mat = scipy.io.loadmat(str(meta_mat_path), squeeze_me=True)
    synsets = mat["synsets"]
    mapping: dict[str, str] = {}
    for entry in synsets:
        wnid = str(entry["WNID"])
        words = str(entry["words"])
        num_children = int(entry["num_children"])
        if not wnid.startswith("n"):
            continue
        if num_children != 0:
            continue
        canonical = words.split(",")[0].strip().lower().replace(" ", "_")
        mapping[wnid] = canonical
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta-mat",
        type=Path,
        default=Path("/opt/ImageNet/devkit/data/meta.mat"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "prompts/imagenet_classes.txt",
    )
    args = parser.parse_args()

    synset_to_name = load_synset_to_name(args.meta_mat)
    sorted_synsets = sorted(synset_to_name.keys())
    if len(sorted_synsets) != 1000:
        raise ValueError(
            f"Expected 1000 leaf synsets, got {len(sorted_synsets)}. "
            f"Check meta.mat at {args.meta_mat}."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for synset in sorted_synsets:
            f.write(synset_to_name[synset] + "\n")

    print(f"Wrote {len(sorted_synsets)} class names to {args.output}")


if __name__ == "__main__":
    main()
