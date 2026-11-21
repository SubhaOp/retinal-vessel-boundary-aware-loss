
"""
Verify the DRIVE retinal vessel segmentation dataset.

Checks:
1. Training and test image counts.
2. Image-to-annotation matching.
3. Image-to-FOV mask matching.
4. Image dimensions.
5. Binary mask values.
6. Availability of test vessel annotations.

Usage:
    python scripts/verify_dataset.py --root /path/to/DRIVE
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def get_files(folder: Path) -> list[Path]:
    """Return all files directly inside a directory."""

    if not folder.is_dir():
        return []

    return sorted(p for p in folder.iterdir() if p.is_file())


def extract_ids(
    folder: Path,
    pattern: str
) -> dict[str, Path]:
    """Extract image IDs and their corresponding file paths."""

    result = {}

    for file in get_files(folder):
        match = re.fullmatch(pattern, file.name)

        if match:
            result[match.group(1)] = file

    return result


def inspect_binary_mask(path: Path) -> bool:
    """Check whether a mask contains only binary values 0 and 255."""

    with Image.open(path) as img:
        array = np.array(img.convert("L"))

    values = np.unique(array)

    print(f"    {path.name}: values={values.tolist()}")

    return set(values.tolist()).issubset({0, 255})


def check_dimensions(
    image_path: Path,
    mask_path: Path
) -> bool:
    """Check whether an image and its mask have matching dimensions."""

    with Image.open(image_path) as image:
        image_size = image.size

    with Image.open(mask_path) as mask:
        mask_size = mask.size

    if image_size != mask_size:
        print(
            f"  SIZE MISMATCH: {image_path.name} "
            f"and {mask_path.name}"
        )
        print(f"    Image: {image_size}, Mask: {mask_size}")
        return False

    return True


def audit_split(
    root: Path,
    split: str,
    check_labels: bool = True
) -> bool:
    """Audit one DRIVE split."""

    base = root / split

    print(f"\n========== {split.upper()} AUDIT ==========")

    if split == "training":
        image_pattern = r"(\d+)_training\.tif"
        mask_pattern = r"(\d+)_training_mask\.gif"
        label_pattern = r"(\d+)_manual1\.gif"
    else:
        image_pattern = r"(\d+)_test\.tif"
        mask_pattern = r"(\d+)_test_mask\.gif"
        label_pattern = r"(\d+)_manual1\.gif"

    images = extract_ids(
        base / "images", image_pattern
    )

    masks = extract_ids(
        base / "mask", mask_pattern
    )

    print("Images found:", len(images))
    print("FOV masks found:", len(masks))

    valid = True

    missing_masks = set(images) - set(masks)
    extra_masks = set(masks) - set(images)

    print("Images missing FOV masks:", sorted(missing_masks))
    print("Extra FOV masks:", sorted(extra_masks))

    if missing_masks or extra_masks:
        valid = False

    labels = {}

    if check_labels:
        labels = extract_ids(
            base / "1st_manual", label_pattern
        )

        print("Manual annotations found:", len(labels))

        missing_labels = set(images) - set(labels)
        extra_labels = set(labels) - set(images)

        print("Images missing labels:", sorted(missing_labels))
        print("Extra labels:", sorted(extra_labels))

        if missing_labels or extra_labels:
            valid = False

    # Check dimensions and mask values.
    print("\nChecking dimensions and mask values...")

    for image_id, image_path in images.items():

        if image_id not in masks:
            continue

        if not check_dimensions(image_path, masks[image_id]):
            valid = False

        if not inspect_binary_mask(masks[image_id]):
            print("    ERROR: Non-binary FOV mask")
            valid = False

        if image_id in labels:

            if not check_dimensions(image_path, labels[image_id]):
                valid = False

            if not inspect_binary_mask(labels[image_id]):
                print("    ERROR: Non-binary vessel annotation")
                valid = False

    print(
        f"\n{split.upper()} RESULT:",
        "PASS" if valid else "CHECK REQUIRED"
    )

    return valid


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DRIVE dataset verification"
    )

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Path to the DRIVE dataset root"
    )

    args = parser.parse_args()
    root = args.root

    if not root.is_dir():
        print(f"ERROR: Dataset directory not found: {root}")
        return 1

    print("DRIVE DATASET AUDIT")
    print("Dataset root:", root.resolve())

    training_ok = audit_split(
        root, "training", check_labels=True
    )

    test_label_dir = root / "test" / "1st_manual"

    if test_label_dir.is_dir():
        test_labels_ok = audit_split(
            root, "test", check_labels=True
        )
    else:
        print("\nTEST VESSEL ANNOTATIONS: NOT AVAILABLE")
        test_labels_ok = None

        # Still verify test images and FOV masks.
        test_ok = audit_split(
            root, "test", check_labels=False
        )
    if test_labels_ok is not None:
        test_ok = test_labels_ok

    print("\n========== FINAL SUMMARY ==========")
    print("Training:", "PASS" if training_ok else "FAIL")
    print("Test images/FOV:", "PASS" if test_ok else "FAIL")
    print(
        "Test annotations:",
        "AVAILABLE" if test_labels_ok is not None else "NOT FOUND"
    )

    if not training_ok or not test_ok:
        return 1

    print("\nDataset structural audit completed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
