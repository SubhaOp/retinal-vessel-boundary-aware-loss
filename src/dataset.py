
"""
DRIVE Dataset Loader
--------------------
Retinal vessel segmentation using the DRIVE dataset.

Supports:
1. RGB images
2. Green-channel images
3. Manual vessel annotations
4. Field of View (FOV) masks
5. Synchronized data augmentation
6. PyTorch DataLoader compatibility

Expected dataset structure:

DRIVE/
    training/
        images/
        1st_manual/
        mask/
    test/
        images/
        mask/
"""

from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class DRIVEDataset(Dataset):

    def __init__(
        self,
        root,
        split="train",
        image_mode="rgb",
        augment=False,
        mean=None,
        std=None
    ):
        """
        Args:
            root: Path to DRIVE dataset root.
            split: 'train' or 'test'.
            image_mode: 'rgb' or 'green'.
            augment: Apply training augmentation.
            mean: Optional channel-wise normalization mean.
            std: Optional channel-wise normalization std.
        """

        self.root = Path(root)
        self.split = split.lower()
        self.image_mode = image_mode.lower()
        self.augment = augment

        self.mean = mean
        self.std = std

        if self.split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")

        if self.image_mode not in ("rgb", "green"):
            raise ValueError("image_mode must be 'rgb' or 'green'")

        if (mean is None) != (std is None):
            raise ValueError("mean and std must be provided together")

        if mean is not None:
            if len(mean) != self.num_channels or len(std) != self.num_channels:
                raise ValueError("Normalization values must match image channels")

            if any(s <= 0 for s in std):
                raise ValueError("Standard deviations must be positive")

        self.split_dir = self.root / (
            "training" if self.split == "train" else "test"
        )

        self.image_dir = self.split_dir / "images"
        self.fov_dir = self.split_dir / "mask"

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )

        if not self.fov_dir.is_dir():
            raise FileNotFoundError(
                f"FOV mask directory not found: {self.fov_dir}"
            )

        if self.split == "train":
            self.label_dir = self.split_dir / "1st_manual"

            if not self.label_dir.is_dir():
                raise FileNotFoundError(
                    f"Annotation directory not found: {self.label_dir}"
                )

        self.samples = self._collect_samples()

        if not self.samples:
            raise RuntimeError(
                f"No valid images found in {self.image_dir}"
            )

        print(
            f"Loaded {len(self.samples)} DRIVE "
            f"{self.split} samples | Mode: {self.image_mode}"
        )

    @property
    def num_channels(self):
        return 3 if self.image_mode == "rgb" else 1

    def _collect_samples(self):
        """Match images with their corresponding masks."""

        samples = []

        image_paths = sorted(self.image_dir.glob("*.tif"))

        for image_path in image_paths:

            image_id = image_path.stem.replace("_training", "").replace(
                "_test", ""
            )

            # DRIVE names training FOV masks with "_training_mask",
            # not "_train_mask". Test masks use "_test_mask".
            fov_suffix = "training" if self.split == "train" else "test"
            fov_path = self.fov_dir / f"{image_id}_{fov_suffix}_mask.gif"

            if not fov_path.is_file():
                raise FileNotFoundError(
                    f"Missing FOV mask for {image_path.name}: {fov_path}"
                )

            sample = {
                "image": image_path,
                "fov": fov_path,
                "id": image_id
            }

            if self.split == "train":

                label_path = self.label_dir / f"{image_id}_manual1.gif"

                if not label_path.is_file():
                    raise FileNotFoundError(
                        f"Missing vessel annotation: {label_path}"
                    )

                sample["label"] = label_path

            samples.append(sample)

        return samples

    @staticmethod
    def _load_grayscale(path):
        """Load a grayscale mask and convert it to a binary array."""

        with Image.open(path) as image:
            array = np.array(image.convert("L"))

        return (array > 127).astype(np.float32)

    def _load_image(self, path):
        """Load RGB or green-channel retinal image."""

        with Image.open(path) as image:

            if self.image_mode == "rgb":
                image = image.convert("RGB")
                array = np.array(image, dtype=np.float32)

                # H x W x 3 -> C x H x W
                array = array.transpose(2, 0, 1)

            else:
                image = image.convert("RGB")
                array = np.array(image, dtype=np.float32)

                # Extract green channel
                array = array[:, :, 1]
                array = array[np.newaxis, :, :]

        # Scale pixel values from [0, 255] to [0, 1]
        array /= 255.0

        return array

    def _augment(self, image, label, fov):
        """Apply identical spatial transformations to all arrays."""

        # Random horizontal flip
        if random.random() < 0.5:
            image = np.flip(image, axis=2)
            label = np.flip(label, axis=1)
            fov = np.flip(fov, axis=1)

        # Random vertical flip
        if random.random() < 0.5:
            image = np.flip(image, axis=1)
            label = np.flip(label, axis=0)
            fov = np.flip(fov, axis=0)

        # Use 0 or 180 degrees so every sample keeps the original
        # (584, 565) spatial dimensions for batching.
        k = random.choice([0, 2])

        image = np.rot90(image, k=k, axes=(1, 2))
        label = np.rot90(label, k=k, axes=(0, 1))
        fov = np.rot90(fov, k=k, axes=(0, 1))

        return image.copy(), label.copy(), fov.copy()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        image = self._load_image(sample["image"])
        fov = self._load_grayscale(sample["fov"])

        label = None

        if self.split == "train":
            label = self._load_grayscale(sample["label"])

            if label.shape != fov.shape:
                raise ValueError(
                    f"Label/FOV shape mismatch: {sample['id']}"
                )

        if image.shape[1:] != fov.shape:
            raise ValueError(
                f"Image/FOV shape mismatch: {sample['id']}"
            )

        if self.augment and self.split == "train":
            image, label, fov = self._augment(
                image, label, fov
            )

        # Optional normalization
        if self.mean is not None:
            mean = np.asarray(self.mean, dtype=np.float32)[:, None, None]
            std = np.asarray(self.std, dtype=np.float32)[:, None, None]

            image = (image - mean) / std

        # Convert NumPy arrays into PyTorch tensors
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image)
        ).float()

        fov_tensor = torch.from_numpy(
            np.ascontiguousarray(fov)
        ).unsqueeze(0).float()

        result = {
            "image": image_tensor,
            "fov": fov_tensor,
            "image_id": sample["id"]
        }

        if label is not None:
            label_tensor = torch.from_numpy(
                np.ascontiguousarray(label)
            ).unsqueeze(0).float()

            result["mask"] = label_tensor

        return result