
"""
Preprocessing utilities for DRIVE retinal vessel segmentation.

- Preserves original image proportions.
- Pads images to dimensions divisible by 32.
- Uses reflection padding for images.
- Uses zero padding for vessel and FOV masks.
- Keeps track of original dimensions for evaluation.
"""

import torch
import torch.nn.functional as F


def get_padded_size(height, width, multiple=32):
    """Calculate the next dimensions divisible by multiple."""

    new_height = ((height + multiple - 1) // multiple) * multiple
    new_width = ((width + multiple - 1) // multiple) * multiple

    return new_height, new_width


def pad_tensor(tensor, target_height, target_width, mode="constant"):
    """Pad a CHW tensor on the bottom and right sides."""

    _, height, width = tensor.shape

    pad_bottom = target_height - height
    pad_right = target_width - width

    if pad_bottom < 0 or pad_right < 0:
        raise ValueError("Target dimensions cannot be smaller.")

    if pad_bottom == 0 and pad_right == 0:
        return tensor

    padding = (0, pad_right, 0, pad_bottom)

    if mode == "reflect":
        return F.pad(tensor, padding, mode="reflect")

    return F.pad(tensor, padding, mode="constant", value=0)


def retinal_collate_fn(batch):
    """
    Custom DataLoader collate function.

    Pads images, vessel masks and FOV masks
    to a common size divisible by 32.
    """

    heights = [item["image"].shape[-2] for item in batch]
    widths = [item["image"].shape[-1] for item in batch]

    target_height, target_width = get_padded_size(
        max(heights),
        max(widths),
        multiple=32
    )

    images = []
    masks = []
    fovs = []
    image_ids = []
    original_sizes = []

    for item in batch:

        image = item["image"]
        mask = item["mask"]
        fov = item["fov"]

        # Images are already normalized to [0, 1]
        if image.min() < 0 or image.max() > 1:
            raise ValueError("Image values must be between 0 and 1.")

        # Image padding
        image = pad_tensor(
            image, target_height, target_width,
            mode="reflect"
        )

        # Mask padding
        mask = pad_tensor(
            mask, target_height, target_width,
            mode="constant"
        )

        # FOV padding
        fov = pad_tensor(
            fov, target_height, target_width,
            mode="constant"
        )

        images.append(image)
        masks.append(mask)
        fovs.append(fov)

        image_ids.append(item["image_id"])

        original_sizes.append(
            (item["image"].shape[-2],
             item["image"].shape[-1])
        )

    return {
        "image": torch.stack(images),
        "mask": torch.stack(masks),
        "fov": torch.stack(fovs),
        "image_id": image_ids,
        "original_sizes": original_sizes
    }


def crop_to_original(tensor, original_size):
    """Remove padding after model prediction."""

    height, width = original_size

    return tensor[..., :height, :width]
