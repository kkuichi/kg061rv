"""
Data Generator for Image Segmentation
Supports separate geometric and color augmentations via Albumentations.
Geometric augmentations are applied to both images and masks;
color augmentations are applied to images only.
"""

import numpy as np
import albumentations as A


class DataGenerator:
    """
    Data generator for image segmentation with augmentation support.

    Parameters
    ----------
    x_data : numpy.ndarray
        Input images.
    y_data : numpy.ndarray
        Segmentation masks.
    batch_size : int
        Number of samples per batch.
    augmentations : albumentations.Compose, optional
        Geometric augmentations applied to both image and mask.
    color_augmentations : albumentations.Compose, optional
        Color augmentations applied to the image only.
    shuffle : bool, default=True
        Whether to shuffle data at the start of each epoch.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, x_data, y_data, batch_size=32, augmentations=None,
                 color_augmentations=None, shuffle=True, seed=None):
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.color_augmentations = color_augmentations
        self.shuffle = shuffle
        self.seed = seed
        self.num_samples = len(x_data)

        if seed is not None:
            np.random.seed(seed)

    def __len__(self):
        """Returns the number of batches per epoch."""
        return int(np.ceil(self.num_samples / self.batch_size))

    def __iter__(self):
        """Allows iteration over the generator."""
        return self.generate()

    def generate(self):
        """
        Generates batches of augmented images and masks.

        Yields
        ------
        tuple
            (batch_x, batch_y) - batch of images and corresponding masks.
        """
        while True:
            # Shuffle indices at the start of each epoch
            if self.shuffle:
                indices = np.random.permutation(self.num_samples)
            else:
                indices = np.arange(self.num_samples)

            for start_idx in range(0, self.num_samples, self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]

                batch_x = []
                batch_y = []

                for idx in batch_indices:
                    image = self.x_data[idx].copy()
                    mask = self.y_data[idx].copy()

                    if self.augmentations is not None:
                        # Apply geometric augmentations to both image and mask
                        augmented = self.augmentations(image=image, mask=mask)
                        image = augmented['image']
                        mask = augmented['mask']

                    if self.color_augmentations is not None:
                        # Apply color augmentations to image only
                        augmented_color = self.color_augmentations(image=image)
                        image = augmented_color['image']

                    batch_x.append(image)
                    batch_y.append(mask)

                yield np.array(batch_x), np.array(batch_y)

    def __call__(self):
        """Allows the generator to be called as a function."""
        return self.generate()



def create_augmentations(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_limit=0,
    border_mode=0,
    separate_color=True
):
    """
    Creates Albumentations augmentation pipelines.

    Geometric augmentations (flips, rotation) are intended for use on both
    images and masks. Color augmentations (gamma, brightness, contrast) are
    intended for use on images only, as mask pixel values must remain binary.

    Parameters
    ----------
    horizontal_flip : bool
        Whether to apply horizontal flip.
    vertical_flip : bool
        Whether to apply vertical flip.
    rotation_limit : int or tuple
        Rotation range in degrees. Set to 0 to disable.
        Rotation is disabled for EPB segmentation because EPB structures
        appear exclusively in a top-to-bottom orientation.
    border_mode : int
        Border padding mode (cv2.BORDER_*).
    separate_color : bool
        If True, returns a (geometric_aug, color_aug) tuple.
        If False, returns a single combined augmentation pipeline.

    Returns
    -------
    albumentations.Compose or tuple of albumentations.Compose
    """
    geometric_transforms = []

    if horizontal_flip:
        geometric_transforms.append(A.HorizontalFlip(p=0.5))

    if vertical_flip:
        geometric_transforms.append(A.VerticalFlip(p=0.5))

    if rotation_limit:
        if isinstance(rotation_limit, int):
            rotation_limit = (-rotation_limit, rotation_limit)
        geometric_transforms.append(A.Rotate(
            limit=rotation_limit,
            p=0.5,
            border_mode=border_mode
        ))

    color_transforms = [
        A.RandomGamma(gamma_limit=(100, 150), p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.4,
            p=0.5
        )
    ]

    if separate_color:
        geometric_aug = A.Compose(geometric_transforms) if geometric_transforms else None
        color_aug = A.Compose(color_transforms)
        return geometric_aug, color_aug
    else:
        return A.Compose(geometric_transforms + color_transforms)
