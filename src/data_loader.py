import glob
import os
import numpy as np
from PIL import Image


def load_images(img_dir, img_size=512):
    """
    Loads images from the specified directory.
    
    Parameters:
        img_dir  : path to a directory containing PNG images (or a glob pattern)
        img_size : target side length for square resizing (default 512)
    Returns:
        imgs_test_list : list of images as np.array (H, W), uint8
        img_names      : list of filenames without extensions
        x_test         : normalized array of shape (N, H, W, 1), float32 scaled to [0, 1]
    """
    if os.path.isdir(img_dir):
        paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    else:
        # A glob pattern was provided directly
        paths = sorted(glob.glob(img_dir))

    print(f"Number of images: {len(paths)}")

    imgs_test_list = []
    img_names = []

    for image_path in paths:
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        img_names.append(img_name)

        rgba = Image.open(image_path).convert("RGBA")
        background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        composited = Image.alpha_composite(background, rgba)

        img = composited.convert("L").resize((img_size, img_size), Image.NEAREST)
        imgs_test_list.append(np.array(img))

    x_test = np.asarray(imgs_test_list, dtype=np.float32) / 255
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    return imgs_test_list, img_names, x_test


def load_masks(masks_dir, img_names, img_size=512):
    """
    Loads ground-truth masks corresponding to the provided list of image names.
    
    Parameters:
        masks_dir : path to the directory containing mask files
        img_names : list of filenames (without extensions) — defines order and pairing
        img_size  : target side length for square resizing (default 512)
    Returns:
        masks_list : list of masks as np.array (H, W), float32 scaled to [0, 1],
                     or None if the directory does not exist or contains no valid masks
        y_test     : array of shape (N, H, W, 1) prepared for metric computation,
                     or None
    """
    if not os.path.isdir(masks_dir):
        print(f"Mask directory '{masks_dir}' does not exist. Continuing without masks.")
        return None, None

    masks_list = []
    for img_name in img_names:
        mask_path = os.path.join(masks_dir, f"{img_name}.png")
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L").resize((img_size, img_size), Image.NEAREST)
            masks_list.append(np.array(mask, dtype=np.float32) / 255)
        else:
            masks_list.append(np.zeros((img_size, img_size), dtype=np.float32))

    if all(np.all(m == 0) for m in masks_list):
        print(f"Directory '{masks_dir}' contains no matching masks. Continuing without masks.")
        return None, None

    print(f"Masks loaded from: {masks_dir}")

    y_test = np.stack(masks_list, axis=0)
    y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2], 1)

    return masks_list, y_test
