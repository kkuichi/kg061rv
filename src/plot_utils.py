import numpy as np
import matplotlib.pyplot as plt
from metrics import dice_np, iou_np


def _squeeze2d(arr):
    """Ensures shape (H, W) - removes the channel dimension from (H, W, 1) if present."""
    arr = np.array(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[:, :, 0]
    return arr


def plot_imgs(imgs, masks=None, predictions=None, n_imgs=10):
    """
    Plots images, masks, predicted masks, and overlays side by side.

    Parameters:
        imgs        : array of images of shape (N, H, W) or (N, H, W, 1)
        masks       : array of ground-truth masks of the same shape, or None
        predictions : array of predicted masks of the same shape, or None
        n_imgs      : number of images to display

    Columns depending on available data:
        images only                -> 1 column:  Image
        images + masks             -> 3 columns: Image | Mask | Overlay
        images + predictions       -> 3 columns: Image | Prediction | Overlay
        images + masks + preds     -> 4 columns: Image | Mask | Prediction | Overlay
                                      + Dice/IoU displayed on the overlay panel

    Returns:
        matplotlib.pyplot  ->  call .show() in a notebook
    """
    n_imgs = min(n_imgs, len(imgs))
    has_masks = masks is not None
    has_preds = predictions is not None

    if has_masks and has_preds:
        n_cols = 4
        col_titles = ["Image", "Mask", "Prediction", "Overlay"]
    elif has_masks or has_preds:
        n_cols = 3
        col_titles = ["Image",
                      "Mask" if has_masks else "Prediction",
                      "Overlay"]
    else:
        n_cols = 1
        col_titles = ["Image"]

    fig, axes = plt.subplots(
        n_imgs, n_cols,
        figsize=(n_cols * 4, n_imgs * 4),
        squeeze=False,
    )

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=22)

    for i in range(n_imgs):
        img_2d  = _squeeze2d(imgs[i])
        mask_2d = _squeeze2d(masks[i]) if has_masks else None
        pred_2d = _squeeze2d(predictions[i]) if has_preds else None

        overlap = pred_2d if has_preds else mask_2d
        if overlap is not None:
            zero   = np.zeros_like(overlap)
            masked = np.stack((overlap, zero, zero, overlap), axis=-1)

        col = 0

        # Obrázok
        axes[i, col].imshow(img_2d, cmap="gray", interpolation=None)
        axes[i, col].set_axis_off()
        col += 1

        if n_cols == 1:
            continue

        if has_masks and has_preds:
            # Ground-truth mask
            axes[i, col].imshow(mask_2d, cmap="gray", interpolation=None)
            axes[i, col].set_axis_off()
            col += 1
            # Prediction
            axes[i, col].imshow(pred_2d, cmap="gray", interpolation=None)
            axes[i, col].set_axis_off()
            col += 1
        elif has_masks:
            axes[i, col].imshow(mask_2d, cmap="gray", interpolation=None)
            axes[i, col].set_axis_off()
            col += 1
        else:
            axes[i, col].imshow(pred_2d, cmap="gray", interpolation=None)
            axes[i, col].set_axis_off()
            col += 1

        # Overlay
        axes[i, col].imshow(img_2d, cmap="gray", interpolation=None)
        axes[i, col].imshow(masked, alpha=0.5, interpolation=None)
        axes[i, col].set_axis_off()

        # Dice + IoU (only when both masks and predictions are available)
        if has_masks and has_preds:
            dice = np.round(dice_np(mask_2d, pred_2d), 4)
            iou  = np.round(iou_np(mask_2d, pred_2d), 4)
            axes[i, col].text(
                0.1, 0.9,
                f"Dice: {dice:.4f}\nIoU:  {iou:.4f}",
                fontsize=15,
                verticalalignment="top",
                transform=axes[i, col].transAxes,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
            )

    plt.tight_layout()
    return plt


def plot_metrics(model):
    """
    Plots training history of Keras model.
    :param model: trained model
    :return matplotlib.pyplot: pyplot of training history
    """
    plt.style.use("ggplot")
    fig, axs = plt.subplots(1, 2, figsize=(18, 5))
    # Plot metrics
    axs[0].plot(model.history["iou"])
    axs[0].plot(model.history["dice"])
    axs[0].plot(model.history["val_iou"])
    axs[0].plot(model.history["val_dice"])
    axs[1].plot(model.history["loss"])
    axs[1].plot(model.history["val_loss"])
    # Set titles
    axs[1].set_title("Loss over epochs", fontsize=20)
    axs[1].set_ylabel("loss", fontsize=20)
    axs[1].set_xlabel("epochs", fontsize=20)
    axs[0].set_title("Metrics over epochs", fontsize=20)
    axs[0].set_ylabel("metrics", fontsize=20)
    axs[0].set_xlabel("epochs", fontsize=20)
    # Set legend
    axs[1].legend(["loss", "val_loss"], loc="center right", fontsize=15)
    axs[0].legend(
        ["iou", "dice", "val_iou", "val_dice"], loc="center right", fontsize=15
    )
    return plt
