import numpy as np
from tensorflow.keras import backend as K



def dice_np(y_true, y_pred, smooth=1.0):
    intersection = np.sum(y_true * y_pred)
    return 2 * (intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)


def iou_np(y_true, y_pred, smooth=1.0):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true + y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


def dice(y_true, y_pred, smooth=1.0):
    intersection = K.sum(y_true * y_pred)
    return (2 * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)


def iou(y_true, y_pred, smooth=1.0):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true + y_pred) - intersection
    return (intersection + smooth) / (union + smooth)



def compute_metrics(y_true, y_pred, threshold=0.5):
    """
    Computes and prints Dice and IoU metrics (with and without thresholding) on the full test set.
    
    Parameters:
        y_true    : ground-truth masks of shape (N, H, W, 1), values scaled to [0, 1]
        y_pred    : predicted masks of shape (N, H, W, 1), values scaled to [0, 1]
        threshold : binarization threshold applied to predictions (default 0.5)
        
    Returns:
        None
    """
    if y_true is None:
        print("Metrics: ground-truth masks are not available, skipping computation.")
        return None

    y_pred_bin = (y_pred > threshold).astype(np.float32)

    dice_val        = np.round(dice_np(y_true, y_pred),     4)
    dice_thresh_val = np.round(dice_np(y_true, y_pred_bin), 4)
    iou_val         = np.round(iou_np(y_true, y_pred),      4)
    iou_thresh_val  = np.round(iou_np(y_true, y_pred_bin),  4)

    print(
        f"  Dice: {dice_val:.4f}  Dice (thresh):  {dice_thresh_val:.4f}\n"
        f"  IoU: {iou_val:.4f}   IoU  (thresh):  {iou_thresh_val:.4f}\n"
    )
