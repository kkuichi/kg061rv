import os
import numpy as np
import tensorflow as tf
from PIL import Image


def predict_batch(model, x_test, img_names, predictions_dir, batch_size=1):
    """
    Runs model prediction in batches and saves predicted masks as PNG files.
    
    Parameters:
        model           : trained Keras model
        x_test          : normalized images of shape (N, H, W, 1)
        img_names       : list of filenames (without extensions)
        predictions_dir : directory where predicted masks will be saved
        batch_size      : batch size (default 1)
        
    Returns:
        y_pred : predicted masks of shape (N, H, W, 1), float32 scaled to [0, 1]
    """
    os.makedirs(predictions_dir, exist_ok=True)
    print("Starting prediction...")

    y_pred_list = []

    for i in range(0, len(x_test), batch_size):
        batch_end = min(i + batch_size, len(x_test))
        print(f"Processing images {i + 1} to {batch_end} of {len(x_test)}")

        batch_x = x_test[i:batch_end]
        batch_pred = model.predict(batch_x, verbose=0)
        y_pred_list.append(batch_pred)

        # Uloženie predikovaných masiek ako PNG
        for j, pred in enumerate(batch_pred):
            img_idx = i + j
            pred_mask = (pred[:, :, 0] * 255).astype(np.uint8)
            pred_image = Image.fromarray(pred_mask, mode="L")
            output_path = os.path.join(predictions_dir, f"{img_names[img_idx]}.png")
            pred_image.save(output_path)

        tf.keras.backend.clear_session()

    y_pred = np.concatenate(y_pred_list, axis=0)
    print(f"Prediction completed. Masks saved to: {predictions_dir}")

    return y_pred
