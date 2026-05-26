import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def create_pdf(
    pdf_path,
    imgs_test_list,
    img_names,
    x_test,
    y_pred,
    masks_list=None,
    rows_per_page=4,
):
    """
    Creates a PDF displaying prediction results.
    
    Parameters:
        pdf_path      : path to the output PDF file
        imgs_test_list: list of original images (as np.array)
        img_names     : list of filenames (without extensions)
        x_test        : normalized images of shape (N, H, W, 1)
        y_pred        : predicted masks of shape (N, H, W, 1)
        masks_list    : list of ground-truth masks (as np.array) or None
                        -> None : 3 columns (Image | Prediction | Overlay)
                        -> list : 4 columns (Image | Ground-truth mask | Prediction | Overlay)
        rows_per_page : number of rows (images) per page
    """
    has_masks = masks_list is not None and len(masks_list) > 0

    if has_masks:
        col_titles = ["Image", "Mask", "Prediction", "Overlay"]
        n_cols = 4
    else:
        col_titles = ["Image", "Prediction", "Overlay"]
        n_cols = 3

    print("Creating PDF...")

    with PdfPages(pdf_path) as pdf:
        n_images = len(imgs_test_list)

        for page_start in range(0, n_images, rows_per_page):
            page_end = min(page_start + rows_per_page, n_images)
            n_rows = page_end - page_start

            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(n_cols * 4, n_rows * 4),
                squeeze=False,
            )

            for col_idx, title in enumerate(col_titles):
                axes[0, col_idx].set_title(title, fontsize=22)

            for row_idx, img_idx in enumerate(range(page_start, page_end)):
                pred_2d = y_pred[img_idx, :, :, 0]
                zero    = np.zeros_like(pred_2d)
                masked  = np.stack((pred_2d, zero, zero, pred_2d), axis=-1)

                col = 0

                # Image + filename label
                axes[row_idx, col].imshow(
                    x_test[img_idx, :, :, 0],
                    cmap="gray", interpolation="nearest", aspect="equal",
                )
                axes[row_idx, col].set_axis_off()
                axes[row_idx, col].text(
                    -0.1, 0.5, img_names[img_idx],
                    fontsize=12, fontweight="bold",
                    transform=axes[row_idx, col].transAxes,
                    rotation=90, va="center", ha="right",
                )
                col += 1

                # Ground-truth mask (optional)
                if has_masks:
                    axes[row_idx, col].imshow(
                        masks_list[img_idx],
                        cmap="gray", interpolation="nearest", aspect="equal",
                    )
                    axes[row_idx, col].set_axis_off()
                    col += 1

                # Prediction
                axes[row_idx, col].imshow(
                    pred_2d,
                    cmap="gray", interpolation="nearest", aspect="equal",
                )
                axes[row_idx, col].set_axis_off()
                col += 1

                # Overlay
                axes[row_idx, col].imshow(
                    x_test[img_idx, :, :, 0],
                    cmap="gray", interpolation="nearest", aspect="equal",
                )
                axes[row_idx, col].imshow(
                    masked, interpolation="nearest", aspect="equal", alpha=0.5,
                )
                axes[row_idx, col].set_axis_off()

            plt.subplots_adjust(
                left=0.15, right=0.95, top=0.95, bottom=0.05,
                hspace=0.3, wspace=0.1,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"PDF saved to: {pdf_path}")
