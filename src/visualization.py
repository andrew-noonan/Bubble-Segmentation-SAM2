import numpy as np
import matplotlib.pyplot as plt

def plot_detected_circles(image, properties, circularity_thresh=0.6, diamOffset = 4):
    """
    Plots a side-by-side comparison of:
    - Original image
    - Image with overlaid circles for high-circularity regions

    Parameters:
        image: ndarray
            Original image (grayscale or RGB).
        properties: list of dicts
            Each dict should have 'centroid', 'diameter', 'circularity'.
        circularity_thresh: float
            Minimum circularity to include region in the plot.
    """
    if image.ndim == 2:
        image_rgb = np.stack([image]*3, axis=-1)
    else:
        image_rgb = image.copy()

    fig, axs = plt.subplots(1, 2, figsize=(18, 9))

    # Original image
    axs[0].imshow(image_rgb)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    # Overlayed image
    axs[1].imshow(image_rgb)
    for prop in properties:
        if prop['circularity'] >= circularity_thresh:
            y, x = prop['centroid']
            r = prop['diameter'] / 2
            circle = plt.Circle((x, y), r-diamOffset/2, edgecolor='cyan', fill=False, linewidth=0.5)
            axs[1].add_patch(circle)
            axs[1].plot(x, y, 'r.', markersize=3)

    axs[1].set_title(f"Detected Circles (circularity ≥ {circularity_thresh})")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

def plot_mask_stages(image, masks_dict, props, title_prefix=""):
    """
    Visualizes mask processing stages: initial, after filtering, after containment, after splitting.

    Args:
        image: Original image
        masks_dict: Dictionary of mask lists for each stage
        props: Final computed properties (e.g., for detected circles)
        title_prefix: Frame title
    """
    fig, axs = plt.subplots(1, 4, figsize=(18, 5))
    stages = ["initial", "filtered", "contained", "split"]
    titles = [
        "1. Initial Masks",
        "2. After IOU/Stability Filtering",
        "3. After Containment Filtering",
        "4. After Splitting"
    ]

    for i, stage in enumerate(stages):
        overlay = image.copy()
        for mask in masks_dict.get(stage, []):
            alpha = 0.6
            mask_layer = np.zeros_like(overlay)
            mask_layer[mask] = [255, 0, 255]
            overlay = overlay.astype(np.float32)
            overlay[mask] = (1 - alpha) * overlay[mask] + alpha * mask_layer[mask]
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        axs[i].imshow(overlay)
        axs[i].set_title(titles[i])
        axs[i].axis("off")

    plt.suptitle(title_prefix)
    plt.tight_layout()
    plt.show()

def visualize_prompts_on_image(image: np.ndarray,
                               boxes: list[list[int]],
                               points_for_box: list[list[tuple[float, float]]]):
    """
    Display the image with bounding boxes and prompt points overlaid.

    Args:
      image: HxWxC or HxW numpy array
      boxes: list of [x0, y0, x1, y1]
      points_for_box: list of lists of (x, y) points inside each box
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    # show image
    if image.ndim == 2:
        ax.imshow(image, cmap='gray')
    else:
        ax.imshow(image)
    # draw boxes
    for (x0, y0, x1, y1) in boxes:
        rect = plt.Rectangle((x0, y0), x1-x0, y1-y0,
                             edgecolor='yellow', facecolor='none', linewidth=1)
        ax.add_patch(rect)
    # draw points
    for pts in points_for_box:
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        ax.scatter(xs, ys, marker='+', color='red', s=20)
    ax.set_title("Boxes and Prompt Points")
    ax.axis('off')
    plt.show()