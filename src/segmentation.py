import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops, label

def sobel_edge(image, ksize=9):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    return (sobel_mag / sobel_mag.max() * 255).astype(np.uint8)

def generate_boxes_and_points(image, sobel_mag, edge_thresh=0.5, min_contour_len=10, aspect_ratio_thresh=1.75):
    sobel_norm = sobel_mag / sobel_mag.max()
    _, edge_mask = cv2.threshold((sobel_norm * 255).astype(np.uint8), int(edge_thresh * 255), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(edge_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = image.shape[:2]
    
    # Convert image to grayscale for darkness comparison
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    
    boxes, points_for_box = [], []
    
    for cnt in contours:
        if len(cnt) < min_contour_len: 
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        
        # Check if box is too oblong
        width = x1 - x0
        height = y1 - y0
        aspect_ratio = max(width, height) / min(width, height)
        
        if aspect_ratio > aspect_ratio_thresh:
            # Determine which dimension is shorter
            if width < height:
                # Width is shorter, extend horizontally
                target_width = height  # Make it square
                width_increase = target_width - width
                
                # Sample darkness to left and right of current box
                cx = (x0 + x1) // 2
                left_sample_x = max(0, x0 - width_increase // 2)
                right_sample_x = min(W - 1, x1 + width_increase // 2)
                
                # Sample a vertical strip to get average darkness
                sample_height = min(10, height)
                sample_y_start = y0 + (height - sample_height) // 2
                sample_y_end = sample_y_start + sample_height
                
                left_darkness = gray_image[sample_y_start:sample_y_end, left_sample_x:x0].mean() if left_sample_x < x0 else 255
                right_darkness = gray_image[sample_y_start:sample_y_end, x1:right_sample_x].mean() if x1 < right_sample_x else 255
                
                # Extend toward the darker side (lower pixel values = darker)
                if left_darkness < right_darkness:
                    # Extend more to the left
                    new_x0 = max(0, x0 - int(width_increase * 0.7))
                    new_x1 = min(W, new_x0 + target_width)
                else:
                    # Extend more to the right
                    new_x1 = min(W, x1 + int(width_increase * 0.7))
                    new_x0 = max(0, new_x1 - target_width)
                
                x0, x1 = new_x0, new_x1
                
            else:
                # Height is shorter, extend vertically
                target_height = width  # Make it square
                height_increase = target_height - height
                
                # Sample darkness above and below current box
                cy = (y0 + y1) // 2
                top_sample_y = max(0, y0 - height_increase // 2)
                bottom_sample_y = min(H - 1, y1 + height_increase // 2)
                
                # Sample a horizontal strip to get average darkness
                sample_width = min(10, width)
                sample_x_start = x0 + (width - sample_width) // 2
                sample_x_end = sample_x_start + sample_width
                
                top_darkness = gray_image[top_sample_y:y0, sample_x_start:sample_x_end].mean() if top_sample_y < y0 else 255
                bottom_darkness = gray_image[y1:bottom_sample_y, sample_x_start:sample_x_end].mean() if y1 < bottom_sample_y else 255
                
                # Extend toward the darker side (lower pixel values = darker)
                if top_darkness < bottom_darkness:
                    # Extend more to the top
                    new_y0 = max(0, y0 - int(height_increase * 0.7))
                    new_y1 = min(H, new_y0 + target_height)
                else:
                    # Extend more to the bottom
                    new_y1 = min(H, y1 + int(height_increase * 0.7))
                    new_y0 = max(0, new_y1 - target_height)
                
                y0, y1 = new_y0, new_y1
        
        boxes.append([x0, y0, x1, y1])
        cx, cy = x0 + (x1 - x0)/2, y0 + (y1 - y0)/2
        points_for_box.append([(cx, cy)])
    
    return boxes, points_for_box

def multi_scale_box_masks(predictor, image, box, point, pad_ratios):
    H, W = image.shape[:2]
    masks, ious, logits_all = [], [], []
    for pr in pad_ratios:
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        dx, dy = w * pr, h * pr
        crop = [max(0, x0 - dx), max(0, y0 - dy), min(W, x1 + dx), min(H, y1 + dy)]
        pc = np.array([[point]], dtype=float)
        pl = np.array([[1]], dtype=int)
        masks_pred, ious_pred, logits = predictor.predict(
            box=np.array([crop], dtype=float), point_coords=pc, point_labels=pl,
            multimask_output=False, return_logits=True)
        masks.append(masks_pred[0])
        ious.append(float(ious_pred[0]))
        logits_all.append(logits[0])
    best = int(np.argmax(ious))
    return masks[best], ious[best], logits_all[best]



def watershed_split(mask, peak_filter_size=15, min_region_area=30, max_region_area=None):
    from scipy import ndimage
    from skimage.measure import regionprops, label

    if mask.sum() < min_region_area:
        return []

    # Distance transform
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)

    # Local maxima
    local_max = ndimage.maximum_filter(dist, size=peak_filter_size) == dist
    local_max &= mask
    markers, num_peaks = ndimage.label(local_max)

    if num_peaks < 2:
        return []

    # Watershed
    mask_rgb = np.stack([mask * 255]*3, axis=-1).astype(np.uint8)
    wshed = cv2.watershed(mask_rgb, markers.astype(np.int32))

    # Extract regions
    split_masks = []
    for label_id in np.unique(wshed):
        if label_id <= 0:
            continue
        region = (wshed == label_id)
        area = region.sum()
        if area < min_region_area:
            continue
        if max_region_area is not None and area > max_region_area:
            continue
        split_masks.append(region.astype(bool))
    return split_masks


def should_split(mask, ecc_thresh=0.85, solidity_thresh=0.95):
    props = regionprops(label(mask.astype(int)))
    if not props:
        return False

    region = props[0]
    if region.eccentricity > ecc_thresh:
        return False
    if region.solidity > solidity_thresh:
        return False
    return True


def filter_contained_masks(anns, containment_thresh=0.9):
    """
    Remove masks that are more than `containment_thresh` contained within another.

    Args:
        anns: list of dicts with "segmentation" keys
        containment_thresh: float in (0, 1)

    Returns:
        Filtered list of annotations
    """
    keep = [True] * len(anns)
    for i, ann_i in enumerate(anns):
        mask_i = ann_i["segmentation"]
        area_i = mask_i.sum()
        for j, ann_j in enumerate(anns):
            if i == j or not keep[j]:
                continue
            mask_j = ann_j["segmentation"]
            area_j = mask_j.sum()

            intersection = np.logical_and(mask_i, mask_j).sum()

            # Compute % of i inside j and vice versa
            containment_i = intersection / area_i
            containment_j = intersection / area_j

            if containment_i > containment_thresh or containment_j > containment_thresh:
                # Remove the smaller one
                if area_i < area_j:
                    keep[i] = False
                    break
                else:
                    keep[j] = False  # j gets removed
    return [ann for ann, k in zip(anns, keep) if k]