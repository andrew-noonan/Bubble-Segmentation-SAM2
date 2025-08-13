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


import cv2
import numpy as np
from scipy import ndimage
from skimage.measure import regionprops, label
from skimage.morphology import erosion, dilation, disk, remove_small_objects
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def improved_watershed_split(mask, peak_filter_size=15, min_region_area=30, max_region_area=None, 
                           erosion_radius=2, min_peak_distance=10, distance_method='geodesic'):
    """
    Improved watershed splitting with better seed detection and preprocessing.
    
    Args:
        mask: Binary mask to split
        peak_filter_size: Size for local maxima detection
        min_region_area: Minimum area for valid regions
        max_region_area: Maximum area for valid regions
        erosion_radius: Radius for morphological erosion before distance transform
        min_peak_distance: Minimum distance between peaks
        distance_method: 'euclidean' or 'geodesic' distance transform
    """
    if mask.sum() < min_region_area:
        return []

    # Preprocessing: smooth the mask to reduce noise
    mask_clean = mask.astype(np.uint8)
    
    # Light erosion followed by dilation to smooth boundaries
    if erosion_radius > 0:
        struct_elem = disk(erosion_radius)
        mask_clean = erosion(mask_clean, struct_elem)
        mask_clean = dilation(mask_clean, struct_elem)
    
    # Remove small noise
    mask_clean = remove_small_objects(mask_clean.astype(bool), min_size=min_region_area//4)
    
    if mask_clean.sum() < min_region_area:
        return []

    # Better distance transform
    if distance_method == 'geodesic':
        # Geodesic distance is better for irregular shapes
        dist = cv2.distanceTransform(mask_clean.astype(np.uint8), cv2.DIST_L2, 5)
        # Apply Gaussian smoothing to distance transform
        dist = cv2.GaussianBlur(dist, (5, 5), 1.0)
    else:
        dist = cv2.distanceTransform(mask_clean.astype(np.uint8), cv2.DIST_L2, 5)

    # Improved peak detection using skimage's peak_local_maxima
    peaks = peak_local_max(dist, min_distance=min_peak_distance, 
                             threshold_abs=dist.max() * 0.3,  # Only consider significant peaks
                             exclude_border=3)
    
    if len(peaks[0]) < 2:  # Need at least 2 peaks to split
        return []

    # Create markers from peaks
    markers = np.zeros_like(dist, dtype=np.int32)
    for i, (y, x) in enumerate(zip(peaks[0], peaks[1])):
        if mask_clean[y, x]:  # Ensure peak is within mask
            markers[y, x] = i + 1

    # Apply watershed
    try:
        wshed = watershed(-dist, markers, mask=mask_clean)
    except:
        # Fallback to OpenCV watershed if skimage fails
        mask_rgb = np.stack([mask_clean * 255] * 3, axis=-1).astype(np.uint8)
        wshed = cv2.watershed(mask_rgb, markers)

    # Extract and validate regions
    split_masks = []
    for label_id in np.unique(wshed):
        if label_id <= 0:
            continue
        
        region = (wshed == label_id)
        area = region.sum()
        
        # Area filtering
        if area < min_region_area:
            continue
        if max_region_area is not None and area > max_region_area:
            continue
            
        # Shape validation - ensure reasonably circular regions
        props = regionprops(label(region.astype(int)))
        if props:
            # Reject very elongated regions (likely splitting artifacts)
            if props[0].eccentricity > 0.9:
                continue
            # Reject regions with very low solidity (likely fragments)
            if props[0].solidity < 0.7:
                continue
                
        split_masks.append(region.astype(bool))
    
    return split_masks


def enhanced_should_split(mask, ecc_thresh=0.85, solidity_thresh=0.85, 
                         area_ratio_thresh=2.0, convexity_thresh=0.8):
    """
    Enhanced splitting decision with multiple geometric criteria.
    
    Args:
        mask: Binary mask to evaluate
        ecc_thresh: Maximum eccentricity (lower = more circular required)
        solidity_thresh: Minimum solidity (higher = more solid required) 
        area_ratio_thresh: Maximum ratio of convex hull area to actual area
        convexity_thresh: Minimum convexity score
    """
    props = regionprops(label(mask.astype(int)))
    if not props:
        return False

    region = props[0]
    
    # Multi-criteria evaluation
    criteria = {
        'eccentricity': region.eccentricity < ecc_thresh,  # Not too elongated
        'solidity': region.solidity < solidity_thresh,     # Has concavities
        'area_ratio': region.convex_area / region.area > area_ratio_thresh,  # Significant concavity
        'extent': region.extent < 0.8,  # Doesn't fill its bounding box well
    }
    
    # Additional convexity check
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = contours[0]
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)
        if hull_area > 0:
            convexity = contour_area / hull_area
            criteria['convexity'] = convexity < convexity_thresh
    
    # Require at least 2 criteria to be met for splitting
    return sum(criteria.values()) >= 2


def smart_filter_contained_masks(anns, containment_thresh=0.8, area_similarity_thresh=0.3):
    """
    Smarter filtering that considers both containment and area similarity.
    
    Args:
        anns: list of dicts with "segmentation" keys
        containment_thresh: float in (0, 1) for containment filtering
        area_similarity_thresh: float in (0, 1) for area similarity filtering
    """
    if len(anns) <= 1:
        return anns
        
    keep = [True] * len(anns)
    
    for i, ann_i in enumerate(anns):
        if not keep[i]:
            continue
            
        mask_i = ann_i["segmentation"]
        area_i = mask_i.sum()
        
        for j, ann_j in enumerate(anns):
            if i >= j or not keep[j]:  # Only check each pair once
                continue
                
            mask_j = ann_j["segmentation"]
            area_j = mask_j.sum()
            
            intersection = np.logical_and(mask_i, mask_j).sum()
            union = np.logical_or(mask_i, mask_j).sum()
            
            # IoU for very similar masks
            iou = intersection / union if union > 0 else 0
            
            # Containment ratios
            containment_i_in_j = intersection / area_i if area_i > 0 else 0
            containment_j_in_i = intersection / area_j if area_j > 0 else 0
            
            # Area similarity ratio
            area_ratio = min(area_i, area_j) / max(area_i, area_j) if max(area_i, area_j) > 0 else 0
            
            should_remove = False
            
            # High IoU suggests very similar masks
            if iou > 0.7:
                should_remove = True
            # High containment
            elif containment_i_in_j > containment_thresh or containment_j_in_i > containment_thresh:
                should_remove = True
            # Very similar areas with significant overlap
            elif area_ratio > (1 - area_similarity_thresh) and max(containment_i_in_j, containment_j_in_i) > 0.5:
                should_remove = True
            
            if should_remove:
                # Keep the one with better shape properties
                props_i = regionprops(label(mask_i.astype(int)))[0]
                props_j = regionprops(label(mask_j.astype(int)))[0]
                
                # Score based on circularity and solidity
                score_i = props_i.solidity * (1 - props_i.eccentricity)
                score_j = props_j.solidity * (1 - props_j.eccentricity)
                
                if score_i > score_j:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    
    return [ann for ann, k in zip(anns, keep) if k]


def multi_scale_watershed_split(mask, scales=[3, 5, 7], **kwargs):
    """
    Apply watershed at multiple scales and combine results.
    
    Args:
        mask: Binary mask to split
        scales: List of scales (peak_filter_sizes) to try
        **kwargs: Additional arguments for improved_watershed_split
    """
    all_splits = []
    
    for scale in scales:
        splits = improved_watershed_split(mask, peak_filter_size=scale, **kwargs)
        all_splits.extend(splits)
    
    if not all_splits:
        return []
    
    # Convert to annotation format for filtering
    anns = [{"segmentation": split} for split in all_splits]
    
    # Filter overlapping results
    filtered_anns = smart_filter_contained_masks(anns, containment_thresh=0.7)
    
    return [ann["segmentation"] for ann in filtered_anns]


def adaptive_bubble_splitter(mask, **kwargs):
    """
    Main function that adaptively chooses splitting strategy based on mask properties.
    """
    # First check if splitting is needed
    if not enhanced_should_split(mask, **kwargs):
        return []
    
    # Try improved single-scale watershed first
    splits = improved_watershed_split(mask)
    
    # If that doesn't work well, try multi-scale approach
    if len(splits) < 2:
        splits = multi_scale_watershed_split(mask)
    
    return splits


# Usage example:
def process_bubbles(masks):
    """
    Example usage of the improved functions.
    """
    all_split_masks = []
    
    for mask in masks:
        # Check if mask should be split
        if enhanced_should_split(mask):
            # Try to split it
            split_masks = adaptive_bubble_splitter(mask)
            if split_masks:
                all_split_masks.extend(split_masks)
            else:
                # If splitting failed, keep original
                all_split_masks.append(mask)
        else:
            # No splitting needed
            all_split_masks.append(mask)
    
    # Final filtering to remove duplicates/contained masks
    anns = [{"segmentation": mask} for mask in all_split_masks]
    filtered_anns = smart_filter_contained_masks(anns)
    
    return [ann["segmentation"] for ann in filtered_anns]


def filter_contained_masks(anns, containment_thresh=0.9):
    """
    Remove masks that are more than containment_thresh contained within another.

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

