import cv2
import numpy as np
from scipy import ndimage
from skimage.measure import regionprops, label
from skimage.morphology import erosion, dilation, disk, remove_small_objects
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def improved_watershed_split(mask, peak_filter_size=15, min_region_area=30, max_region_area=None):
    """
    Simplified watershed splitting that just works.
    """
    if mask.sum() < min_region_area:
        return []

    # Clean up the mask
    mask_clean = mask.astype(np.uint8)
    mask_clean = remove_small_objects(mask_clean.astype(bool), min_size=min_region_area//4)
    
    if mask_clean.sum() < min_region_area:
        return []

    # Distance transform
    dist = cv2.distanceTransform(mask_clean.astype(np.uint8), cv2.DIST_L2, 5)
    dist = cv2.GaussianBlur(dist, (5, 5), 1.0)

    # Find peaks
    peaks = peak_local_max(dist, min_distance=10, threshold_abs=dist.max() * 0.3, exclude_border=3)
    
    if len(peaks) < 2:  # Need at least 2 peaks to split
        return []

    # Create markers
    markers = np.zeros_like(dist, dtype=np.int32)
    for i, (y, x) in enumerate(peaks):
        if mask_clean[y, x]:
            markers[y, x] = i + 1

    # Watershed
    try:
        wshed = watershed(-dist, markers, mask=mask_clean)
    except:
        # Fallback to OpenCV
        mask_rgb = np.stack([mask_clean * 255] * 3, axis=-1).astype(np.uint8)
        wshed = cv2.watershed(mask_rgb, markers)

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
            
        # Basic shape validation
        props = regionprops(label(region.astype(int)))
        if props and props[0].eccentricity < 0.9 and props[0].solidity > 0.7:
            split_masks.append(region.astype(bool))
    
    return split_masks


def should_split(mask, ecc_thresh=0.85, solidity_thresh=0.85):
    """
    Simplified splitting decision - same as your original but better thresholds.
    """
    props = regionprops(label(mask.astype(int)))
    if not props:
        return False

    region = props[0]
    
    # Simple criteria: if it's not too circular AND not too solid, consider splitting
    not_too_circular = region.eccentricity < ecc_thresh
    has_concavities = region.solidity < solidity_thresh
    significant_concavity = region.convex_area / region.area > 1.5
    
    return not_too_circular and (has_concavities or significant_concavity)


def smart_filter_contained_masks(anns, containment_thresh=0.8):
    """
    Simplified smart filtering - removes the complex area similarity logic.
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
            if i >= j or not keep[j]:
                continue
                
            mask_j = ann_j["segmentation"]
            area_j = mask_j.sum()
            
            intersection = np.logical_and(mask_i, mask_j).sum()
            union = np.logical_or(mask_i, mask_j).sum()
            
            # IoU check
            iou = intersection / union if union > 0 else 0
            
            # Containment check
            containment_i_in_j = intersection / area_i if area_i > 0 else 0
            containment_j_in_i = intersection / area_j if area_j > 0 else 0
            
            # Remove if high overlap
            if iou > 0.7 or containment_i_in_j > containment_thresh or containment_j_in_i > containment_thresh:
                # Keep the one with better shape (more circular and solid)
                props_i = regionprops(label(mask_i.astype(int)))[0]
                props_j = regionprops(label(mask_j.astype(int)))[0]
                
                score_i = props_i.solidity * (1 - props_i.eccentricity)
                score_j = props_j.solidity * (1 - props_j.eccentricity)
                
                if score_i > score_j:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    
    return [ann for ann, k in zip(anns, keep) if k]


def filter_contained_masks(anns, containment_thresh=0.9):
    """
    Your original function - keeping it unchanged.
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

            containment_i = intersection / area_i
            containment_j = intersection / area_j

            if containment_i > containment_thresh or containment_j > containment_thresh:
                if area_i < area_j:
                    keep[i] = False
                    break
                else:
                    keep[j] = False
    return [ann for ann, k in zip(anns, keep) if k]