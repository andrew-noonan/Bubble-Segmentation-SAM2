import cv2
import numpy as np
from scipy import ndimage
from skimage.measure import regionprops, label
from skimage.morphology import erosion, dilation, disk, remove_small_objects
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def improved_watershed_split(mask, peak_filter_size=15, min_region_area=30, max_region_area=None):
    """
    More aggressive watershed splitting.
    """
    if mask.sum() < min_region_area:
        return []

    # Clean up the mask
    mask_clean = mask.astype(np.uint8)
    mask_clean = remove_small_objects(mask_clean.astype(bool), min_size=min_region_area//8)  # Less aggressive cleanup
    
    if mask_clean.sum() < min_region_area:
        return []

    # Distance transform
    dist = cv2.distanceTransform(mask_clean.astype(np.uint8), cv2.DIST_L2, 5)
    dist = cv2.GaussianBlur(dist, (3, 3), 0.8)  # Less smoothing to preserve peaks

    # More aggressive peak detection
    peaks = peak_local_max(
        dist, 
        min_distance=max(5, min(mask.shape) // 8),  # Adaptive distance based on mask size
        threshold_abs=dist.max() * 0.2,  # Lower threshold to find more peaks
        exclude_border=2
    )
    
    if len(peaks) < 2:
        # Try even more aggressive settings
        peaks = peak_local_max(
            dist, 
            min_distance=3,  # Very close peaks allowed
            threshold_abs=dist.max() * 0.15,  # Even lower threshold
            exclude_border=1
        )
    
    if len(peaks) < 2:  # Still need at least 2 peaks
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

    # Extract regions with relaxed validation
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
            
        # More lenient shape validation
        props = regionprops(label(region.astype(int)))
        if props and props[0].eccentricity < 0.95 and props[0].solidity > 0.5:  # More lenient
            split_masks.append(region.astype(bool))
    
    return split_masks


def should_split(mask, ecc_thresh=0.90, solidity_thresh=0.90):
    """
    More aggressive splitting decision for attached bubbles.
    """
    props = regionprops(label(mask.astype(int)))
    if not props:
        return False

    region = props[0]
    
    # Multiple criteria - only need ONE to trigger splitting
    criteria = []
    
    # 1. Has significant concavities (convex hull much larger than actual area)
    if region.area > 0:
        concavity_ratio = region.convex_area / region.area
        criteria.append(concavity_ratio > 1.3)  # Lowered threshold
    
    # 2. Low solidity (has indentations)
    criteria.append(region.solidity < solidity_thresh)
    
    # 3. Elongated but not extremely so (could be two circles touching)
    criteria.append(0.3 < region.eccentricity < ecc_thresh)
    
    # 4. Large area relative to bounding box but with gaps
    criteria.append(region.extent < 0.75)
    
    # 5. Check aspect ratio of bounding box (elongated shapes often need splitting)
    bbox_aspect = max(region.bbox[2] - region.bbox[0], region.bbox[3] - region.bbox[1]) / \
                 min(region.bbox[2] - region.bbox[0], region.bbox[3] - region.bbox[1])
    criteria.append(bbox_aspect > 1.4)
    
    # Trigger splitting if ANY criteria is met (much more aggressive)
    should_attempt = any(criteria)
    
    # Debug info - remove this later
    #if should_attempt:
        #print(f"Should split: concavity={region.convex_area/region.area:.2f}, "
        #      f"solidity={region.solidity:.2f}, ecc={region.eccentricity:.2f}, "
        #      f"extent={region.extent:.2f}, aspect={bbox_aspect:.2f}")
    return should_attempt


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