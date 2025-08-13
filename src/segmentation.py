import cv2
import numpy as np
from scipy import ndimage
from skimage.measure import regionprops, label
from skimage.morphology import erosion, dilation, disk, remove_small_objects
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

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