def compute_props(masks):
    props = []
    for m in masks:
        if m.sum() == 0: continue
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours or len(contours[0]) < 5: continue
        perimeter = cv2.arcLength(contours[0], True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * m.sum() / (perimeter ** 2)
        reg = regionprops(label(m))[0]
        props.append({
            'centroid': reg.centroid,
            'diameter': 2 * np.sqrt(m.sum() / np.pi),
            'circularity': circularity
        })
    return props

def summarize_props(props):
    diameters = [p['diameter'] for p in props if p['circularity'] >= 0.7]
    if len(diameters) == 0:
        return {"count": 0}

    diameters = np.array(diameters)
    d32 = np.sum(diameters**3) / np.sum(diameters**2)  # Surface-volume mean (D[3,2])
    dv = (np.mean(diameters**3))**(1/3)                # Volume-equivalent mean (D[3,0])
    log_d = np.log(diameters)

    return {
        "count": len(diameters),
        "d32": float(d32),
        "dv": float(dv),
        "log_mu": float(np.mean(log_d)),
        "log_sigma": float(np.std(log_d)),
        "diameters": diameters.tolist()