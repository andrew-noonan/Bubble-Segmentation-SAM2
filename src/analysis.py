import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm, norm
from skimage.measure import regionprops, label
import ast
from sklearn.metrics import r2_score
import cv2
def compute_props(masks,diamOffsetProps=2):
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
            'diameter': 2 * np.sqrt(m.sum() / np.pi)-diamOffsetProps,
            'circularity': circularity
        })
    return props

def summarize_props(props, circularity_thresh = 0.6):
    diameters = [p['diameter'] for p in props if p['circularity'] >= circularity_thresh]
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
    }

def plot_diameter_histogram_from_summary(base_dir):
    summary_path = os.path.join(base_dir, "experiment_summary.csv")
    if not os.path.exists(summary_path):
        print(f"Summary not found at {summary_path}")
        return

    df = pd.read_csv(summary_path)
    if 'diameters' not in df.columns:
        print("No 'diameters' column found in summary.")
        return

    # Parse stringified list
    try:
        diameters = ast.literal_eval(df['diameters'].values[0])
        diameters = np.array(diameters)
    except Exception as e:
        print(f"Error reading diameters: {e}")
        return

    if diameters.size == 0:
        print("No diameters found.")
        return

    log_mu = df['log_mu'].values[0]
    log_sigma = df['log_sigma'].values[0]

    # Lognormal fit in diameter-space
    shape = log_sigma
    scale = np.exp(log_mu)
    x = np.linspace(diameters.min(), diameters.max(), 500)
    pdf = lognorm.pdf(x, s=shape, loc=0, scale=scale)

    # Plot histogram in original space
    plt.figure(figsize=(8, 5))
    plt.hist(diameters, bins=30, density=True, alpha=0.6, color='gray', edgecolor='black', label='Observed')
    plt.plot(x, pdf, 'r-', lw=2, label=f'Lognormal Fit\nlog μ={log_mu:.2f}, log σ={log_sigma:.2f}')
    plt.xlabel("Bubble Diameter (pixels)")
    plt.ylabel("Probability Density")
    plt.title("Histogram of Bubble Diameters with Lognormal Fit")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Log-space histogram + fit
    log_d = np.log(diameters)
    log_x = np.linspace(log_d.min(), log_d.max(), 500)
    normal_pdf = stats.norm.pdf(log_x, loc=log_mu, scale=log_sigma)

    # Compute predicted densities for R²
    hist_vals, hist_bins = np.histogram(log_d, bins=30, density=True)
    hist_centers = 0.5 * (hist_bins[1:] + hist_bins[:-1])
    pred_vals = stats.norm.pdf(hist_centers, loc=log_mu, scale=log_sigma)
    r2 = r2_score(hist_vals, pred_vals)

    # Plot in log-space
    plt.figure(figsize=(8, 5))
    plt.hist(log_d, bins=30, density=True, alpha=0.6, color='gray', edgecolor='black', label='Observed (log-space)')
    plt.plot(log_x, normal_pdf, 'r-', lw=2, label=f'Normal Fit in Log-space\nμ={log_mu:.2f}, σ={log_sigma:.2f}\n$R^2$={r2:.3f}')
    plt.xlabel("log(Diameter)")
    plt.ylabel("Probability Density")
    plt.title("Log-space Histogram of Bubble Diameters with Normal Fit")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
