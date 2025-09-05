import os
import json
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.integrate import quad

# ---- CONFIGURATION ----
experiment_dir = r"G:\My Drive\Master's Data Processing\Both Viscosities\10 cSt\15 Degree\180F\0_2 Percent Trial 1\4_5"
json_path = os.path.join(experiment_dir, "per_frame_props.json")
matlab_path = os.path.join(experiment_dir, "MATLAB Results", "median_image_stats.mat")
output_dir = r"C:\Users\anoon\Pictures\Publication Figures"  # <-- Set output directory here
output_name = "diameter_distribution_10cst_180_45GPM.png"
num_bins = 30

# ---- LOAD DATA ----
with open(json_path, "r") as f:
    frame_data = json.load(f)

diameters_pix = [
    bubble["diameter"]
    for frame in frame_data.values()
    for bubble in frame
]

mat = scipy.io.loadmat(matlab_path)
scale = float(mat["median_image_stats"]["scale"][0][0].item())
diameters_um = np.array(diameters_pix) * scale

# ---- CALCULATE STATS ----
D30 = (np.sum(diameters_um ** 3) / len(diameters_um)) ** (1 / 3)
D32 = np.sum(diameters_um ** 3) / np.sum(diameters_um ** 2)

log_diameters = np.log(diameters_um)
logmu = np.mean(log_diameters)
logsigma = np.std(log_diameters)

# ---- PDF Integration Check ----
pdf_func = lambda d: lognorm.pdf(d, s=logsigma, scale=np.exp(logmu))
integral, _ = quad(pdf_func, 0, 1000)
print(f"Integral of lognormal PDF from 0 to 1000 µm: {integral:.4f}")

# ---- PLOTTING CONFIG ----
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 14,
    "axes.linewidth": 1.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
})

x = np.linspace(0.1, np.percentile(diameters_um, 99) * 1.5, 500)
pdf = lognorm.pdf(x, s=logsigma, scale=np.exp(logmu))

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# --- Normal Scale Plot (left) ---
axs[0].hist(
    diameters_um,
    bins=num_bins,
    density=True,
    alpha=0.4,
    label=f"Histogram (N={len(diameters_um)})",
    color="gray",
    edgecolor='black',
    linewidth=0.5,
)
axs[0].plot(x, pdf, label="Lognormal PDF", color='black', linewidth=2)
#axs[0].axvline(D32, linestyle='--', color='black', linewidth=2, label=f'D32 = {D32:.1f} µm')
axs[0].axvline(D30, linestyle=':', color='black', linewidth=2, label=f'D30 = {D30:.0f} µm')
#axs[0].set_title("Diameter Distribution (Normal Scale)")
axs[0].set_xlabel("Diameter (µm)")
axs[0].set_ylabel("Probability Density")
axs[0].legend(loc='upper right')
legend = axs[0].legend(frameon=True)
legend.get_frame().set_facecolor('white')   # Solid background
legend.get_frame().set_alpha(1.0)           # No transparency
legend.get_frame().set_edgecolor('black')   # Thin black stroke
legend.get_frame().set_linewidth(0.8)       # Stroke thickness
axs[0].grid(True, linestyle=':', color='gray', alpha=0.3)
axs[0].autoscale(enable=True, axis='x', tight=True)

# --- Log(Diameter) Plot (right) ---
log_bins = np.logspace(np.log10(np.min(diameters_um)), np.log10(np.max(diameters_um)), num_bins)
from sklearn.metrics import r2_score

# --- Log(Diameter) Plot (right) ---
log_d = np.log(diameters_um)
hist_vals, hist_edges = np.histogram(log_d, bins=num_bins, density=True)
hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2

normal_pdf = lambda x: (1 / (logsigma * np.sqrt(2 * np.pi))) * np.exp(-(x - logmu) ** 2 / (2 * logsigma ** 2))
normal_vals = normal_pdf(hist_centers)

r2 = r2_score(hist_vals, normal_vals)

axs[1].hist(
    log_d,
    bins=num_bins,
    density=True,
    alpha=0.4,
    label=f"Histogram (N={len(log_d)})",
    color="gray",
    edgecolor='black',
    linewidth=0.5,
)
xlog = np.linspace(min(log_d)*0.9, max(log_d)*1.2, 500)
axs[1].plot(xlog, normal_pdf(xlog), color='black', linewidth=2,
            label=f"Normal PDF\n$\\mu = {logmu:.2f},\\ \\sigma = {logsigma:.2f},\\ \\mathrm{{R}}^2 = {r2:.3f}$")
axs[1].set_xlabel("log(Diameter) (ln µm)")
axs[1].set_ylabel("Probability Density")
axs[1].legend(loc='upper right')
legend.get_frame().set_facecolor('white')   # Solid background
legend.get_frame().set_alpha(1.0)           # No transparency
legend.get_frame().set_edgecolor('black')   # Thin black stroke
legend.get_frame().set_linewidth(0.8)       # Stroke thickness
axs[1].grid(True, linestyle=':', color='gray', alpha=0.3)
axs[1].autoscale(enable=True, axis='x', tight=True)
plt.tight_layout()

# ---- EXPORT TO PNG ----
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, output_name)
plt.savefig(output_path, dpi=500)
print(f"Figure saved to: {output_path}")

plt.show()
