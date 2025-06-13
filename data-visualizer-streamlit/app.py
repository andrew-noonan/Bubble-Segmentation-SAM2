import os
import json
import itertools
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import cv2
from scipy.io import loadmat
import streamlit.components.v1 as components
import re
# --- Utility Functions ---
def getDirectories(basePath, flowRates, temps, angles, aeration_rates, trials):
    all_combinations = list(itertools.product(angles, temps, aeration_rates, trials, flowRates))
    all_dirs = []
    for angle, temp, aeration, trial, flow in all_combinations:
        aeration_str = str(aeration).replace('.', '_')
        flow_str = str(flow).replace('.', '_')
        rel_path = f"{angle} Degree/{temp}F/{aeration_str} Percent Trial {trial}/{flow_str}"
        full_path = os.path.join(basePath, rel_path)
        if os.path.exists(full_path):
            all_dirs.append((full_path, angle, temp, aeration, trial, flow))
    return all_dirs

def parse_mat_struct(matobj):
    return {field: matobj[field][0, 0] if matobj[field].ndim == 2 else matobj[field] for field in matobj.dtype.names}

def safe_get_scalar(obj, field):
    val = obj.get(field, np.nan)
    return float(val) if np.isscalar(val) or (isinstance(val, np.ndarray) and val.size == 1) else np.nan

@st.cache_data
def load_sam2_json(path):
    with open(path, "r") as f:
        return json.load(f)

@st.cache_data
def load_matlab_bubbles(path):
    raw = loadmat(path, squeeze_me=True)["frameData_Reviewed"]
    mat_bubbles = {}
    for entry in raw:
        if not hasattr(entry, "normalizedPath") or not hasattr(entry, "circles"):
            continue
        frame_name = os.path.basename(os.path.normpath(entry.normalizedPath))
        circles = []
        for c in entry.circles:
            centroid = tuple(c.Centroid)
            diameter = float(c.DiameterPixels)
            circles.append((centroid, diameter))
        mat_bubbles[frame_name] = circles
    return mat_bubbles

@st.cache_data(show_spinner=False)
def scan_valid_directories(base_dir):
    angles, temps, aerations, trials, flows = set(), set(), set(), set(), set()
    for root, dirs, _ in os.walk(base_dir):
        parts = root.replace("\\", "/").split("/")
        if len(parts) < 5:
            continue
        try:
            angle = int(parts[-4].split()[0])
            temp = int(parts[-3].replace("F", ""))
            aer = float(parts[-2].split()[0].replace("_", "."))
            trial = int(parts[-2].split()[-1])
            flow = float(parts[-1].replace("_", "."))
        except Exception:
            continue
        angles.add(angle)
        temps.add(temp)
        aerations.add(aer)
        trials.add(trial)
        flows.add(flow)
    return sorted(angles), sorted(temps), sorted(aerations), sorted(trials), sorted(flows)

# --- UI Setup ---
st.title("🔬 Experiment Viewer")

# --- Step 1: Base directory input ---
base_dir = st.text_input("Enter base directory", value=r"G:\My Drive\Master's Data Processing\Thesis Data")

if not base_dir or not os.path.isdir(base_dir):
    st.warning("Please enter a valid base directory.")
    st.stop()

# --- Step 2: Scan for valid values ---
angles, temps, aerations, trials, flows = scan_valid_directories(base_dir)
if not angles:
    st.warning("No valid experiment directories found.")
    st.stop()

# --- Step 3: User filter controls ---
st.subheader("🎛️ Filter Experiment")
col1, col2 = st.columns(2)
with col1:
    angle_sel = st.selectbox("Venturi Angle", angles)
    temp_sel = st.selectbox("Temperature (F)", temps)
    aeration_sel = st.selectbox("Aeration Rate", aerations)
with col2:
    trial_sel = st.selectbox("Trial Number", trials)
    flow_sel = st.selectbox("Flow Rate", flows)

# --- Step 4: Experiment selection button ---
if st.button("Select Experiment"):
    dirs = getDirectories(base_dir, [flow_sel], [temp_sel], [angle_sel], [aeration_sel], [trial_sel])
    if dirs:
        selected_path, *_ = dirs[0]
        st.session_state["selected_path"] = selected_path
        st.success(f"✅ Selected: {selected_path}")
    else:
        st.error("No matching experiment directory found.")

# --- Step 4: Load SAM summary ---
if "selected_path" in st.session_state:
    selected_path = st.session_state["selected_path"]
    summary_path = os.path.join(selected_path, "experiment_summary.csv")
    sam_data = {}
    if os.path.isfile(summary_path):
        try:
            df = pd.read_csv(summary_path)
            sam_data = {
                "Count": int(df["count"].values[0]),
                "D32": float(df["d32"].values[0]),
                "D_v (D30)": float(df["dv"].values[0]),
                "LogMu": float(df["log_mu"].values[0]),
                "LogSigma": float(df["log_sigma"].values[0]),
            }
        except Exception as e:
            st.error(f"Error reading SAM summary: {e}")
    else:
        st.warning("experiment_summary.csv not found.")

    # --- Step 5: Load MATLAB parameters ---
    mat_path = os.path.join(selected_path, "MATLAB Results", "lognormal_fit_params2.mat")
    mat_data = {}
    if os.path.isfile(mat_path):
        try:
            mat = loadmat(mat_path)
            if "tempLogData" in mat:
                raw = mat["tempLogData"]
                if isinstance(raw, np.ndarray) and raw.dtype.names:
                    logdata = parse_mat_struct(raw[0, 0])
                    mat_data = {
                        "LogMu": safe_get_scalar(logdata, 'mu'),
                        "LogSigma": safe_get_scalar(logdata, 'sigma'),
                        "D32": safe_get_scalar(logdata, 'SauterMeanDiameter'),
                        "D_v (D30)": safe_get_scalar(logdata, 'D_v'),
                    }
        except Exception as e:
            st.error(f"Error reading MATLAB file: {e}")
    else:
        st.warning("MATLAB lognormal_fit_params2.mat not found.")

    # --- Step 6: Side-by-side comparison ---
    if sam_data and mat_data:
        st.subheader("📈 Comparison: SAM2 vs MATLAB")
        comparison_df = pd.DataFrame({
            "SAM2": sam_data,
            "MATLAB": mat_data
        })
        st.dataframe(comparison_df.round(1))

# --- Step 5: Viewer logic ---
# --- Step 5: Viewer logic ---
if "selected_path" in st.session_state:
    selected_path = st.session_state["selected_path"]
    norm_path = os.path.join(selected_path, "3 - Normalized")
    sam2_path = os.path.join(selected_path, "per_frame_props.json")
    matlab_path = os.path.join(selected_path, "MATLAB Results", "experiment_data_reviewed.mat")

    if not os.path.isfile(sam2_path) or not os.path.isfile(matlab_path):
        st.warning("Missing overlay data.")
        st.stop()

    # Load data
    sam2_data = load_sam2_json(os.path.normpath(sam2_path))
    mat_data = loadmat(os.path.normpath(matlab_path))
    frame_data = mat_data["frameData_Reviewed"]

    frame_files = sorted(sam2_data.keys())
    if not frame_files:
        st.error("No frame data found.")
        st.stop()

    # Frame navigation state
    if "frame_index" not in st.session_state:
        st.session_state.frame_index = 0


    # Slider
    st.session_state.frame_index = st.slider("Frame #", 0, len(frame_files) - 1, st.session_state.frame_index)

    frame_name = frame_files[st.session_state.frame_index]
    image_path = os.path.normpath(os.path.join(norm_path, frame_name))
    if not os.path.isfile(image_path):
        st.error(f"Image not found: {image_path}")
        st.stop()

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # Prepare MATLAB circle data
    def get_matlab_circles(frame_name):
        frame_match = re.search(r"Frame_(\d+)\.png", frame_name)
        if not frame_match:
            return []
        frame_num = int(frame_match.group(1))
        for i in range(frame_data.shape[1]):
            entry = frame_data[0, i]
            if entry["frameNumber"].item() == frame_num:
                circles = entry["circles"]
                if circles.size == 0:
                    return []
                return [
                    (tuple(c["Centroid"].flatten()), c["DiameterPixels"].item())
                    for i in range(circles.shape[1])
                    for c in [circles[0, i]]
                ]
        return []

    matlab_circles = get_matlab_circles(frame_name)
    sam_circles = [(tuple(ann["centroid"]), ann["diameter"]) for ann in sam2_data.get(frame_name, [])]

    # Plot 2 subplots: SAM and MATLAB overlays
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    titles = ["SAM2 Overlay", "MATLAB Overlay"]

    for ax, title in zip(axs, titles):
        ax.imshow(image)
        ax.set_title(f"{title}\n{frame_name}")
        ax.axis("off")

    # --- SAM circles (lime, dotted) ---
    for (y, x), d in sam_circles:
        circle = plt.Circle((x, y), d / 2, color="red", fill=False, linewidth=2.0, linestyle=":")
        axs[0].add_patch(circle)

    # --- MATLAB circles (blue, dotted) ---
    for (x, y), d in matlab_circles:
        circle = plt.Circle((x, y), d / 2, color="blue", fill=False, linewidth=2.0, linestyle=":")
        axs[1].add_patch(circle)


    st.pyplot(fig)

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Prev") and st.session_state.frame_index > 0:
            st.session_state.frame_index -= 1
    with col3:
        if st.button("Next ➡️") and st.session_state.frame_index < len(frame_files) - 1:
            st.session_state.frame_index += 1
else:
    st.info("📂 Select an experiment to begin.")