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

def getDirectories_v2(basePath, viscosities, angles, temps, aeration_rates, flowRates, times):
    """
    Generate valid experiment folder paths for Phase 8 Recirculation based on parameter combinations.
    Only returns paths that exist on the filesystem.

    Folder structure:
    {viscosity} cSt/{angle} Degree/{temp}F/{aeration} Aeration/{flow} GPM/{time} min
    """
    all_combinations = list(itertools.product(viscosities, angles, temps, aeration_rates, flowRates, times))
    all_dirs = []

    for visc, angle, temp, aeration, flow, time in all_combinations:
        aeration_str = str(aeration).replace('.', '_')
        flow_str = str(flow).replace('.', '_')
        time_str = str(time).replace('.', '_')

        rel_path = os.path.join(
            f"{visc} cSt",
            f"{angle} Degree",
            f"{temp}F",
            f"{aeration_str} Aeration",
            f"{flow_str} GPM",
            f"{time_str} min"
        )
        full_path = os.path.join(basePath, rel_path)

        print("Trying path:", full_path) 
        if os.path.exists(full_path):
            all_dirs.append(full_path)

    print(f"Found {len(all_dirs)} valid directories out of {len(all_combinations)} total combinations.")
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
def scan_valid_directories_v2(base_dir):
    viscosities, angles, temps, aerations, flowRates, times = set(), set(), set(), set(), set(), set()
    for root, dirs, _ in os.walk(base_dir):
        parts = root.replace("\\", "/").split("/")
        if len(parts) < 6:
            continue
        try:
            visc = int(parts[-6].split()[0])  # e.g. "20 cSt"
            angle = int(parts[-5].split()[0])  # e.g. "20 Degree"
            temp = int(parts[-4].replace("F", ""))
            aer = float(parts[-3].split()[0].replace("_", "."))
            flow = float(parts[-2].split()[0].replace("_", "."))
            time_raw = parts[-1].split()[0]  # e.g. "3", "3_5"
            time = int(time_raw) if time_raw.isdigit() else float(time_raw.replace("_", "."))
        except Exception:
            continue
        viscosities.add(visc)
        angles.add(angle)
        temps.add(temp)
        aerations.add(aer)
        flowRates.add(flow)
        times.add(time)
    return (sorted(viscosities), sorted(angles), sorted(temps),
            sorted(aerations), sorted(flowRates), sorted(times))


# --- Interface Controls ---
continuous = False
# --- UI Setup ---
st.title("🔬 Experiment Viewer")

# --- Step 1: Base directory input ---
base_dir = st.text_input("Enter base directory", value=r"G:\My Drive\Master's Data Processing\Phase 8 - Recirculation")

if not base_dir or not os.path.isdir(base_dir):
    st.warning("Please enter a valid base directory.")
    st.stop()

# --- Step 2: Scan for valid values ---
viscosities, angles, temps, aerations, flows, times = scan_valid_directories_v2(base_dir)

if not angles:
    st.warning("No valid experiment directories found.")
    st.stop()

col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

with col1:
    viscosity_sel = st.selectbox("Viscosity (cSt)", viscosities)
with col2:
    angle_sel = st.selectbox("Venturi Angle", angles)
with col3:
    temp_sel = st.selectbox("Temperature (F)", temps)
with col4:
    aeration_sel = st.selectbox("Aeration Rate (%)", aerations)
with col5:
    flow_sel = st.selectbox("Flow Rate (GPM)", flows)
with col6:
    time_sel = st.selectbox("Time (min)", times)

# --- Step 4: Experiment selection button ---
if st.button("Select Experiment"):
    dirs = getDirectories_v2(base_dir, [viscosity_sel], [angle_sel], [temp_sel],
                              [aeration_sel], [flow_sel], [time_sel])
    if dirs:
        selected_path = dirs[0]
        st.session_state["selected_path"] = selected_path
        st.success(f"✅ Selected: {selected_path}")
    else:
        st.error("No matching experiment directory found.")

if "selected_path" in st.session_state:
    selected_path = st.session_state["selected_path"]

    # Load relevant paths
    norm_path = os.path.join(selected_path, "3a - Normalized Continuous") if continuous else os.path.join(selected_path, "3 - Normalized")
    sam2_path = os.path.join(selected_path, "per_frame_props.json")
    matlab_path = os.path.join(selected_path, "MATLAB Results", "raw_frame_data.mat")
    selected_path = st.session_state["selected_path"]
    sam_data = {}
    mat_data = {}

    # --- Load SAM summary ---
    summary_path = os.path.join(selected_path, "experiment_summary.csv")
    UM_PER_PIXEL = 5.71  # micron conversion
    if os.path.isfile(summary_path):
        try:
            df = pd.read_csv(summary_path)
            log_mu_pixels = float(df["log_mu"].values[0])
            log_sigma = float(df["log_sigma"].values[0])
            scale = UM_PER_PIXEL
            sam_data = {
                "Count": int(df["count"].values[0]),
                "D32": float(df["d32"].values[0]) * scale,
                "D_v": float(df["dv"].values[0]) * scale,
                "LogMu": log_mu_pixels + np.log(scale),
                "LogSigma": log_sigma,
            }
        except Exception as e:
            st.error(f"Error reading SAM summary: {e}")
    else:
        st.warning("SAM summary (experiment_summary.csv) not found.")

    # --- Load MATLAB parameters ---
    mat_path = os.path.join(selected_path, "MATLAB Results", "lognormal_fit_params2.mat")
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
                        "D_v": safe_get_scalar(logdata, 'D_v'),
                    }
        except Exception as e:
            st.error(f"Error reading MATLAB file: {e}")
    else:
        st.warning("MATLAB lognormal_fit_params2.mat not found.")

    # Load data
    sam2_data = load_sam2_json(sam2_path) if os.path.isfile(sam2_path) else None
    frame_data = loadmat(matlab_path)["frameData"] if os.path.isfile(matlab_path) else None

    # Derive available frames
    if sam2_data:
        frame_files = sorted(sam2_data.keys())
    elif frame_data is not None:
        frame_files = [f"Frame_{int(frame_data[0, i]['frameNumber'])}.png" for i in range(frame_data.shape[1])]
    else:
        st.error("No valid overlay data found.")
        st.stop()

    if not frame_files:
        st.error("No frame data found.")
        st.stop()

    # Init index
    if "frame_index" not in st.session_state:
        st.session_state.frame_index = 0



    # --- Tabs ---
    tab1, tab2, tab3 = st.tabs(["🖼️ Frame Comparison", "🎞️ Export GIF", "📊 Histogram & R²"])

    with tab1:
        st.session_state.frame_index = st.slider("Frame #", 0, len(frame_files) - 1, st.session_state.frame_index)
        frame_name = frame_files[st.session_state.frame_index]
        image_path = os.path.join(norm_path, frame_name)

        if not os.path.isfile(image_path):
            st.error(f"Image not found: {image_path}")
            st.stop()

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        def get_matlab_circles(frame_name):
            if frame_data is None:
                return []
            match = re.search(r"Frame_(\d+)\.png", frame_name)
            if not match:
                return []
            frame_num = int(match.group(1))
            for i in range(frame_data.shape[1]):
                entry = frame_data[0, i]
                if entry["frameNumber"].item() == frame_num:
                    circles = entry["circles"]
                    if circles.size == 0:
                        return []
                    return [(tuple(c["Centroid"].flatten()), c["DiameterPixels"].item()) for i in range(circles.shape[1]) for c in [circles[0, i]]]
            return []

        sam_circles = [(tuple(ann["centroid"]), ann["diameter"]) for ann in sam2_data[frame_name]] if sam2_data and frame_name in sam2_data else []
        if sam2_data and frame_name in sam2_data:
            sam_circles = [(tuple(ann["centroid"]), ann["diameter"]) for ann in sam2_data[frame_name]]
        matlab_circles = get_matlab_circles(frame_name)

        fig, axs = plt.subplots(1, 2, figsize=(18, 9))
        titles = ["SAM2 Overlay", "MATLAB Overlay"]
        for ax, title in zip(axs, titles):
            ax.imshow(image)
            ax.set_title(f"{title}\n{frame_name}")
            ax.axis("off")

        for (y, x), d in sam_circles:
            axs[0].add_patch(plt.Circle((x, y), d / 2, color="blue", fill=True, alpha=0.2))

        for (x, y), d in matlab_circles:
            axs[1].add_patch(plt.Circle((x, y), d / 2, color="red", fill=True, alpha=0.2))

        st.pyplot(fig)
        plt.close(fig)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ Prev") and st.session_state.frame_index > 0:
                st.session_state.frame_index -= 1
        with col3:
            if st.button("Next ➡️") and st.session_state.frame_index < len(frame_files) - 1:
                st.session_state.frame_index += 1
        plt.close(fig)

    with tab2:
        overlay_choice = st.radio("Overlay Type", ["SAM2", "MATLAB"], horizontal=True)
        export_gif = st.button("Generate and Download GIF")

        if export_gif:
            import imageio.v2 as imageio
            from tempfile import TemporaryDirectory

            output_frames = []
            max_frames = 50

            with st.spinner("Generating GIF..."):
                for i, frame_name in enumerate(frame_files[:max_frames]):
                    image_path = os.path.join(norm_path, frame_name)
                    if not os.path.isfile(image_path):
                        continue
                    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                    fig, ax = plt.subplots(figsize=(8, 6.4), dpi=160)
                    ax.imshow(img)
                    ax.axis("off")
                    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

                    if overlay_choice == "SAM2" and sam2_data and frame_name in sam2_data:
                        for (y, x), d in [(tuple(ann["centroid"]), ann["diameter"]) for ann in sam2_data[frame_name]]:
                            ax.add_patch(plt.Circle((x, y), d / 2, color="blue", fill=True, alpha=0.2))

                    elif overlay_choice == "MATLAB" and frame_data is not None:
                        for (x, y), d in get_matlab_circles(frame_name):
                            ax.add_patch(plt.Circle((x, y), d / 2, color="red", fill=True, alpha=0.3))

                    fig.canvas.draw()
                    renderer = fig.canvas.get_renderer()
                    frame = np.frombuffer(renderer.buffer_rgba(), dtype=np.uint8)
                    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                    frame_rgb = frame[:, :, :3]                    
                    output_frames.append(frame)
                    plt.close(fig)

                with TemporaryDirectory() as tmpdir:
                    gif_path = os.path.join(tmpdir, "bubbles_overlay.gif")
                    imageio.mimsave(gif_path, output_frames, fps=3,loop=0)
                    with open(gif_path, "rb") as f:
                        st.download_button("📥 Download GIF", f, file_name="bubbles_overlay.gif", mime="image/gif")

    with tab3:

        st.subheader("📊 Histogram & R² Comparison")
        def format_with_units(df: pd.DataFrame) -> pd.DataFrame:
            renamed = df.rename(index={
                "D32": "D32 (µm)",
                "D_v": "D_v (D30, µm)",
                "LogMu": "LogMu (µm)",
                "LogSigma": "LogSigma",
                "Count": "Count"
            })
            return renamed

        if sam_data and mat_data:
            st.markdown("### 🔬 Summary Comparison Table")
            comparison_df = pd.DataFrame({
                "SAM2": sam_data,
                "MATLAB": mat_data
            })
            st.dataframe(format_with_units(comparison_df).round(1))

        elif sam_data:
            st.markdown("### 🔬 SAM2 Summary Only")
            df = pd.DataFrame(sam_data, index=["SAM2"]).T
            st.dataframe(format_with_units(df).round(1))

        elif mat_data:
            st.markdown("### 🔬 MATLAB Summary Only")
            df = pd.DataFrame(mat_data, index=["MATLAB"]).T
            st.dataframe(format_with_units(df).round(1))

        else:
            st.info("No summary data found from either SAM2 or MATLAB.")

        fig, ax = plt.subplots(figsize=(10, 6))
        plotted = False

        # --- MATLAB histogram ---
        mat_diam_path = os.path.join(selected_path, "MATLAB Results", "diameter_data_reviewed.mat")
        if os.path.isfile(mat_diam_path):
            try:
                diam_data = loadmat(mat_diam_path)
                if "all_diameters" in diam_data:
                    mat_diameters = diam_data["all_diameters"].flatten()
                    count = len(mat_diameters)
                    bins = 40
                    bin_range = (min(mat_diameters), max(mat_diameters))
                    bin_edges = np.linspace(*bin_range, bins + 1)
                    bin_width = bin_edges[1] - bin_edges[0]

                    ax.hist(mat_diameters, bins=bin_edges, density=True, alpha=0.5,
                            label=f"MATLAB ({count} bubbles)", color="red", width=bin_width * 0.9)

                    if mat_data and "LogMu" in mat_data and "LogSigma" in mat_data:
                        from scipy.stats import lognorm
                        mu, sigma = mat_data["LogMu"], mat_data["LogSigma"]
                        x_vals = np.linspace(min(mat_diameters), max(mat_diameters), 500)
                        pdf_vals = lognorm.pdf(x_vals, sigma, scale=np.exp(mu))
                        ax.plot(x_vals, pdf_vals, color="darkred", linestyle="--",
                                label=f"MATLAB Lognormal Fit\nμ = {mu:.2f}, σ = {sigma:.2f}")

                    plotted = True
            except Exception as e:
                st.warning(f"Error loading MATLAB diameters: {e}")

        # --- SAM2 histogram ---
        sam_diameters = []
        if sam2_data:
            for anns in sam2_data.values():
                sam_diameters.extend([ann["diameter"] * UM_PER_PIXEL for ann in anns])
            count = len(sam_diameters)
            if sam_diameters:
                bin_range = (min(sam_diameters), max(sam_diameters))
                bin_edges = np.linspace(*bin_range, bins + 1)
                bin_width = bin_edges[1] - bin_edges[0]

                ax.hist(sam_diameters, bins=bin_edges, density=True, alpha=0.5,
                        label=f"SAM2 ({count} bubbles)", color="blue", width=bin_width * 0.9)

                if sam_data and "LogMu" in sam_data and "LogSigma" in sam_data:
                    mu, sigma = sam_data["LogMu"], sam_data["LogSigma"]
                    x_vals = np.linspace(min(sam_diameters), max(sam_diameters), 500)
                    pdf_vals = lognorm.pdf(x_vals, sigma, scale=np.exp(mu))
                    ax.plot(x_vals, pdf_vals, color="navy", linestyle="--",
                            label=f"SAM2 Lognormal Fit\nμ = {mu:.2f}, σ = {sigma:.2f}")

                plotted = True

        # --- Display combined plot ---
        if plotted:
            ax.set_xlabel("Diameter (µm)")
            ax.set_ylabel("Normalized Count")
            ax.set_title("Bubble Size Distribution")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No diameter data available to plot.")