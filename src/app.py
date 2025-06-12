import os
import re
import json
import ast
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import lognorm, norm
from sklearn.metrics import r2_score

import os
import re
import streamlit as st

def scan_experiments(base_dir):
    results = []
    angle_re = re.compile(r"^(\d+)\s*Degree$", re.IGNORECASE)
    temp_re = re.compile(r"^(\d+)F$", re.IGNORECASE)
    aer_trial_re = re.compile(r"^([\d_]+)\s+Percent\s+Trial\s+(\d+)$", re.IGNORECASE)

    st.write(f"🔍 Scanning: `{base_dir}`")
    if not os.path.isdir(base_dir):
        st.error(f"❌ Not a directory: {base_dir}")
        return results

    for angle_name in os.listdir(base_dir):
        angle_path = os.path.join(base_dir, angle_name)
        if not os.path.isdir(angle_path):
            continue
        am = angle_re.fullmatch(angle_name.strip())
        if not am:
            st.warning(f"⚠️ Skipped angle folder: '{angle_name}'")
            continue
        angle = int(am.group(1))

        for temp_name in os.listdir(angle_path):
            temp_path = os.path.join(angle_path, temp_name)
            if not os.path.isdir(temp_path):
                continue
            tm = temp_re.fullmatch(temp_name.strip())
            if not tm:
                st.warning(f"⚠️ Skipped temp folder: '{temp_name}'")
                continue
            temp = int(tm.group(1))

            for aer_trial_name in os.listdir(temp_path):
                aer_trial_path = os.path.join(temp_path, aer_trial_name)
                if not os.path.isdir(aer_trial_path):
                    continue
                atm = aer_trial_re.fullmatch(aer_trial_name.strip())
                if not atm:
                    st.warning(f"⚠️ Skipped aeration/trial folder: '{aer_trial_name}'")
                    continue
                aeration = float(atm.group(1).replace('_', '.'))
                trial = int(atm.group(2))

                for flow_name in os.listdir(aer_trial_path):
                    flow_path = os.path.join(aer_trial_path, flow_name)
                    if not os.path.isdir(flow_path):
                        continue
                    try:
                        flow = float(flow_name.replace('_', '.'))
                    except ValueError:
                        st.warning(f"⚠️ Skipped flow folder: '{flow_name}'")
                        continue
                    results.append({
                        "path": flow_path,
                        "angle": angle,
                        "temp": temp,
                        "aeration": aeration,
                        "trial": trial,
                        "flow": flow,
                    })
    return results

# ------------------- Streamlit UI -------------------
st.title("📁 Experiment Scanner")

# Let user choose the base directory
base_dir = st.text_input("Enter the base directory path:", "/content/drive/My Drive/YourPathHere")

if st.button("Scan"):
    experiments = scan_experiments(base_dir)
    st.success(f"✅ Found {len(experiments)} experiments.")
    st.json(experiments[:5])  # show a preview




st.title("Bubble Visualization")
base_dir = st.text_input("Base folder path")

if base_dir:
    experiments = scan_experiments(base_dir)
    if not experiments:
        st.write("No experiments found or invalid directory structure.")
    else:
        df = pd.DataFrame(experiments)
        angle = st.selectbox("Angle", sorted(df["angle"].unique()))
        df_t = df[df["angle"] == angle]
        temp = st.selectbox("Temperature (F)", sorted(df_t["temp"].unique()))
        df_t = df_t[df_t["temp"] == temp]
        aer = st.selectbox("Aeration", sorted(df_t["aeration"].unique()))
        df_t = df_t[df_t["aeration"] == aer]
        trial = st.selectbox("Trial", sorted(df_t["trial"].unique()))
        df_t = df_t[df_t["trial"] == trial]
        flow = st.selectbox("Flow", sorted(df_t["flow"].unique()))
        df_sel = df_t[df_t["flow"] == flow]
        if not df_sel.empty:
            exp_path = df_sel.iloc[0]["path"]
            st.write(f"Experiment directory: {exp_path}")
            summary_path = os.path.join(exp_path, "experiment_summary.csv")
            json_path = os.path.join(exp_path, "per_frame_props.json")
            img_dir = os.path.join(exp_path, "3 - Normalized")

            if os.path.isfile(summary_path):
                df_sum = pd.read_csv(summary_path)
                if "diameters" in df_sum.columns:
                    try:
                        diameters = np.array(ast.literal_eval(df_sum["diameters"].iloc[0]))
                    except Exception as e:
                        st.error(f"Could not parse diameters: {e}")
                        diameters = np.array([])
                    if diameters.size > 0:
                        log_mu = df_sum["log_mu"].iloc[0]
                        log_sigma = df_sum["log_sigma"].iloc[0]
                        shape = log_sigma
                        scale = np.exp(log_mu)
                        x = np.linspace(diameters.min(), diameters.max(), 500)
                        pdf = lognorm.pdf(x, s=shape, loc=0, scale=scale)
                        log_d = np.log(diameters)
                        hist_vals, hist_bins = np.histogram(log_d, bins=30, density=True)
                        hist_centers = 0.5 * (hist_bins[1:] + hist_bins[:-1])
                        pred_vals = norm.pdf(hist_centers, loc=log_mu, scale=log_sigma)
                        r2 = r2_score(hist_vals, pred_vals)
                        fig, ax = plt.subplots()
                        ax.hist(diameters, bins=30, density=True, alpha=0.6, edgecolor="black")
                        ax.plot(x, pdf, "r-", label=f"Lognormal Fit (R²={r2:.3f})")
                        ax.set_xlabel("Bubble Diameter (pixels)")
                        ax.set_ylabel("Density")
                        ax.legend()
                        st.pyplot(fig)
            if os.path.isfile(json_path):
                with open(json_path, "r") as f:
                    props_all = json.load(f)
            else:
                props_all = {}

            if os.path.isdir(img_dir):
                frame_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
                if frame_files:
                    index = st.slider("Frame index", 0, len(frame_files)-1, 0)
                    frame_name = frame_files[index]
                    img_path = os.path.join(img_dir, frame_name)
                    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                    show = st.checkbox("Show circles (press 'c' to toggle)", value=True, key="circles")
                    st.markdown(
                        """
                        <script>
                        document.addEventListener('keydown', function(e) {
                            if (e.key === 'c') {
                                const cb = window.parent.document.querySelector('input[id^="circles"]');
                                if (cb) { cb.click(); }
                            }
                        });
                        </script>
                        """,
                        unsafe_allow_html=True,
                    )
                    overlay = image.copy()
                    if show and frame_name in props_all:
                        for p in props_all.get(frame_name, []):
                            y, x = p["centroid"]
                            r = p["diameter"] / 2
                            cv2.circle(overlay, (int(x), int(y)), int(r), (0, 255, 255), 1)
                    st.image(overlay if show else image, caption=frame_name)
        else:
            st.write("No matching experiment directory found.")
else:
    st.info("Enter base folder path above to begin.")