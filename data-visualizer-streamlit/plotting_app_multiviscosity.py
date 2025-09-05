# Updated Streamlit app main entry (refactored for SAM2-only processing with viscosity folder separation)

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# --- Constants ---
UM_PER_PIXEL = 5.71
D_t = 6e-3
D_p = 15.8e-3
T_ref = 298.15
A_throat = np.pi * D_t**2 / 4
GPM_to_m3_s = 1 / (264.172053 * 60)

# --- Fluid property definitions (from screenshot)
fluid_properties = {
    "10 cSt": {"A": 687, "mu_ref": 10, "alpha": 0.00105, "rho_ref": 934, "surfaceTensionYIntercept": 21.6, "surfaceTensionSlope": -0.06},
    "20 cSt": {"A": 752, "mu_ref": 19.7, "alpha": 0.00103, "rho_ref": 949, "surfaceTensionYIntercept": 22.1, "surfaceTensionSlope": -0.06},
    "50 cSt": {"A": 732, "mu_ref": 45, "alpha": 0.000994, "rho_ref": 959, "surfaceTensionYIntercept": 22.3, "surfaceTensionSlope": -0.06}
}

def compute_fluid_properties(tempK, fluid_key):
    props = fluid_properties[fluid_key]
    A, mu_ref, alpha, rho_ref = props["A"], props["mu_ref"], props["alpha"], props["rho_ref"]
    B = np.log10(mu_ref / 1000) - A / T_ref
    mu = 10 ** (A / tempK + B)
    rho = rho_ref * (1 - alpha * (tempK - T_ref))
    gamma = (props["surfaceTensionYIntercept"] + props["surfaceTensionSlope"] * (tempK - 273.15)) / 1000
    return mu, rho, gamma

@st.cache_data
def import_yin_data():
    mu_water = 0.001
    sigma_water = 0.0728
    rho_water = 997
    D_t_yin = 0.023
    D_upstream = 0.053
    theta_yin = 8
    L_yin = (53 - 23) / 2 / np.tan(np.radians(theta_yin)) / 1000  # m

    yin_raw = np.array([
        [138057.9483, 0.9731],
        [168598.2772, 0.8130],
        [199295.2232, 0.6924],
        [229992.1691, 0.5537],
        [260689.1151, 0.4993],
        [291386.0611, 0.4438]
    ])

    Re_upstream = yin_raw[:, 0]
    Re_throat = Re_upstream * (D_upstream / D_t_yin)
    d_v_m = yin_raw[:, 1] / 1000

    V_throat = (Re_throat * mu_water) / (rho_water * D_t_yin)
    Ca = (mu_water * V_throat) / sigma_water
    We = (rho_water * V_throat**2 * D_t_yin) / sigma_water

    return pd.DataFrame({
        'Re_upstream': Re_upstream,
        'Re_t': Re_throat,
        'D_v': d_v_m,
        'Velocity_m_per_s': V_throat,
        'Ca': Ca,
        'We': We,
        'ThroatDiameter_m': D_t_yin,
        'DivergingL_m': L_yin
    })

@st.cache_data
def import_sun_data():
    mu_water = 0.001
    sigma_water = 0.0728
    rho_water = 997
    D_t = 0.025
    D_upstream = 0.05
    theta = 7.5
    L = (50 - 25) / 2 / np.tan(np.radians(theta)) / 1000  # m

    sun_raw = np.array([
        [229646.4949, 0.038018832],
        [244925.1049, 0.036647834],
        [260263.6309, 0.032188324],
        [275542.2409, 0.031450094],
        [290880.7669, 0.029551789],
        [306159.3769, 0.024429379],
        [321437.9868, 0.021943503],
    ])

    Re_water = sun_raw[:, 0]
    d_v_m = sun_raw[:, 1] * D_t
    V_throat = (Re_water * mu_water) / (rho_water * D_t)
    Ca = (mu_water * V_throat) / sigma_water
    We = (rho_water * V_throat**2 * D_t) / sigma_water

    return pd.DataFrame({
        'Re': Re_water,
        'D_v': d_v_m,
        'Velocity_m_per_s': V_throat,
        'Ca': Ca,
        'We': We,
        'ThroatDiameter_m': D_t,
        'DivergingL_m': L
    })

# --- Streamlit Sidebar Toggle ---
st.set_page_config(layout="wide")
st.title("🧪 Microbubble Experiment Loader (SAM2 Only)")
debug_mode = st.sidebar.checkbox("🔍 Debug Mode", value=False)
page = st.sidebar.radio("Select Page", ["Filter Data", "Plot Results"])

# --- Data Loading Function ---
@st.cache_data

def load_all_experiment_data(base_dir):
    records = []
    for fluid_folder in ["10 cSt", "50 cSt"]:
        full_path = os.path.join(base_dir, fluid_folder)
        if not os.path.isdir(full_path):
            continue
        for root, _, files in os.walk(full_path):
            if "experiment_summary.csv" not in files:
                continue
            try:
                parts = root.replace("\\", "/").split("/")
                angle = int(parts[-4].split()[0])
                temp = int(parts[-3].replace("F", ""))
                aer = float(parts[-2].split()[0].replace("_", "."))
                trial = int(parts[-2].split()[-1])
                flow = float(parts[-1].replace("_", "."))
                record = {
                    'Temp': temp, 'FlowRate': flow, 'VenturiAngle': angle,
                    'AeratedFlow': aer, 'Trial': trial, 'Viscosity_cSt': fluid_folder, 'Valid': True
                }
                lv_path = os.path.join(root, 'labview.txt')
                if os.path.isfile(lv_path):
                    try:
                        data = pd.read_csv(lv_path, encoding='utf-8', on_bad_lines='skip')
                        record.update({
                            'MeanTemp': data['Temp (F)'].mean(),
                            'MeanFlow': data['Oil Flow Rate'].mean(),
                            'MeanP1': data['P1'].mean(),
                            'MeanP2': data['P2'].mean(),
                        })
                    except Exception as e:
                        if debug_mode:
                            st.warning(f"Error reading LabVIEW data in {root}: {e}")
                sam_path = os.path.join(root, 'experiment_summary.csv')
                try:
                    sam_df = pd.read_csv(sam_path)
                    record.update({
                        'LogMu': sam_df['log_mu'].iloc[0] + np.log(UM_PER_PIXEL),
                        'LogSigma': sam_df['log_sigma'].iloc[0],
                        'D32': sam_df['d32'].iloc[0] * UM_PER_PIXEL,
                        'D_v': sam_df['dv'].iloc[0] * UM_PER_PIXEL
                    })
                except Exception as e:
                    if debug_mode:
                        st.warning(f"Error reading SAM data in {root}: {e}")
                records.append(record)
            except Exception as e:
                if debug_mode:
                    st.warning(f"Directory parse error: {e}")
                continue

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df

    df['deltaP'] = df.get('MeanP1', np.nan) - df.get('MeanP2', np.nan)
    df['deltaP_Pa'] = df['deltaP'] * 6894.75729
    tempF = df['MeanTemp'].combine_first(df['Temp'])
    df['tempK'] = (tempF - 32) / 1.8 + 273.15

    df['mu'] = np.nan
    df['rho'] = np.nan
    df['Gamma'] = np.nan

    for fluid_key in fluid_properties.keys():
        idx = df['Viscosity_cSt'] == fluid_key
        if idx.any():
            mu, rho, gamma = compute_fluid_properties(df.loc[idx, 'tempK'], fluid_key)
            df.loc[idx, 'mu'] = mu
            df.loc[idx, 'rho'] = rho
            df.loc[idx, 'Gamma'] = gamma

    df['nu'] = df['mu'] / df['rho']
    flow = df['MeanFlow'].combine_first(df['FlowRate'])
    df['V_throat'] = flow * GPM_to_m3_s / A_throat
    df['Reynolds'] = df['V_throat'] * D_t / df['nu']
    df['dynamicPressure'] = 0.5 * df['rho'] * df['V_throat']**2
    df['deltaP_normalized'] = df['deltaP_Pa'] / df['dynamicPressure']
    df['Ca'] = df['mu'] * df['V_throat'] / df['Gamma']
    df['We_D'] = df['rho'] * df['V_throat']**2 * D_t / df['Gamma']
    df['L'] = (D_p - D_t) / np.tan(np.radians(df['VenturiAngle']))
    df['We_L'] = df['rho'] * df['V_throat']**2 * df['L'] / df['Gamma']
    df['ThroatDiameter_m'] = D_t

    return df

# --- Load the data ---
if page == "Filter Data":
    # --- Directory Input ---
    base_dir = st.text_input("Base directory", value=r"G:\My Drive\Master's Data Processing\Both Viscosities")
    if not os.path.isdir(base_dir):
        st.warning("Invalid base path.")
        st.stop()
    df = load_all_experiment_data(base_dir)
    if df.empty:
        st.warning("No experiments found.")
        st.stop()

    st.success(f"Loaded {len(df)} experiments.")

    # --- Filter UI ---
    st.subheader("📊 Filter Data")

    viscosities = sorted(df['Viscosity_cSt'].unique())
    angles_all = sorted(df['VenturiAngle'].dropna().unique())
    temps_all = sorted(df['Temp'].dropna().unique())
    aerations_all = sorted(df['AeratedFlow'].dropna().unique())
    flows_all = sorted(df['FlowRate'].dropna().unique())

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("**Viscosity (cSt)**")
        selected_visc = [v for v in viscosities if st.checkbox(v, value=True)]
    with col2:
        st.markdown("**Angles (°)**")
        selected_angles = [a for a in angles_all if st.checkbox(str(a), value=True, key=f"a{a}")]
    with col3:
        st.markdown("**Temperatures (°F)**")
        selected_temps = [t for t in temps_all if st.checkbox(str(t), value=True, key=f"t{t}")]
    with col4:
        st.markdown("**Aeration (% GPM)**")
        selected_aers = [a for a in aerations_all if st.checkbox(str(a), value=True, key=f"aer{a}")]
    with col5:
        st.markdown("**Flow Rates (GPM)**")
        selected_flows = [f for f in flows_all if st.checkbox(str(f), value=True, key=f"flow{f}")]

    # Apply filter
    filtered_df = df[
        df['Viscosity_cSt'].isin(selected_visc) &
        df['VenturiAngle'].isin(selected_angles) &
        df['Temp'].isin(selected_temps) &
        df['AeratedFlow'].isin(selected_aers) &
        df['FlowRate'].isin(selected_flows)
    ]

    st.success(f"Filtered to {len(filtered_df)} experiments.")
    st.download_button("📅 Export Filtered CSV", data=filtered_df.to_csv(index=False), file_name="filtered_data.csv")
    st.dataframe(filtered_df.head(50))
    yin_data = import_yin_data()
    sun_data = import_sun_data()

    st.session_state['filtered_df'] = filtered_df
    st.session_state['yin_data'] = yin_data
    st.session_state['sun_data'] = sun_data

from scipy.optimize import curve_fit

if page == "Plot Results":
    if 'filtered_df' not in st.session_state:
        st.warning("Please go to 'Filter Data' page first to load data.")
        st.stop()
    
    filtered_df = st.session_state['filtered_df']
    yin_data = st.session_state.get('yin_data')
    sun_data = st.session_state.get('sun_data')
    # --- Tabs for plotting ---
    tabs = st.tabs(["Repeatability", "Flow Rate", "Temperature", "Angle", "Reynolds", "Weber", "Capillary", "Universal ReWe", "PDFs Fixed Flow", "Universal ReCa", "Universal WeCa"])

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.linewidth'] = 1.5  # thicker axes
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linewidth'] = 0.5
    # Define consistent color scheme
    fluid_colors = {
        "10 cSt": 'red',
        "50 cSt": 'black'
    }
    linestyles = ['-', '--', ':', '-.']
    marker_styles = ['o', 's', '^', 'D', 'v', 'P', 'X']

    with tabs[0]:
        st.markdown("### Trial Repeatability (SAM2)")
        var_labels = [r'$\mu_{LN}$', r'$\sigma_{LN}$', r'$d_{32}$', r'$d_{30}$']
        var_keys = ['LogMu', 'LogSigma', 'D32', 'D_v']
        param_cols = ['Viscosity_cSt', 'Temp', 'FlowRate', 'VenturiAngle', 'AeratedFlow']

        filtered_df['param_id'] = filtered_df[param_cols].apply(tuple, axis=1)
        unique_params = filtered_df['param_id'].unique()
        t1_data, t2_data, labels, colors = [], [], [], []

        for param in unique_params:
            subset = filtered_df[filtered_df['param_id'] == param]
            trial1 = subset[subset['Trial'] == 1]
            trial2 = subset[subset['Trial'] == 2]
            if len(trial1) == 1 and len(trial2) == 1:
                t1_data.append([trial1.iloc[0][key] for key in var_keys])
                t2_data.append([trial2.iloc[0][key] for key in var_keys])
                label = f"{param[0]}, {param[1]}°F, {param[2]} GPM"
                labels.append(label)
                colors.append('black' if param[0] == '50 cSt' else 'red')  # 50 cSt = black, 10 cSt = red

        t1_data, t2_data = np.array(t1_data), np.array(t2_data)

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()

        for i in range(4):
            ax = axes[i]
            x, y = t1_data[:, i], t2_data[:, i]
            for xi, yi, ci in zip(x, y, colors):
                ax.scatter(xi, yi, color=ci, s=40, alpha=1, marker='x',linewidth=1.2)
            min_val, max_val = min(x.min(), y.min()), max(x.max(), y.max())
            margin = 0.05 * (max_val - min_val)
            ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], 'k--', linewidth=1)
            ax.set_xlim(min_val - margin, max_val + margin)
            ax.set_ylim(min_val - margin, max_val + margin)
            ax.set_xlabel(f'Trial 1 {var_labels[i]}', fontsize=11)
            ax.set_ylabel(f'Trial 2 {var_labels[i]}', fontsize=11)
            ax.set_title(f'{var_labels[i]} Repeatability', fontsize=12)
            r2 = np.corrcoef(x, y)[0, 1] ** 2
            ax.text(0.02, 0.95, f'$R^2$ = {r2:.3f}', transform=ax.transAxes, fontsize=10, va='top')
            ax.grid(True)

        legend_elements = [
            plt.Line2D([0], [0], marker='x', color='black', label='50 cSt',
               markersize=8, linestyle='None', linewidth=1.5),
            plt.Line2D([0], [0], marker='x', color='red', label='10 cSt',
               markersize=8, linestyle='None', linewidth=1.5)
        ]
        for i in range(4):
            ax = axes[i]
            ax.legend(handles=legend_elements, frameon=True, facecolor='white', edgecolor='black', fontsize=9)


        plt.tight_layout()
        st.pyplot(fig)

    with tabs[1]:
        st.markdown("### Flow Rate Analysis (Trial-Averaged)")

        # Step 1: Average across trials
        group_cols = ['Viscosity_cSt', 'Temp', 'FlowRate', 'VenturiAngle', 'AeratedFlow']
        grouped = filtered_df.groupby(group_cols)

        trial_avg_records = []
        for key, group in grouped:
            if group['Trial'].nunique() == 2:
                record = group.mean(numeric_only=True)
                record['Viscosity_cSt'] = key[0]
                record['Temp'] = key[1]
                record['FlowRate'] = key[2]
                record['VenturiAngle'] = key[3]
                record['AeratedFlow'] = key[4]
                trial_avg_records.append(record)

        if not trial_avg_records:
            st.warning("No trial-averaged data available.")
        else:
            avg_df = pd.DataFrame(trial_avg_records)

            # Step 2: Group by viscosity + temp
            viscosity_levels = sorted(avg_df['Viscosity_cSt'].unique())

            

            # Step 3: Plot
            fig, axes = plt.subplots(2, 1, figsize=(10, 9))
            for i, (col, ylabel, ylim) in enumerate([
                ('D_v', r'$d_{30}$ ($\mu$m)', (0, 1100)),
                ('LogSigma', r'$\sigma_{LN}$', (0, 2))
            ]):
                ax = axes[i]
                for vi, visc in enumerate(viscosity_levels):
                    temp_levels = sorted(avg_df[avg_df['Viscosity_cSt'] == visc]['Temp'].unique())
                    color = fluid_colors.get(visc, 'gray')  # default to gray if not in dict

                    for ti, temp in enumerate(temp_levels):
                        subset = avg_df[(avg_df['Viscosity_cSt'] == visc) & (avg_df['Temp'] == temp)].sort_values('FlowRate')
                        if not subset.empty:
                            label = f"{visc} - {temp}°F"
                            ax.plot(subset['FlowRate'], subset[col],
                                    linestyle='-', marker=marker_styles[ti % len(marker_styles)],
                                    color=color, linewidth=1.0, markersize=6,
                                    markeredgewidth=1.5, markerfacecolor='none', label=label)

                ax.set_xlabel('Flow Rate (GPM)', fontsize=11)
                ax.set_ylabel(ylabel, fontsize=11)
                ax.set_ylim(ylim)
                ax.grid(True)
                ax.legend(fontsize=9, frameon=True, facecolor='white', edgecolor='black')

            plt.tight_layout()
            st.pyplot(fig)

    with tabs[2]:
        st.markdown("### Temperature Analysis")
        st.markdown("#### $d_{30}$ and $\\sigma_{LN}$ vs Measured Temperature (Trial-Averaged)")

        # 1) average the two trials of identical setups
        group_cols = ['Viscosity_cSt', 'Temp', 'FlowRate', 'VenturiAngle', 'AeratedFlow']
        grouped = filtered_df.groupby(group_cols)

        trial_avg_records = []
        for key, group in grouped:
            if group['Trial'].nunique() == 2:
                rec = group.mean(numeric_only=True)
                rec['MeanTemp_exp'] = group['MeanTemp'].mean()
                rec['Viscosity_cSt'], rec['Temp'], rec['FlowRate'] = key[0], key[1], key[2]
                rec['VenturiAngle'],  rec['AeratedFlow'] = key[3], key[4]
                trial_avg_records.append(rec)

        if not trial_avg_records:
            st.warning("No trial-averaged data found.")
            st.stop()

        avg_df = pd.DataFrame(trial_avg_records)

        # 2) keys = (viscosity, flow)
        vis_flow_keys = sorted(avg_df.groupby(['Viscosity_cSt', 'FlowRate']).groups.keys())

        # 3) plot d30 and log-sigma
        fig, axes = plt.subplots(2, 1, figsize=(10, 9))
        for k, (col, ylabel, ylim) in enumerate([
            ('D_v',      r'$d_{30}$ ($\mu$m)', (0, 1100)),
            ('LogSigma', r'$\sigma_{LN}$',     (0, 2))
        ]):
            ax = axes[k]
            for idx, (visc, flow) in enumerate(vis_flow_keys):
                subset = avg_df[(avg_df['Viscosity_cSt'] == visc) &
                                (avg_df['FlowRate']     == flow)].sort_values('MeanTemp_exp')
                if subset.empty:
                    continue

                color  = fluid_colors.get(visc, 'gray')          # red / black default gray
                marker = marker_styles[idx % len(marker_styles)] # cycle markers

                ax.plot(subset['MeanTemp_exp'], subset[col],
                        linestyle='-',  linewidth=1.0,
                        marker=marker, markerfacecolor='none',
                        markeredgewidth=1.5, markersize=6,
                        color=color,
                        label=f"{visc} – {flow} GPM")

            ax.set_xlabel('Measured Temperature (°F)', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_ylim(ylim)
            ax.grid(True)
            ax.legend(frameon=True, facecolor='white', edgecolor='black', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)


    with tabs[3]:
        st.markdown("### Venturi Angle Analysis")
        st.markdown("#### $d_{30}$ and $\sigma_{LN}$ vs Angle (Trial-Averaged)")

        # Group by trial-independent setup
        group_cols = ['Viscosity_cSt', 'Temp', 'FlowRate', 'VenturiAngle', 'AeratedFlow']
        grouped = filtered_df.groupby(group_cols)

        trial_avg_records = []
        for key, group in grouped:
            if group['Trial'].nunique() == 2:
                record = group.mean(numeric_only=True)
                record['Viscosity_cSt'] = key[0]
                record['Temp'] = key[1]
                record['FlowRate'] = key[2]
                record['VenturiAngle'] = key[3]
                record['AeratedFlow'] = key[4]
                trial_avg_records.append(record)

        if not trial_avg_records:
            st.warning("No trial-averaged data available.")
            st.stop()

        avg_df = pd.DataFrame(trial_avg_records)

        # Group by viscosity + temp
        groups = sorted(avg_df.groupby(['Viscosity_cSt', 'Temp']).groups.keys())
        color_map = plt.colormaps.get_cmap('tab20').resampled(len(groups))

        fig, axes = plt.subplots(2, 1, figsize=(10, 9))
        for i, (col, ylabel, ylim) in enumerate([
            ('D_v', r'$d_{30}$ ($\mu$m)', (0, 1100)),
            ('LogSigma', r'$\sigma_{LN}$', (0, 2))
        ]):
            ax = axes[i]
            for j, (visc, temp) in enumerate(groups):
                subset = avg_df[(avg_df['Viscosity_cSt'] == visc) & (avg_df['Temp'] == temp)].sort_values('VenturiAngle')
                if not subset.empty:
                    label = f"{visc} - {temp}°F"
                    ax.plot(subset['VenturiAngle'], subset[col], '-o', label=label, color=color_map(j))
            ax.set_xlabel('Venturi Angle (°)')
            ax.set_ylabel(ylabel)
            ax.set_ylim(ylim)
            ax.grid(True)
            ax.legend(fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)

    with tabs[4]:
        st.markdown("### Reynolds Number Analysis (SAM-only, Trial-Averaged)")

        # Controls
        ext_data_opt = st.radio("Include External Data?", ["None", "Yin", "Sun", "Both"], index=0, horizontal=True)
        fit_opt = st.radio("Include Fit (d/D = A·Re^b)?", ["No", "Yes"], index=1, horizontal=True)
        flow_fit_opt = st.radio("Add constant flow rate lines?", ["No", "Yes"], index=0, horizontal=True)
        legend_opt = st.radio("Legend Style", ["Full", "Simplified"], index=0, horizontal=True)
        scale_opt = st.radio("Scale", ["Linear", "Log"], index=0, horizontal=True)

        # Trial-averaging
        group_cols = ['Viscosity_cSt', 'Temp', 'FlowRate', 'VenturiAngle', 'AeratedFlow']
        grouped = filtered_df.groupby(group_cols)

        trial_avg_records = []
        for key, group in grouped:
            if group['Trial'].nunique() == 2:
                record = group.mean(numeric_only=True)
                record['Viscosity_cSt'], record['Temp'] = key[0], key[1]
                record['FlowRate'], record['VenturiAngle'], record['AeratedFlow'] = key[2], key[3], key[4]
                trial_avg_records.append(record)

        if not trial_avg_records:
            st.warning("No trial-averaged SAM data available.")
            st.stop()

        df_combined = pd.DataFrame(trial_avg_records)

        # Identify unique (visc, temp) groups
        groups = sorted(df_combined.groupby(['Viscosity_cSt', 'Temp']).groups.keys())
        temp_list = sorted(df_combined['Temp'].unique())
        temp_to_marker = {temp: marker_styles[i % len(marker_styles)] for i, temp in enumerate(temp_list)}

        # Plotting
        fig, ax = plt.subplots(figsize=(9, 6))

        for j, (visc, temp) in enumerate(groups):
            subset = df_combined[(df_combined['Viscosity_cSt'] == visc) & (df_combined['Temp'] == temp)]
            subset = subset.dropna(subset=['D_v', 'Reynolds', 'ThroatDiameter_m']).sort_values('Reynolds')

            if len(subset) == 0:
                continue

            x = subset['Reynolds']
            y = subset['D_v'] * 1e-6 / subset['ThroatDiameter_m']

            color = fluid_colors.get(visc, 'gray')
            marker = temp_to_marker[temp]
            
            if legend_opt == "Simplified":
                label = visc if temp == min(temp_list) else None  # Only label first occurrence of each viscosity
            else:
                label = f"{visc} {temp}°F"

            # Scatter with solid line
            ax.scatter(x, y, marker=marker,s=20,
                    color=color, label=label)

            # Fit
            if fit_opt == "Yes" and len(x) >= 3:
                try:
                    def model_fn(Re, A, b): return A * Re**b
                    popt, _ = curve_fit(model_fn, x, y, maxfev=10000)
                    x_fit = np.linspace(min(x), max(x), 200)
                    y_fit = model_fn(x_fit, *popt)

                    fit_style = linestyles[j % len(linestyles)]
                    fit_label = fr"{label} Fit: $A$={popt[0]:.2e}, $b$={popt[1]:.3f}"
                    ax.plot(x_fit, y_fit,
                            color=color, linestyle=fit_style,
                            linewidth=1, label=fit_label)
                except Exception as e:
                    st.warning(f"Fit failed for {label}: {e}")
        
        # Add constant flow rate lines
        if flow_fit_opt == "Yes":
            flow_rates = sorted(df_combined['FlowRate'].unique())
            for flow_rate in flow_rates:
                flow_data = df_combined[df_combined['FlowRate'] == flow_rate].sort_values('Reynolds')
                if len(flow_data) >= 2:
                    x_flow = flow_data['Reynolds']
                    y_flow = flow_data['D_v'] * 1e-6 / flow_data['ThroatDiameter_m']
                    ax.plot(x_flow, y_flow, 'k:', linewidth=1, alpha=0.7)

        # External Data
        if ext_data_opt in ["Yin", "Both"]:
            x = yin_data['Re_t']
            y = yin_data['D_v'] / yin_data['ThroatDiameter_m']
            ax.scatter(x, y, label="Yin et al. 2015", marker='s', color='tab:green', edgecolor='k')

        if ext_data_opt in ["Sun", "Both"]:
            x = sun_data['Re']
            y = sun_data['D_v'] / sun_data['ThroatDiameter_m']
            ax.scatter(x, y, label="Sun et al. 2017", marker='s', color='tab:orange', edgecolor='k')

        # Formatting
        ax.axhline(1/6, color='k', linestyle='--', linewidth=1.2, label='Air Injection Diameter')
        ax.set_xlabel("Reynolds Number", fontsize=13)
        ax.set_ylabel(r"$d_{30}/D_t$", fontsize=13)
        ax.set_title("Normalized Diameter vs Reynolds Number", fontsize=14)
        ax.grid(True)
        ax.legend(fontsize=9, frameon=True, facecolor='white', edgecolor='black')

        if scale_opt == "Log":
            ax.set_xscale('log')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with tabs[5]:
        st.markdown("### Weber Number Analysis (SAM-only, Trial-Averaged)")

        ext_data_opt = st.radio("Include External Data?", ["None", "Yin", "Sun", "Both"], index=0, horizontal=True, key="weber_ext")
        fit_opt = st.radio("Include Fit (d/D = A·We^b)?", ["No", "Yes"], index=1, horizontal=True, key="weber_fit")
        flow_fit_opt = st.radio("Add constant flow rate lines?", ["No", "Yes"], index=0, horizontal=True, key="weber_flow_fit")
        legend_opt = st.radio("Legend Style", ["Full", "Simplified"], index=0, horizontal=True, key="weber_legend")
        scale_opt = st.radio("Scale", ["Linear", "Log"], index=0, horizontal=True, key="weber_scale")


        group_cols = ['Viscosity_cSt', 'Temp', 'FlowRate', 'VenturiAngle', 'AeratedFlow']
        grouped = filtered_df.groupby(group_cols)

        trial_avg_records = []
        for key, group in grouped:
            if group['Trial'].nunique() == 2:
                record = group.mean(numeric_only=True)
                record['Viscosity_cSt'] = key[0]
                record['Temp'] = key[1]
                record['FlowRate'] = key[2]
                record['VenturiAngle'] = key[3]
                record['AeratedFlow'] = key[4]
                trial_avg_records.append(record)

        if not trial_avg_records:
            st.warning("No trial-averaged SAM data available.")
            st.stop()

        df_combined = pd.DataFrame(trial_avg_records)

        fig, ax = plt.subplots(figsize=(9, 6))
        groups = sorted(df_combined.groupby(['Viscosity_cSt', 'Temp']).groups.keys())
        temp_list = sorted(df_combined['Temp'].unique())
        temp_to_marker = {temp: marker_styles[i % len(marker_styles)] for i, temp in enumerate(temp_list)}

        for j, (visc, temp) in enumerate(groups):
            subset = df_combined[(df_combined['Viscosity_cSt'] == visc) & (df_combined['Temp'] == temp)]
            subset = subset.dropna(subset=['D_v', 'We_D', 'ThroatDiameter_m'])

            if len(subset) == 0:
                continue

            x = subset['We_D']
            y = subset['D_v'] * 1e-6 / subset['ThroatDiameter_m']
            color = fluid_colors.get(visc, 'gray')
            marker = temp_to_marker[temp]
            
            if legend_opt == "Simplified":
                label = visc if temp == min(temp_list) else None
            else:
                label = f"{visc} - {temp}°F"
            
            ax.scatter(x, y, label=label, color=color, alpha=0.7, marker=marker, s=20)

            if fit_opt == "Yes" and len(x) >= 3:
                try:
                    def model_fn(We, A, b): return A * We**b
                    popt, _ = curve_fit(model_fn, x, y, maxfev=10000)
                    x_fit = np.linspace(min(x), max(x), 200)
                    y_fit = model_fn(x_fit, *popt)
                    fit_style = linestyles[j % len(linestyles)]
                    fit_label = fr"{label} Fit: $A$={popt[0]:.2e}, $b$={popt[1]:.3f}"
                    ax.plot(x_fit, y_fit, color=color, linestyle=fit_style,
                            linewidth=1, label=fit_label)
                except Exception as e:
                    st.warning(f"Fit failed for {label}: {e}")
        
        # Add constant flow rate lines
        if flow_fit_opt == "Yes":
            flow_rates = sorted(df_combined['FlowRate'].unique())
            for flow_rate in flow_rates:
                flow_data = df_combined[df_combined['FlowRate'] == flow_rate].sort_values('We_D')
                if len(flow_data) >= 2:
                    x_flow = flow_data['We_D']
                    y_flow = flow_data['D_v'] * 1e-6 / flow_data['ThroatDiameter_m']
                    ax.plot(x_flow, y_flow, 'k:', linewidth=1, alpha=0.7)

        if ext_data_opt in ["Yin", "Both"]:
            x = yin_data['We']
            y = yin_data['D_v'] / yin_data['ThroatDiameter_m']
            ax.scatter(x, y, label="Yin et al. 2015", marker='s', color='tab:green', edgecolor='k')

        if ext_data_opt in ["Sun", "Both"]:
            x = sun_data['We']
            y = sun_data['D_v'] / sun_data['ThroatDiameter_m']
            ax.scatter(x, y, label="Sun et al. 2017", marker='s', color='tab:orange', edgecolor='k')

        ax.axhline(1/6, color='k', linestyle='--', linewidth=1.2, label='Air Injection Diameter')
        ax.set_xlabel("Weber Number", fontsize=13)
        ax.set_ylabel(r"$d_{30}/D_t$", fontsize=13)
        ax.set_title("Normalized Diameter vs Weber Number", fontsize=14)
        ax.grid(True)
        ax.legend(fontsize=9, loc='best')
        if scale_opt == "Log":
            ax.set_xscale('log')
        st.pyplot(fig)
        plt.close(fig)

    with tabs[6]:
        st.markdown("### Capillary Number Analysis (SAM-only, Trial-Averaged)")

        ext_data_opt = st.radio("Include External Data?", ["None", "Yin", "Sun", "Both"], index=0, horizontal=True, key="capillary_ext")
        fit_opt = st.radio("Include Fit (d/D = A·Ca^b)?", ["No", "Yes"], index=1, horizontal=True, key="capillary_fit")
        flow_fit_opt = st.radio("Add constant flow rate lines?", ["No", "Yes"], index=0, horizontal=True, key="capillary_flow_fit")
        legend_opt = st.radio("Legend Style", ["Full", "Simplified"], index=0, horizontal=True, key="capillary_legend")
        scale_opt = st.radio("Scale", ["Linear", "Log"], index=0, horizontal=True, key="capillary_scale")


        group_cols = ['Viscosity_cSt', 'Temp', 'FlowRate', 'VenturiAngle', 'AeratedFlow']
        grouped = filtered_df.groupby(group_cols)

        trial_avg_records = []
        for key, group in grouped:
            if group['Trial'].nunique() == 2:
                record = group.mean(numeric_only=True)
                record['Viscosity_cSt'] = key[0]
                record['Temp'] = key[1]
                record['FlowRate'] = key[2]
                record['VenturiAngle'] = key[3]
                record['AeratedFlow'] = key[4]
                trial_avg_records.append(record)

        if not trial_avg_records:
            st.warning("No trial-averaged SAM data available.")
            st.stop()

        df_combined = pd.DataFrame(trial_avg_records)

        fig, ax = plt.subplots(figsize=(9, 6))
        groups = sorted(df_combined.groupby(['Viscosity_cSt', 'Temp']).groups.keys())
        temp_list = sorted(df_combined['Temp'].unique())
        temp_to_marker = {temp: marker_styles[i % len(marker_styles)] for i, temp in enumerate(temp_list)}

        for j, (visc, temp) in enumerate(groups):
            subset = df_combined[(df_combined['Viscosity_cSt'] == visc) & (df_combined['Temp'] == temp)]
            subset = subset.dropna(subset=['D_v', 'Ca', 'ThroatDiameter_m'])

            if len(subset) == 0:
                continue

            x = subset['Ca']
            y = subset['D_v'] * 1e-6 / subset['ThroatDiameter_m']
            color = fluid_colors.get(visc, 'gray')
            marker = temp_to_marker[temp]
            
            if legend_opt == "Simplified":
                label = visc if temp == min(temp_list) else None
            else:
                label = f"{visc} - {temp}°F"
            
            ax.scatter(x, y, label=label, color=color, alpha=0.7, marker=marker, s=20)

            if fit_opt == "Yes" and len(x) >= 3:
                try:
                    def model_fn(Ca, A, b): return A * Ca**b
                    popt, _ = curve_fit(model_fn, x, y, maxfev=10000)
                    x_fit = np.linspace(min(x), max(x), 200)
                    y_fit = model_fn(x_fit, *popt)
                    fit_style = linestyles[j % len(linestyles)]
                    fit_label = fr"{label} Fit: $A$={popt[0]:.2e}, $b$={popt[1]:.3f}"
                    ax.plot(x_fit, y_fit, color=color, linestyle=fit_style,
                            linewidth=1, label=fit_label)
                except Exception as e:
                    st.warning(f"Fit failed for {label}: {e}")
        
        # Add constant flow rate lines
        if flow_fit_opt == "Yes":
            flow_rates = sorted(df_combined['FlowRate'].unique())
            for flow_rate in flow_rates:
                flow_data = df_combined[df_combined['FlowRate'] == flow_rate].sort_values('Ca')
                if len(flow_data) >= 2:
                    x_flow = flow_data['Ca']
                    y_flow = flow_data['D_v'] * 1e-6 / flow_data['ThroatDiameter_m']
                    ax.plot(x_flow, y_flow, 'k:', linewidth=1, alpha=0.7)

        if ext_data_opt in ["Yin", "Both"]:
            x = yin_data['Ca']
            y = yin_data['D_v'] / yin_data['ThroatDiameter_m']
            ax.scatter(x, y, label="Yin et al. 2015", marker='s', color='tab:green', edgecolor='k')

        if ext_data_opt in ["Sun", "Both"]:
            x = sun_data['Ca']
            y = sun_data['D_v'] / sun_data['ThroatDiameter_m']
            ax.scatter(x, y, label="Sun et al. 2017", marker='s', color='tab:orange', edgecolor='k')

        ax.axhline(1/6, color='k', linestyle='--', linewidth=1.2, label='Air Injection Diameter')
        ax.set_xlabel("Capillary Number", fontsize=13)
        ax.set_ylabel(r"$d_{30}/D_t$", fontsize=13)
        ax.set_title("Normalized Diameter vs Capillary Number", fontsize=14)
        ax.grid(True)
        ax.legend(fontsize=9, loc='best')
        if scale_opt == "Log":
            ax.set_xscale('log')
        st.pyplot(fig)
        plt.close(fig)

    def create_universal_plot(x_params, x_labels, y_label, plot_title, tab_name):
        """Universal function to create collapsed scaling plots"""
        
        # Controls
        ext_data_opt = st.radio("Include External Data?", ["None", "Yin", "Sun", "Both"], index=0, horizontal=True, key=f"{tab_name}_ext")
        include_in_fit = st.radio("Include external data in fit?", ["No", "Yes"], index=0, horizontal=True, key=f"{tab_name}_fit_ext")
        air_injection_opt = st.radio("Show air injection diameter?", ["No", "Yes"], index=1, horizontal=True, key=f"{tab_name}_air")
        scale_opt = st.radio("Scale", ["Linear", "Log"], index=1, horizontal=True, key=f"{tab_name}_scale")
        
        # Trial-averaged data preparation
        group_cols = ['Viscosity_cSt', 'Temp', 'FlowRate', 'VenturiAngle', 'AeratedFlow']
        grouped = filtered_df.groupby(group_cols)

        trial_avg_records = []
        for key, group in grouped:
            if group['Trial'].nunique() == 2:
                record = group.mean(numeric_only=True)
                record['Viscosity_cSt'] = key[0]
                record['Temp'] = key[1]
                record['FlowRate'] = key[2]
                record['VenturiAngle'] = key[3]
                record['AeratedFlow'] = key[4]
                trial_avg_records.append(record)

        if not trial_avg_records:
            st.warning("No trial-averaged SAM data available.")
            st.stop()

        # Prepare fit data
        all_data = []
        df_internal = pd.DataFrame(trial_avg_records)
        required_cols = ['D_v', 'ThroatDiameter_m'] + x_params
        df_internal = df_internal.dropna(subset=required_cols)
        
        # Add internal data
        for _, row in df_internal.iterrows():
            all_data.append({
                x_params[0]: row[x_params[0]],
                x_params[1]: row[x_params[1]],
                'D_v': row['D_v'] * 1e-6,  # Convert to meters
                'ThroatDiameter_m': row['ThroatDiameter_m'],
                'source': 'Internal',
                'Viscosity_cSt': row['Viscosity_cSt']
            })
        
        # Add external data if requested for fitting
        if include_in_fit == "Yes":
            if ext_data_opt in ["Yin", "Both"]:
                for _, row in yin_data.iterrows():
                    param_map = {'Reynolds': 'Re_t', 'We_D': 'We', 'Ca': 'Ca'}
                    all_data.append({
                        x_params[0]: row[param_map.get(x_params[0], x_params[0])],
                        x_params[1]: row[param_map.get(x_params[1], x_params[1])],
                        'D_v': row['D_v'],
                        'ThroatDiameter_m': row['ThroatDiameter_m'],
                        'source': 'Yin et al. 2015',
                        'Viscosity_cSt': None
                    })
            
            if ext_data_opt in ["Sun", "Both"]:
                for _, row in sun_data.iterrows():
                    param_map = {'Reynolds': 'Re', 'We_D': 'We', 'Ca': 'Ca'}
                    all_data.append({
                        x_params[0]: row[param_map.get(x_params[0], x_params[0])],
                        x_params[1]: row[param_map.get(x_params[1], x_params[1])],
                        'D_v': row['D_v'],
                        'ThroatDiameter_m': row['ThroatDiameter_m'],
                        'source': 'Sun et al. 2017',
                        'Viscosity_cSt': None
                    })
        
        df_combined = pd.DataFrame(all_data)
        df_combined = df_combined.dropna(subset=['D_v', 'ThroatDiameter_m'] + x_params)

        # Fit the model
        xdata = df_combined[x_params].values.T  # shape (2, N)
        ydata = df_combined['D_v'] / df_combined['ThroatDiameter_m']  # Normalize to d/D

        def model_fn(X, A, a, b):
            return A * X[0]**a * X[1]**b

        try:
            popt, pcov = curve_fit(model_fn, xdata, ydata, p0=[1e-3, -0.5, -0.3], maxfev=10000)
            A, a, b = popt
            
            # Calculate R-squared
            y_pred = model_fn(xdata, *popt)
            ss_res = np.sum((ydata - y_pred) ** 2)
            ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            st.success(f"Best Fit: $A$ = {A:.2e}, $a$ = {a:.3f}, $b$ = {b:.3f}")
            st.info(f"R² = {r_squared:.4f}")
        except Exception as e:
            st.error(f"Fit failed: {e}")
            st.stop()

        # Compute collapsed x-axis values
        df_combined['CollapseX'] = df_combined[x_params[0]]**a * df_combined[x_params[1]]**b
        df_combined['NormDiameter'] = df_combined['D_v'] / df_combined['ThroatDiameter_m']

        # Plotting
        fig, ax = plt.subplots(figsize=(9, 6))
        markers = {'10 cSt': 'o', '50 cSt': 's'}

        # Plot internal data by viscosity
        internal_data = df_combined[df_combined['source'] == 'Internal']
        for visc in sorted(internal_data['Viscosity_cSt'].unique()):
            subset = internal_data[internal_data['Viscosity_cSt'] == visc]
            color = fluid_colors.get(visc, 'gray')
            marker = markers.get(visc, 'x')
            ax.scatter(subset['CollapseX'], subset['NormDiameter'],
                    label=visc, alpha=0.7, color=color, marker=marker, s=30)

        # Plot external data if requested for display
        if ext_data_opt in ["Yin", "Both"]:
            yin_subset = df_combined[df_combined['source'] == 'Yin et al. 2015']
            if not yin_subset.empty:
                ax.scatter(yin_subset['CollapseX'], yin_subset['NormDiameter'],
                        label="Yin et al. 2015", marker='s', color='tab:green', 
                        edgecolor='k', alpha=0.7, s=30)

        if ext_data_opt in ["Sun", "Both"]:
            sun_subset = df_combined[df_combined['source'] == 'Sun et al. 2017']
            if not sun_subset.empty:
                ax.scatter(sun_subset['CollapseX'], sun_subset['NormDiameter'],
                        label="Sun et al. 2017", marker='s', color='tab:orange', 
                        edgecolor='k', alpha=0.7, s=30)

        # Plot the best-fit curve
        x_fit = np.linspace(df_combined['CollapseX'].min(), df_combined['CollapseX'].max(), 200)
        y_fit = A * x_fit
        ax.plot(x_fit, y_fit, 'k--', linewidth=2, 
                label=f"Best Fit: $d/D = {A:.2e} \\cdot {x_labels}$")

        # Add air injection diameter reference
        if air_injection_opt == "Yes":
            ax.axhline(1/6, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='Air Injection Diameter')

        ax.set_xlabel(f"${x_labels}$", fontsize=13)
        ax.set_ylabel(r"$d_{30} / D_t$", fontsize=13)
        ax.set_title(plot_title, fontsize=14)
        ax.grid(True)
        ax.legend(fontsize=9, frameon=True, facecolor='white', edgecolor='black')
        
        if scale_opt == "Log":
            ax.set_xscale('log')
            ax.set_yscale('log')

        st.pyplot(fig)
        plt.close(fig)
    
    with tabs[7]:
        st.markdown("### Universal Scaling: $d/D = A \\cdot Re^a \\cdot We^b$")
        create_universal_plot(
            ['Reynolds', 'We_D'], 
            "Re^a \\cdot We^b", 
            r"$d_{30} / D_t$", 
            "Collapsed Plot Using Fitted $Re^a We^b$ Scaling",
            "rewe"
        )
    from scipy.stats import lognorm

    with tabs[8]:

        st.markdown("### PDF Comparison at Fixed Flow Rate")
        st.markdown("#### Log-normal distribution of $d$ based on Trial-Averaged $\mu_{LN}$ and $\sigma_{LN}$")

        # User selects flow rate
        available_flow_rates = sorted(filtered_df['FlowRate'].unique())
        selected_flow = st.selectbox("Select a Flow Rate (GPM)", available_flow_rates)

        # Trial-averaged data for selected flow rate
        group_cols = ['Viscosity_cSt', 'Temp', 'FlowRate', 'VenturiAngle', 'AeratedFlow']
        grouped = filtered_df[filtered_df['FlowRate'] == selected_flow].groupby(group_cols)

        trial_avg_records = []
        for key, group in grouped:
            if group['Trial'].nunique() == 2:
                rec = group.mean(numeric_only=True)
                rec['Viscosity_cSt'], rec['Temp'] = key[0], key[1]
                rec['FlowRate'], rec['VenturiAngle'], rec['AeratedFlow'] = key[2], key[3], key[4]
                trial_avg_records.append(rec)

        if not trial_avg_records:
            st.warning("No trial-averaged data found for selected flow rate.")
            st.stop()

        avg_df = pd.DataFrame(trial_avg_records)

        # Build consistent linestyle mapping based on unique temperatures
        unique_temps = sorted(avg_df['Temp'].unique())
        temp_linestyle_map = {temp: linestyles[i % len(linestyles)]
                            for i, temp in enumerate(unique_temps)}

        # Plot PDFs
        fig, ax = plt.subplots(figsize=(8, 5))
        x_vals = np.linspace(1e-3, 600, 800)

        for _, row in avg_df.iterrows():
            mu_ln = row['LogMu']
            sigma_ln = row['LogSigma']
            d30 = row['D_v']
            temp = row['Temp']

            # Converted units
            mu_mPas = row['mu'] * 1000
            gamma_mNm = row['Gamma'] * 1000

            color = fluid_colors.get(row['Viscosity_cSt'], 'gray')
            linestyle = temp_linestyle_map[temp]

            label = f"{row['Viscosity_cSt']} {int(temp)}°F " \
                    f"(μ={mu_mPas:.1f} mPa·s, σ={gamma_mNm:.1f} mN/m)"

            pdf_vals = lognorm.pdf(x_vals, s=sigma_ln, scale=np.exp(mu_ln))

            ax.plot(x_vals, pdf_vals,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.0,
                    label=label)

            # Vertical line at d30
            ax.axvline(d30, color=color, linestyle=linestyle, linewidth=1)

        ax.set_xlabel(r'Diameter ($\mu$m)', fontsize=11)
        ax.set_ylabel('Probability Density', fontsize=11)
        ax.set_xlim(0, 600)
        ax.grid(True)
        ax.legend(fontsize=9, frameon=True, facecolor='white', edgecolor='black')

        plt.tight_layout()
        st.pyplot(fig)

    with tabs[9]:
        st.markdown("### Universal Scaling: $d/D = A \\cdot Re^a \\cdot Ca^b$")
        create_universal_plot(
            ['Reynolds', 'Ca'], 
            "Re^a \\cdot Ca^b", 
            r"$d_{30} / D_t$", 
            "Collapsed Plot Using Fitted $Re^a Ca^b$ Scaling",
            "reca"
        )
    with tabs[10]:
        st.markdown("### Universal Scaling: $d/D = A \\cdot We^a \\cdot Ca^b$")
        create_universal_plot(
            ['We_D', 'Ca'], 
            "We^a \\cdot Ca^b", 
            r"$d_{30} / D_t$", 
            "Collapsed Plot Using Fitted $We^a Ca^b$ Scaling",
            "weca"
        )
        