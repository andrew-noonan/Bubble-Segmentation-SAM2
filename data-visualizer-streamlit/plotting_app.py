import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from scipy.io import loadmat
from scipy.stats import lognorm
import matplotlib.pyplot as plt
from matplotlib import colormaps

# --- Constants ---
UM_PER_PIXEL = 5.71
D_t = 6e-3
A_throat = np.pi * D_t**2 / 4
GPM_to_m3_s = 1 / (264.172053 * 60)
A, B = 732, -3.803
alpha, rho_ref = 0.000994, 959
T_ref = 298.15
D_p = 15.8e-3

# --- Utility Functions ---
def parse_mat_struct(mat_struct):
    return {name: mat_struct[name][0, 0] if mat_struct[name].ndim == 2 else mat_struct[name]
            for name in mat_struct.dtype.names}

def safe_get_scalar(d, key):
    try:
        return float(np.squeeze(d[key]))
    except Exception:
        return np.nan

@st.cache_data
def load_all_experiment_data(base_dir):
    records = []
    for root, _, files in os.walk(base_dir):
        if "labview.txt" not in files:
            continue
        try:
            parts = root.replace("\\", "/").split("/")
            angle = int(parts[-4].split()[0])
            temp = int(parts[-3].replace("F", ""))
            aer = float(parts[-2].split()[0].replace("_", "."))
            trial = int(parts[-2].split()[-1])
            flow = float(parts[-1].replace("_", "."))

            record = {
                'Temp': temp,
                'FlowRate': flow,
                'VenturiAngle': angle,
                'AeratedFlow': aer,
                'Trial': trial,
                'Valid': True
            }

            # --- LabVIEW CSV ---
            lv_path = os.path.join(root, 'labview.txt')
            try:
                data = pd.read_csv(lv_path, encoding='utf-8', on_bad_lines='skip')
                record.update({
                    'MeanTemp': data['Temp (F)'].mean(),
                    'MeanFlow': data['Oil Flow Rate'].mean(),
                    'MeanP1': data['P1'].mean(),
                    'MeanP2': data['P2'].mean(),
                    'StdTemp': data['Temp (F)'].std(),
                    'StdFlow': data['Oil Flow Rate'].std(),
                    'StdP1': data['P1'].std(),
                    'StdP2': data['P2'].std()
                })
            except Exception as e:
                print(f"LabVIEW error in {root}: {e}")

            # --- MATLAB ---
            mat_path = os.path.join(root, 'MATLAB Results', 'lognormal_fit_params2.mat')
            if os.path.isfile(mat_path):
                mat = loadmat(mat_path)
                if 'tempLogData' in mat:
                    raw = mat['tempLogData']
                    if isinstance(raw, np.ndarray) and raw.dtype.names:
                        logdata = parse_mat_struct(raw[0, 0])
                        record.update({
                            'LogMu': safe_get_scalar(logdata, 'mu'),
                            'LogSigma': safe_get_scalar(logdata, 'sigma'),
                            'D32': safe_get_scalar(logdata, 'SauterMeanDiameter'),
                            'D_v': safe_get_scalar(logdata, 'D_v'),
                            'MedianDiameter': safe_get_scalar(logdata, 'MedianDiameter')
                        })

            # --- SAM CSV ---
            sam_path = os.path.join(root, 'experiment_summary.csv')
            if os.path.isfile(sam_path):
                sam_df = pd.read_csv(sam_path)
                record.update({
                    'LogMu_sam': sam_df['log_mu'].iloc[0] + np.log(UM_PER_PIXEL) if 'log_mu' in sam_df else np.nan,
                    'LogSigma_sam': sam_df['log_sigma'].iloc[0] if 'log_sigma' in sam_df else np.nan,
                    'D32_sam': sam_df['d32'].iloc[0] * UM_PER_PIXEL if 'd32' in sam_df else np.nan,
                    'D_v_sam': sam_df['dv'].iloc[0] * UM_PER_PIXEL if 'dv' in sam_df else np.nan
                })

            records.append(record)
        except:
            continue

    df = pd.DataFrame.from_records(records)

    # Use .unique() to only get values present in the data
    unique_angles = sorted(df['VenturiAngle'].dropna().unique().tolist())
    unique_flows = sorted(df['FlowRate'].dropna().unique().tolist())
    unique_temps = sorted(df['Temp'].dropna().unique().tolist())
    unique_aeration = sorted(df['AeratedFlow'].dropna().unique().tolist())

    if df.empty:
        return df

    # --- Post Processing ---
    df['deltaP'] = df.get('MeanP1', np.nan) - df.get('MeanP2', np.nan)
    df['deltaP_Pa'] = df['deltaP'] * 6894.75729
    tempF = df['MeanTemp'].combine_first(df['Temp'])
    df['tempK'] = (tempF - 32) / 1.8 + 273.15
    df['mu'] = 10 ** (A / df['tempK'] + B)
    df['rho'] = rho_ref * (1 - alpha * (df['tempK'] - T_ref))
    df['nu'] = df['mu'] / df['rho']
    flow = df['MeanFlow'].combine_first(df['FlowRate'])
    df['V_throat'] = flow * GPM_to_m3_s / A_throat
    df['Reynolds'] = df['V_throat'] * D_t / df['nu']
    df['dynamicPressure'] = 0.5 * df['rho'] * df['V_throat']**2
    df['deltaP_normalized'] = df['deltaP_Pa'] / df['dynamicPressure']
    df['Gamma'] = (22.3 - 0.06 * (df['tempK'] - 273.15)) / 1000
    df['Ca'] = df['mu'] * df['V_throat'] / df['Gamma']
    df['We_D'] = df['rho'] * df['V_throat']**2 * D_t / df['Gamma']
    df['L'] = (D_p - D_t) / np.tan(np.radians(df['VenturiAngle']))
    df['We_L'] = df['rho'] * df['V_throat']**2 * df['L'] / df['Gamma']
    df['ThroatDiameter_m'] = D_t

    return df

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

def plot_repeatability(df, plot_orig=True, plot_sam=False):
    # Define variable names from your dataset
    var_labels = [r'$\mu_{LN}$', r'$\sigma_{LN}$', r'$d_{32}$', r'$d_{30}$']
    matlab_cols = ['LogMu', 'LogSigma', 'D32', 'D_v']
    sam_cols = ['LogMu_sam', 'LogSigma_sam', 'D32_sam', 'D_v_sam']

    # Identify unique experimental conditions (excluding trial number)
    param_cols = ['Temp', 'FlowRate', 'VenturiAngle', 'AeratedFlow']
    df['param_id'] = df[param_cols].apply(tuple, axis=1)
    unique_params = df['param_id'].unique()

    def extract_trials(df, use_sam=False):
        t1_data, t2_data = [], []
        cols = sam_cols if use_sam else matlab_cols

        for param in unique_params:
            subset = df[df['param_id'] == param]
            trial1 = subset[subset['Trial'] == 1]
            trial2 = subset[subset['Trial'] == 2]
            if len(trial1) == 1 and len(trial2) == 1:
                try:
                    t1_data.append([trial1[cols[i]].values[0] for i in range(4)])
                    t2_data.append([trial2[cols[i]].values[0] for i in range(4)])
                except KeyError as e:
                    print(f"Missing column: {e}")
        return np.array(t1_data), np.array(t2_data)

    trial1_orig, trial2_orig = extract_trials(df, use_sam=False)
    trial1_sam, trial2_sam = extract_trials(df, use_sam=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i in range(4):
        ax = axes[i]

        x_all, y_all = [], []

        if plot_orig and trial1_orig.size > 0:
            x_orig, y_orig = trial1_orig[:, i], trial2_orig[:, i]
            ax.scatter(x_orig, y_orig, s=50, label='MATLAB', alpha=0.7)
            x_all.append(x_orig)
            y_all.append(y_orig)

        if plot_sam and trial1_sam.size > 0:
            x_sam, y_sam = trial1_sam[:, i], trial2_sam[:, i]
            ax.scatter(x_sam, y_sam, s=50, marker='x', label='SAM', alpha=0.7)
            x_all.append(x_sam)
            y_all.append(y_sam)

        # Axis limits and diagonal
        if x_all and y_all:
            all_x = np.concatenate(x_all)
            all_y = np.concatenate(y_all)
            min_val, max_val = min(all_x.min(), all_y.min()), max(all_x.max(), all_y.max())
            margin = 0.05 * (max_val - min_val)
            ax.plot([min_val - margin, max_val + margin],
                    [min_val - margin, max_val + margin], 'k--')
            ax.set_xlim([min_val - margin, max_val + margin])
            ax.set_ylim([min_val - margin, max_val + margin])

        ax.set_xlabel(f'Trial 1 {var_labels[i]}', fontsize=12)
        ax.set_ylabel(f'Trial 2 {var_labels[i]}', fontsize=12)
        ax.set_title(f'{var_labels[i]} Repeatability', fontsize=13)
        ax.grid(True)

        if plot_orig and trial1_orig.size > 0:
            r2 = np.corrcoef(trial1_orig[:, i], trial2_orig[:, i])[0, 1]**2
            ax.text(0.02, 0.95, f'MATLAB $R^2$ = {r2:.3f}', transform=ax.transAxes, fontsize=9, verticalalignment='top')

        if plot_sam and trial1_sam.size > 0:
            r2 = np.corrcoef(trial1_sam[:, i], trial2_sam[:, i])[0, 1]**2
            ax.text(0.02, 0.85, f'SAM $R^2$ = {r2:.3f}', transform=ax.transAxes, fontsize=9, verticalalignment='top')

        if plot_orig and plot_sam:
            ax.legend()

    fig.suptitle('Repeatability Analysis: Trial 2 vs Trial 1', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_sam_vs_matlab(df):
    import matplotlib.pyplot as plt
    import numpy as np

    # Define the variable mappings
    matlab_cols = ['LogMu', 'LogSigma', 'D32', 'D_v']
    sam_cols = ['LogMu_sam', 'LogSigma_sam', 'D32_sam', 'D_v_sam']
    var_labels = [r'$\mu_{LN}$', r'$\sigma_{LN}$', r'$d_{32}$', r'$d_{30}$']

    # Identify unique experimental conditions (excluding trial number)
    param_cols = ['Temp', 'FlowRate', 'VenturiAngle', 'AeratedFlow']
    df['param_id'] = df[param_cols].apply(tuple, axis=1)
    unique_params = df['param_id'].unique()

    sam_data = []
    matlab_data = []

    for param in unique_params:
        subset = df[df['param_id'] == param]
        trials = subset['Trial'].values

        if set(trials) == {1, 2}:
            try:
                sam_avg = subset[sam_cols].mean().values
                matlab_avg = subset[matlab_cols].mean().values
                sam_data.append(sam_avg)
                matlab_data.append(matlab_avg)
            except KeyError as e:
                print(f"Missing column for param {param}: {e}")

    sam_data = np.array(sam_data)
    matlab_data = np.array(matlab_data)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i in range(4):
        ax = axes[i]
        x = matlab_data[:, i]
        y = sam_data[:, i]

        ax.scatter(x, y, s=50, alpha=0.8)
        min_val, max_val = min(x.min(), y.min()), max(x.max(), y.max())
        margin = 0.05 * (max_val - min_val)
        min_val -= margin
        max_val += margin

        ax.plot([min_val, max_val], [min_val, max_val], 'k--')
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])

        ax.set_xlabel(f'MATLAB Avg {var_labels[i]}', fontsize=12)
        ax.set_ylabel(f'SAM Avg {var_labels[i]}', fontsize=12)
        ax.set_title(f'SAM vs MATLAB: {var_labels[i]}', fontsize=13)
        ax.grid(True)

        r_squared = np.corrcoef(x, y)[0, 1]**2
        ax.text(min_val + 0.05*(max_val - min_val), max_val - 0.1*(max_val - min_val),
                f'$R^2$ = {r_squared:.3f}', fontsize=10)

    fig.suptitle('Comparison: SAM vs MATLAB (Averaged Across Trials)', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_flow_rate_effect(df, angle, aeration, plot_orig=True, plot_sam=False):
    import matplotlib.pyplot as plt
    import numpy as np

    filtered = df[(df['VenturiAngle'] == angle) & (df['AeratedFlow'] == aeration)]
    if filtered.empty:
        raise ValueError('No data found for specified parameters.')

    avg_df = filtered.groupby(['Temp', 'FlowRate'], as_index=False).mean(numeric_only=True)
    temps = sorted(avg_df['Temp'].unique())
    color_map = plt.cm.get_cmap('tab10', len(temps))

    # --- Define plot configuration ---
    variable_map = {
        'd30': ('D_v', 'D_v_sam', r'$\mathrm{d}_{30}$', r'Volume-Equivalent Diameter ($\mu m$)', (0, 1100)),
        'sigma': ('LogSigma', 'LogSigma_sam', r'$\sigma_{LN}$', r'Log-Normal $\sigma_{LN}$', (0, 2))
    }

    sources = []
    if plot_orig:
        sources.append(('MATLAB', 'D_v', 'LogSigma'))
    if plot_sam:
        sources.append(('SAM', 'D_v_sam', 'LogSigma_sam'))

    ncols = len(sources)
    nrows = 2  # d30 and sigma rows

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if ncols == 1:
        axes = np.array(axes).reshape((2, 1))  # force 2D array for consistency

    for row_idx, (key, (col_orig, col_sam, title, ylabel, ylim)) in enumerate(variable_map.items()):
        for col_idx, (label, d_col, sigma_col) in enumerate(sources):
            ax = axes[row_idx, col_idx]
            col_name = col_orig if label == 'MATLAB' else col_sam

            for j, temp in enumerate(temps):
                subset = avg_df[avg_df['Temp'] == temp].sort_values(by='FlowRate')
                ax.plot(subset['FlowRate'], subset[col_name], '-o',
                        label=f'{temp}°F', color=color_map(j), markerfacecolor=color_map(j))

            ax.set_xlabel('Flow Rate (GPM)')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{title} vs Flow Rate ({label})')
            ax.grid(True)
            ax.set_xlim([0, 6])
            ax.set_ylim(ylim)

            if col_name == 'D_v':
                ax.axhline(1000, linestyle='--', color='k', label='Injection Hole Diameter')

            if row_idx == 0:
                ax.legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)



def plot_flow_rate_analysis(df, angle, temperature, aeration_percent, colors, plot_orig=True, plot_sam=False):
    filtered = df[(df['VenturiAngle'] == angle) & (df['Temp'] == temperature) &
                  (df['AeratedFlow'] == aeration_percent)]
    if filtered.empty:
        raise ValueError('No data found for specified parameters.')

    avg_df = filtered.groupby('FlowRate', as_index=False).mean(numeric_only=True).sort_values('FlowRate')

    x = np.linspace(0, 1200, 1000)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    if plot_orig:
        for i, row in avg_df.iterrows():
            mu, sigma = row['LogMu'], row['LogSigma']
            pdf_vals = lognorm.pdf(x, s=sigma, scale=np.exp(mu))
            color = colors[i % len(colors)]
            ax1.plot(x, pdf_vals, label=f"{row['FlowRate']} GPM (MATLAB)", color=color, linewidth=2)
            if not np.isnan(row['D_v']):
                ax1.axvline(row['D_v'], linestyle='-', color=color, linewidth=1.5)

    if plot_sam:
        for i, row in avg_df.iterrows():
            mu, sigma = row['LogMu_sam'], row['LogSigma_sam']
            pdf_vals = lognorm.pdf(x, s=sigma, scale=np.exp(mu))
            color = colors[i % len(colors)]
            ax1.plot(x, pdf_vals, linestyle=':', label=f"{row['FlowRate']} GPM (SAM)", color=color, linewidth=2)
            if not np.isnan(row['D_v_sam']):
                ax1.axvline(row['D_v_sam'], linestyle=':', color=color, linewidth=1.5)

    ax1.set_xlabel(r'Diameter ($\mu m$)', fontsize=14)
    ax1.set_ylabel('Probability Density', fontsize=14)
    ax1.set_title(f'Log-Normal Distributions for {angle}° Venturi, {temperature}°F, {aeration_percent}% Aeration', fontsize=16)
    ax1.grid(True)
    ax1.set_xlim([0, 1200])
    ax1.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_temperature_effect(df, angle, aeration, plot_orig=True, plot_sam=False):
    import matplotlib.pyplot as plt
    import numpy as np

    filtered = df[(df['VenturiAngle'] == angle) & (df['AeratedFlow'] == aeration)]
    if filtered.empty:
        raise ValueError('No data found for specified parameters.')

    avg_df = filtered.groupby(['Temp', 'FlowRate'], as_index=False).mean(numeric_only=True)
    flow_rates = sorted(avg_df['FlowRate'].unique())
    color_map = plt.cm.get_cmap('Dark2', len(flow_rates))

    # --- Define plot configuration ---
    variable_map = {
        'd30': ('D_v', 'D_v_sam', r'$\mathrm{d}_{30}$', r'Volume-Equivalent Diameter ($\mu m$)', (0, 1100)),
        'sigma': ('LogSigma', 'LogSigma_sam', r'$\sigma_{LN}$', r'Log-Normal $\sigma_{LN}$', (0, 2))
    }

    sources = []
    if plot_orig:
        sources.append(('MATLAB', 'D_v', 'LogSigma'))
    if plot_sam:
        sources.append(('SAM', 'D_v_sam', 'LogSigma_sam'))

    ncols = len(sources)
    nrows = 2  # d30 and sigma

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if ncols == 1:
        axes = np.array(axes).reshape((2, 1))  # ensure it's 2D (2 rows, 1 column)

    for row_idx, (key, (col_orig, col_sam, title, ylabel, ylim)) in enumerate(variable_map.items()):
        for col_idx, (label, d_col, sigma_col) in enumerate(sources):
            ax = axes[row_idx, col_idx]
            col_name = col_orig if label == 'MATLAB' else col_sam

            for j, flow in enumerate(flow_rates):
                subset = avg_df[avg_df['FlowRate'] == flow].sort_values(by='Temp')
                ax.plot(subset['Temp'], subset[col_name], '-o',
                        label=f'{flow:.1f} GPM', color=color_map(j), markerfacecolor=color_map(j))

            ax.set_xlabel('Temperature (°F)')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{title} vs Temperature ({label})')
            ax.grid(True)
            ax.set_ylim(ylim)
            ax.set_xlim([avg_df['Temp'].min() - 5, avg_df['Temp'].max() + 5])

            if 'D_v' in col_name:
                ax.axhline(1000, linestyle='--', color='k', label='Injection Hole Diameter')

            if row_idx == 0:
                ax.legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)



def plot_temperature_analysis(df, angle, flow_rate, aeration_percent, colors, plot_orig=True, plot_sam=False):

    filtered = df[(df['VenturiAngle'] == angle) &
                  (df['FlowRate'] == flow_rate) &
                  (df['AeratedFlow'] == aeration_percent)]
    if filtered.empty:
        raise ValueError('No data found for specified parameters.')

    avg_df = filtered.groupby('Temp', as_index=False).mean(numeric_only=True).sort_values('Temp')
    x = np.linspace(0, 1200, 1000)

    # Plot 1: Log-normal PDF
    fig, ax = plt.subplots(figsize=(10, 6))

    if plot_orig:
        for i, row in avg_df.iterrows():
            mu, sigma = row['LogMu'], row['LogSigma']
            pdf_vals = lognorm.pdf(x, s=sigma, scale=np.exp(mu))
            color = colors[i % len(colors)]
            ax.plot(x, pdf_vals, label=f"{row['Temp']}°F (MATLAB)", color=color, linewidth=2)
            if not np.isnan(row['D_v']):
                ax.axvline(row['D_v'], linestyle='-', color=color, linewidth=1.5)

    if plot_sam:
        for i, row in avg_df.iterrows():
            mu, sigma = row['LogMu_sam'], row['LogSigma_sam']
            pdf_vals = lognorm.pdf(x, s=sigma, scale=np.exp(mu))
            color = colors[i % len(colors)]
            ax.plot(x, pdf_vals, linestyle=':', label=f"{row['Temp']}°F (SAM)", color=color, linewidth=2)
            if not np.isnan(row['D_v_sam']):
                ax.axvline(row['D_v_sam'], linestyle=':', color=color, linewidth=1.5)

    ax.legend() 
    ax.set_xlabel(r'Diameter ($\mu m$)', fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.set_title(f'Log-Normal Distributions for {angle}°, {flow_rate} GPM, {aeration_percent}% Aeration', fontsize=16)
    ax.grid(True)
    ax.set_xlim([0, 1200])
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_angle_effect(df, flow_rate, aeration, plot_orig=True, plot_sam=False):
    import matplotlib.pyplot as plt
    import numpy as np

    filtered = df[(df['FlowRate'] == flow_rate) & (df['AeratedFlow'] == aeration)]
    if filtered.empty:
        raise ValueError('No data found for specified parameters.')

    avg_df = filtered.groupby(['VenturiAngle', 'Temp'], as_index=False).mean(numeric_only=True)
    temps = sorted(avg_df['Temp'].unique())
    color_map = plt.cm.get_cmap('Set2', len(temps))

    variable_map = {
        'd30': ('D_v', 'D_v_sam', r'$\mathrm{d}_{30}$', r'Volume-Equivalent Diameter ($\mu m$)', (0, 1100)),
        'sigma': ('LogSigma', 'LogSigma_sam', r'$\sigma_{LN}$', r'Log-Normal $\sigma_{LN}$', (0, 2))
    }

    sources = []
    if plot_orig:
        sources.append(('MATLAB', 'D_v', 'LogSigma'))
    if plot_sam:
        sources.append(('SAM', 'D_v_sam', 'LogSigma_sam'))

    ncols = len(sources)
    nrows = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if ncols == 1:
        axes = np.array(axes).reshape((2, 1))

    for row_idx, (key, (col_orig, col_sam, title, ylabel, ylim)) in enumerate(variable_map.items()):
        for col_idx, (label, d_col, sigma_col) in enumerate(sources):
            ax = axes[row_idx, col_idx]
            col_name = col_orig if label == 'MATLAB' else col_sam

            for j, temp in enumerate(temps):
                subset = avg_df[avg_df['Temp'] == temp].sort_values(by='VenturiAngle')
                ax.plot(subset['VenturiAngle'], subset[col_name], '-o',
                        label=f'{temp}°F', color=color_map(j), markerfacecolor=color_map(j))

            ax.set_xlabel('Venturi Angle (deg)')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{title} vs Angle ({label})')
            ax.grid(True)
            ax.set_ylim(ylim)
            ax.set_xlim([avg_df['VenturiAngle'].min() - 2, avg_df['VenturiAngle'].max() + 2])

            if 'D_v' in col_name:
                ax.axhline(1000, linestyle='--', color='k', label='Injection Hole Diameter')

            if row_idx == 0:
                ax.legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_angle_analysis(df, temperature, flow_rate, aeration_percent, colors, plot_orig=True, plot_sam=False):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import lognorm

    filtered = df[(df['Temp'] == temperature) &
                  (df['FlowRate'] == flow_rate) &
                  (df['AeratedFlow'] == aeration_percent)]
    if filtered.empty:
        raise ValueError('No data found for specified parameters.')

    avg_df = filtered.groupby('VenturiAngle', as_index=False).mean(numeric_only=True).sort_values('VenturiAngle')
    x = np.linspace(0, 1200, 1000)

    # Plot 1: Log-normal PDF
    fig, ax = plt.subplots(figsize=(10, 6))
    if plot_orig:
        for i, row in avg_df.iterrows():
            mu, sigma = row['LogMu'], row['LogSigma']
            pdf_vals = lognorm.pdf(x, s=sigma, scale=np.exp(mu))
            color = colors[i % len(colors)]
            ax.plot(x, pdf_vals, label=f"{row['VenturiAngle']}° (MATLAB)", color=color, linewidth=2)
            if not np.isnan(row['D_v']):
                ax.axvline(row['D_v'], linestyle='--', color=color, linewidth=1.5)

    if plot_sam:
        for i, row in avg_df.iterrows():
            mu, sigma = row['LogMu_sam'], row['LogSigma_sam']
            pdf_vals = lognorm.pdf(x, s=sigma, scale=np.exp(mu))
            color = colors[i % len(colors)]
            ax.plot(x, pdf_vals, linestyle=':', label=f"{row['VenturiAngle']}° (SAM)", color=color, linewidth=2)
            if not np.isnan(row['D_v_sam']):
                ax.axvline(row['D_v_sam'], linestyle=':', color=color, linewidth=1.5)

    ax.set_xlabel(r'Diameter ($\mu m$)', fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.set_title(f'Log-Normal Distributions vs Angle ({temperature}°F, {flow_rate} GPM, {aeration_percent}%)', fontsize=16)
    ax.grid(True)
    ax.set_xlim([0, 1200])
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# --- Streamlit App ---
st.title("🧪 Experiment Data Loader & Filter")

base_dir = st.text_input("Base directory", value=r"G:\My Drive\Master's Data Processing\Thesis Data")

if not os.path.isdir(base_dir):
    st.warning("Invalid path.")
    st.stop()

df = load_all_experiment_data(base_dir)
if df.empty:
    st.warning("No valid data found.")
    st.stop()

# --- Import external data ---
yin_data = import_yin_data()
sun_data = import_sun_data()

# --- Filter UI ---
st.subheader("📊 Filter Data")

# Dynamic filter options from data
angles_all = sorted(df['VenturiAngle'].dropna().unique())
temps_all = sorted(df['Temp'].dropna().unique())
aerations_all = sorted(df['AeratedFlow'].dropna().unique())
flows_all = sorted(df['FlowRate'].dropna().unique())

# Create three columns for better layout
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**Venturi Angles (°)**")
    plotAngles = [
        angle for angle in angles_all
        if st.checkbox(f"{angle}", value=True, key=f"angle_{angle}")
    ]

with col2:
    st.markdown("**Temperatures (°F)**")
    plotTemps = [
        temp for temp in temps_all
        if st.checkbox(f"{temp}", value=True, key=f"temp_{temp}")
    ]

with col3:
    st.markdown("**Aeration Rates (%)**")
    plotAerations = [
        aer for aer in aerations_all
        if st.checkbox(f"{aer}", value=True, key=f"aer_{aer}")
    ]
with col4:
    st.markdown("**Flow Rates (GPM)**")
    plotFlows = [
        flow for flow in flows_all
        if st.checkbox(f"{flow}", value=True, key=f"flow_{flow}")
    ]

# Reynolds number input
minRe = st.number_input("Minimum Reynolds #", value=0.0)

# --- Filtering ---
df_filtered = df[
    df['VenturiAngle'].isin(plotAngles) &
    df['Temp'].isin(plotTemps) &
    df['AeratedFlow'].isin(plotAerations) &
    df['FlowRate'].isin(plotFlows) &
    (df['Reynolds'] > minRe)
].copy()


st.success(f"Filtered to {len(df_filtered)} experiments.")
st.dataframe(df_filtered.head())

st.header("Plotting")

tab1, tab2 = st.tabs(["📏 Dimensional Data", "🔢 Nondimensional Data"])

with tab1:
    dropdown1 = st.selectbox(
        "Choose Dimensional Plot Type",
        ["Repeatability Comparison", "SAM vs MATLAB Comparison", 
         "Temperature", "Angle", "Flow Rate"]
    )

    if dropdown1 == "Repeatability Comparison":
        st.markdown("### Repeatability Comparison")
        col1, col2 = st.columns(2)
        with col1:
            plot_orig = st.checkbox("Show MATLAB data", value=True)
        with col2:
            plot_sam = st.checkbox("Show SAM data", value=True)

        # Run the repeatability plot with filtered data
        if plot_orig or plot_sam:
            plot_repeatability(df_filtered, plot_orig=plot_orig, plot_sam=plot_sam)
        else:
            st.warning("Please select at least one data source to plot.")
    
    elif dropdown1 == "SAM vs MATLAB Comparison":
        st.markdown("### SAM vs MATLAB Comparison")
        plot_sam_vs_matlab(df_filtered)
    elif dropdown1 == "Flow Rate":
        st.markdown("## Flow Rate Analysis")

        # --- Selection Parameters from df_filtered ---
        unique_angles = sorted(df_filtered['VenturiAngle'].dropna().unique())
        unique_aerations = sorted(df_filtered['AeratedFlow'].dropna().unique())
        unique_temps = sorted(df_filtered['Temp'].dropna().unique())

        angle = st.selectbox("Select Venturi Angle", unique_angles, index=0, key='angle_temp')
        aeration = st.selectbox("Select Aeration Rate (%)", unique_aerations, index=0, key='aeration_temp')

        st.markdown("### Data Source Options")
        col1, col2 = st.columns(2)
        with col1:
            plot_orig = st.checkbox("Include MATLAB Results", value=True, key='orig_temp')
        with col2:
            plot_sam = st.checkbox("Include SAM Results", value=True, key='sam_temp')

        st.divider()
        st.markdown("### Flow Rate Effect at Fixed Angle & Aeration")
        try:
            plot_flow_rate_effect(df_filtered, angle=angle, aeration=aeration,
                                plot_orig=plot_orig, plot_sam=plot_sam)
        except ValueError as e:
            st.warning(str(e))

        st.divider()
        st.markdown("### Flow Rate Analysis at Fixed Temp, Angle & Aeration")
        angle = st.selectbox("Select Venturi Angle", unique_angles, index=0, key='angle_temp2')
        aeration = st.selectbox("Select Aeration Rate (%)", unique_aerations, index=0, key='aeration_temp2')
        temperature = st.selectbox("Select Temperature (°F)", unique_temps, index=0, key='temp_temp2')
        try:
            plot_flow_rate_analysis(df_filtered, angle=angle, temperature=temperature,
                                    aeration_percent=aeration, colors=plt.cm.tab10.colors,
                                    plot_orig=plot_orig, plot_sam=plot_sam)
        except ValueError as e:
            st.warning(str(e))
    elif dropdown1 == "Temperature":
        st.markdown("## Temperature Analysis")

        # --- Selection Parameters from df_filtered ---
        unique_angles = sorted(df_filtered['VenturiAngle'].dropna().unique())
        unique_aerations = sorted(df_filtered['AeratedFlow'].dropna().unique())
        unique_flows = sorted(df_filtered['FlowRate'].dropna().unique())

        angle = st.selectbox("Select Venturi Angle", unique_angles, index=0, key='angle_temp3')
        aeration = st.selectbox("Select Aeration Rate (%)", unique_aerations, index=0, key='aeration_temp3')

        st.markdown("### Data Source Options")
        col1, col2 = st.columns(2)
        with col1:
            plot_orig = st.checkbox("Include MATLAB Results", value=True, key='orig_temp')
        with col2:
            plot_sam = st.checkbox("Include SAM Results", value=True, key='sam_temp')

        st.divider()
        st.markdown("### Temperature Effect at Fixed Angle & Aeration")
        try:
            plot_temperature_effect(df_filtered, angle=angle, aeration=aeration, plot_orig=plot_orig, plot_sam=plot_sam)
        except ValueError as e:
            st.warning(str(e))

        st.divider()
        st.markdown("### Temperature Analysis at Fixed Flow, Angle & Aeration")
        angle = st.selectbox("Select Venturi Angle", unique_angles, index=0, key='angle_temp4')
        aeration = st.selectbox("Select Aeration Rate (%)", unique_aerations, index=0, key='aeration_temp4')
        flow_rate = st.selectbox("Select Flow Rate (GPM)", unique_flows, index=0, key='flow_temp4')
        try:
            plot_temperature_analysis(df_filtered, angle=angle, flow_rate=flow_rate,
                                    aeration_percent=aeration, colors=plt.cm.tab10.colors,
                                    plot_orig=plot_orig, plot_sam=plot_sam)
        except ValueError as e:
            st.warning(str(e))
    elif dropdown1 == "Angle":
        st.markdown("## Angle Analysis")

        # --- Selection Parameters from df_filtered ---
        unique_angles = sorted(df_filtered['VenturiAngle'].dropna().unique())
        unique_aerations = sorted(df_filtered['AeratedFlow'].dropna().unique())
        unique_flows = sorted(df_filtered['FlowRate'].dropna().unique())
        unique_temps = sorted(df_filtered['Temp'].dropna().unique())

        flow_rate = st.selectbox("Select Flow Rate (GPM)", unique_flows, index=0, key='flow_temp5')
        aeration = st.selectbox("Select Aeration Rate (%)", unique_aerations, index=0, key='aeration_temp5')

        st.markdown("### Data Source Options")
        col1, col2 = st.columns(2)
        with col1:
            plot_orig = st.checkbox("Include MATLAB Results", value=True, key='orig_temp')
        with col2:
            plot_sam = st.checkbox("Include SAM Results", value=True, key='sam_temp')

        st.divider()
        st.markdown("### Angle Effect at Fixed Flow Rate & Aeration")
        try:
            plot_angle_effect(df_filtered,flow_rate=flow_rate, aeration=aeration, plot_orig=plot_orig, plot_sam=plot_sam)
        except ValueError as e:
            st.warning(str(e))

        st.divider()
        st.markdown("### Angle Analysis at Fixed Flow, Angle & Aeration")
        temperature = st.selectbox("Select Temperature (°F)", unique_temps, index=0, key='temp_temp2')
        aeration = st.selectbox("Select Aeration Rate (%)", unique_aerations, index=0, key='aeration_temp6')
        flow_rate = st.selectbox("Select Flow Rate (GPM)", unique_flows, index=0, key='flow_temp6')
        try:
            plot_angle_analysis(df_filtered,temperature=temperature, flow_rate=flow_rate,
                                    aeration_percent=aeration, colors=plt.cm.tab10.colors,
                                    plot_orig=plot_orig, plot_sam=plot_sam)
        except ValueError as e:
            st.warning(str(e))
    else:
        st.info(f"Placeholder for: **{dropdown1}**")

with tab2:
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    group_cols = ['Temp', 'FlowRate', 'VenturiAngle', 'AeratedFlow']
    # Helper function to get averaged D_v and D_v_sam per experiment set
    def get_d30_averaged_df(df):
        grouped = df.groupby(group_cols)
        records = []

        for key, group in grouped:
            base_row = group.iloc[0].copy()
            if group['Trial'].nunique() == 2:
                base_row['D_v'] = group['D_v'].mean(skipna=True)
                base_row['D_v_sam'] = group['D_v_sam'].mean(skipna=True)
                base_row['Trial'] = 0  # optional: mark as averaged
            records.append(base_row)

        return pd.DataFrame(records)

    # Apply it only for Reynolds plot
    df_combined = get_d30_averaged_df(df_filtered)

    dropdown2 = st.selectbox(
        "Choose Nondimensional Plot Type",
        ["Reynolds", "Weber", "Capillary", "Re^a We^b"]
    )

    if dropdown2 == "Reynolds":
        st.markdown("## Reynolds Number Analysis")
        # User options
        plot_orig = st.checkbox("Include MATLAB Results", value=True, key='re_orig')
        plot_sam = st.checkbox("Include SAM Results", value=True, key='re_sam')
        ext_data_opt = st.radio("Include External Data?", ["None", "Yin", "Sun", "Both"], index=3, horizontal=True)
        fit_opt = st.radio("Include Fit (d/D = A·Re^b)?", ["No", "Yes"], index=0, horizontal=True)
        scale_opt = st.radio("Scale", ["Linear", "Log"], index=0, horizontal=True)

        fig, ax = plt.subplots(figsize=(9, 6))

        color_map = plt.cm.get_cmap('viridis', len(df_filtered['Temp'].unique()))
        temp_to_color = {t: color_map(i) for i, t in enumerate(sorted(df_filtered['Temp'].unique()))}

        def add_internal(df, d_col, label_base):
            grouped = df.dropna(subset=[d_col, 'Reynolds', 'ThroatDiameter_m', 'Temp']) \
                        .groupby('Temp')

            for temp, group in grouped:
                x = group['Reynolds']
                y = group[d_col] * 1e-6 / group['ThroatDiameter_m']  # Convert μm to m
                color = temp_to_color[temp]
                label = f"{temp}°F ({label_base})"
                ax.scatter(x, y, label=label, marker='o' if label_base == "MATLAB" else 'x', color=color, alpha=0.7)

                if fit_opt == "Yes" and len(x) >= 3:
                    try:
                        def model_fn(Re, A, b): return A * Re ** b
                        popt, _ = curve_fit(model_fn, x, y, maxfev=10000)
                        x_fit = np.linspace(min(x), max(x), 200)
                        y_fit = model_fn(x_fit, *popt)
                        ax.plot(x_fit, y_fit, '--', color=color,
                                label=fr"{temp}°F {label_base} Fit: $A$={popt[0]:.2e}, $b$={popt[1]:.3f}")
                    except Exception as e:
                        st.warning(f"Fit failed for {label_base} at {temp}°F: {e}")

        if plot_orig:
            add_internal(df_combined, 'D_v', "MATLAB")
        if plot_sam:
            add_internal(df_combined, 'D_v_sam', "SAM")

        # External data
        def add_external(df, label, color):
            x = df['Re_t']
            y = df['D_v'] / df['ThroatDiameter_m']  # Convert μm to m
            ax.scatter(x, y, label=label, marker='s', color=color, edgecolor='k')
            return x.values, y.values

        if ext_data_opt in ["Yin", "Both"]:
            add_external(yin_data, "Yin et al. 2015", "tab:green")
        if ext_data_opt in ["Sun", "Both"]:
            add_external(sun_data.rename(columns={'Re': 'Re_t'}), "Sun et al. 2017", "tab:orange")

        # Axes formatting
        ax.axhline(y=1/6, color='k', linestyle='--', linewidth=1.2, label='Air Injection Diameter')
        ax.set_xlabel("Reynolds Number", fontsize=13)
        ax.set_ylabel(r"$d_{30}/D_t$", fontsize=13)
        ax.set_title("Normalized Diameter vs Reynolds Number", fontsize=14)
        ax.grid(True)
        ax.legend(fontsize=9, loc='best')

        if scale_opt == "Log":
            ax.set_xscale('log')

        st.pyplot(fig)
        plt.close(fig)
    elif dropdown2 == "Weber":
        st.markdown("## Weber Number Analysis")
        # User options
        plot_orig = st.checkbox("Include MATLAB Results", value=True, key='re_orig')
        plot_sam = st.checkbox("Include SAM Results", value=True, key='re_sam')
        ext_data_opt = st.radio("Include External Data?", ["None", "Yin", "Sun", "Both"], index=3, horizontal=True)
        fit_opt = st.radio("Include Fit (d/D = A·We^b)?", ["No", "Yes"], index=0, horizontal=True)
        scale_opt = st.radio("Scale", ["Linear", "Log"], index=0, horizontal=True)

        fig, ax = plt.subplots(figsize=(9, 6))

        color_map = plt.cm.get_cmap('viridis', len(df_filtered['Temp'].unique()))
        temp_to_color = {t: color_map(i) for i, t in enumerate(sorted(df_filtered['Temp'].unique()))}

        def add_internal(df, d_col, label_base):
            grouped = df.dropna(subset=[d_col, 'We_D', 'ThroatDiameter_m', 'Temp']) \
                        .groupby('Temp')

            for temp, group in grouped:
                x = group['We_D']
                y = group[d_col] * 1e-6 / group['ThroatDiameter_m']  # Convert μm to m
                color = temp_to_color[temp]
                label = f"{temp}°F ({label_base})"
                ax.scatter(x, y, label=label, marker='o' if label_base == "MATLAB" else 'x', color=color, alpha=0.7)

                if fit_opt == "Yes" and len(x) >= 3:
                    try:
                        def model_fn(Re, A, b): return A * Re ** b
                        popt, _ = curve_fit(model_fn, x, y, maxfev=10000)
                        x_fit = np.linspace(min(x), max(x), 200)
                        y_fit = model_fn(x_fit, *popt)
                        ax.plot(x_fit, y_fit, '--', color=color,
                                label=fr"{temp}°F {label_base} Fit: $A$={popt[0]:.2e}, $b$={popt[1]:.3f}")
                    except Exception as e:
                        st.warning(f"Fit failed for {label_base} at {temp}°F: {e}")

        if plot_orig:
            add_internal(df_combined, 'D_v', "MATLAB")
        if plot_sam:
            add_internal(df_combined, 'D_v_sam', "SAM")

        # External data
        def add_external(df, label, color):
            x = df['We']
            y = df['D_v'] / df['ThroatDiameter_m']  # Convert μm to m
            ax.scatter(x, y, label=label, marker='s', color=color, edgecolor='k')
            return x.values, y.values

        if ext_data_opt in ["Yin", "Both"]:
            add_external(yin_data, "Yin et al. 2015", "tab:green")
        if ext_data_opt in ["Sun", "Both"]:
            add_external(sun_data.rename(columns={'Re': 'Re_t'}), "Sun et al. 2017", "tab:orange")

        # Axes formatting
        ax.axhline(y=1/6, color='k', linestyle='--', linewidth=1.2, label='Air Injection Diameter')
        ax.set_xlabel("Weber Number", fontsize=13)
        ax.set_ylabel(r"$d_{30}/D_t$", fontsize=13)
        ax.set_title("Normalized Diameter vs Weber Number", fontsize=14)
        ax.grid(True)
        ax.legend(fontsize=9, loc='best')

        if scale_opt == "Log":
            ax.set_xscale('log')

        st.pyplot(fig)
        plt.close(fig)
    elif dropdown2 == "Capillary":
        st.markdown("## Capillary Number Analysis")
        # User options
        plot_orig = st.checkbox("Include MATLAB Results", value=True, key='re_orig')
        plot_sam = st.checkbox("Include SAM Results", value=True, key='re_sam')
        ext_data_opt = st.radio("Include External Data?", ["None", "Yin", "Sun", "Both"], index=3, horizontal=True)
        fit_opt = st.radio("Include Fit (d/D = A·We^b)?", ["No", "Yes"], index=0, horizontal=True)
        scale_opt = st.radio("Scale", ["Linear", "Log"], index=0, horizontal=True)

        fig, ax = plt.subplots(figsize=(9, 6))

        color_map = plt.cm.get_cmap('viridis', len(df_filtered['Temp'].unique()))
        temp_to_color = {t: color_map(i) for i, t in enumerate(sorted(df_filtered['Temp'].unique()))}

        def add_internal(df, d_col, label_base):
            grouped = df.dropna(subset=[d_col, 'Ca', 'ThroatDiameter_m', 'Temp']) \
                        .groupby('Temp')

            for temp, group in grouped:
                x = group['Ca']
                y = group[d_col] * 1e-6 / group['ThroatDiameter_m']  # Convert μm to m
                color = temp_to_color[temp]
                label = f"{temp}°F ({label_base})"
                ax.scatter(x, y, label=label, marker='o' if label_base == "MATLAB" else 'x', color=color, alpha=0.7)

                if fit_opt == "Yes" and len(x) >= 3:
                    try:
                        def model_fn(Re, A, b): return A * Re ** b
                        popt, _ = curve_fit(model_fn, x, y, maxfev=10000)
                        x_fit = np.linspace(min(x), max(x), 200)
                        y_fit = model_fn(x_fit, *popt)
                        ax.plot(x_fit, y_fit, '--', color=color,
                                label=fr"{temp}°F {label_base} Fit: $A$={popt[0]:.2e}, $b$={popt[1]:.3f}")
                    except Exception as e:
                        st.warning(f"Fit failed for {label_base} at {temp}°F: {e}")

        if plot_orig:
            add_internal(df_combined, 'D_v', "MATLAB")
        if plot_sam:
            add_internal(df_combined, 'D_v_sam', "SAM")

        # External data
        def add_external(df, label, color):
            x = df['Ca']
            y = df['D_v'] / df['ThroatDiameter_m']  # Convert μm to m
            ax.scatter(x, y, label=label, marker='s', color=color, edgecolor='k')
            return x.values, y.values

        if ext_data_opt in ["Yin", "Both"]:
            add_external(yin_data, "Yin et al. 2015", "tab:green")
        if ext_data_opt in ["Sun", "Both"]:
            add_external(sun_data, "Sun et al. 2017", "tab:orange")

        # Axes formatting
        ax.axhline(y=1/6, color='k', linestyle='--', linewidth=1.2, label='Air Injection Diameter')
        ax.set_xlabel("Capillary Number", fontsize=13)
        ax.set_ylabel(r"$d_{30}/D_t$", fontsize=13)
        ax.set_title("Normalized Diameter vs Capillary Number", fontsize=14)
        ax.grid(True)
        ax.legend(fontsize=9, loc='best')

        if scale_opt == "Log":
            ax.set_xscale('log')

        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info(f"Placeholder for: **{dropdown2}**")