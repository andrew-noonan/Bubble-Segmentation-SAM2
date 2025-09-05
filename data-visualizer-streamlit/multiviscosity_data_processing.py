#!/usr/bin/env python3
"""
Refactored multiviscosity data processing GUI
Improved performance, readability, and maintainability
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from scipy.stats import lognorm
from scipy.optimize import curve_fit
import threading
from datetime import datetime
import itertools
from functools import lru_cache
from typing import Dict, Tuple, Optional, List
import io
from PIL import Image

# --- Constants ---
UM_PER_PIXEL = 5.71
D_t = 6e-3
D_p = 15.8e-3
T_ref = 298.15
A_throat = np.pi * D_t**2 / 4
GPM_to_m3_s = 1 / (264.172053 * 60)

# --- Fluid property definitions ---
fluid_properties = {
    "10 cSt": {"A": 687, "mu_ref": 10, "alpha": 0.00105, "rho_ref": 934, "surfaceTensionYIntercept": 21.6, "surfaceTensionSlope": -0.06},
    "20 cSt": {"A": 752, "mu_ref": 19.7, "alpha": 0.00103, "rho_ref": 949, "surfaceTensionYIntercept": 22.1, "surfaceTensionSlope": -0.06},
    "50 cSt": {"A": 732, "mu_ref": 45, "alpha": 0.000994, "rho_ref": 959, "surfaceTensionYIntercept": 22.3, "surfaceTensionSlope": -0.06}
}

# --- Plot styling ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linewidth'] = 0.5

class DataProcessor:
    """Handles data loading, caching, and processing operations"""
    
    def __init__(self):
        self._data_cache = {}
        self._external_data = {'yin': None, 'sun': None}
    
    @lru_cache(maxsize=32)
    def compute_fluid_properties(self, temp_k: float, fluid_key: str) -> Tuple[float, float, float]:
        """Compute fluid properties with caching for performance"""
        props = fluid_properties[fluid_key]
        A, mu_ref, alpha, rho_ref = props["A"], props["mu_ref"], props["alpha"], props["rho_ref"]
        B = np.log10(mu_ref / 1000) - A / T_ref
        mu = 10 ** (A / temp_k + B)
        rho = rho_ref * (1 - alpha * (temp_k - T_ref))
        gamma = (props["surfaceTensionYIntercept"] + props["surfaceTensionSlope"] * (temp_k - 273.15)) / 1000
        return mu, rho, gamma
    
    def load_experiment_data(self, base_dir: str) -> pd.DataFrame:
        """Load and process all experiment data with caching"""
        cache_key = f"{base_dir}_{os.path.getmtime(base_dir) if os.path.exists(base_dir) else 0}"
        
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        records = []
        
        # Process each fluid type
        for fluid_folder in ["10 cSt", "20 cSt", "50 cSt"]:
            full_path = os.path.join(base_dir, fluid_folder)
            if not os.path.isdir(full_path):
                continue
                
            records.extend(self._process_fluid_folder(full_path, fluid_folder))
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = self._compute_derived_properties(df)
        
        self._data_cache[cache_key] = df
        return df
    
    def _process_fluid_folder(self, fluid_path: str, fluid_key: str) -> List[Dict]:
        """Process a single fluid folder and return records"""
        records = []
        
        for root, _, files in os.walk(fluid_path):
            if "experiment_summary.csv" not in files:
                continue
                
            try:
                record = self._parse_directory_structure(root, fluid_key)
                record.update(self._load_labview_data(root))
                record.update(self._load_sam_data(root))
                records.append(record)
            except Exception as e:
                print(f"Warning: Failed to process {root}: {e}")
                continue
        
        return records
    
    def _parse_directory_structure(self, root: str, fluid_key: str) -> Dict:
        """Parse experiment parameters from directory structure"""
        parts = root.replace("\\", "/").split("/")
        angle = int(parts[-4].split()[0])
        temp = int(parts[-3].replace("F", ""))
        aer = float(parts[-2].split()[0].replace("_", "."))
        trial = int(parts[-2].split()[-1])
        flow = float(parts[-1].replace("_", "."))
        
        return {
            'Temp': temp, 'FlowRate': flow, 'VenturiAngle': angle,
            'AeratedFlow': aer, 'Trial': trial, 'Viscosity_cSt': fluid_key, 'Valid': True
        }
    
    def _load_labview_data(self, root: str) -> Dict:
        """Load LabVIEW data from experiment directory"""
        lv_path = os.path.join(root, 'labview.txt')
        if not os.path.isfile(lv_path):
            return {}
        
        try:
            data = pd.read_csv(lv_path, encoding='utf-8', on_bad_lines='skip')
            return {
                'MeanTemp': data['Temp (F)'].mean(),
                'MeanFlow': data['Oil Flow Rate'].mean(),
                'MeanP1': data['P1'].mean(),
                'MeanP2': data['P2'].mean(),
            }
        except Exception:
            return {}
    
    def _load_sam_data(self, root: str) -> Dict:
        """Load SAM analysis results"""
        sam_path = os.path.join(root, 'experiment_summary.csv')
        try:
            sam_df = pd.read_csv(sam_path)
            return {
                'LogMu': sam_df['log_mu'].iloc[0] + np.log(UM_PER_PIXEL),
                'LogSigma': sam_df['log_sigma'].iloc[0],
                'D32': sam_df['d32'].iloc[0] * UM_PER_PIXEL,
                'D_v': sam_df['dv'].iloc[0] * UM_PER_PIXEL
            }
        except Exception:
            return {}
    
    def _compute_derived_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute derived properties and dimensionless numbers"""
        # Basic derived properties
        df['deltaP'] = df.get('MeanP1', np.nan) - df.get('MeanP2', np.nan)
        df['deltaP_Pa'] = df['deltaP'] * 6894.75729
        tempF = df['MeanTemp'].combine_first(df['Temp'])
        df['tempK'] = (tempF - 32) / 1.8 + 273.15
        
        # Initialize fluid property columns
        df[['mu', 'rho', 'Gamma']] = np.nan
        
        # Compute fluid properties by viscosity (vectorized)
        for fluid_key in fluid_properties.keys():
            mask = df['Viscosity_cSt'] == fluid_key
            if mask.any():
                temps = df.loc[mask, 'tempK'].values
                mu_vals, rho_vals, gamma_vals = zip(*[self.compute_fluid_properties(t, fluid_key) for t in temps])
                df.loc[mask, 'mu'] = mu_vals
                df.loc[mask, 'rho'] = rho_vals
                df.loc[mask, 'Gamma'] = gamma_vals
        
        # Dimensionless numbers (vectorized)
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
        
        # Calculate new dimensionless length scales
        # Weber1 = We*Dt
        df['Weber1'] = df['We_D'] * D_t
        
        # Weber2 = sqrt(We)*Dt  
        df['Weber2'] = np.sqrt(df['We_D']) * D_t
        
        # Capillary1 = Ca*Dt
        df['Capillary1'] = df['Ca'] * D_t
        
        # Capillary2 = sqrt(Ca)*D_t
        df['Capillary2'] = np.sqrt(df['Ca']) * D_t
        
        # Reynolds1 = Re*D_t
        df['Reynolds1'] = df['Reynolds'] * D_t
        
        # Reynolds2 = D_t/Re
        df['Reynolds2'] = D_t / df['Reynolds']
        
        return df
    
    def get_trial_averaged_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get trial-averaged data for analysis"""
        group_cols = ['Viscosity_cSt', 'Temp', 'FlowRate', 'VenturiAngle', 'AeratedFlow']
        grouped = df.groupby(group_cols)
        
        trial_avg_records = []
        for key, group in grouped:
            if group['Trial'].nunique() == 2:
                record = group.mean(numeric_only=True)
                for i, col in enumerate(group_cols):
                    record[col] = key[i]
                trial_avg_records.append(record)
        
        return pd.DataFrame(trial_avg_records)
    
    @property
    def yin_data(self) -> pd.DataFrame:
        """Get Yin et al. external data"""
        if self._external_data['yin'] is None:
            self._external_data['yin'] = self._load_yin_data()
        return self._external_data['yin']
    
    @property
    def sun_data(self) -> pd.DataFrame:
        """Get Sun et al. external data"""
        if self._external_data['sun'] is None:
            self._external_data['sun'] = self._load_sun_data()
        return self._external_data['sun']
    
    def _load_yin_data(self) -> pd.DataFrame:
        """Load Yin et al. 2015 data"""
        mu_water = 0.001
        sigma_water = 0.0728
        rho_water = 997
        D_t_yin = 0.023
        D_upstream = 0.053
        
        yin_raw = np.array([
            [138057.9483, 0.9731], [168598.2772, 0.8130], [199295.2232, 0.6924],
            [229992.1691, 0.5537], [260689.1151, 0.4993], [291386.0611, 0.4438]
        ])
        
        Re_upstream = yin_raw[:, 0]
        Re_throat = Re_upstream * (D_upstream / D_t_yin)
        d_v_m = yin_raw[:, 1] / 1000
        
        V_throat = (Re_throat * mu_water) / (rho_water * D_t_yin)
        Ca = (mu_water * V_throat) / sigma_water
        We = (rho_water * V_throat**2 * D_t_yin) / sigma_water
        
        # Calculate additional length scales for normalization
        nu_water = mu_water / rho_water
        
        # Calculate new dimensionless length scales for Yin data
        # Weber1 = We*Dt
        Weber1 = We * D_t_yin
        
        # Weber2 = sqrt(We)*Dt  
        Weber2 = np.sqrt(We) * D_t_yin
        
        # Capillary1 = Ca*Dt
        Capillary1 = Ca * D_t_yin
        
        # Capillary2 = sqrt(Ca)*D_t
        Capillary2 = np.sqrt(Ca) * D_t_yin
        
        # Reynolds1 = Re*D_t
        Reynolds1 = Re_throat * D_t_yin
        
        # Reynolds2 = D_t/Re
        Reynolds2 = D_t_yin / Re_throat
        
        return pd.DataFrame({
            'Re_upstream': Re_upstream, 'Re_t': Re_throat, 'D_v': d_v_m,
            'Velocity_m_per_s': V_throat, 'Ca': Ca, 'We': We,
            'ThroatDiameter_m': D_t_yin, 'DivergingL_m': (53 - 23) / 2 / np.tan(np.radians(8)) / 1000,
            'Weber1': Weber1, 'Weber2': Weber2, 'Capillary1': Capillary1,
            'Capillary2': Capillary2, 'Reynolds1': Reynolds1, 'Reynolds2': Reynolds2
        })
    
    def _load_sun_data(self) -> pd.DataFrame:
        """Load Sun et al. 2017 data"""
        mu_water = 0.001
        sigma_water = 0.0728
        rho_water = 997
        D_t = 0.025
        
        sun_raw = np.array([
            [229646.4949, 0.038018832], [244925.1049, 0.036647834], [260263.6309, 0.032188324],
            [275542.2409, 0.031450094], [290880.7669, 0.029551789], [306159.3769, 0.024429379],
            [321437.9868, 0.021943503],
        ])
        
        Re_water = sun_raw[:, 0]
        d_v_m = sun_raw[:, 1] * D_t
        V_throat = (Re_water * mu_water) / (rho_water * D_t)
        Ca = (mu_water * V_throat) / sigma_water
        We = (rho_water * V_throat**2 * D_t) / sigma_water
        
        # Calculate additional length scales for normalization
        nu_water = mu_water / rho_water
        
        # Calculate new dimensionless length scales for Sun data
        # Weber1 = We*Dt
        Weber1 = We * D_t
        
        # Weber2 = sqrt(We)*Dt  
        Weber2 = np.sqrt(We) * D_t
        
        # Capillary1 = Ca*Dt
        Capillary1 = Ca * D_t
        
        # Capillary2 = sqrt(Ca)*D_t
        Capillary2 = np.sqrt(Ca) * D_t
        
        # Reynolds1 = Re*D_t
        Reynolds1 = Re_water * D_t
        
        # Reynolds2 = D_t/Re
        Reynolds2 = D_t / Re_water
        
        return pd.DataFrame({
            'Re': Re_water, 'D_v': d_v_m, 'Velocity_m_per_s': V_throat,
            'Ca': Ca, 'We': We, 'ThroatDiameter_m': D_t, 
            'DivergingL_m': (50 - 25) / 2 / np.tan(np.radians(7.5)) / 1000,
            'Weber1': Weber1, 'Weber2': Weber2, 'Capillary1': Capillary1,
            'Capillary2': Capillary2, 'Reynolds1': Reynolds1, 'Reynolds2': Reynolds2
        })


class PlottingManager:
    """Enhanced plotting configuration and utilities with better visibility"""
    
    # Color schemes - enhanced for better visibility
    FLUID_COLORS = {"50 cSt": '#D32F2F', "20 cSt": '#000000', "10 cSt": '#1976D2'}  # Red, Black, Blue
    EXTERNAL_COLORS = {'Yin': '#4CAF50', 'Sun': '#FF9800'}  # Green, Orange
    
    # Style arrays - improved markers for visibility
    LINESTYLES = ['-', '--', ':', '-.']
    MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '+']
    
    # Enhanced font sizes for better readability
    FONT_SIZES = {
        'title': 11,
        'label': 10,
        'tick': 8,
        'legend': 8,
        'text': 8
    }
    
    # Improved marker sizes for visibility
    MARKER_SIZES = {
        'scatter': 20,  # Increased from 15-16
        'legend': 1.2,  # Increased legend marker scale
        'line_width': 1.2  # Increased line width
    }
    
    @classmethod
    def setup_plot_style(cls, ax, title="", xlabel="", ylabel="", grid=True):
        """Apply consistent styling to plot with enhanced readability"""
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=cls.FONT_SIZES['label'])
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=cls.FONT_SIZES['label'])
        
        ax.tick_params(labelsize=cls.FONT_SIZES['tick'])
        if grid:
            ax.grid(True, alpha=0.3)
        
        # Enhanced grid and spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        
        # Force standard notation
        from matplotlib.ticker import ScalarFormatter
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
    
    @classmethod
    def get_color_marker_iterator(cls, data_groups):
        """Get iterator for consistent color/marker assignment with enhanced visibility"""
        colors = list(cls.FLUID_COLORS.values()) + ['#9C27B0', '#00BCD4', '#795548', '#607D8B']
        markers = cls.MARKERS
        
        color_cycle = itertools.cycle(colors)
        marker_cycle = itertools.cycle(markers)
        
        return [(next(color_cycle), next(marker_cycle)) for _ in data_groups]
    
    @classmethod
    def plot_scatter_enhanced(cls, ax, x, y, color, marker, label, alpha=0.8, size=None):
        """Enhanced scatter plot with better visibility"""
        if size is None:
            size = cls.MARKER_SIZES['scatter']
        
        ax.scatter(x, y, c=color, marker=marker, s=size, alpha=alpha,
                  label=label, edgecolors='white', linewidths=0.8)
    
    @classmethod
    def plot_line_enhanced(cls, ax, x, y, color, linestyle, label, linewidth=None):
        """Enhanced line plot with better visibility"""
        if linewidth is None:
            linewidth = cls.MARKER_SIZES['line_width']
        
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, 
               label=label, alpha=0.9)
    
    @classmethod
    def combine_external_data(cls, internal_df, yin_data, sun_data, x_cols, include_external_plot=True, include_external_fit=False):
        """Combine internal and external data for universal scaling plots"""
        combined_data = []
        
        # Add internal data - include all length scale columns
        required_cols = x_cols + ['D_v', 'ThroatDiameter_m', 'Weber1', 'Weber2', 'Capillary1', 'Capillary2', 'Reynolds1', 'Reynolds2']
        for _, row in internal_df.iterrows():
            data_row = {'source': 'Internal', 'Viscosity_cSt': row['Viscosity_cSt']}
            for col in required_cols:
                if col in row:
                    data_row[col] = row[col]
            combined_data.append(data_row)
        
        # Add external data if requested
        if include_external_plot or include_external_fit:
            # Map column names for external data
            col_mapping = {
                'Reynolds': {'Yin': 'Re_t', 'Sun': 'Re'},
                'We_D': {'Yin': 'We', 'Sun': 'We'},
                'Ca': {'Yin': 'Ca', 'Sun': 'Ca'}
            }
            
            # Add Yin data with length scale columns
            if yin_data is not None:
                for _, row in yin_data.iterrows():
                    data_row = {'source': 'Yin et al. 2015', 'Viscosity_cSt': None}
                    for col in x_cols:
                        mapped_col = col_mapping.get(col, {}).get('Yin', col)
                        if mapped_col in row:
                            data_row[col] = row[mapped_col]
                    data_row['D_v'] = row['D_v'] * 1e6  # Convert back to microns for consistency
                    data_row['ThroatDiameter_m'] = row['ThroatDiameter_m']
                    
                    # Add length scale columns if they exist
                    for length_col in ['Weber1', 'Weber2', 'Capillary1', 'Capillary2', 'Reynolds1', 'Reynolds2']:
                        if length_col in row:
                            data_row[length_col] = row[length_col]
                    
                    combined_data.append(data_row)
            
            # Add Sun data with length scale columns
            if sun_data is not None:
                for _, row in sun_data.iterrows():
                    data_row = {'source': 'Sun et al. 2017', 'Viscosity_cSt': None}
                    for col in x_cols:
                        mapped_col = col_mapping.get(col, {}).get('Sun', col)
                        if mapped_col in row:
                            data_row[col] = row[mapped_col]
                    data_row['D_v'] = row['D_v'] * 1e6  # Convert back to microns for consistency
                    data_row['ThroatDiameter_m'] = row['ThroatDiameter_m']
                    
                    # Add length scale columns if they exist
                    for length_col in ['Weber1', 'Weber2', 'Capillary1', 'Capillary2', 'Reynolds1', 'Reynolds2']:
                        if length_col in row:
                            data_row[length_col] = row[length_col]
                    
                    combined_data.append(data_row)
        
        return pd.DataFrame(combined_data)
    
    @classmethod
    def create_universal_fixed_exponent_plot(cls, ax, df_fit, x_cols, y_col, a, b, yin_data=None, sun_data=None, include_external_plot=True, include_external_fit=False, show_air_injection=True, normalization_func=None, external_norm_func=None):
        """Create universal scaling plot with fixed user-specified exponents"""
        if len(x_cols) != 2:
            raise ValueError("Universal scaling requires exactly 2 x columns")
        
        # Combine with external data if needed
        if (include_external_plot or include_external_fit) and yin_data is not None and sun_data is not None:
            df_combined = cls.combine_external_data(df_fit, yin_data, sun_data, x_cols, include_external_plot, include_external_fit)
        else:
            df_combined = df_fit.copy()
            df_combined['source'] = 'Internal'
        
        # Prepare data for fitting with fixed exponents
        fit_subset = df_combined.copy()
        
        # Use normalization function if provided
        if normalization_func is not None:
            # Separate internal and external data for normalization
            internal_mask = fit_subset['source'] == 'Internal'
            fit_subset['y_norm'] = np.nan
            
            # Normalize internal data
            if internal_mask.any():
                try:
                    internal_data = fit_subset[internal_mask]
                    y_vals, _, _ = normalization_func(internal_data)
                    fit_subset.loc[internal_mask, 'y_norm'] = y_vals
                except Exception:
                    # Fall back to throat diameter
                    internal_data = fit_subset[internal_mask]
                    fit_subset.loc[internal_mask, 'y_norm'] = internal_data[y_col] * 1e-6 / internal_data['ThroatDiameter_m']
            
            # Normalize external data
            if external_norm_func is not None and (~internal_mask).any():
                try:
                    external_data = fit_subset[~internal_mask]
                    y_vals = external_norm_func(external_data)
                    fit_subset.loc[~internal_mask, 'y_norm'] = y_vals
                except Exception:
                    # Fall back to throat diameter
                    external_data = fit_subset[~internal_mask]
                    fit_subset.loc[~internal_mask, 'y_norm'] = external_data[y_col] * 1e-6 / external_data['ThroatDiameter_m']
        else:
            # Default throat diameter normalization
            fit_subset['y_norm'] = fit_subset[y_col] * 1e-6 / fit_subset['ThroatDiameter_m']
        
        # Calculate collapse coordinate with fixed exponents for both datasets
        try:
            df_combined['CollapseX'] = np.power(df_combined[x_cols[0]], a) * np.power(df_combined[x_cols[1]], b)
            df_combined['CollapseX'] = df_combined['CollapseX'].replace([np.inf, -np.inf, np.nan], 1e-10)
            
            fit_subset['CollapseX'] = np.power(fit_subset[x_cols[0]], a) * np.power(fit_subset[x_cols[1]], b)
            fit_subset['CollapseX'] = fit_subset['CollapseX'].replace([np.inf, -np.inf, np.nan], 1e-10)
        except Exception:
            df_combined['CollapseX'] = np.exp(a * np.log(df_combined[x_cols[0]]) + b * np.log(df_combined[x_cols[1]]))
            df_combined['CollapseX'] = df_combined['CollapseX'].replace([np.inf, -np.inf, np.nan], 1e-10)
            
            fit_subset['CollapseX'] = np.exp(a * np.log(fit_subset[x_cols[0]]) + b * np.log(fit_subset[x_cols[1]]))
            fit_subset['CollapseX'] = fit_subset['CollapseX'].replace([np.inf, -np.inf, np.nan], 1e-10)
        
        # Normalize diameter data for plotting using selected normalization
        if normalization_func is not None:
            # Separate internal and external data for normalization
            internal_mask = df_combined['source'] == 'Internal'
            df_combined['NormDiameter'] = np.nan
            
            # Normalize internal data
            if internal_mask.any():
                try:
                    internal_data = df_combined[internal_mask]
                    y_norm, _, _ = normalization_func(internal_data)
                    df_combined.loc[internal_mask, 'NormDiameter'] = y_norm
                except Exception:
                    internal_data = df_combined[internal_mask]
                    df_combined.loc[internal_mask, 'NormDiameter'] = internal_data[y_col] * 1e-6 / internal_data['ThroatDiameter_m']
            
            # Normalize external data
            if external_norm_func is not None and (~internal_mask).any():
                try:
                    external_data = df_combined[~internal_mask]
                    df_combined.loc[~internal_mask, 'NormDiameter'] = external_norm_func(external_data)
                except Exception:
                    external_data = df_combined[~internal_mask]
                    df_combined.loc[~internal_mask, 'NormDiameter'] = external_data[y_col] * 1e-6 / external_data['ThroatDiameter_m']
        else:
            # Default throat diameter normalization
            df_combined['NormDiameter'] = df_combined[y_col] * 1e-6 / df_combined['ThroatDiameter_m']
        
        # Solve for A using linear regression on log-transformed data
        try:
            # Remove invalid data
            valid_data = fit_subset.dropna(subset=['CollapseX', 'y_norm'])
            valid_data = valid_data[(valid_data['CollapseX'] > 0) & (valid_data['y_norm'] > 0)]
            
            if len(valid_data) < 3:
                raise ValueError("Insufficient valid data for fitting")
            
            # Linear fit: log(y) = log(A) + log(x) 
            x_fit_data = np.log(valid_data['CollapseX'])
            y_fit_data = np.log(valid_data['y_norm'])
            
            # Simple linear regression to find A
            A = np.exp(np.mean(y_fit_data - x_fit_data))
            
            # Calculate R²
            y_pred = A * valid_data['CollapseX']
            y_actual = valid_data['y_norm']
            r_squared = 1 - (np.sum((y_actual - y_pred) ** 2) / np.sum((y_actual - np.mean(y_actual)) ** 2))
            
            # Plot data points
            internal_data = df_combined[df_combined['source'] == 'Internal']
            markers = {'10 cSt': 'o', '20 cSt': 's', '50 cSt': '^'}
            
            for visc in sorted(internal_data['Viscosity_cSt'].unique()):
                subset = internal_data[internal_data['Viscosity_cSt'] == visc]
                if not subset.empty:
                    color = cls.FLUID_COLORS.get(visc, '#666666')
                    marker = markers.get(visc, 'x')
                    cls.plot_scatter_enhanced(ax, subset['CollapseX'], subset['NormDiameter'],
                                             color, marker, f"{visc}")
            
            # Plot external data if requested
            if include_external_plot:
                external_data = df_combined[df_combined['source'] != 'Internal']
                yin_data_plot = external_data[external_data['source'] == 'Yin et al. 2015']
                sun_data_plot = external_data[external_data['source'] == 'Sun et al. 2017']
                
                if not yin_data_plot.empty:
                    cls.plot_scatter_enhanced(ax, yin_data_plot['CollapseX'], yin_data_plot['NormDiameter'],
                                             cls.EXTERNAL_COLORS['Yin'], 's', "Yin et al. 2015")
                
                if not sun_data_plot.empty:
                    cls.plot_scatter_enhanced(ax, sun_data_plot['CollapseX'], sun_data_plot['NormDiameter'],
                                             cls.EXTERNAL_COLORS.get('Sun', '#FF9800'), 's', "Sun et al. 2017")
            
            # Plot fit line
            x_min, x_max = df_combined['CollapseX'].min(), df_combined['CollapseX'].max()
            if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min and x_min > 0:
                # Minimal extension in log space, or none if range is already large
                log_x_min, log_x_max = np.log10(x_min), np.log10(x_max)
                log_range = log_x_max - log_x_min
                
                # Only extend if the range is small (< 2 decades)
                if log_range < 2.0:
                    extension = min(log_range * 0.05, 0.1)  # Max 5% or 0.1 decades
                    log_x_fit_min = log_x_min - extension
                    log_x_fit_max = log_x_max + extension
                else:
                    # Large range - no extension needed
                    log_x_fit_min = log_x_min
                    log_x_fit_max = log_x_max
                
                # Use more points for large log ranges
                num_points = max(500, int(log_range * 200))  # Scale points with decades
                x_fit = np.logspace(log_x_fit_min, log_x_fit_max, min(num_points, 2000))  # Cap at 2000 points
                y_fit = A * x_fit
                ax.plot(x_fit, y_fit, 'k-', linewidth=2, alpha=0.7, 
                       label=f'Fit: A = {A:.2f}, R² = {r_squared:.2f}')
            
            # Get appropriate y-label based on normalization
            if normalization_func is not None:
                sample_internal = df_combined[df_combined['source'] == 'Internal'].head(1)
                if not sample_internal.empty:
                    _, y_label, _ = normalization_func(sample_internal)
                else:
                    y_label = r"$d_{30} / D_t$"
            else:
                y_label = r"$d_{30} / D_t$"
            
            x_label = f"$Re^{{{a:.1f}}} \\times We^{{{b:.1f}}}$"
            
            cls.setup_plot_style(ax, xlabel=x_label, ylabel=y_label)
            ax.set_xscale('log')
            ax.set_yscale('log')
            
            # Enhanced grid and tick spacing
            ax.grid(True, alpha=0.3, which='both')
            ax.grid(True, alpha=0.1, which='minor')
            
            # Better tick spacing for log plots
            from matplotlib.ticker import LogLocator
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=8))
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs='auto', numticks=20))
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs='auto', numticks=20))
            
            cls.create_legend(ax)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Fixed exponent fit failed: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    @classmethod
    def create_universal_scaling_plot(cls, ax, df_fit, x_cols, y_col, title_prefix="", yin_data=None, sun_data=None, include_external_plot=True, include_external_fit=False, fluid_plot_include=None, fluid_fit_include=None, show_air_injection=True, normalization_func=None, external_norm_func=None, x_scale='log', y_scale='log'):
        """Create collapsed scaling plot with curve fitting"""
        if len(x_cols) != 2:
            raise ValueError("Universal scaling requires exactly 2 x columns")
        
        # Set default fluid inclusion if not provided
        if fluid_plot_include is None:
            fluid_plot_include = {fluid: True for fluid in df_fit['Viscosity_cSt'].unique()}
        if fluid_fit_include is None:
            fluid_fit_include = {fluid: True for fluid in df_fit['Viscosity_cSt'].unique()}
        
        # Filter internal data based on fluid inclusion settings
        df_fit_filtered = df_fit.copy()
        
        # Combine with external data if needed
        if (include_external_plot or include_external_fit) and yin_data is not None and sun_data is not None:
            df_combined = cls.combine_external_data(df_fit_filtered, yin_data, sun_data, x_cols, include_external_plot, include_external_fit)
            # Filter for fitting: only include fluids selected for fit + external if requested
            fit_data = df_combined.copy()
            if include_external_fit:
                # Keep external data and selected internal fluids for fitting
                fit_mask = (fit_data['source'] != 'Internal') | (fit_data['Viscosity_cSt'].isin([f for f, include in fluid_fit_include.items() if include]))
                fit_data = fit_data[fit_mask]
            else:
                # Only use selected internal fluids for fitting
                fit_data = fit_data[fit_data['source'] == 'Internal']
                fit_data = fit_data[fit_data['Viscosity_cSt'].isin([f for f, include in fluid_fit_include.items() if include])]
        else:
            df_combined = df_fit_filtered.copy()
            df_combined['source'] = 'Internal'
            # Only use selected fluids for fitting
            fit_data = df_combined[df_combined['Viscosity_cSt'].isin([f for f, include in fluid_fit_include.items() if include])]
        
        # Prepare data for fitting
        fit_subset = fit_data.dropna(subset=x_cols + [y_col])
        # Normalize diameter data for fitting using selected normalization
        fit_subset_norm = fit_subset.copy()
        
        # Use normalization function if provided, otherwise default to throat diameter
        if normalization_func is not None:
            # Separate internal and external data for normalization
            internal_mask = fit_subset_norm['source'] == 'Internal'
            fit_subset_norm['y_norm'] = np.nan
            
            # Normalize internal data
            if internal_mask.any():
                internal_data = fit_subset_norm[internal_mask]
                y_norm, _, _ = normalization_func(internal_data)  # Get normalization parameters
                fit_subset_norm.loc[internal_mask, 'y_norm'] = y_norm
            
            # Normalize external data
            if external_norm_func is not None and (~internal_mask).any():
                external_data = fit_subset_norm[~internal_mask]
                # External data in combined df has D_v converted to micrometers (line 449, 467)
                # but get_normalized_external_data expects D_v in meters, so we need to convert back
                external_data_corrected = external_data.copy()
                external_data_corrected['D_v'] = external_data_corrected['D_v'] * 1e-6  # Convert μm back to m
                fit_subset_norm.loc[~internal_mask, 'y_norm'] = external_norm_func(external_data_corrected)
        else:
            # Default throat diameter normalization
            fit_subset_norm['y_norm'] = fit_subset_norm[y_col] * 1e-6 / fit_subset_norm['ThroatDiameter_m']
        
        xdata = fit_subset_norm[x_cols].values.T  # shape (2, N)
        ydata = fit_subset_norm['y_norm'].values
        
        def model_fn(X, A, a, b):
            # Add bounds checking to prevent overflow
            try:
                result = A * np.power(X[0], a) * np.power(X[1], b)
                return np.where(np.isfinite(result), result, 1e-10)
            except (OverflowError, RuntimeWarning):
                return np.full_like(X[0], 1e-10)
        
        try:
            # For universal scaling, use better initial guesses based on typical values
            if 'We' in x_cols[0] or 'We' in x_cols[1]:
                # Re-We scaling typically has negative exponents
                p0 = [0.1, -0.6, -0.2]
            else:
                # Re-Ca scaling
                p0 = [0.1, -0.6, -0.3]
            
            # Fit the model without bounds (like MATLAB)
            popt, pcov = curve_fit(model_fn, xdata, ydata, p0=p0, maxfev=20000)
            A, a, b = popt
            
            # Check fit quality
            y_pred = model_fn(xdata, A, a, b)
            r_squared = 1 - (np.sum((ydata - y_pred) ** 2) / np.sum((ydata - np.mean(ydata)) ** 2))
            
            # Compute collapsed x-axis for all data with overflow protection
            try:
                with np.errstate(over='raise', invalid='raise'):
                    df_combined['CollapseX'] = np.power(df_combined[x_cols[0]], a) * np.power(df_combined[x_cols[1]], b)
                    # Replace any infinite or NaN values
                    df_combined['CollapseX'] = df_combined['CollapseX'].replace([np.inf, -np.inf, np.nan], 1e-10)
            except (OverflowError, FloatingPointError):
                # Fallback to safer calculation
                df_combined['CollapseX'] = np.exp(a * np.log(df_combined[x_cols[0]]) + b * np.log(df_combined[x_cols[1]]))
                df_combined['CollapseX'] = df_combined['CollapseX'].replace([np.inf, -np.inf, np.nan], 1e-10)
            
            # Normalize diameter data for plotting using selected normalization
            if normalization_func is not None:
                # Separate internal and external data for normalization
                internal_mask = df_combined['source'] == 'Internal'
                df_combined['NormDiameter'] = np.nan
                
                # Normalize internal data
                if internal_mask.any():
                    internal_data = df_combined[internal_mask]
                    y_norm, _, _ = normalization_func(internal_data)
                    df_combined.loc[internal_mask, 'NormDiameter'] = y_norm
                
                # Normalize external data
                if external_norm_func is not None and (~internal_mask).any():
                    external_data = df_combined[~internal_mask]
                    # External data in combined df has D_v converted to micrometers (line 449, 467)
                    # but get_normalized_external_data expects D_v in meters, so we need to convert back
                    external_data_corrected = external_data.copy()
                    external_data_corrected['D_v'] = external_data_corrected['D_v'] * 1e-6  # Convert μm back to m
                    df_combined.loc[~internal_mask, 'NormDiameter'] = external_norm_func(external_data_corrected)
            else:
                # Default throat diameter normalization
                df_combined['NormDiameter'] = df_combined['D_v'] * 1e-6 / df_combined['ThroatDiameter_m']
            
            # Plot internal data by viscosity (only show selected fluids)
            internal_data = df_combined[df_combined['source'] == 'Internal']
            markers = {'10 cSt': 'o', '20 cSt': 's', '50 cSt': '^'}
            
            # Get fluids to show
            fluids_to_show = [f for f, include in fluid_plot_include.items() if include]
            for visc in sorted(fluids_to_show):
                if visc in internal_data['Viscosity_cSt'].values:
                    subset = internal_data[internal_data['Viscosity_cSt'] == visc]
                    color = cls.FLUID_COLORS.get(visc, '#666666')
                    marker = markers.get(visc, 'x')
                    
                    cls.plot_scatter_enhanced(ax, subset['CollapseX'], subset['NormDiameter'],
                                             color, marker, f"{visc}")
            
            # Plot external data if requested
            if include_external_plot:
                yin_data_plot = df_combined[df_combined['source'] == 'Yin et al. 2015']
                if not yin_data_plot.empty:
                    cls.plot_scatter_enhanced(ax, yin_data_plot['CollapseX'], yin_data_plot['NormDiameter'],
                                             cls.EXTERNAL_COLORS['Yin'], 's', "Yin et al. 2015")
                
                sun_data_plot = df_combined[df_combined['source'] == 'Sun et al. 2017']
                if not sun_data_plot.empty:
                    cls.plot_scatter_enhanced(ax, sun_data_plot['CollapseX'], sun_data_plot['NormDiameter'],
                                             cls.EXTERNAL_COLORS.get('Sun', '#FF9800'), 's', "Sun et al. 2017")
            
            # Plot best-fit line with bounds checking
            # The relationship is d/D = A * (Re^a * We^b), so on the collapsed plot it should be d/D = A * x
            # Use only the x-range from the data that was actually used in the fit
            fit_collapsed_x = np.power(xdata[0], a) * np.power(xdata[1], b)
            x_min, x_max = fit_collapsed_x.min(), fit_collapsed_x.max()
            if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min and x_min > 0:
                # Create logarithmically spaced points for better visualization on log-log plot
                # No extension - line should only span the actual fit data range
                log_x_min, log_x_max = np.log10(x_min), np.log10(x_max)
                log_range = log_x_max - log_x_min
                
                # Use appropriate number of points for smooth line
                num_points = max(100, int(log_range * 100))  # Scale points with decades
                x_fit = np.logspace(log_x_min, log_x_max, min(num_points, 1000))  # Cap at 1000 points
                y_fit = A * x_fit  # This is the correct relationship for universal scaling
                # Only plot if y_fit values are reasonable
                if np.all(np.isfinite(y_fit)) and np.all(y_fit > 0):
                    # Create full equation label with proper LaTeX formatting
                    x1_name = x_cols[0].replace('Reynolds', 'Re').replace('We_D', 'We')
                    x2_name = x_cols[1].replace('Reynolds', 'Re').replace('We_D', 'We').replace('Ca', 'Ca')
                    equation = f"$d/D_t = {A:.1f} \cdot {x1_name}^{{{a:.1f}}} \cdot {x2_name}^{{{b:.1f}}}, R^2 = {r_squared:.2f}$"
                    ax.plot(x_fit, y_fit, 'k--', linewidth=2, alpha=0.8, label=equation)
            
            # Air injection reference line (optional)
            if show_air_injection:
                air_injection_ratio = 0.001 / 0.006  # d_air/D_t ≈ 1/6
                ax.axhline(air_injection_ratio, color='k', linestyle=':', linewidth=1.5, alpha=0.7, 
                          label=r'$D_{air}/D_t$ ≈ 1/6')
            
            # Styling
            x_label = fr"${x_cols[0].replace('Reynolds', 'Re').replace('We_D', 'We')}^{{{a:.1f}}} \cdot {x_cols[1].replace('Reynolds', 'Re').replace('We_D', 'We').replace('Ca', 'Ca')}^{{{b:.1f}}}$"
            
            # Create fit description
            fit_fluids = [f for f, include in fluid_fit_include.items() if include]
            fit_desc = f"Fit: {', '.join([f'{f}' for f in fit_fluids])}"
            if include_external_fit:
                fit_desc += " + Ext."
            
            # Get appropriate y-label based on normalization
            if normalization_func is not None:
                # Use a sample of internal data to get the y-label
                sample_internal = df_combined[df_combined['source'] == 'Internal'].head(1)
                if not sample_internal.empty:
                    _, y_label, _ = normalization_func(sample_internal)
                else:
                    y_label = r"$d_{30} / D_t$"
            else:
                y_label = r"$d_{30} / D_t$"
            
            cls.setup_plot_style(ax, 
                               xlabel=x_label,
                               ylabel=y_label)
            
            ax.set_xscale(x_scale.lower())
            ax.set_yscale(y_scale.lower())
            
            # Add enhanced grid and tick spacing
            ax.grid(True, alpha=0.3, which='both')
            ax.grid(True, alpha=0.1, which='minor')
            
            # Configure tick spacing based on scale type
            if x_scale.lower() == 'log':
                from matplotlib.ticker import LogLocator
                ax.xaxis.set_major_locator(LogLocator(base=10, numticks=8))
                ax.xaxis.set_minor_locator(LogLocator(base=10, subs='auto', numticks=20))
            
            if y_scale.lower() == 'log':
                from matplotlib.ticker import LogLocator
                ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
                ax.yaxis.set_minor_locator(LogLocator(base=10, subs='auto', numticks=20))
            
            cls.create_legend(ax)
            
            return True, (A, a, b)
            
        except Exception as e:
            print(f"Universal scaling fit failed: {e}")
            print(f"Data shapes - X: {xdata.shape}, Y: {ydata.shape}")
            print(f"Data ranges - X1: {xdata[0].min():.2f}-{xdata[0].max():.2f}, X2: {xdata[1].min():.2f}-{xdata[1].max():.2f}, Y: {ydata.min():.2f}-{ydata.max():.2f}")
            
            # Still plot the data points even if fitting fails - use simple scatter plot
            internal_data = df_combined[df_combined['source'] == 'Internal']
            markers = {'10 cSt': 'o', '20 cSt': 's', '50 cSt': '^'}
            
            for visc in sorted(internal_data['Viscosity_cSt'].unique()):
                subset = internal_data[internal_data['Viscosity_cSt'] == visc]
                if not subset.empty:
                    color = cls.FLUID_COLORS.get(visc, '#666666')
                    marker = markers.get(visc, 'x')
                    
                    # Plot against first dimensionless number as fallback with selected normalization
                    if normalization_func is not None:
                        y_vals, y_label, _ = normalization_func(subset)
                    else:
                        y_vals = subset['D_v'] * 1e-6 / subset['ThroatDiameter_m']
                        y_label = r"$d_{30} / D_t$"
                    
                    cls.plot_scatter_enhanced(ax, subset[x_cols[0]], y_vals,
                                             color, marker, f"{visc}")
            
            cls.setup_plot_style(ax, 
                               xlabel=x_cols[0],
                               ylabel=y_label)
            ax.text(0.05, 0.95, f"Curve fitting failed: {str(e)[:100]}", 
                   ha='left', va='top', transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                   fontsize=8, wrap=True)
            ax.set_xscale('log')
            ax.set_yscale('log')
            
            # Add enhanced grid and tick spacing for universal plots (error case)
            ax.grid(True, alpha=0.3, which='both')
            ax.grid(True, alpha=0.1, which='minor')
            
            # Better tick spacing for log plots
            from matplotlib.ticker import LogLocator, LogFormatter
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=8))
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs='auto', numticks=20))
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs='auto', numticks=20))
            
            cls.create_legend(ax)
            return False, None
    
    @classmethod  
    def fit_and_plot_curve(cls, ax, x, y, color, label, linestyle_idx=0, x_name="x", 
                          hide_from_legend=False, x_range=None):
        """Fit power law and plot curve with full equation and enhanced options"""
        if len(x) < 3:
            return
            
        try:
            def model_fn(x_val, A, b): 
                return A * x_val**b
            popt, _ = curve_fit(model_fn, x, y, maxfev=10000)
            
            # Calculate R²
            y_pred = model_fn(x, *popt)
            r_squared = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
            
            # Use custom range if provided, otherwise use data range
            if x_range and len(x_range) == 2 and x_range[0] is not None and x_range[1] is not None:
                x_min, x_max = x_range
            else:
                x_min, x_max = min(x), max(x)
            
            # Respect manual range if provided, otherwise use data range with minimal extension
            if x_range and len(x_range) == 2 and x_range[0] is not None and x_range[1] is not None:
                x_fit_min, x_fit_max = x_range
            else:
                # Only minimal extension for automatic range
                x_range_extended = (x_max - x_min) * 0.05  # Reduced to 5% extension
                x_fit_min = x_min - x_range_extended
                x_fit_max = x_max + x_range_extended
            
            # Use more points for large ranges (especially on log scales)
            range_ratio = x_fit_max / x_fit_min if x_fit_min > 0 else 1000
            if range_ratio > 100:  # Large range, likely for log scale
                num_points = 1000  # Extra points for large ranges
            else:
                num_points = 500
            
            x_fit = np.linspace(x_fit_min, x_fit_max, num_points)
            y_fit = model_fn(x_fit, *popt)
            
            linestyle = cls.LINESTYLES[linestyle_idx % len(cls.LINESTYLES)]
            
            # Create label based on legend preference
            if hide_from_legend:
                fit_label = None  # No label = no legend entry
            else:
                fit_label = f"{label}: $d/D_t = {popt[0]:.1f} \\cdot {x_name}^{{{popt[1]:.1f}}}$, $R^2 = {r_squared:.2f}$"
            
            cls.plot_line_enhanced(ax, x_fit, y_fit, color, linestyle, fit_label)
        except Exception:
            pass
    
    @classmethod
    def create_legend(cls, ax, force_two_column=False, **kwargs):
        """Create enhanced legend with better visibility"""
        handles, labels = ax.get_legend_handles_labels()
        num_items = len(handles)
        
        # Enhanced positioning logic
        if force_two_column and num_items >= 4:
            loc = 'lower right'
            ncol = 2
            fontsize = cls.FONT_SIZES['legend']
        elif num_items <= 3:
            loc = 'lower right'
            ncol = 1
            fontsize = cls.FONT_SIZES['legend']
        elif num_items <= 6:
            loc = 'lower right'
            ncol = 2 if num_items > 4 else 1
            fontsize = cls.FONT_SIZES['legend']
        else:
            loc = 'lower right'
            ncol = 2 if num_items > 8 else 1
            fontsize = cls.FONT_SIZES['legend']
        
        default_props = {
            'fontsize': fontsize,
            'frameon': True,
            'facecolor': 'white',
            'edgecolor': 'black',
            'loc': loc,
            'ncol': ncol,
            'markerscale': cls.MARKER_SIZES['legend'],  # Enhanced marker scale
            'columnspacing': 0.4,
            'handletextpad': 0.3,
            'borderpad': 0.4,
            'handlelength': 1.5,
            'shadow': True,  # Add shadow for better visibility
            'framealpha': 0.95  # Slight transparency
        }
        default_props.update(kwargs)
        return ax.legend(**default_props)
    
    @classmethod
    def get_fluid_color(cls, viscosity_cst):
        """Get consistent color for fluid viscosity"""
        return cls.FLUID_COLORS.get(viscosity_cst, '#666666')  # Default gray if not found
    
    @classmethod
    def format_axis(cls, ax, xlabel=None, ylabel=None, title=None):
        """Format axis with consistent styling"""
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=cls.FONT_SIZES['label'])
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=cls.FONT_SIZES['label'])
        if title:
            ax.set_title(title, fontsize=cls.FONT_SIZES['title'])

    @classmethod
    def get_color_marker_iterator(cls, data_groups):
        """Generate color-marker pairs for data groups"""
        colors = ['#D32F2F', '#000000', '#1976D2', '#4CAF50', '#FF9800', '#9C27B0', '#607D8B']
        markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
        
        color_cycle = itertools.cycle(colors)
        marker_cycle = itertools.cycle(markers)
        
        return [(next(color_cycle), next(marker_cycle)) for _ in data_groups]

    @classmethod
    def create_universal_3_parameter_plot(cls, ax, df_fit, x_cols, y_col, title_prefix="", yin_data=None, sun_data=None, include_external_plot=True, include_external_fit=False, fluid_plot_include=None, fluid_fit_include=None, show_air_injection=True, normalization_func=None, external_norm_func=None, x_scale='log', y_scale='log'):
        """Create universal 3-parameter scaling plot: d/D = A·Re^a·We^b·Ca^c"""
        
        # Apply normalization to get proper y-values and labels
        if normalization_func is not None:
            # Get a representative sample for getting the y-label
            sample_internal = df_fit.head(1) if not df_fit.empty else None
            if sample_internal is not None:
                _, y_label, air_injection_ref = normalization_func(sample_internal)
            else:
                y_label = r"$d_{30} / D_t$"
                air_injection_ref = 1/6
        else:
            y_label = r"$d_{30} / D_t$"
            air_injection_ref = 1/6

        # Combine internal and external data for fitting
        all_fit_data = []
        
        # Add internal data for fitting (respecting fluid inclusion options)
        if fluid_fit_include:
            for fluid, include_in_fit in fluid_fit_include.items():
                if include_in_fit:
                    fluid_data = df_fit[df_fit['Viscosity_cSt'] == fluid].copy()
                    if not fluid_data.empty:
                        # Apply normalization
                        if normalization_func is not None:
                            y_vals, _, _ = normalization_func(fluid_data)
                            fluid_data['NormDiameter'] = y_vals
                        else:
                            fluid_data['NormDiameter'] = fluid_data[y_col] * 1e-6 / fluid_data['ThroatDiameter_m']
                        
                        # Keep only essential columns for consistent concatenation
                        fluid_fit = fluid_data[['Reynolds', 'We_D', 'Ca', 'NormDiameter']].copy()
                        fluid_fit['source'] = f'Internal_{fluid}'
                        all_fit_data.append(fluid_fit)
        
        # Add external data for fitting if requested
        if include_external_fit:
            if yin_data is not None and include_external_plot:
                yin_copy = yin_data.copy()
                if external_norm_func is not None:
                    yin_y_vals = external_norm_func(yin_copy)
                    yin_copy['NormDiameter'] = yin_y_vals
                else:
                    yin_copy['NormDiameter'] = yin_copy['D_v'] / yin_copy['ThroatDiameter_m']
                
                # Standardize column names for concatenation
                if 'Re_t' in yin_copy.columns:
                    yin_copy['Reynolds'] = yin_copy['Re_t']
                elif 'Re' in yin_copy.columns:
                    yin_copy['Reynolds'] = yin_copy['Re']
                
                if 'We' in yin_copy.columns and 'We_D' not in yin_copy.columns:
                    yin_copy['We_D'] = yin_copy['We']
                
                # Ensure only necessary columns are kept to avoid conflicts
                yin_fit = yin_copy[['Reynolds', 'We_D', 'Ca', 'NormDiameter']].copy()
                yin_fit['source'] = 'Yin'
                all_fit_data.append(yin_fit)
                
            if sun_data is not None and include_external_plot:
                sun_copy = sun_data.copy()
                if external_norm_func is not None:
                    sun_y_vals = external_norm_func(sun_copy)
                    sun_copy['NormDiameter'] = sun_y_vals
                else:
                    sun_copy['NormDiameter'] = sun_copy['D_v'] / sun_copy['ThroatDiameter_m']
                
                # Standardize column names for concatenation
                if 'Re' in sun_copy.columns and 'Reynolds' not in sun_copy.columns:
                    sun_copy['Reynolds'] = sun_copy['Re']
                
                if 'We' in sun_copy.columns and 'We_D' not in sun_copy.columns:
                    sun_copy['We_D'] = sun_copy['We']
                
                # Ensure only necessary columns are kept to avoid conflicts
                sun_fit = sun_copy[['Reynolds', 'We_D', 'Ca', 'NormDiameter']].copy()
                sun_fit['source'] = 'Sun'
                all_fit_data.append(sun_fit)
        
        # Combine fit data
        if all_fit_data:
            df_combined_fit = pd.concat(all_fit_data, ignore_index=True)
            df_combined_fit = df_combined_fit.dropna(subset=['NormDiameter', 'Reynolds', 'We_D', 'Ca'])
        else:
            df_combined_fit = pd.DataFrame()

        # Fit the 3-parameter model
        fit_success = False
        A, a, b, c = 1e-3, -0.5, -0.3, -0.2  # Default values
        fit_text = "No data for fitting"
        
        if not df_combined_fit.empty and len(df_combined_fit) >= 4:  # Need at least 4 points for 4 parameters
            try:
                # Use standardized column names (now all data should have these)
                reynolds_col = 'Reynolds'
                weber_col = 'We_D' 
                capillary_col = 'Ca'
                
                # Get the data and ensure it's clean
                fit_data_clean = df_combined_fit[['Reynolds', 'We_D', 'Ca', 'NormDiameter']].dropna()
                
                if len(fit_data_clean) < 4:
                    raise ValueError(f"Insufficient clean data points: {len(fit_data_clean)}")
                
                # Extract arrays and check for valid values
                reynolds_vals = fit_data_clean['Reynolds'].values
                weber_vals = fit_data_clean['We_D'].values
                capillary_vals = fit_data_clean['Ca'].values
                y_vals = fit_data_clean['NormDiameter'].values
                
                # Check for invalid values
                if not (np.all(np.isfinite(reynolds_vals)) and np.all(reynolds_vals > 0)):
                    raise ValueError("Invalid Reynolds values")
                if not (np.all(np.isfinite(weber_vals)) and np.all(weber_vals > 0)):
                    raise ValueError("Invalid Weber values")
                if not (np.all(np.isfinite(capillary_vals)) and np.all(capillary_vals > 0)):
                    raise ValueError("Invalid Capillary values")
                if not (np.all(np.isfinite(y_vals)) and np.all(y_vals > 0)):
                    raise ValueError("Invalid y values")
                
                def model_fn(X, A, a, b, c):
                    Re, We, Ca = X
                    with np.errstate(invalid='raise', over='raise'):
                        return A * np.power(Re, a) * np.power(We, b) * np.power(Ca, c)
                
                # Prepare data for fitting
                xdata = np.array([reynolds_vals, weber_vals, capillary_vals])
                ydata = y_vals
                
                popt, pcov = curve_fit(model_fn, xdata, ydata, p0=[1e-3, -0.5, -0.3, -0.2], 
                                     maxfev=20000, bounds=([-np.inf, -5, -5, -5], [np.inf, 5, 5, 5]))
                A, a, b, c = popt
                
                # Calculate R-squared
                y_pred = model_fn(xdata, *popt)
                ss_res = np.sum((ydata - y_pred) ** 2)
                ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                
                # Calculate parameter uncertainties
                param_errors = np.sqrt(np.diag(pcov))
                
                fit_success = True
                fit_text = f"3-param fit: A={A:.2e}±{param_errors[0]:.1e}, a={a:.2f}±{param_errors[1]:.2f}, b={b:.2f}±{param_errors[2]:.2f}, c={c:.2f}±{param_errors[3]:.2f}, R²={r_squared:.2f}"
                
            except Exception as e:
                # Fallback to 2-parameter fit (Re-Ca)
                try:
                    fit_data_2param = df_combined_fit[['Reynolds', 'Ca', 'NormDiameter']].dropna()
                    
                    if len(fit_data_2param) < 3:
                        raise ValueError(f"Insufficient data for 2-param fit: {len(fit_data_2param)}")
                    
                    reynolds_2p = fit_data_2param['Reynolds'].values
                    capillary_2p = fit_data_2param['Ca'].values
                    y_2p = fit_data_2param['NormDiameter'].values
                    
                    def model_fn_2param(X, A, a, c):
                        Re, Ca = X
                        with np.errstate(invalid='raise', over='raise'):
                            return A * np.power(Re, a) * np.power(Ca, c)
                    
                    xdata_2param = np.array([reynolds_2p, capillary_2p])
                    
                    popt, pcov = curve_fit(model_fn_2param, xdata_2param, y_2p, p0=[1e-3, -0.5, -0.2], 
                                         maxfev=20000, bounds=([-np.inf, -5, -5], [np.inf, 5, 5]))
                    A, a, c = popt
                    b = 0  # No Weber dependence
                    
                    y_pred = model_fn_2param(xdata_2param, *popt)
                    ss_res = np.sum((y_2p - y_pred) ** 2)
                    ss_tot = np.sum((y_2p - np.mean(y_2p)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    
                    param_errors = np.sqrt(np.diag(pcov))
                    fit_success = True
                    fit_text = f"Fallback 2-param: A={A:.2e}±{param_errors[0]:.1e}, a={a:.2f}±{param_errors[1]:.2f}, c={c:.2f}±{param_errors[2]:.2f} (b=0), R²={r_squared:.2f}"
                    
                except Exception as e2:
                    fit_text = f"Fit failed: {str(e)[:50]}... Fallback also failed: {str(e2)[:30]}..."

        # Now plot all data (regardless of fit inclusion)
        # Internal data
        if fluid_plot_include:
            for fluid, include_in_plot in fluid_plot_include.items():
                if include_in_plot:
                    fluid_data = df_fit[df_fit['Viscosity_cSt'] == fluid].copy()
                    if not fluid_data.empty:
                        # Apply normalization
                        if normalization_func is not None:
                            y_vals, _, _ = normalization_func(fluid_data)
                        else:
                            y_vals = fluid_data[y_col] * 1e-6 / fluid_data['ThroatDiameter_m']
                        
                        # Compute collapsed x-axis values safely
                        try:
                            reynolds = fluid_data['Reynolds'].values
                            weber = fluid_data['We_D'].values
                            capillary = fluid_data['Ca'].values
                            
                            # Check for valid values
                            if np.any(reynolds <= 0) or np.any(weber <= 0) or np.any(capillary <= 0):
                                continue  # Skip invalid data
                            
                            x_vals = np.power(reynolds, a) * np.power(weber, b) * np.power(capillary, c)
                            
                            if not np.all(np.isfinite(x_vals)):
                                continue  # Skip if computation failed
                        except:
                            continue  # Skip on any error
                        
                        # Get color and marker
                        color = {'10 cSt': '#D32F2F', '50 cSt': '#000000', '20 cSt': '#1976D2'}.get(fluid, '#4CAF50')
                        marker = {'10 cSt': 'o', '50 cSt': 's', '20 cSt': '^'}.get(fluid, 'D')
                        
                        cls.plot_scatter_enhanced(ax, x_vals, y_vals, color, marker, f"{fluid}", alpha=0.7, size=40)

        # External data
        if include_external_plot:
            if yin_data is not None:
                # Apply normalization to external data
                if external_norm_func is not None:
                    yin_y_vals = external_norm_func(yin_data)
                else:
                    yin_y_vals = yin_data['D_v'] / yin_data['ThroatDiameter_m']
                
                # Use appropriate column names for external data and compute safely
                try:
                    yin_reynolds = yin_data['Re_t'].values if 'Re_t' in yin_data.columns else yin_data['Re'].values
                    yin_weber = yin_data['We'].values
                    yin_capillary = yin_data['Ca'].values
                    
                    if np.any(yin_reynolds <= 0) or np.any(yin_weber <= 0) or np.any(yin_capillary <= 0):
                        yin_x_vals = None  # Skip invalid data
                    else:
                        yin_x_vals = np.power(yin_reynolds, a) * np.power(yin_weber, b) * np.power(yin_capillary, c)
                        if not np.all(np.isfinite(yin_x_vals)):
                            yin_x_vals = None
                except:
                    yin_x_vals = None
                
                if yin_x_vals is not None:
                    cls.plot_scatter_enhanced(ax, yin_x_vals, yin_y_vals, '#4CAF50', 's', "Yin et al. 2015", alpha=0.7, size=40)
            
            if sun_data is not None:
                # Apply normalization to external data
                if external_norm_func is not None:
                    sun_y_vals = external_norm_func(sun_data)
                else:
                    sun_y_vals = sun_data['D_v'] / sun_data['ThroatDiameter_m']
                
                try:
                    sun_reynolds = sun_data['Re'].values
                    sun_weber = sun_data['We'].values
                    sun_capillary = sun_data['Ca'].values
                    
                    if np.any(sun_reynolds <= 0) or np.any(sun_weber <= 0) or np.any(sun_capillary <= 0):
                        sun_x_vals = None
                    else:
                        sun_x_vals = np.power(sun_reynolds, a) * np.power(sun_weber, b) * np.power(sun_capillary, c)
                        if not np.all(np.isfinite(sun_x_vals)):
                            sun_x_vals = None
                except:
                    sun_x_vals = None
                
                if sun_x_vals is not None:
                    cls.plot_scatter_enhanced(ax, sun_x_vals, sun_y_vals, '#FF9800', 's', "Sun et al. 2017", alpha=0.7, size=40)

        # Plot best-fit curve if successful
        if fit_success and not df_combined_fit.empty:
            all_x_vals = []
            
            # Collect all x-values for range
            if fluid_plot_include:
                for fluid, include_in_plot in fluid_plot_include.items():
                    if include_in_plot:
                        fluid_data = df_fit[df_fit['Viscosity_cSt'] == fluid]
                        if not fluid_data.empty:
                            try:
                                reynolds = fluid_data['Reynolds'].values
                                weber = fluid_data['We_D'].values
                                capillary = fluid_data['Ca'].values
                                
                                if not (np.any(reynolds <= 0) or np.any(weber <= 0) or np.any(capillary <= 0)):
                                    x_vals = np.power(reynolds, a) * np.power(weber, b) * np.power(capillary, c)
                                    if np.all(np.isfinite(x_vals)):
                                        all_x_vals.extend(x_vals)
                            except:
                                continue
            
            if include_external_plot:
                if yin_data is not None:
                    try:
                        yin_reynolds = yin_data['Re_t'].values if 'Re_t' in yin_data.columns else yin_data['Re'].values
                        yin_weber = yin_data['We'].values
                        yin_capillary = yin_data['Ca'].values
                        
                        if not (np.any(yin_reynolds <= 0) or np.any(yin_weber <= 0) or np.any(yin_capillary <= 0)):
                            yin_x_vals = np.power(yin_reynolds, a) * np.power(yin_weber, b) * np.power(yin_capillary, c)
                            if np.all(np.isfinite(yin_x_vals)):
                                all_x_vals.extend(yin_x_vals)
                    except:
                        pass
                        
                if sun_data is not None:
                    try:
                        sun_reynolds = sun_data['Re'].values
                        sun_weber = sun_data['We'].values
                        sun_capillary = sun_data['Ca'].values
                        
                        if not (np.any(sun_reynolds <= 0) or np.any(sun_weber <= 0) or np.any(sun_capillary <= 0)):
                            sun_x_vals = np.power(sun_reynolds, a) * np.power(sun_weber, b) * np.power(sun_capillary, c)
                            if np.all(np.isfinite(sun_x_vals)):
                                all_x_vals.extend(sun_x_vals)
                    except:
                        pass
            
            # Fit line is now created in the later section with proper LaTeX formatting

        # Reference line for air injection diameter
        if show_air_injection:
            ax.axhline(air_injection_ref, color='gray', linestyle=':', linewidth=1, alpha=0.7, 
                      label='Air Injection Diameter')

        # Formatting
        cls.setup_plot_style(ax, title=f"{title_prefix}Universal 3-Parameter Collapsed Scaling",
                           xlabel=r"$Re^a \cdot We^b \cdot Ca^c$", ylabel=y_label, grid=True)
        ax.set_xscale(x_scale.lower())
        ax.set_yscale(y_scale.lower())
        
        # Add fit results to plot as a fit line in legend instead of text box
        if fit_success:
            try:
                # Calculate collapsed x values for all data points
                reynolds_vals = df_combined_fit['Reynolds'].values
                weber_vals = df_combined_fit['We_D'].values 
                capillary_vals = df_combined_fit['Ca'].values
                
                # Calculate the actual collapsed x-axis values from the data
                collapsed_x_data = np.power(reynolds_vals, a) * np.power(weber_vals, b) * np.power(capillary_vals, c)
                
                # Use the actual range of collapsed x values, not the individual parameter ranges
                x_min = np.nanmin(collapsed_x_data)
                x_max = np.nanmax(collapsed_x_data)
                
                # Create fit line over the actual data range
                x_fit = np.logspace(np.log10(x_min), np.log10(x_max), 100)
                y_fit = A * x_fit
                
                # Create fit equation for legend with proper LaTeX formatting
                fit_equation = f"$d/D_t = {A:.1e} \cdot Re^{{{a:.1f}}} \cdot We^{{{b:.1f}}} \cdot Ca^{{{c:.1f}}}, R^2 = {r_squared:.2f}$"
                ax.plot(x_fit, y_fit, 'k--', linewidth=2, alpha=0.8, label=fit_equation)
            except:
                pass  # Skip if fit line creation fails
        
        cls.create_legend(ax)

class ExportManager:
    """Handles consistent export functionality"""
    
    @staticmethod
    def get_publication_figure_size(format_type: str, plot_type: str) -> Tuple[float, float]:
        """Get figure size based on publication format and plot type"""
        is_subplot = plot_type in ['Repeatability', 'Flow Rate', 'Temperature', 'Angle']
        
        if format_type == "Column Width":
            return (4.5, 5.0) if is_subplot else (3.8, 3.2)
        else:  # Full Width
            return (8.5, 6.5) if is_subplot else (7.5, 5.0)
    
    @staticmethod
    def setup_publication_style(format_type: str):
        """Configure matplotlib for publication-quality plots"""
        if format_type == "Column Width":
            plt.rcParams.update({
                'font.size': 9,
                'axes.titlesize': 10,
                'axes.labelsize': 9,
                'xtick.labelsize': 8,
                'ytick.labelsize': 8,
                'legend.fontsize': 7,
                'lines.linewidth': 1.2,
                'lines.markersize': 3, 
                'axes.linewidth': 1.0,
            })
        else:  # Full Width
            plt.rcParams.update({
                'font.size': 11,
                'axes.titlesize': 13,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 9,
                'lines.linewidth': 1.4,
                'lines.markersize': 6,
                'axes.linewidth': 1.2,
            })
    
    @staticmethod
    def prepare_figure_for_export(fig, format_type: str, plot_type: str):
        """Prepare figure with export settings and return original size for restoration"""
        current_size = fig.get_size_inches()
        pub_size = ExportManager.get_publication_figure_size(format_type, plot_type)
        ExportManager.setup_publication_style(format_type)
        fig.set_size_inches(pub_size)
        
        # Layout adjustments by plot type (same as export_figure)
        if plot_type == 'Repeatability':
            fig.tight_layout(pad=0.3)
            fig.subplots_adjust(hspace=0.3, wspace=0.3)
            # Reapply aspect ratios for repeatability plots after layout adjustment
            for ax in fig.get_axes():
                ax.set_aspect('equal', adjustable='box')
            

        elif plot_type in ['Flow Rate', 'Temperature', 'Angle']:
            fig.tight_layout(pad=0.4)
            fig.subplots_adjust(hspace=0.45)
        else:
            fig.tight_layout(pad=0.4)
        
        return current_size
    
    @staticmethod
    def export_figure(fig, filename: str, format_type: str, plot_type: str, dpi: int = 500):
        """Export figure with consistent settings"""
        try:
            current_size = ExportManager.prepare_figure_for_export(fig, format_type, plot_type)
            
            # Export based on file extension
            if filename.lower().endswith('.pdf'):
                fig.savefig(filename, bbox_inches='tight', facecolor='white', 
                           edgecolor='none', format='pdf')
            else:
                fig.savefig(filename, dpi=dpi, bbox_inches='tight', 
                           facecolor='white', edgecolor='none', format='png',
                           pil_kwargs={'optimize': True})
            
            # Reset to original state
            fig.set_size_inches(current_size)
            plt.rcParams.update(plt.rcParamsDefault)
            
            return True
        except Exception as e:
            plt.rcParams.update(plt.rcParamsDefault)
            raise e


class MultiviscosityAnalyzer:
    """Main GUI application with improved architecture"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Multiviscosity Data Processing - Enhanced GUI")
        self.root.geometry("1600x1000")
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.plotting_manager = PlottingManager()
        self.export_manager = ExportManager()
        
        # Data storage
        self.df = None
        self.filtered_df = None
        
        # GUI setup
        self.setup_gui_fonts()
        self.create_widgets()
    
    def setup_gui_fonts(self):
        """Configure larger fonts for better readability"""
        default_font = ('Segoe UI', 11)
        bold_font = ('Segoe UI', 11, 'bold')
        large_font = ('Segoe UI', 12)
        
        # Configure ttk styles
        style = ttk.Style()
        style.configure('TLabel', font=default_font)
        style.configure('TButton', font=default_font)
        style.configure('TCheckbutton', font=default_font)
        style.configure('TRadiobutton', font=default_font)
        style.configure('TCombobox', font=default_font)
        style.configure('TEntry', font=default_font)
        style.configure('TLabelframe.Label', font=bold_font)
        
        # Configure tk widget default fonts
        self.root.option_add('*Font', default_font)
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel with scrollable controls (fixed width: 400px)
        control_panel = ttk.Frame(main_frame, width=400)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_panel.pack_propagate(False)  # Maintain fixed width
        
        # Create scrollable frame for controls
        canvas = tk.Canvas(control_panel, highlightthickness=0)
        scrollbar = ttk.Scrollbar(control_panel, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollable components
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Directory selection
        dir_frame = ttk.LabelFrame(self.scrollable_frame, text="Data Directory", padding=10)
        dir_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.dir_var = tk.StringVar(value=r"G:\My Drive\Master's Data Processing\Both Viscosities")
        ttk.Entry(dir_frame, textvariable=self.dir_var, width=35).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(dir_frame, text="Browse", command=self.browse_directory).pack(fill=tk.X)
        
        # Load button
        ttk.Button(self.scrollable_frame, text="Load Data", command=self.load_data).pack(fill=tk.X, pady=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(self.scrollable_frame, textvariable=self.status_var, foreground="blue")
        status_label.pack(pady=5)
        
        # Filter section
        self.filter_frame = ttk.LabelFrame(self.scrollable_frame, text="Data Filters", padding=10)
        self.filter_frame.pack(fill=tk.X, pady=10)
        self.filter_widgets = {}
        
        # Plotting section
        plot_frame = ttk.LabelFrame(self.scrollable_frame, text="Analysis Type", padding=10)
        plot_frame.pack(fill=tk.X, pady=10)
        
        self.plot_type = tk.StringVar(value="Repeatability")
        plot_types = [
            "Repeatability", "Flow Rate", "Temperature", "Angle",
            "Reynolds", "Weber", "Capillary", "Universal: ReWe", 
            "PDFs Fixed Flow", "Universal: ReCa", "Universal: WeCa", "Universal: 3-Parameter"
        ]
        
        for plot_type in plot_types:
            ttk.Radiobutton(plot_frame, text=plot_type, variable=self.plot_type, 
                           value=plot_type, command=self.update_plot_options).pack(anchor=tk.W, pady=2)
        
        # Plot options frame
        self.options_frame = ttk.LabelFrame(self.scrollable_frame, text="Plot Options", padding=10)
        self.options_frame.pack(fill=tk.X, pady=10)
        
        # Action buttons at bottom (always visible)
        button_frame = ttk.Frame(self.scrollable_frame)
        button_frame.pack(fill=tk.X, pady=20, side=tk.BOTTOM)
        
        ttk.Button(button_frame, text="Generate Plot", command=self.generate_plot).pack(fill=tk.X, pady=(0, 5))
        
        # Export options
        export_frame = ttk.LabelFrame(button_frame, text="Export Options", padding=5)
        export_frame.pack(fill=tk.X, pady=(5,5))
        
        # Figure format options
        format_frame = ttk.Frame(export_frame)
        format_frame.pack(fill=tk.X)
        ttk.Label(format_frame, text="Format:").pack(side=tk.LEFT)
        self.fig_format_var = tk.StringVar(value="Column Width")
        ttk.Radiobutton(format_frame, text="Column", variable=self.fig_format_var, value="Column Width").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(format_frame, text="Full Width", variable=self.fig_format_var, value="Full Width").pack(side=tk.LEFT)
        
        # Export buttons
        export_button_frame = ttk.Frame(export_frame)
        export_button_frame.pack(fill=tk.X, pady=(5,0))
        ttk.Button(export_button_frame, text="Export PNG (500 DPI)", command=self.export_png).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,1))
        ttk.Button(export_button_frame, text="Copy to Clipboard", command=self.copy_to_clipboard).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(1,1))
        ttk.Button(export_button_frame, text="Export PDF", command=self.export_pdf).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(1,0))
        
        ttk.Button(button_frame, text="Export Data", command=self.export_data).pack(fill=tk.X, pady=(5,0))
        
        # Right panel for plots
        plot_panel = ttk.Frame(main_frame)
        plot_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.canvas = FigureCanvasTkAgg(self.fig, plot_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Navigation toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, plot_panel)
        toolbar.update()
        
    def browse_directory(self):
        directory = filedialog.askdirectory(initialdir=self.dir_var.get())
        if directory:
            self.dir_var.set(directory)
    
    def load_data(self):
        self.status_var.set("Loading data...")
        self.root.update_idletasks()
        
        try:
            # Load in separate thread to avoid freezing UI
            thread = threading.Thread(target=self._load_data_thread)
            thread.daemon = True
            thread.start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")
            self.status_var.set("Error loading data")
    
    def _load_data_thread(self):
        try:
            self.df = self.data_processor.load_experiment_data(self.dir_var.get())
            if self.df.empty:
                self.status_var.set("No data found")
                return
            
            # Create filter widgets on main thread
            self.root.after(0, self.create_filter_widgets)
            self.status_var.set(f"Loaded {len(self.df)} experiments")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
    
    def create_filter_widgets(self):
        # Clear existing filter widgets
        for widget in self.filter_frame.winfo_children():
            widget.destroy()
        
        if self.df is None or self.df.empty:
            return
            
        # Create filter checkboxes for each parameter
        params = {
            'Viscosity_cSt': 'Viscosity (cSt)',
            'VenturiAngle': 'Angle (°)',
            'Temp': 'Temperature (°F)',
            'AeratedFlow': 'Aeration (%)',
            'FlowRate': 'Flow Rate (GPM)'
        }
        
        self.filter_vars = {}
        
        for i, (param, label) in enumerate(params.items()):
            # Parameter label
            ttk.Label(self.filter_frame, text=label, font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))
            
            values = sorted(self.df[param].dropna().unique())
            self.filter_vars[param] = {}
            
            # Create frame for checkboxes with better wrapping
            check_frame = ttk.Frame(self.filter_frame)
            check_frame.pack(fill=tk.X, padx=10)
            
            # Arrange checkboxes in rows for better space usage
            col_count = 0
            row_frame = None
            max_cols = 4 if len(str(max(values))) < 5 else 3  # Adjust based on text length
            
            for value in values:
                if col_count == 0:
                    row_frame = ttk.Frame(check_frame)
                    row_frame.pack(fill=tk.X, pady=2)
                
                var = tk.BooleanVar(value=True)
                self.filter_vars[param][value] = var
                cb = ttk.Checkbutton(row_frame, text=str(value), variable=var, command=self.apply_filters)
                cb.pack(side=tk.LEFT, padx=(0, 15))
                
                col_count = (col_count + 1) % max_cols
        
        # Apply initial filter
        self.apply_filters()
    
    def apply_filters(self):
        if self.df is None or not self.filter_vars:
            return
            
        # Build filter conditions
        mask = pd.Series([True] * len(self.df), index=self.df.index)
        
        for param, var_dict in self.filter_vars.items():
            selected_values = [value for value, var in var_dict.items() if var.get()]
            if selected_values:
                mask &= self.df[param].isin(selected_values)
        
        self.filtered_df = self.df[mask].copy()
        self.status_var.set(f"Filtered to {len(self.filtered_df)} experiments")
    
    def update_plot_options(self):
        # Clear existing options
        for widget in self.options_frame.winfo_children():
            widget.destroy()
        
        plot_type = self.plot_type.get()
        
        # Add options based on plot type
        if plot_type in ["Reynolds", "Weber", "Capillary"]:
            # Normalization options for diameter plots
            ttk.Label(self.options_frame, text="Diameter Normalization:", font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W, pady=(0,2))
            self.normalization_var = tk.StringVar(value="Throat Diameter")
            
            norm_frame = ttk.Frame(self.options_frame)
            norm_frame.pack(fill=tk.X, padx=10)
            
            # Column 1
            col1_frame = ttk.Frame(norm_frame)
            col1_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            ttk.Radiobutton(col1_frame, text="Plain D_t", variable=self.normalization_var, value="Throat Diameter").pack(anchor=tk.W)
            ttk.Radiobutton(col1_frame, text="Weber1 = We*D_t", variable=self.normalization_var, value="Weber1").pack(anchor=tk.W)
            ttk.Radiobutton(col1_frame, text="Weber2 = sqrt(We)*D_t", variable=self.normalization_var, value="Weber2").pack(anchor=tk.W)
            ttk.Radiobutton(col1_frame, text="Capillary1 = Ca*D_t", variable=self.normalization_var, value="Capillary1").pack(anchor=tk.W)
            
            # Column 2  
            col2_frame = ttk.Frame(norm_frame)
            col2_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10,0))
            
            ttk.Radiobutton(col2_frame, text="Capillary2 = sqrt(Ca)*D_t", variable=self.normalization_var, value="Capillary2").pack(anchor=tk.W)
            ttk.Radiobutton(col2_frame, text="Reynolds1 = Re*D_t", variable=self.normalization_var, value="Reynolds1").pack(anchor=tk.W)
            ttk.Radiobutton(col2_frame, text="Reynolds2 = D_t/Re", variable=self.normalization_var, value="Reynolds2").pack(anchor=tk.W)
            
            self.ext_data_var = tk.StringVar(value="None")
            ttk.Label(self.options_frame, text="External Data:").pack(anchor=tk.W)
            for option in ["None", "Yin", "Sun", "Both"]:
                ttk.Radiobutton(self.options_frame, text=option, variable=self.ext_data_var, value=option).pack(anchor=tk.W)
            
            # Fit options
            ttk.Label(self.options_frame, text="Fit Options:").pack(anchor=tk.W, pady=(10,0))
            self.per_series_fit_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(self.options_frame, text="Per Series Fit", variable=self.per_series_fit_var).pack(anchor=tk.W)
            
            self.per_fluid_fit_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.options_frame, text="Per Fluid Fit", variable=self.per_fluid_fit_var).pack(anchor=tk.W)
            
            # Additional fit options
            self.flow_rate_fit_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.options_frame, text="Constant Flow Rate Lines", variable=self.flow_rate_fit_var).pack(anchor=tk.W)
            
            self.flow_rate_line_fit_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.options_frame, text="Flow Rate Line Fits", variable=self.flow_rate_line_fit_var).pack(anchor=tk.W)
            
            # Legend options
            self.hide_fits_from_legend_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.options_frame, text="Hide Fits from Legend", variable=self.hide_fits_from_legend_var).pack(anchor=tk.W)
            
            self.simplified_legend_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.options_frame, text="Simplified Legend (fluid colors only)", variable=self.simplified_legend_var).pack(anchor=tk.W)
            
            # Manual fit range options
            ttk.Label(self.options_frame, text="Manual Fit Range:", font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W, pady=(10,2))
            
            range_frame = ttk.Frame(self.options_frame)
            range_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(range_frame, text="X Min:").pack(side=tk.LEFT)
            self.fit_xmin_var = tk.StringVar(value="")
            ttk.Entry(range_frame, textvariable=self.fit_xmin_var, width=8).pack(side=tk.LEFT, padx=(5,10))
            
            ttk.Label(range_frame, text="X Max:").pack(side=tk.LEFT)
            self.fit_xmax_var = tk.StringVar(value="")
            ttk.Entry(range_frame, textvariable=self.fit_xmax_var, width=8).pack(side=tk.LEFT, padx=(5,0))
            
            ttk.Label(self.options_frame, text="(Leave empty for auto range)", font=('Segoe UI', 8)).pack(anchor=tk.W)
            
            # Air injection line option
            self.air_injection_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(self.options_frame, text="Show Air Injection Diameter", variable=self.air_injection_var).pack(anchor=tk.W, pady=(10,0))
            
            self.scale_var = tk.StringVar(value="Linear")
            ttk.Label(self.options_frame, text="Scale:").pack(anchor=tk.W)
            for scale in ["Linear", "Log"]:
                ttk.Radiobutton(self.options_frame, text=scale, variable=self.scale_var, value=scale).pack(anchor=tk.W)
        
        elif plot_type in ["Universal: ReWe", "Universal: ReCa", "Universal: WeCa", "Universal: 3-Parameter"]:
            # Normalization options for diameter plots
            ttk.Label(self.options_frame, text="Diameter Normalization:", font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W, pady=(0,2))
            self.normalization_var = tk.StringVar(value="Throat Diameter")
            
            norm_frame = ttk.Frame(self.options_frame)
            norm_frame.pack(fill=tk.X, padx=10)
            
            # Column 1
            col1_frame = ttk.Frame(norm_frame)
            col1_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            ttk.Radiobutton(col1_frame, text="Plain D_t", variable=self.normalization_var, value="Throat Diameter").pack(anchor=tk.W)
            ttk.Radiobutton(col1_frame, text="Weber1 = We*D_t", variable=self.normalization_var, value="Weber1").pack(anchor=tk.W)
            ttk.Radiobutton(col1_frame, text="Weber2 = sqrt(We)*D_t", variable=self.normalization_var, value="Weber2").pack(anchor=tk.W)
            ttk.Radiobutton(col1_frame, text="Capillary1 = Ca*D_t", variable=self.normalization_var, value="Capillary1").pack(anchor=tk.W)
            
            # Column 2  
            col2_frame = ttk.Frame(norm_frame)
            col2_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10,0))
            
            ttk.Radiobutton(col2_frame, text="Capillary2 = sqrt(Ca)*D_t", variable=self.normalization_var, value="Capillary2").pack(anchor=tk.W)
            ttk.Radiobutton(col2_frame, text="Reynolds1 = Re*D_t", variable=self.normalization_var, value="Reynolds1").pack(anchor=tk.W)
            ttk.Radiobutton(col2_frame, text="Reynolds2 = D_t/Re", variable=self.normalization_var, value="Reynolds2").pack(anchor=tk.W)
            
            # External data options for universal plots
            self.ext_data_plot_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(self.options_frame, text="Show External Data", variable=self.ext_data_plot_var).pack(anchor=tk.W)
            
            self.ext_data_fit_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.options_frame, text="Include External in Fit", variable=self.ext_data_fit_var).pack(anchor=tk.W)
            
            # Air injection line option
            self.air_injection_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(self.options_frame, text="Show Air Injection Diameter", variable=self.air_injection_var).pack(anchor=tk.W, pady=(10,0))
            
            # Scale options
            ttk.Label(self.options_frame, text="Axis Scales:", font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W, pady=(10,2))
            
            scale_frame = ttk.Frame(self.options_frame)
            scale_frame.pack(fill=tk.X, padx=10)
            
            # X-axis scale
            x_scale_frame = ttk.Frame(scale_frame)
            x_scale_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Label(x_scale_frame, text="X-axis:").pack(anchor=tk.W)
            self.x_scale_var = tk.StringVar(value="Log")
            ttk.Radiobutton(x_scale_frame, text="Linear", variable=self.x_scale_var, value="Linear").pack(anchor=tk.W)
            ttk.Radiobutton(x_scale_frame, text="Log", variable=self.x_scale_var, value="Log").pack(anchor=tk.W)
            
            # Y-axis scale  
            y_scale_frame = ttk.Frame(scale_frame)
            y_scale_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(20,0))
            ttk.Label(y_scale_frame, text="Y-axis:").pack(anchor=tk.W)
            self.y_scale_var = tk.StringVar(value="Log")
            ttk.Radiobutton(y_scale_frame, text="Linear", variable=self.y_scale_var, value="Linear").pack(anchor=tk.W)
            ttk.Radiobutton(y_scale_frame, text="Log", variable=self.y_scale_var, value="Log").pack(anchor=tk.W)
            
            # Fluid-specific inclusion options
            if self.filtered_df is not None and not self.filtered_df.empty:
                available_fluids = sorted(self.filtered_df['Viscosity_cSt'].unique())
                
                # Plot inclusion checkboxes
                ttk.Label(self.options_frame, text="Show in Plot:", font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W, pady=(10,2))
                self.fluid_plot_vars = {}
                for fluid in available_fluids:
                    var = tk.BooleanVar(value=True)
                    self.fluid_plot_vars[fluid] = var
                    ttk.Checkbutton(self.options_frame, text=f"{fluid} cSt", variable=var).pack(anchor=tk.W, padx=10)
                
                # Fit inclusion checkboxes
                ttk.Label(self.options_frame, text="Include in Fit:", font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W, pady=(10,2))
                self.fluid_fit_vars = {}
                for fluid in available_fluids:
                    var = tk.BooleanVar(value=True)
                    self.fluid_fit_vars[fluid] = var
                    ttk.Checkbutton(self.options_frame, text=f"{fluid} cSt", variable=var).pack(anchor=tk.W, padx=10)
        
        elif plot_type == "Universal: ReWe Fixed Exp":
            # Normalization options for diameter plots
            ttk.Label(self.options_frame, text="Diameter Normalization:", font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W, pady=(0,2))
            self.normalization_var = tk.StringVar(value="Throat Diameter")
            
            norm_frame = ttk.Frame(self.options_frame)
            norm_frame.pack(fill=tk.X, padx=10)
            
            # Column 1
            col1_frame = ttk.Frame(norm_frame)
            col1_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            ttk.Radiobutton(col1_frame, text="Plain D_t", variable=self.normalization_var, value="Throat Diameter").pack(anchor=tk.W)
            ttk.Radiobutton(col1_frame, text="Weber1 = We*D_t", variable=self.normalization_var, value="Weber1").pack(anchor=tk.W)
            ttk.Radiobutton(col1_frame, text="Weber2 = sqrt(We)*D_t", variable=self.normalization_var, value="Weber2").pack(anchor=tk.W)
            ttk.Radiobutton(col1_frame, text="Capillary1 = Ca*D_t", variable=self.normalization_var, value="Capillary1").pack(anchor=tk.W)
            
            # Column 2  
            col2_frame = ttk.Frame(norm_frame)
            col2_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10,0))
            
            ttk.Radiobutton(col2_frame, text="Capillary2 = sqrt(Ca)*D_t", variable=self.normalization_var, value="Capillary2").pack(anchor=tk.W)
            ttk.Radiobutton(col2_frame, text="Reynolds1 = Re*D_t", variable=self.normalization_var, value="Reynolds1").pack(anchor=tk.W)
            ttk.Radiobutton(col2_frame, text="Reynolds2 = D_t/Re", variable=self.normalization_var, value="Reynolds2").pack(anchor=tk.W)
            
            # Fixed exponent inputs
            ttk.Label(self.options_frame, text="Fixed Exponents:", font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W, pady=(10,2))
            
            exp_frame = ttk.Frame(self.options_frame)
            exp_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(exp_frame, text="Reynolds exp (a):").pack(side=tk.LEFT)
            self.reynolds_exp_var = tk.StringVar(value="0.5")
            ttk.Entry(exp_frame, textvariable=self.reynolds_exp_var, width=8).pack(side=tk.LEFT, padx=(5,10))
            
            ttk.Label(exp_frame, text="Weber exp (b):").pack(side=tk.LEFT)
            self.weber_exp_var = tk.StringVar(value="-0.6")
            ttk.Entry(exp_frame, textvariable=self.weber_exp_var, width=8).pack(side=tk.LEFT, padx=(5,0))
            
            ttk.Label(self.options_frame, text="Form: d/D_t = A × Re^a × We^b", font=('Segoe UI', 8)).pack(anchor=tk.W)
            
            # External data options
            self.ext_data_plot_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(self.options_frame, text="Show External Data", variable=self.ext_data_plot_var).pack(anchor=tk.W, pady=(10,0))
            
            self.ext_data_fit_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.options_frame, text="Include External in Fit", variable=self.ext_data_fit_var).pack(anchor=tk.W)
            
            # Air injection line option
            self.air_injection_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(self.options_frame, text="Show Air Injection Diameter", variable=self.air_injection_var).pack(anchor=tk.W, pady=(10,0))
        
        elif plot_type in ["Flow Rate", "Temperature", "Angle"]:
            # Air injection line option for these plot types
            self.air_injection_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(self.options_frame, text="Show Air Injection Diameter", variable=self.air_injection_var).pack(anchor=tk.W)
        
        elif plot_type == "PDFs Fixed Flow":
            if self.filtered_df is not None:
                flows = sorted(self.filtered_df['FlowRate'].unique())
                self.flow_var = tk.StringVar(value=flows[0] if flows else "")
                ttk.Label(self.options_frame, text="Flow Rate (GPM):").pack(anchor=tk.W)
                ttk.Combobox(self.options_frame, textvariable=self.flow_var, values=flows, state="readonly").pack(fill=tk.X)
        
        elif plot_type == "Weber Fixed Fit":
            # Normalization options for diameter plots
            ttk.Label(self.options_frame, text="Diameter Normalization:", font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W, pady=(0,2))
            self.normalization_var = tk.StringVar(value="Throat Diameter")
            
            norm_frame = ttk.Frame(self.options_frame)
            norm_frame.pack(fill=tk.X, padx=10)
            
            # Column 1
            col1_frame = ttk.Frame(norm_frame)
            col1_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            ttk.Radiobutton(col1_frame, text="Plain D_t", variable=self.normalization_var, value="Throat Diameter").pack(anchor=tk.W)
            ttk.Radiobutton(col1_frame, text="Weber1 = We*D_t", variable=self.normalization_var, value="Weber1").pack(anchor=tk.W)
            ttk.Radiobutton(col1_frame, text="Weber2 = sqrt(We)*D_t", variable=self.normalization_var, value="Weber2").pack(anchor=tk.W)
            ttk.Radiobutton(col1_frame, text="Capillary1 = Ca*D_t", variable=self.normalization_var, value="Capillary1").pack(anchor=tk.W)
            
            # Column 2  
            col2_frame = ttk.Frame(norm_frame)
            col2_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10,0))
            
            ttk.Radiobutton(col2_frame, text="Capillary2 = sqrt(Ca)*D_t", variable=self.normalization_var, value="Capillary2").pack(anchor=tk.W)
            ttk.Radiobutton(col2_frame, text="Reynolds1 = Re*D_t", variable=self.normalization_var, value="Reynolds1").pack(anchor=tk.W)
            ttk.Radiobutton(col2_frame, text="Reynolds2 = D_t/Re", variable=self.normalization_var, value="Reynolds2").pack(anchor=tk.W)
            
            # Fixed exponent input
            ttk.Label(self.options_frame, text="Fixed Weber Exponent (b):").pack(anchor=tk.W, pady=(10,0))
            self.weber_exponent_var = tk.StringVar(value="-0.6")
            ttk.Entry(self.options_frame, textvariable=self.weber_exponent_var, width=10).pack(anchor=tk.W, pady=2)
            
            # Scale options
            ttk.Label(self.options_frame, text="Scale:").pack(anchor=tk.W, pady=(10,0))
            self.scale_var = tk.StringVar(value="Log")
            for option in ["Linear", "Log"]:
                ttk.Radiobutton(self.options_frame, text=option, variable=self.scale_var, value=option).pack(anchor=tk.W)
            
            # External data options
            self.ext_data_var = tk.StringVar(value="None")
            ttk.Label(self.options_frame, text="External Data:").pack(anchor=tk.W, pady=(10,0))
            for option in ["None", "Yin", "Sun", "Both"]:
                ttk.Radiobutton(self.options_frame, text=option, variable=self.ext_data_var, value=option).pack(anchor=tk.W)

        elif plot_type == "Universal: 3-Parameter":
            # Normalization options for diameter plots
            ttk.Label(self.options_frame, text="Diameter Normalization:", font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W, pady=(0,2))
            self.normalization_var = tk.StringVar(value="Throat Diameter")
            
            norm_frame = ttk.Frame(self.options_frame)
            norm_frame.pack(fill=tk.X, padx=10)
            
            # Column 1
            col1_frame = ttk.Frame(norm_frame)
            col1_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            ttk.Radiobutton(col1_frame, text="Plain D_t", variable=self.normalization_var, value="Throat Diameter").pack(anchor=tk.W)
            ttk.Radiobutton(col1_frame, text="Weber1 = We*D_t", variable=self.normalization_var, value="Weber1").pack(anchor=tk.W)
            ttk.Radiobutton(col1_frame, text="Weber2 = sqrt(We)*D_t", variable=self.normalization_var, value="Weber2").pack(anchor=tk.W)
            ttk.Radiobutton(col1_frame, text="Capillary1 = Ca*D_t", variable=self.normalization_var, value="Capillary1").pack(anchor=tk.W)
            
            # Column 2  
            col2_frame = ttk.Frame(norm_frame)
            col2_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10,0))
            
            ttk.Radiobutton(col2_frame, text="Capillary2 = sqrt(Ca)*D_t", variable=self.normalization_var, value="Capillary2").pack(anchor=tk.W)
            ttk.Radiobutton(col2_frame, text="Reynolds1 = Re*D_t", variable=self.normalization_var, value="Reynolds1").pack(anchor=tk.W)
            ttk.Radiobutton(col2_frame, text="Reynolds2 = D_t/Re", variable=self.normalization_var, value="Reynolds2").pack(anchor=tk.W)
            
            # External data options for universal plots
            self.ext_data_plot_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(self.options_frame, text="Show External Data", variable=self.ext_data_plot_var).pack(anchor=tk.W, pady=(10,0))
            
            self.ext_data_fit_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.options_frame, text="Include External in Fit", variable=self.ext_data_fit_var).pack(anchor=tk.W)
            
            # Air injection line option
            self.air_injection_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(self.options_frame, text="Show Air Injection Diameter", variable=self.air_injection_var).pack(anchor=tk.W, pady=(10,0))
            
            # Fluid-specific inclusion options
            if self.filtered_df is not None and not self.filtered_df.empty:
                available_fluids = sorted(self.filtered_df['Viscosity_cSt'].unique())
                
                # Plot inclusion checkboxes
                ttk.Label(self.options_frame, text="Show in Plot:", font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W, pady=(10,2))
                self.fluid_plot_vars = {}
                for fluid in available_fluids:
                    var = tk.BooleanVar(value=True)
                    self.fluid_plot_vars[fluid] = var
                    ttk.Checkbutton(self.options_frame, text=f"{fluid} cSt", variable=var).pack(anchor=tk.W, padx=10)
                
                # Fit inclusion checkboxes
                ttk.Label(self.options_frame, text="Include in Fit:", font=('Segoe UI', 10, 'bold')).pack(anchor=tk.W, pady=(10,2))
                self.fluid_fit_vars = {}
                for fluid in available_fluids:
                    var = tk.BooleanVar(value=True)
                    self.fluid_fit_vars[fluid] = var
                    ttk.Checkbutton(self.options_frame, text=f"{fluid} cSt", variable=var).pack(anchor=tk.W, padx=10)
    
    def generate_plot(self):
        if self.filtered_df is None or self.filtered_df.empty:
            messagebox.showwarning("Warning", "No filtered data available. Please load and filter data first.")
            return
        
        try:
            self.fig.clear()
            plot_type = self.plot_type.get()
            
            if plot_type == "Repeatability":
                self.plot_repeatability()
            elif plot_type == "Flow Rate":
                self.plot_flow_rate()
            elif plot_type == "Temperature":
                self.plot_temperature()
            elif plot_type == "Angle":
                self.plot_angle()
            elif plot_type == "Reynolds":
                self.plot_reynolds()
            elif plot_type == "Weber":
                self.plot_weber()
            elif plot_type == "Capillary":
                self.plot_capillary()
            elif plot_type == "Universal: ReWe":
                self.plot_universal_rewe()
            elif plot_type == "PDFs Fixed Flow":
                self.plot_pdfs_fixed_flow()
            elif plot_type == "Universal: ReCa":
                self.plot_universal_reca()
            elif plot_type == "Universal: WeCa":
                self.plot_universal_weca()
            elif plot_type == "Universal: 3-Parameter":
                self.plot_universal_3_parameter()
            
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to generate plot: {e}")
    
    def plot_repeatability(self):
        var_labels = [r'$\mu_{LN}$', r'$\sigma_{LN}$', r'$d_{30}$']
        var_keys = ['LogMu', 'LogSigma', 'D_v']
        param_cols = ['Viscosity_cSt', 'Temp', 'FlowRate', 'VenturiAngle', 'AeratedFlow']
        
        self.filtered_df['param_id'] = self.filtered_df[param_cols].apply(tuple, axis=1)
        unique_params = self.filtered_df['param_id'].unique()
        t1_data, t2_data, colors = [], [], []
        
        for param in unique_params:
            subset = self.filtered_df[self.filtered_df['param_id'] == param]
            trial1 = subset[subset['Trial'] == 1]
            trial2 = subset[subset['Trial'] == 2]
            if len(trial1) == 1 and len(trial2) == 1:
                t1_data.append([trial1.iloc[0][key] for key in var_keys])
                t2_data.append([trial2.iloc[0][key] for key in var_keys])
                colors.append(self.plotting_manager.FLUID_COLORS.get(param[0], '#666666'))
        
        if not t1_data:
            self.ax.text(0.5, 0.5, "No paired trial data available", ha='center', va='center', transform=self.ax.transAxes)
            return
        
        t1_data, t2_data = np.array(t1_data), np.array(t2_data)
        axes = self.fig.subplots(1, 3)
        # Improve subplot spacing
        self.fig.subplots_adjust(wspace=0.3)
        
        for i in range(3):
            ax = axes[i]
            x, y = t1_data[:, i], t2_data[:, i]
            
            # Plot data points
            for xi, yi, ci in zip(x, y, colors):
                self.plotting_manager.plot_scatter_enhanced(ax, [xi], [yi], ci, 'x', '')
            
            # Styling and 1:1 line with square aspect ratio
            min_val, max_val = min(x.min(), y.min()), max(x.max(), y.max())
            margin = 0.05 * (max_val - min_val)
            self.plotting_manager.plot_line_enhanced(ax, [min_val - margin, max_val + margin], 
                                                   [min_val - margin, max_val + margin], 
                                                   'k', '--', '', linewidth=1.5)
            ax.set_xlim(min_val - margin, max_val + margin)
            ax.set_ylim(min_val - margin, max_val + margin)
            # Set equal aspect ratio to make grid squares
            ax.set_aspect('equal', adjustable='box')
            
            # Apply consistent styling
            self.plotting_manager.setup_plot_style(ax, xlabel=f'Trial 1 {var_labels[i]}',
                                                  ylabel=f'Trial 2 {var_labels[i]}')
            
            # Set consistent decimal formatting for repeatability plots
            from matplotlib.ticker import FormatStrFormatter
            if i < 2:  # First two plots (LogMu, LogSigma) - 1 decimal place
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            else:  # Third plot (D_v) - whole numbers
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            # Add subplot label instead of title
            ax.text(0.05, 0.95, f'{var_labels[i]}', transform=ax.transAxes, 
                   fontsize=self.plotting_manager.FONT_SIZES['text'], weight='bold', va='top')
            
            # R² calculation and display  
            r2 = np.corrcoef(x, y)[0, 1] ** 2
            ax.text(0.05, 0.85, f'$R^2$ = {r2:.2f}', transform=ax.transAxes, 
                   fontsize=self.plotting_manager.FONT_SIZES['text'], va='top')
        
        # Create unified legend
        legend_elements = []
        for visc, color in self.plotting_manager.FLUID_COLORS.items():
            if visc in [param[0] for param in unique_params]:
                legend_elements.append(plt.Line2D([0], [0], marker='x', color=color, 
                                                label=visc, markersize=8, linestyle='None', linewidth=1.5))
        
        if legend_elements:
            self.plotting_manager.create_legend(axes[0], handles=legend_elements)
        
        # Use tight_layout but preserve aspect ratios
        plt.tight_layout()
        
        # Reapply aspect ratios after tight_layout (which can reset them)
        for i in range(3):
            axes[i].set_aspect('equal', adjustable='box')
    
    def get_trial_averaged_data(self):
        """Get trial-averaged data using data processor"""
        return self.data_processor.get_trial_averaged_data(self.filtered_df)
    
    def get_normalized_diameter(self, df):
        """Get normalized diameter values and axis label based on selected normalization"""
        normalization = getattr(self, 'normalization_var', tk.StringVar(value="Throat Diameter")).get()
        
        if normalization == "Weber1":
            # Normalize by Weber1 = We*D_t
            y_values = df['D_v'] * 1e-6 / df['Weber1']
            y_label = r"$d_{30} / (We \cdot D_t)$"
            air_injection_ratio = 0.001 / df['Weber1'].mean()  # Approximate
            
        elif normalization == "Weber2":
            # Normalize by Weber2 = sqrt(We)*D_t
            y_values = df['D_v'] * 1e-6 / df['Weber2']
            y_label = r"$d_{30} / (\sqrt{We} \cdot D_t)$"
            air_injection_ratio = 0.001 / df['Weber2'].mean()  # Approximate
            
        elif normalization == "Capillary1":
            # Normalize by Capillary1 = Ca*D_t
            y_values = df['D_v'] * 1e-6 / df['Capillary1']
            y_label = r"$d_{30} / (Ca \cdot D_t)$"
            air_injection_ratio = 0.001 / df['Capillary1'].mean()  # Approximate
            
        elif normalization == "Capillary2":
            # Normalize by Capillary2 = sqrt(Ca)*D_t
            y_values = df['D_v'] * 1e-6 / df['Capillary2']
            y_label = r"$d_{30} / (\sqrt{Ca} \cdot D_t)$"
            air_injection_ratio = 0.001 / df['Capillary2'].mean()  # Approximate
            
        elif normalization == "Reynolds1":
            # Normalize by Reynolds1 = Re*D_t
            y_values = df['D_v'] * 1e-6 / df['Reynolds1']
            y_label = r"$d_{30} / (Re \cdot D_t)$"
            air_injection_ratio = 0.001 / df['Reynolds1'].mean()  # Approximate
            
        elif normalization == "Reynolds2":
            # Normalize by Reynolds2 = D_t/Re
            y_values = df['D_v'] * 1e-6 / df['Reynolds2']
            y_label = r"$d_{30} / (D_t / Re)$"
            air_injection_ratio = 0.001 / df['Reynolds2'].mean()  # Approximate
            
        else:
            # Default: normalize by throat diameter 
            y_values = df['D_v'] * 1e-6 / df['ThroatDiameter_m']
            y_label = r"$d_{30} / D_t$"
            air_injection_ratio = 0.001 / 0.006  # d_air/D_t ≈ 1/6
        
        return y_values, y_label, air_injection_ratio
    
    def get_normalized_external_data(self, ext_data):
        """Get normalized external data based on selected normalization"""
        # Note: External data D_v is already in meters, unlike internal data which is in micrometers
        normalization = getattr(self, 'normalization_var', tk.StringVar(value="Throat Diameter")).get()
        
        if normalization == "Weber1" and 'Weber1' in ext_data.columns:
            return ext_data['D_v'] / ext_data['Weber1']
        elif normalization == "Weber2" and 'Weber2' in ext_data.columns:
            return ext_data['D_v'] / ext_data['Weber2']
        elif normalization == "Capillary1" and 'Capillary1' in ext_data.columns:
            return ext_data['D_v'] / ext_data['Capillary1']
        elif normalization == "Capillary2" and 'Capillary2' in ext_data.columns:
            return ext_data['D_v'] / ext_data['Capillary2']
        elif normalization == "Reynolds1" and 'Reynolds1' in ext_data.columns:
            return ext_data['D_v'] / ext_data['Reynolds1']
        elif normalization == "Reynolds2" and 'Reynolds2' in ext_data.columns:
            return ext_data['D_v'] / ext_data['Reynolds2']
        else:
            # Fall back to throat diameter normalization
            # External data D_v already in meters, so direct division is correct
            return ext_data['D_v'] / ext_data['ThroatDiameter_m']
    
    def plot_flow_rate(self):
        avg_df = self.get_trial_averaged_data()
        if avg_df.empty:
            self.ax.text(0.5, 0.5, "No trial-averaged data available", ha='center', va='center', transform=self.ax.transAxes)
            return
        
        axes = self.fig.subplots(2, 1)
        self.fig.subplots_adjust(hspace=0.35)  # Better vertical spacing
        variables = [
            ('D_v', r'$d_{30}$ ($\mu$m)', (0, 400)),
            ('LogSigma', r'$\sigma_{LN}$', (0, 1))
        ]
        
        # Get unique combinations of viscosity and temperature
        visc_temp_groups = sorted(avg_df.groupby(['Viscosity_cSt', 'Temp']).groups.keys())
        color_marker_pairs = self.plotting_manager.get_color_marker_iterator(visc_temp_groups)
        
        for i, (col, ylabel, ylim) in enumerate(variables):
            ax = axes[i]
            
            for (visc, temp), (color, marker) in zip(visc_temp_groups, color_marker_pairs):
                # Override color for viscosity consistency
                color = self.plotting_manager.FLUID_COLORS.get(visc, color)
                
                subset = avg_df[(avg_df['Viscosity_cSt'] == visc) & (avg_df['Temp'] == temp)].sort_values('FlowRate')
                if not subset.empty:
                    label = f"{visc} base: {temp}°F"
                    ax.plot(subset['FlowRate'], subset[col],
                           linestyle='-', marker=marker, color=color, 
                           linewidth=self.plotting_manager.MARKER_SIZES['line_width'], 
                           markersize=5, markeredgewidth=1, 
                           markerfacecolor='none', label=label)
            
            # Apply consistent styling
            self.plotting_manager.setup_plot_style(ax, xlabel='Flow Rate (GPM)', ylabel=ylabel)
            ax.set_ylim(ylim)
            self.plotting_manager.create_legend(axes[0])
        
        plt.tight_layout()
    
    def plot_temperature(self):
        # Similar implementation as flow_rate but vs temperature
        avg_df = self.get_trial_averaged_data()
        if avg_df.empty:
            self.ax.text(0.5, 0.5, "No trial-averaged data available", ha='center', va='center', transform=self.ax.transAxes)
            return
        
        axes = self.fig.subplots(2, 1)
        self.fig.subplots_adjust(hspace=0.35)  # Better vertical spacing
        vis_flow_keys = sorted(avg_df.groupby(['Viscosity_cSt', 'FlowRate']).groups.keys())
        
        for k, (col, ylabel, ylim) in enumerate([
            ('D_v', r'$d_{30}$ ($\mu$m)', (0, 400)),
            ('LogSigma', r'$\sigma_{LN}$', (0, 1))
        ]):
            ax = axes[k]
            for idx, (visc, flow) in enumerate(vis_flow_keys):
                subset = avg_df[(avg_df['Viscosity_cSt'] == visc) & (avg_df['FlowRate'] == flow)].sort_values('Temp')
                if subset.empty:
                    continue
                
                color = self.plotting_manager.get_fluid_color(visc)
                marker = self.plotting_manager.MARKERS[idx % len(self.plotting_manager.MARKERS)]
                
                ax.plot(subset['Temp'], subset[col],
                       linestyle='-', linewidth=self.plotting_manager.MARKER_SIZES['line_width'],
                       marker=marker, markerfacecolor='none',
                       markeredgewidth=1.2, markersize=7,
                       color=color, label=f"{visc} – {flow} GPM")
            
            ax.set_xlabel('Temperature (°F)', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_ylim(ylim)
            ax.grid(True,alpha=0.3)
            ax.legend(frameon=True, facecolor='white', edgecolor='black', fontsize=9)
        
        plt.tight_layout()
    
    def plot_angle(self):
        # Angle vs bubble properties
        avg_df = self.get_trial_averaged_data()
        if avg_df.empty:
            self.ax.text(0.5, 0.5, "No trial-averaged data available", ha='center', va='center', transform=self.ax.transAxes)
            return
        
        axes = self.fig.subplots(2, 1)
        self.fig.subplots_adjust(hspace=0.35)  # Better vertical spacing
        groups = sorted(avg_df.groupby(['Viscosity_cSt', 'Temp']).groups.keys())
        
        for i, (col, ylabel, ylim) in enumerate([
            ('D_v', r'$d_{30}$ ($\mu$m)', (0, 400)),
            ('LogSigma', r'$\sigma_{LN}$', (0, 1))
        ]):
            ax = axes[i]
            for j, (visc, temp) in enumerate(groups):
                subset = avg_df[(avg_df['Viscosity_cSt'] == visc) & (avg_df['Temp'] == temp)].sort_values('VenturiAngle')
                if not subset.empty:
                    color = self.plotting_manager.get_fluid_color(visc)
                    marker = self.plotting_manager.MARKERS[j % len(self.plotting_manager.MARKERS)]
                    label = f"{visc} - {temp}°F"
                    ax.plot(subset['VenturiAngle'], subset[col], '-', marker=marker, 
                           label=label, color=color, linewidth=self.plotting_manager.MARKER_SIZES['line_width'],
                           markersize=7, markeredgewidth=1.2, markerfacecolor='none')
            
            self.plotting_manager.setup_plot_style(ax, xlabel='Venturi Angle (°)', ylabel=ylabel)
            ax.set_ylim(ylim)
            self.plotting_manager.create_legend(ax)
        
        plt.tight_layout()
    
    def plot_dimensionless_number(self, x_col, x_label, ext_x_col_mapping):
        """Generic method for plotting dimensionless number analyses"""
        avg_df = self.get_trial_averaged_data()
        if avg_df.empty:
            self.ax.text(0.5, 0.5, "No trial-averaged data available", ha='center', va='center', transform=self.ax.transAxes)
            return
        
        # Get options
        ext_data_opt = getattr(self, 'ext_data_var', tk.StringVar(value="None")).get()
        per_series_fit = getattr(self, 'per_series_fit_var', tk.BooleanVar(value=False)).get()
        per_fluid_fit = getattr(self, 'per_fluid_fit_var', tk.BooleanVar(value=False)).get()
        flow_rate_fit = getattr(self, 'flow_rate_fit_var', tk.BooleanVar(value=False)).get()
        flow_rate_line_fit = getattr(self, 'flow_rate_line_fit_var', tk.BooleanVar(value=False)).get()
        simplified_legend = getattr(self, 'simplified_legend_var', tk.BooleanVar(value=False)).get()
        scale_opt = getattr(self, 'scale_var', tk.StringVar(value="Linear")).get()
        hide_fits = getattr(self, 'hide_fits_from_legend_var', tk.BooleanVar(value=False)).get()
        
        # Get manual fit range if specified
        fit_range = None
        try:
            x_min_str = getattr(self, 'fit_xmin_var', tk.StringVar(value="")).get().strip()
            x_max_str = getattr(self, 'fit_xmax_var', tk.StringVar(value="")).get().strip()
            
            if x_min_str and x_max_str:
                x_min = float(x_min_str)
                x_max = float(x_max_str)
                if x_max > x_min:
                    fit_range = (x_min, x_max)
        except (ValueError, AttributeError):
            pass  # Use auto range if manual range is invalid
        
        # Prepare data groups
        groups = sorted(avg_df.groupby(['Viscosity_cSt', 'Temp']).groups.keys())
        color_marker_pairs = self.plotting_manager.get_color_marker_iterator(groups)
        
        ax = self.fig.add_subplot(111)
        
        # Per fluid fit data collection
        fluid_data = {}
        flow_rate_data = {}  # For flow rate lines
        
        # Track which viscosities have been labeled for simplified legend
        viscosity_labeled = set()
        
        for j, ((visc, temp), (color, marker)) in enumerate(zip(groups, color_marker_pairs)):
            # Override color for viscosity consistency
            color = self.plotting_manager.FLUID_COLORS.get(visc, color)
            
            subset = avg_df[(avg_df['Viscosity_cSt'] == visc) & (avg_df['Temp'] == temp)]
            subset = subset.dropna(subset=['D_v', x_col, 'ThroatDiameter_m'])
            
            if len(subset) == 0:
                continue
            
            x = subset[x_col]
            y, _, _ = self.get_normalized_diameter(subset)
            
            # Choose label based on simplified legend option
            if simplified_legend:
                if visc not in viscosity_labeled:
                    label = visc
                    viscosity_labeled.add(visc)
                else:
                    label = None  # Don't label subsequent points of same viscosity
            else:
                label = f"{visc} {temp}°F"
            
            self.plotting_manager.plot_scatter_enhanced(ax, x, y, color, marker, label)
            
            # Collect data for flow rate lines
            if flow_rate_fit or flow_rate_line_fit:
                for idx, (_, row) in enumerate(subset.iterrows()):
                    flow_rate = row['FlowRate']
                    if flow_rate not in flow_rate_data:
                        flow_rate_data[flow_rate] = {'x': [], 'y': []}
                    flow_rate_data[flow_rate]['x'].append(row[x_col])
                    flow_rate_data[flow_rate]['y'].append(y.iloc[idx] if hasattr(y, 'iloc') else y[idx])
            
            # Per series fit
            if per_series_fit:
                x_name = x_col.replace('Reynolds', 'Re').replace('We_D', 'We').replace('Ca', 'Ca')
                self.plotting_manager.fit_and_plot_curve(ax, x, y, color, label, j, x_name, 
                                                       hide_fits, fit_range)
            
            # Collect data for per fluid fit
            if per_fluid_fit:
                if visc not in fluid_data:
                    fluid_data[visc] = {'x': [], 'y': [], 'color': color}
                fluid_data[visc]['x'].extend(x.values)
                fluid_data[visc]['y'].extend(y.values)
        
        # Per fluid fits
        if per_fluid_fit:
            x_name = x_col.replace('Reynolds', 'Re').replace('We_D', 'We').replace('Ca', 'Ca')
            for visc, data in fluid_data.items():
                if len(data['x']) >= 3:  # Need at least 3 points for fitting
                    x_vals = np.array(data['x'])
                    y_vals = np.array(data['y'])
                    self.plotting_manager.fit_and_plot_curve(ax, x_vals, y_vals, data['color'], f"{visc} fit", 0, x_name,
                                                            hide_fits, fit_range)
        
        # Draw constant flow rate lines
        if flow_rate_fit:
            for flow_rate, data in flow_rate_data.items():
                if len(data['x']) >= 2:  # Need at least 2 points for a line
                    x_vals = np.array(data['x'])
                    y_vals = np.array(data['y'])
                    # Sort by x values for proper line drawing
                    sort_indices = np.argsort(x_vals)
                    x_sorted = x_vals[sort_indices]
                    y_sorted = y_vals[sort_indices]
                    ax.plot(x_sorted, y_sorted, 'k:', linewidth=1, alpha=0.7)
        
        # Draw flow rate line fits (linear)
        if flow_rate_line_fit:
            for flow_rate, data in flow_rate_data.items():
                if len(data['x']) >= 2:  # Need at least 2 points for linear fitting
                    x_vals = np.array(data['x'])
                    y_vals = np.array(data['y'])
                    
                    # Linear fit: y = mx + b
                    coeffs = np.polyfit(x_vals, y_vals, 1)
                    m, b = coeffs
                    
                    # Create fit line
                    x_fit = np.linspace(min(x_vals), max(x_vals), 100)
                    y_fit = m * x_fit + b
                    
                    # Plot the linear fit
                    if not hide_fits:
                        label = f"{flow_rate} GPM: y = {m:.3e}x + {b:.3e}"
                    else:
                        label = None
                    
                    ax.plot(x_fit, y_fit, color='gray', linestyle='--', linewidth=1.5, alpha=0.8, label=label)
        
        # Add external data with selected normalization (when available)
        if ext_data_opt in ["Yin", "Both"] and self.data_processor.yin_data is not None:
            yin_x_col = ext_x_col_mapping.get('Yin', x_col)
            yin_y = self.get_normalized_external_data(self.data_processor.yin_data)
            self.plotting_manager.plot_scatter_enhanced(ax, self.data_processor.yin_data[yin_x_col], yin_y, 
                                                       self.plotting_manager.EXTERNAL_COLORS['Yin'], 's', "Yin et al. 2015")
        
        if ext_data_opt in ["Sun", "Both"] and self.data_processor.sun_data is not None:
            sun_x_col = ext_x_col_mapping.get('Sun', x_col)
            sun_y = self.get_normalized_external_data(self.data_processor.sun_data)
            self.plotting_manager.plot_scatter_enhanced(ax, self.data_processor.sun_data[sun_x_col], sun_y, 
                                                       self.plotting_manager.EXTERNAL_COLORS['Sun'], 's', "Sun et al. 2017")
        
        # Get normalization-specific parameters  
        _, y_label, air_injection_ratio = self.get_normalized_diameter(avg_df)
        
        # Air injection reference line (optional)
        show_air_line = getattr(self, 'air_injection_var', tk.BooleanVar(value=True)).get()
        if show_air_line:
            normalization = getattr(self, 'normalization_var', tk.StringVar(value="Throat Diameter")).get()
            
            # Create appropriate air injection label based on normalization
            air_labels = {
                "Throat Diameter": r'$D_{air}/D_t$ ≈ 1/6',
                "Kolmogorov": r'$D_{air}/\eta$',
                "Viscous": r'$D_{air}/l_v$', 
                "Weber": r'$D_{air}/l_{We}$',
                "ModReynolds": r'$D_{air}/l_{Re}$'
            }
            air_label = air_labels.get(normalization, r'$D_{air}/D_t$ ≈ 1/6')
            
            ax.axhline(air_injection_ratio, color='k', linestyle=':', linewidth=1.5, alpha=0.7, 
                      label=air_label)
        
        self.plotting_manager.setup_plot_style(ax, xlabel=x_label, ylabel=y_label)
        
        if scale_opt == "Log":
            ax.set_xscale('log')
            
        # Use two-column legend for single plots with many items
        plot_type = self.plot_type.get()
        if plot_type in ['Reynolds', 'Weber', 'Capillary', 'PDFs Fixed Flow', 'Weber Fixed Fit']:
            self.plotting_manager.create_legend(ax, force_two_column=True, loc='upper right')
        else:
            self.plotting_manager.create_legend(ax, loc='upper right')
    
    def plot_reynolds(self):
        ext_mapping = {'Yin': 'Re_t', 'Sun': 'Re'}
        self.plot_dimensionless_number('Reynolds', 'Reynolds Number', ext_mapping)
    
    def plot_weber(self):
        ext_mapping = {'Yin': 'We', 'Sun': 'We'}
        self.plot_dimensionless_number('We_D', 'Weber Number', ext_mapping)
    
    def plot_capillary(self):
        ext_mapping = {'Yin': 'Ca', 'Sun': 'Ca'}
        self.plot_dimensionless_number('Ca', 'Capillary Number', ext_mapping)
    
    def plot_universal_rewe(self):
        """Universal scaling Re^a * We^b"""
        avg_df = self.get_trial_averaged_data()
        if avg_df.empty:
            self.ax.text(0.5, 0.5, "No trial-averaged data available", ha='center', va='center', transform=self.ax.transAxes)
            return
        
        # Get options
        show_external = getattr(self, 'ext_data_plot_var', tk.BooleanVar(value=True)).get()
        include_external_fit = getattr(self, 'ext_data_fit_var', tk.BooleanVar(value=False)).get()
        show_air_injection = getattr(self, 'air_injection_var', tk.BooleanVar(value=True)).get()
        
        # Get fluid inclusion options
        fluid_plot_include = {}
        fluid_fit_include = {}
        if hasattr(self, 'fluid_plot_vars') and hasattr(self, 'fluid_fit_vars'):
            for fluid, var in self.fluid_plot_vars.items():
                fluid_plot_include[fluid] = var.get()
            for fluid, var in self.fluid_fit_vars.items():
                fluid_fit_include[fluid] = var.get()
        else:
            # Default to including all fluids if options not available
            available_fluids = avg_df['Viscosity_cSt'].unique()
            fluid_plot_include = {f: True for f in available_fluids}
            fluid_fit_include = {f: True for f in available_fluids}
        
        df_fit = avg_df.dropna(subset=['D_v', 'ThroatDiameter_m', 'Reynolds', 'We_D']).copy()
        
        # Get scale options
        x_scale = getattr(self, 'x_scale_var', tk.StringVar(value="Log")).get()
        y_scale = getattr(self, 'y_scale_var', tk.StringVar(value="Log")).get()
        
        ax = self.fig.add_subplot(111)
        self.plotting_manager.create_universal_scaling_plot(ax, df_fit, ['Reynolds', 'We_D'], 'D_v', "Re-We ", 
                                                           self.data_processor.yin_data, self.data_processor.sun_data, 
                                                           show_external, include_external_fit,
                                                           fluid_plot_include, fluid_fit_include, show_air_injection,
                                                           normalization_func=self.get_normalized_diameter,
                                                           external_norm_func=self.get_normalized_external_data,
                                                           x_scale=x_scale, y_scale=y_scale)
    
    def plot_universal_reca(self):
        """Universal scaling Re^a * Ca^b"""
        avg_df = self.get_trial_averaged_data()
        if avg_df.empty:
            self.ax.text(0.5, 0.5, "No trial-averaged data available", ha='center', va='center', transform=self.ax.transAxes)
            return
        
        # Get options
        show_external = getattr(self, 'ext_data_plot_var', tk.BooleanVar(value=True)).get()
        include_external_fit = getattr(self, 'ext_data_fit_var', tk.BooleanVar(value=False)).get()
        show_air_injection = getattr(self, 'air_injection_var', tk.BooleanVar(value=True)).get()
        
        # Get fluid inclusion options
        fluid_plot_include = {}
        fluid_fit_include = {}
        if hasattr(self, 'fluid_plot_vars') and hasattr(self, 'fluid_fit_vars'):
            for fluid, var in self.fluid_plot_vars.items():
                fluid_plot_include[fluid] = var.get()
            for fluid, var in self.fluid_fit_vars.items():
                fluid_fit_include[fluid] = var.get()
        else:
            # Default to including all fluids if options not available
            available_fluids = avg_df['Viscosity_cSt'].unique()
            fluid_plot_include = {f: True for f in available_fluids}
            fluid_fit_include = {f: True for f in available_fluids}
        
        df_fit = avg_df.dropna(subset=['D_v', 'ThroatDiameter_m', 'Reynolds', 'Ca']).copy()
        
        # Get scale options
        x_scale = getattr(self, 'x_scale_var', tk.StringVar(value="Log")).get()
        y_scale = getattr(self, 'y_scale_var', tk.StringVar(value="Log")).get()
        
        ax = self.fig.add_subplot(111)
        self.plotting_manager.create_universal_scaling_plot(ax, df_fit, ['Reynolds', 'Ca'], 'D_v', "Re-Ca ", 
                                                           self.data_processor.yin_data, self.data_processor.sun_data, 
                                                           show_external, include_external_fit,
                                                           fluid_plot_include, fluid_fit_include, show_air_injection,
                                                           normalization_func=self.get_normalized_diameter,
                                                           external_norm_func=self.get_normalized_external_data,
                                                           x_scale=x_scale, y_scale=y_scale)
    
    def plot_universal_weca(self):
        """Universal scaling We^a * Ca^b"""
        avg_df = self.get_trial_averaged_data()
        if avg_df.empty:
            self.ax.text(0.5, 0.5, "No trial-averaged data available", ha='center', va='center', transform=self.ax.transAxes)
            return
        
        # Get options
        show_external = getattr(self, 'ext_data_plot_var', tk.BooleanVar(value=True)).get()
        include_external_fit = getattr(self, 'ext_data_fit_var', tk.BooleanVar(value=False)).get()
        show_air_injection = getattr(self, 'air_injection_var', tk.BooleanVar(value=True)).get()
        
        # Get fluid inclusion options
        fluid_plot_include = {}
        fluid_fit_include = {}
        if hasattr(self, 'fluid_plot_vars') and hasattr(self, 'fluid_fit_vars'):
            for fluid, var in self.fluid_plot_vars.items():
                fluid_plot_include[fluid] = var.get()
            for fluid, var in self.fluid_fit_vars.items():
                fluid_fit_include[fluid] = var.get()
        else:
            # Default to including all fluids if options not available
            available_fluids = avg_df['Viscosity_cSt'].unique()
            fluid_plot_include = {f: True for f in available_fluids}
            fluid_fit_include = {f: True for f in available_fluids}
        
        df_fit = avg_df.dropna(subset=['D_v', 'ThroatDiameter_m', 'We_D', 'Ca']).copy()
        
        # Get scale options
        x_scale = getattr(self, 'x_scale_var', tk.StringVar(value="Log")).get()
        y_scale = getattr(self, 'y_scale_var', tk.StringVar(value="Log")).get()
        
        ax = self.fig.add_subplot(111)
        self.plotting_manager.create_universal_scaling_plot(ax, df_fit, ['We_D', 'Ca'], 'D_v', "We-Ca ", 
                                                           self.data_processor.yin_data, self.data_processor.sun_data, 
                                                           show_external, include_external_fit,
                                                           fluid_plot_include, fluid_fit_include, show_air_injection,
                                                           normalization_func=self.get_normalized_diameter,
                                                           external_norm_func=self.get_normalized_external_data,
                                                           x_scale=x_scale, y_scale=y_scale)
    
    def plot_pdfs_fixed_flow(self):
        # PDF comparison at fixed flow rate
        if not hasattr(self, 'flow_var'):
            self.ax.text(0.5, 0.5, "Please select a flow rate", ha='center', va='center', transform=self.ax.transAxes)
            return
        
        selected_flow = float(self.flow_var.get())
        flow_data = self.filtered_df[self.filtered_df['FlowRate'] == selected_flow]
        
        avg_df = self.get_trial_averaged_data()
        avg_df = avg_df[avg_df['FlowRate'] == selected_flow]
        
        if avg_df.empty:
            self.ax.text(0.5, 0.5, "No data for selected flow rate", ha='center', va='center', transform=self.ax.transAxes)
            return
        
        unique_temps = sorted(avg_df['Temp'].unique())
        temp_linestyle_map = {temp: self.plotting_manager.LINESTYLES[i % len(self.plotting_manager.LINESTYLES)] for i, temp in enumerate(unique_temps)}
        
        ax = self.fig.add_subplot(111)
        x_vals = np.linspace(1e-3, 600, 800)
        
        for _, row in avg_df.iterrows():
            mu_ln = row['LogMu']
            sigma_ln = row['LogSigma']
            d30 = row['D_v']
            temp = row['Temp']
            
            mu_mPas = row['mu'] * 1000
            gamma_mNm = row['Gamma'] * 1000
            
            color = self.plotting_manager.get_fluid_color(row['Viscosity_cSt'])
            linestyle = temp_linestyle_map[temp]
            
            label = f"{row['Viscosity_cSt']} {int(temp)}°F \n(μ={mu_mPas:.1f} mPa·s), σ={gamma_mNm:.1f} mN/m)"
            
            pdf_vals = lognorm.pdf(x_vals, s=sigma_ln, scale=np.exp(mu_ln))
            ax.plot(x_vals, pdf_vals, color=color, linestyle=linestyle, linewidth=1.2, label=label)
            ax.axvline(d30, color=color, linestyle=linestyle, linewidth=1)
        
        ax.set_xlabel(r'Diameter ($\mu$m)', fontsize=11)
        ax.set_ylabel('Probability Density', fontsize=11)
        ax.set_xlim(0, 500)
        ax.grid(True,alpha=0.3)
        
        # Enhanced legend for PDF plots
        self.plotting_manager.create_legend(ax, fontsize=7, frameon=True, 
                                           facecolor='white', edgecolor='black', 
                                           loc='upper right', ncol=1, 
                                           columnspacing=0.3, handletextpad=0.2, 
                                           borderpad=0.3, handlelength=2.0)
    
    def plot_weber_fixed_fit(self):
        """Weber number plot with fixed exponent fit: d/D = A*We^b"""
        avg_df = self.get_trial_averaged_data()
        if avg_df.empty:
            self.ax.text(0.5, 0.5, "No trial-averaged data available", ha='center', va='center', transform=self.ax.transAxes)
            return
        
        # Get fixed exponent value
        try:
            fixed_b = float(self.weber_exponent_var.get())
        except (ValueError, AttributeError):
            fixed_b = -0.6  # Default value
        
        # Get options
        ext_data_opt = getattr(self, 'ext_data_var', tk.StringVar(value="None")).get()
        scale_opt = getattr(self, 'scale_var', tk.StringVar(value="Log")).get()
        
        ax = self.fig.add_subplot(111)
        
        # Prepare data groups
        groups = sorted(avg_df.groupby(['Viscosity_cSt', 'Temp']).groups.keys())
        color_marker_pairs = self.plotting_manager.get_color_marker_iterator(groups)
        
        # Plot individual data points and fits
        for j, ((visc, temp), (color, marker)) in enumerate(zip(groups, color_marker_pairs)):
            color = self.plotting_manager.FLUID_COLORS.get(visc, color)
            subset = avg_df[(avg_df['Viscosity_cSt'] == visc) & (avg_df['Temp'] == temp)]
            subset = subset.dropna(subset=['D_v', 'We_D', 'ThroatDiameter_m'])
            
            if len(subset) == 0:
                continue
            
            x = subset['We_D']
            y = subset['D_v'] * 1e-6 / subset['ThroatDiameter_m']
            label = f"{visc} {temp}°F"
            
            self.plotting_manager.plot_scatter_enhanced(ax, x, y, color, marker, label)
            
            # Individual series fit
            if len(x) >= 2:
                try:
                    x_vals = np.array(x)
                    y_vals = np.array(y)
                    
                    # Remove any invalid values
                    valid_mask = (x_vals > 0) & (y_vals > 0) & np.isfinite(x_vals) & np.isfinite(y_vals)
                    x_vals = x_vals[valid_mask]
                    y_vals = y_vals[valid_mask]
                    
                    if len(x_vals) >= 2:
                        # For y = A * x^b with fixed b: ln(y) = ln(A) + b*ln(x)
                        # So A = exp(mean(ln(y) - b*ln(x)))
                        ln_x = np.log(x_vals)
                        ln_y = np.log(y_vals)
                        ln_A = np.mean(ln_y - fixed_b * ln_x)
                        A = np.exp(ln_A)
                        
                        # Calculate R-squared
                        y_pred = A * (x_vals ** fixed_b)
                        r_squared = 1 - (np.sum((y_vals - y_pred) ** 2) / np.sum((y_vals - np.mean(y_vals)) ** 2))
                        
                        # Plot fit line with appropriate point density for log scale
                        x_min_vals, x_max_vals = min(x_vals), max(x_vals)
                        log_x_min, log_x_max = np.log10(x_min_vals), np.log10(x_max_vals)
                        log_range = log_x_max - log_x_min
                        
                        # Only extend if the range is small (< 2 decades)
                        if log_range < 2.0:
                            extension = min(log_range * 0.05, 0.1)  # Max 5% or 0.1 decades
                            log_x_fit_min = log_x_min - extension
                            log_x_fit_max = log_x_max + extension
                        else:
                            # Large range - no extension needed
                            log_x_fit_min = log_x_min
                            log_x_fit_max = log_x_max
                        
                        # Use more points for large log ranges
                        num_points = max(500, int(log_range * 200))  # Scale points with decades
                        x_fit = np.logspace(log_x_fit_min, log_x_fit_max, min(num_points, 2000))  # Cap at 2000 points
                        y_fit = A * (x_fit ** fixed_b)
                        
                        # Use different linestyles for different series
                        linestyle = self.plotting_manager.LINESTYLES[j % len(self.plotting_manager.LINESTYLES)]
                        fit_label = f"{label}: $A = {A:.2f}$, $R^2 = {r_squared:.2f}$"
                        self.plotting_manager.plot_line_enhanced(ax, x_fit, y_fit, color, linestyle, fit_label)
                        
                except Exception as e:
                    print(f"Fitting error for {label}: {e}")
        
        # Add external data if requested (no fits for external data)
        if ext_data_opt in ["Yin", "Both"] and self.data_processor.yin_data is not None:
            yin_y = self.data_processor.yin_data['D_v'] / self.data_processor.yin_data['ThroatDiameter_m']
            self.plotting_manager.plot_scatter_enhanced(ax, self.data_processor.yin_data['We'], yin_y,
                                                       self.plotting_manager.EXTERNAL_COLORS['Yin'], 's', "Yin et al. 2015")
        
        if ext_data_opt in ["Sun", "Both"] and self.data_processor.sun_data is not None:
            sun_y = self.data_processor.sun_data['D_v'] / self.data_processor.sun_data['ThroatDiameter_m']
            self.plotting_manager.plot_scatter_enhanced(ax, self.data_processor.sun_data['We'], sun_y,
                                                       self.plotting_manager.EXTERNAL_COLORS['Sun'], 's', "Sun et al. 2017")
        
        # Add fixed exponent equation to plot title or text box
        ax.text(0.02, 0.98, f'Fixed Exponent: $b = {fixed_b:.2f}$\n$d/D_t = A \\cdot We^{{{fixed_b:.2f}}}$', 
               transform=ax.transAxes, fontsize=8, va='top', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.8))
        
        # Styling
        self.plotting_manager.setup_plot_style(ax, xlabel='Weber Number', ylabel=r'$d/D_t$')
        
        # Apply scaling based on user selection
        if scale_opt == "Log":
            ax.set_xscale('log')
            ax.set_yscale('log')
        else:  # Linear
            ax.set_xscale('linear')
            ax.set_yscale('linear')
        
        # Use two-column legend for this plot type
        self.plotting_manager.create_legend(ax, force_two_column=True, loc='upper right')
    
    def plot_universal_capillary(self):
        """Universal scaling Re^a * Ca^b"""
        avg_df = self.get_trial_averaged_data()
        if avg_df.empty:
            self.ax.text(0.5, 0.5, "No trial-averaged data available", ha='center', va='center', transform=self.ax.transAxes)
            return
        
        # Get options
        show_external = getattr(self, 'ext_data_plot_var', tk.BooleanVar(value=True)).get()
        include_external_fit = getattr(self, 'ext_data_fit_var', tk.BooleanVar(value=False)).get()
        show_air_injection = getattr(self, 'air_injection_var', tk.BooleanVar(value=True)).get()
        
        # Get fluid inclusion options
        fluid_plot_include = {}
        fluid_fit_include = {}
        if hasattr(self, 'fluid_plot_vars') and hasattr(self, 'fluid_fit_vars'):
            for fluid, var in self.fluid_plot_vars.items():
                fluid_plot_include[fluid] = var.get()
            for fluid, var in self.fluid_fit_vars.items():
                fluid_fit_include[fluid] = var.get()
        else:
            # Default to including all fluids if options not available
            available_fluids = avg_df['Viscosity_cSt'].unique()
            fluid_plot_include = {f: True for f in available_fluids}
            fluid_fit_include = {f: True for f in available_fluids}
        
        df_fit = avg_df.dropna(subset=['D_v', 'ThroatDiameter_m', 'We', 'Ca']).copy()
        
        # Get scale options
        x_scale = getattr(self, 'x_scale_var', tk.StringVar(value="Log")).get()
        y_scale = getattr(self, 'y_scale_var', tk.StringVar(value="Log")).get()
        
        ax = self.fig.add_subplot(111)
        self.plotting_manager.create_universal_scaling_plot(ax, df_fit, ['We', 'Ca'], 'D_v', "We-Ca ", 
                                                           self.data_processor.yin_data, self.data_processor.sun_data, 
                                                           show_external, include_external_fit,
                                                           fluid_plot_include, fluid_fit_include, show_air_injection,
                                                           normalization_func=self.get_normalized_diameter,
                                                           external_norm_func=self.get_normalized_external_data,
                                                           x_scale=x_scale, y_scale=y_scale)
    
    def plot_universal_fixed_exponents(self):
        """Universal ReWe plot with fixed user-specified exponents"""
        avg_df = self.get_trial_averaged_data()
        if avg_df.empty:
            self.ax.text(0.5, 0.5, "No trial-averaged data available", ha='center', va='center', transform=self.ax.transAxes)
            return

        # Get options
        show_external = getattr(self, 'ext_data_plot_var', tk.BooleanVar(value=False)).get()
        include_external_fit = getattr(self, 'ext_data_fit_var', tk.BooleanVar(value=False)).get()
        show_air_injection = getattr(self, 'air_injection_var', tk.BooleanVar(value=True)).get()

        # Get fixed exponents
        try:
            a = float(getattr(self, 'reynolds_exp_var', tk.StringVar(value="0.5")).get())
            b = float(getattr(self, 'weber_exp_var', tk.StringVar(value="-0.6")).get())
        except ValueError:
            self.ax.text(0.5, 0.5, "Invalid exponent values. Please enter numbers.", ha='center', va='center', transform=self.ax.transAxes)
            return

        df_fit = avg_df.dropna(subset=['D_v', 'ThroatDiameter_m', 'Reynolds', 'We_D']).copy()

        ax = self.fig.add_subplot(111)
        self.plotting_manager.create_universal_fixed_exponent_plot(ax, df_fit, ['Reynolds', 'We_D'], 'D_v', a, b,
                                                                  self.data_processor.yin_data, self.data_processor.sun_data, 
                                                                  show_external, include_external_fit, show_air_injection,
                                                                  normalization_func=self.get_normalized_diameter,
                                                                  external_norm_func=self.get_normalized_external_data)
    
    def plot_universal_3_parameter(self):
        """Universal 3-parameter scaling plot: d/D = A·Re^a·We^b·Ca^c"""
        avg_df = self.get_trial_averaged_data()
        if avg_df.empty:
            self.ax.text(0.5, 0.5, "No trial-averaged data available", ha='center', va='center', transform=self.ax.transAxes)
            return

        # Get options
        show_external = getattr(self, 'ext_data_plot_var', tk.BooleanVar(value=True)).get()
        include_external_fit = getattr(self, 'ext_data_fit_var', tk.BooleanVar(value=False)).get()
        show_air_injection = getattr(self, 'air_injection_var', tk.BooleanVar(value=True)).get()

        # Get fluid inclusion options
        if hasattr(self, 'fluid_plot_vars') and self.fluid_plot_vars:
            fluid_plot_include = {}
            for fluid, var in self.fluid_plot_vars.items():
                fluid_plot_include[fluid] = var.get()
        else:
            # Default to including all fluids if options not available
            available_fluids = avg_df['Viscosity_cSt'].unique()
            fluid_plot_include = {f: True for f in available_fluids}

        if hasattr(self, 'fluid_fit_vars') and self.fluid_fit_vars:
            fluid_fit_include = {}
            for fluid, var in self.fluid_fit_vars.items():
                fluid_fit_include[fluid] = var.get()
        else:
            # Default to including all fluids if options not available
            available_fluids = avg_df['Viscosity_cSt'].unique()
            fluid_fit_include = {f: True for f in available_fluids}
        
        df_fit = avg_df.dropna(subset=['D_v', 'ThroatDiameter_m', 'Reynolds', 'We_D', 'Ca']).copy()
        
        # Get scale options
        x_scale = getattr(self, 'x_scale_var', tk.StringVar(value="Log")).get()
        y_scale = getattr(self, 'y_scale_var', tk.StringVar(value="Log")).get()
        
        ax = self.fig.add_subplot(111)
        self.plotting_manager.create_universal_3_parameter_plot(ax, df_fit, ['Reynolds', 'We_D', 'Ca'], 'D_v', "3-Parameter ", 
                                                               self.data_processor.yin_data, self.data_processor.sun_data, 
                                                               show_external, include_external_fit,
                                                               fluid_plot_include, fluid_fit_include, show_air_injection,
                                                               normalization_func=self.get_normalized_diameter,
                                                               external_norm_func=self.get_normalized_external_data,
                                                               x_scale=x_scale, y_scale=y_scale)
    
    
    def export_png(self):
        """Export current plot as high-DPI PNG using ExportManager"""
        if self.fig is None:
            messagebox.showwarning("Warning", "No plot to export. Generate a plot first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")],
            title="Export plot as PNG"
        )
        
        if filename:
            try:
                format_type = self.fig_format_var.get()
                plot_type = self.plot_type.get()
                
                self.export_manager.export_figure(self.fig, filename, format_type, plot_type, dpi=500)
                self.canvas.draw()
                
                messagebox.showinfo("Success", f"PNG exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export PNG: {e}")
    
    def copy_to_clipboard(self):
        """Copy current plot to clipboard with same settings as PNG export"""
        if self.fig is None:
            messagebox.showwarning("Warning", "No plot to copy. Generate a plot first.")
            return
        
        try:
            # Get same settings as PNG export
            format_type = self.fig_format_var.get()
            plot_type = self.plot_type.get()
            
            # Prepare figure with export settings (same as PNG export)
            current_size = self.export_manager.prepare_figure_for_export(self.fig, format_type, plot_type)
            
            # Save to memory buffer as PNG with same DPI as PNG export
            buf = io.BytesIO()
            self.fig.savefig(buf, format='png', dpi=500, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
            buf.seek(0)
            
            # Convert to PIL Image and copy to clipboard
            image = Image.open(buf)
            
            # Copy to clipboard (Windows/Linux compatible)
            try:
                import win32clipboard
                from io import BytesIO
                
                # Convert to bitmap format for Windows clipboard
                output = BytesIO()
                image.save(output, 'BMP')
                data = output.getvalue()[14:]  # Remove BMP file header
                output.close()
                
                win32clipboard.OpenClipboard()
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
                win32clipboard.CloseClipboard()
                
                messagebox.showinfo("Success", "Plot copied to clipboard with PNG export settings!")
                
            except ImportError:
                # Fallback for systems without win32clipboard
                try:
                    # Save temporary file as fallback
                    temp_path = "temp_clipboard.png"
                    image.save(temp_path, 'PNG')
                    messagebox.showinfo("Clipboard", f"Image saved to: {os.path.abspath(temp_path)}\n(Direct clipboard copying not available on this system)")
                except Exception:
                    messagebox.showerror("Error", "Clipboard functionality not available on this system")
                    return
            
            # Restore original figure settings
            self.fig.set_size_inches(current_size)
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy to clipboard: {e}")
    
    def export_pdf(self):
        """Export current plot as PDF using ExportManager"""
        if self.fig is None:
            messagebox.showwarning("Warning", "No plot to export. Generate a plot first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            title="Export plot as PDF"
        )
        
        if filename:
            try:
                format_type = self.fig_format_var.get()
                plot_type = self.plot_type.get()
                
                self.export_manager.export_figure(self.fig, filename, format_type, plot_type)
                self.canvas.draw()
                
                messagebox.showinfo("Success", f"PDF exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export PDF: {e}")

    def export_data(self):
        """Export filtered data with enhanced error handling"""
        if self.filtered_df is None or self.filtered_df.empty:
            messagebox.showwarning("Warning", "No data to export")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")],
            title="Export filtered data"
        )
        
        if filename:
            try:
                # Add timestamp and metadata
                export_df = self.filtered_df.copy()
                
                if filename.endswith('.xlsx'):
                    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                        export_df.to_excel(writer, sheet_name='Data', index=False)
                        
                        # Add metadata sheet
                        metadata = pd.DataFrame({
                            'Parameter': ['Export Date', 'Total Records', 'Unique Fluids', 'Unique Experiments'],
                            'Value': [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                                    len(export_df),
                                    len(export_df['Viscosity_cSt'].unique()),
                                    len(export_df.groupby(['Viscosity_cSt', 'Temp', 'FlowRate', 'VenturiAngle', 'AeratedFlow']))]
                        })
                        metadata.to_excel(writer, sheet_name='Metadata', index=False)
                else:
                    export_df.to_csv(filename, index=False)
                
                messagebox.showinfo("Success", f"Data exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export data: {e}")
    

def main():
    root = tk.Tk()
    app = MultiviscosityAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main()