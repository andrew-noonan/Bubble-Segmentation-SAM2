#!/usr/bin/env python3
"""
Non-Streamlit GUI version of multiviscosity data processing
Uses tkinter for faster performance compared to Streamlit web interface
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

class PlottingManager:
    """Centralized plotting configuration and utilities"""
    
    # Color schemes
    FLUID_COLORS = {"10 cSt": '#D32F2F', "50 cSt": '#000000', "20 cSt": '#1976D2'}  # Red, Black, Blue
    EXTERNAL_COLORS = {'Yin': '#4CAF50', 'Sun': '#FF9800'}  # Green, Orange
    
    # Style arrays
    LINESTYLES = ['-', '--', ':', '-.']
    MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '+']
    
    # Font sizes - optimized for publication
    FONT_SIZES = {
        'title': 10,
        'label': 9,
        'tick': 7,
        'legend': 7,  # Smaller legend text
        'text': 7
    }
    
    @classmethod
    def setup_plot_style(cls, ax, title="", xlabel="", ylabel="", grid=True):
        """Apply consistent styling to plot"""
        # Skip title for publication format - use captions instead
        # if title:
        #     ax.set_title(title, fontsize=cls.FONT_SIZES['title'])
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=cls.FONT_SIZES['label'])
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=cls.FONT_SIZES['label'])
        
        ax.tick_params(labelsize=cls.FONT_SIZES['tick'])
        if grid:
            ax.grid(True)
        
        # Force standard notation instead of scientific notation
        from matplotlib.ticker import ScalarFormatter
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
    
    @classmethod
    def get_color_marker_iterator(cls, data_groups):
        """Get iterator for consistent color/marker assignment"""
        colors = list(cls.FLUID_COLORS.values()) + ['#9C27B0', '#00BCD4', '#795548', '#607D8B']
        markers = cls.MARKERS
        
        color_cycle = itertools.cycle(colors)
        marker_cycle = itertools.cycle(markers)
        
        return [(next(color_cycle), next(marker_cycle)) for _ in data_groups]
    
    @classmethod
    def combine_external_data(cls, internal_df, yin_data, sun_data, x_cols, include_external_plot=True, include_external_fit=False):
        """Combine internal and external data for universal scaling plots"""
        combined_data = []
        
        # Add internal data
        for _, row in internal_df.iterrows():
            data_row = {'source': 'Internal', 'Viscosity_cSt': row['Viscosity_cSt']}
            for col in x_cols + ['D_v', 'ThroatDiameter_m']:
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
            
            # Add Yin data
            for _, row in yin_data.iterrows():
                data_row = {'source': 'Yin et al. 2015', 'Viscosity_cSt': None}
                for col in x_cols:
                    mapped_col = col_mapping.get(col, {}).get('Yin', col)
                    data_row[col] = row[mapped_col]
                data_row['D_v'] = row['D_v'] * 1e6  # Convert back to microns for consistency
                data_row['ThroatDiameter_m'] = row['ThroatDiameter_m']
                combined_data.append(data_row)
            
            # Add Sun data
            for _, row in sun_data.iterrows():
                data_row = {'source': 'Sun et al. 2017', 'Viscosity_cSt': None}
                for col in x_cols:
                    mapped_col = col_mapping.get(col, {}).get('Sun', col)
                    data_row[col] = row[mapped_col]
                data_row['D_v'] = row['D_v'] * 1e6  # Convert back to microns for consistency
                data_row['ThroatDiameter_m'] = row['ThroatDiameter_m']
                combined_data.append(data_row)
        
        return pd.DataFrame(combined_data)

    @classmethod
    def create_universal_scaling_plot(cls, ax, df_fit, x_cols, y_col, title_prefix="", yin_data=None, sun_data=None, include_external_plot=True, include_external_fit=False, fluid_plot_include=None, fluid_fit_include=None, show_air_injection=True):
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
        # Normalize diameter data for fitting
        fit_subset_norm = fit_subset.copy()
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
                    
                    ax.scatter(subset['CollapseX'], subset['NormDiameter'],
                              label=f"{visc}", alpha=0.8, marker=marker, 
                              color=color, s=16, edgecolors='white', linewidth=0.3)
            
            # Plot external data if requested
            if include_external_plot:
                yin_data_plot = df_combined[df_combined['source'] == 'Yin et al. 2015']
                if not yin_data_plot.empty:
                    ax.scatter(yin_data_plot['CollapseX'], yin_data_plot['NormDiameter'],
                              label="Yin et al. 2015", marker='s', color=cls.EXTERNAL_COLORS['Yin'], 
                              edgecolor='k', s=16, alpha=0.8)
                
                sun_data_plot = df_combined[df_combined['source'] == 'Sun et al. 2017']
                if not sun_data_plot.empty:
                    ax.scatter(sun_data_plot['CollapseX'], sun_data_plot['NormDiameter'],
                              label="Sun et al. 2017", marker='s', color=cls.EXTERNAL_COLORS.get('Sun', '#FF9800'), 
                              edgecolor='k',s=16, alpha=0.8)
            
            # Plot best-fit line with bounds checking
            # The relationship is d/D = A * (Re^a * We^b), so on the collapsed plot it should be d/D = A * x
            x_min, x_max = df_combined['CollapseX'].min(), df_combined['CollapseX'].max()
            if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min and x_min > 0:
                # Create logarithmically spaced points for better visualization on log-log plot
                x_fit = np.logspace(np.log10(x_min), np.log10(x_max), 200)
                y_fit = A * x_fit  # This is the correct relationship for universal scaling
                # Only plot if y_fit values are reasonable
                if np.all(np.isfinite(y_fit)) and np.all(y_fit > 0):
                    # Create full equation label
                    x1_name = x_cols[0].replace('Reynolds', 'Re').replace('We_D', 'We')
                    x2_name = x_cols[1].replace('Reynolds', 'Re').replace('We_D', 'We').replace('Ca', 'Ca')
                    equation = f"$d/D_t = {A:.2f} \\cdot {x1_name}^{{{a:.3f}}} \\cdot {x2_name}^{{{b:.3f}}}$, $R^2 = {r_squared:.3f}$"
                    ax.plot(x_fit, y_fit, 'k--', linewidth=2, alpha=0.8, label=equation)
            
            # Air injection reference line (optional)
            if show_air_injection:
                air_injection_ratio = 0.001 / 0.006  # d_air/D_t ≈ 1/6
                ax.axhline(air_injection_ratio, color='k', linestyle=':', linewidth=1.5, alpha=0.7, 
                          label=r'$D_{air}/D_t$ ≈ 1/6')
            
            # Styling
            x_label = fr"${x_cols[0].replace('Reynolds', 'Re').replace('We_D', 'We')}^{{{a:.2f}}} \cdot {x_cols[1].replace('Reynolds', 'Re').replace('We_D', 'We').replace('Ca', 'Ca')}^{{{b:.2f}}}$"
            
            # Create fit description
            fit_fluids = [f for f, include in fluid_fit_include.items() if include]
            fit_desc = f"Fit: {', '.join([f'{f}' for f in fit_fluids])}"
            if include_external_fit:
                fit_desc += " + Ext."
            
            cls.setup_plot_style(ax, 
                               xlabel=x_label,
                               ylabel=r"$d_{30} / D_t$")
            
            ax.set_xscale('log')
            ax.set_yscale('log')
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
                    
                    # Plot against first dimensionless number as fallback
                    y_vals = subset['D_v'] * 1e-6 / subset['ThroatDiameter_m']
                    ax.scatter(subset[x_cols[0]], y_vals,
                              label=f"{visc}", alpha=0.8, marker=marker, 
                              color=color, s=15, edgecolors='white', linewidth=0.3)
            
            cls.setup_plot_style(ax, 
                               xlabel=x_cols[0],
                               ylabel=r"$d_{30} / D_t$")
            ax.text(0.05, 0.95, f"Curve fitting failed: {str(e)[:100]}", 
                   ha='left', va='top', transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                   fontsize=8, wrap=True)
            ax.set_xscale('log')
            ax.set_yscale('log')
            cls.create_legend(ax)
            return False, None
    
    @classmethod  
    def fit_and_plot_curve(cls, ax, x, y, color, label, linestyle_idx=0, x_name="x"):
        """Fit power law and plot curve with full equation"""
        if len(x) < 3:
            return
            
        try:
            def model_fn(x_val, A, b): 
                return A * x_val**b
            popt, _ = curve_fit(model_fn, x, y, maxfev=10000)
            
            # Calculate R²
            y_pred = model_fn(x, *popt)
            r_squared = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
            
            x_fit = np.linspace(min(x), max(x), 200)
            y_fit = model_fn(x_fit, *popt)
            
            linestyle = cls.LINESTYLES[linestyle_idx % len(cls.LINESTYLES)]
            # Create full equation label
            fit_label = f"{label}: $d/D_t = {popt[0]:.2f} \\cdot {x_name}^{{{popt[1]:.2f}}}$, $R^2 = {r_squared:.3f}$"
            ax.plot(x_fit, y_fit, color=color, linestyle=linestyle, 
                   linewidth=1.5, label=fit_label)
        except Exception:
            pass
    
    @classmethod
    def create_legend(cls, ax, force_two_column=False, **kwargs):
        """Create consistent legend styling with smart positioning"""
        # Count legend entries to determine optimal layout
        handles, labels = ax.get_legend_handles_labels()
        num_items = len(handles)
        
        # Smart positioning based on plot content
        if force_two_column and num_items >= 4:
            # Force two-column layout for single plots with many items
            loc = 'upper right'
            ncol = 2
            fontsize = cls.FONT_SIZES['legend'] - 1  # Even smaller for two-column
        elif num_items <= 3:
            loc = 'upper right'
            ncol = 1
            fontsize = cls.FONT_SIZES['legend']
        elif num_items <= 6:
            loc = 'upper right'
            ncol = 2 if num_items > 4 else 1
            fontsize = cls.FONT_SIZES['legend']
        else:
            # For many items, place outside plot area
            loc = 'center left'
            bbox_to_anchor = (1.02, 0.5)
            kwargs['bbox_to_anchor'] = bbox_to_anchor
            ncol = 1
            fontsize = cls.FONT_SIZES['legend']
        
        default_props = {
            'fontsize': fontsize,
            'frameon': True,
            'facecolor': 'white',
            'edgecolor': 'black',
            'loc': loc,
            'ncol': ncol,
            'markerscale': 1 if force_two_column else 0.6,  # Even smaller for two-column
            'columnspacing': 0.3 if force_two_column else 0.4,  # Tighter for two-column
            'handletextpad': 0.15 if force_two_column else 0.2,  # Less space for two-column
            'borderpad': 0.25 if force_two_column else 0.3,      # Less padding for two-column
            'handlelength': 1.0 if force_two_column else 1.2    # Shorter lines for two-column
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

class MultiviscosityAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Multiviscosity Data Processing - Fast GUI")
        self.root.geometry("1600x1000")
        
        # Configure GUI font scaling
        self.setup_gui_fonts()
        
        # Data storage
        self.df = None
        self.filtered_df = None
        self.yin_data = self.import_yin_data()
        self.sun_data = self.import_sun_data()
        self.pm = PlottingManager()  # Plotting manager instance
        
        # Create UI
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
            "PDFs Fixed Flow", "Universal: ReCa", "Weber Fixed Fit"
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
        ttk.Button(export_button_frame, text="Export PNG (500 DPI)", command=self.export_png).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,2))
        ttk.Button(export_button_frame, text="Export PDF", command=self.export_pdf).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2,0))
        
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
            self.df = self.load_all_experiment_data(self.dir_var.get())
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
            
            # Air injection line option
            self.air_injection_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(self.options_frame, text="Show Air Injection Diameter", variable=self.air_injection_var).pack(anchor=tk.W, pady=(10,0))
            
            self.scale_var = tk.StringVar(value="Linear")
            ttk.Label(self.options_frame, text="Scale:").pack(anchor=tk.W)
            for scale in ["Linear", "Log"]:
                ttk.Radiobutton(self.options_frame, text=scale, variable=self.scale_var, value=scale).pack(anchor=tk.W)
        
        elif plot_type in ["Universal: ReWe", "Universal: ReCa"]:
            # External data options for universal plots
            self.ext_data_plot_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(self.options_frame, text="Show External Data", variable=self.ext_data_plot_var).pack(anchor=tk.W)
            
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
            # Fixed exponent input
            ttk.Label(self.options_frame, text="Fixed Weber Exponent (b):").pack(anchor=tk.W)
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
                self.plot_universal()
            elif plot_type == "PDFs Fixed Flow":
                self.plot_pdfs_fixed_flow()
            elif plot_type == "Universal: ReCa":
                self.plot_universal_capillary()
            elif plot_type == "Weber Fixed Fit":
                self.plot_weber_fixed_fit()
            
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to generate plot: {e}")
    
    def plot_repeatability(self):
        var_labels = [r'$\mu_{LN}$', r'$\sigma_{LN}$', r'$d_{32}$', r'$d_{30}$']
        var_keys = ['LogMu', 'LogSigma', 'D32', 'D_v']
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
                colors.append(self.pm.FLUID_COLORS.get(param[0], '#666666'))
        
        if not t1_data:
            self.ax.text(0.5, 0.5, "No paired trial data available", ha='center', va='center', transform=self.ax.transAxes)
            return
        
        t1_data, t2_data = np.array(t1_data), np.array(t2_data)
        axes = self.fig.subplots(2, 2)
        axes = axes.flatten()
        # Improve subplot spacing
        self.fig.subplots_adjust(hspace=0.4, wspace=0.3)
        
        for i in range(4):
            ax = axes[i]
            x, y = t1_data[:, i], t2_data[:, i]
            
            # Plot data points
            for xi, yi, ci in zip(x, y, colors):
                ax.scatter(xi, yi, color=ci, s=16, alpha=0.8, marker='x', linewidth=1.5)
            
            # Styling and 1:1 line
            min_val, max_val = min(x.min(), y.min()), max(x.max(), y.max())
            margin = 0.05 * (max_val - min_val)
            ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], 
                   'k--', linewidth=1.5, alpha=0.7)
            ax.set_xlim(min_val - margin, max_val + margin)
            ax.set_ylim(min_val - margin, max_val + margin)
            
            # Apply consistent styling - remove title for subplots
            self.pm.setup_plot_style(ax, xlabel=f'Trial 1 {var_labels[i]}',
                                   ylabel=f'Trial 2 {var_labels[i]}')
            # Add subplot label instead of title
            ax.text(0.05, 0.95, f'{var_labels[i]}', transform=ax.transAxes, 
                   fontsize=self.pm.FONT_SIZES['text'], weight='bold', va='top')
            
            # R² calculation and display  
            r2 = np.corrcoef(x, y)[0, 1] ** 2
            ax.text(0.05, 0.85, f'$R^2$ = {r2:.3f}', transform=ax.transAxes, 
                   fontsize=self.pm.FONT_SIZES['text'], va='top')
        
        # Create unified legend
        legend_elements = []
        for visc, color in self.pm.FLUID_COLORS.items():
            if visc in [param[0] for param in unique_params]:
                legend_elements.append(plt.Line2D([0], [0], marker='x', color=color, 
                                                label=visc, markersize=6, linestyle='None', linewidth=1.5))
        
        if legend_elements:
            self.pm.create_legend(axes[0], handles=legend_elements)
        
        plt.tight_layout()
    
    def get_trial_averaged_data(self):
        group_cols = ['Viscosity_cSt', 'Temp', 'FlowRate', 'VenturiAngle', 'AeratedFlow']
        grouped = self.filtered_df.groupby(group_cols)
        
        trial_avg_records = []
        for key, group in grouped:
            if group['Trial'].nunique() == 2:
                record = group.mean(numeric_only=True)
                for i, col in enumerate(group_cols):
                    record[col] = key[i]
                trial_avg_records.append(record)
        
        return pd.DataFrame(trial_avg_records)
    
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
        color_marker_pairs = self.pm.get_color_marker_iterator(visc_temp_groups)
        
        for i, (col, ylabel, ylim) in enumerate(variables):
            ax = axes[i]
            
            for (visc, temp), (color, marker) in zip(visc_temp_groups, color_marker_pairs):
                # Override color for viscosity consistency
                color = self.pm.FLUID_COLORS.get(visc, color)
                
                subset = avg_df[(avg_df['Viscosity_cSt'] == visc) & (avg_df['Temp'] == temp)].sort_values('FlowRate')
                if not subset.empty:
                    label = f"{visc} - {temp}°F"
                    ax.plot(subset['FlowRate'], subset[col],
                           linestyle='-', marker=marker, color=color, 
                           linewidth=1.5, markersize=5, markeredgewidth=1, 
                           markerfacecolor='none', label=label)
            
            # Apply consistent styling
            self.pm.setup_plot_style(ax, xlabel='Flow Rate (GPM)', ylabel=ylabel)
            ax.set_ylim(ylim)
            self.pm.create_legend(ax)
        
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
                
                color = self.pm.get_fluid_color(visc)
                marker = self.pm.MARKERS[idx % len(self.pm.MARKERS)]
                
                ax.plot(subset['Temp'], subset[col],
                       linestyle='-', linewidth=1.0,
                       marker=marker, markerfacecolor='none',
                       markeredgewidth=1, markersize=4,
                       color=color, label=f"{visc} – {flow} GPM")
            
            ax.set_xlabel('Temperature (°F)', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_ylim(ylim)
            ax.grid(True)
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
                    color = self.pm.get_fluid_color(visc)
                    marker = self.pm.MARKERS[j % len(self.pm.MARKERS)]
                    label = f"{visc} - {temp}°F"
                    ax.plot(subset['VenturiAngle'], subset[col], '-', marker=marker, label=label, color=color)
            
            self.pm.format_axis(ax, 'Venturi Angle (°)', ylabel, None)
            ax.set_ylim(ylim)
            ax.grid(True)
            self.pm.create_legend(ax)
        
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
        scale_opt = getattr(self, 'scale_var', tk.StringVar(value="Linear")).get()
        
        # Prepare data groups
        groups = sorted(avg_df.groupby(['Viscosity_cSt', 'Temp']).groups.keys())
        color_marker_pairs = self.pm.get_color_marker_iterator(groups)
        
        ax = self.fig.add_subplot(111)
        
        # Per fluid fit data collection
        fluid_data = {}
        
        for j, ((visc, temp), (color, marker)) in enumerate(zip(groups, color_marker_pairs)):
            # Override color for viscosity consistency
            color = self.pm.FLUID_COLORS.get(visc, color)
            
            subset = avg_df[(avg_df['Viscosity_cSt'] == visc) & (avg_df['Temp'] == temp)]
            subset = subset.dropna(subset=['D_v', x_col, 'ThroatDiameter_m'])
            
            if len(subset) == 0:
                continue
            
            x = subset[x_col]
            y = subset['D_v'] * 1e-6 / subset['ThroatDiameter_m']
            label = f"{visc} {temp}°F"
            
            ax.scatter(x, y, marker=marker, s=16, color=color, label=label, 
                      alpha=0.8, edgecolors='white', linewidth=0.3)
            
            # Per series fit
            if per_series_fit:
                x_name = x_col.replace('Reynolds', 'Re').replace('We_D', 'We').replace('Ca', 'Ca')
                self.pm.fit_and_plot_curve(ax, x, y, color, label, j, x_name)
            
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
                    self.pm.fit_and_plot_curve(ax, x_vals, y_vals, data['color'], f"{visc} fit", 0, x_name)
        
        # Add external data
        if ext_data_opt in ["Yin", "Both"] and self.yin_data is not None:
            yin_x_col = ext_x_col_mapping.get('Yin', x_col)
            yin_y = self.yin_data['D_v'] / self.yin_data['ThroatDiameter_m']
            ax.scatter(self.yin_data[yin_x_col], yin_y, 
                      label="Yin et al. 2015", marker='s', 
                      color=self.pm.EXTERNAL_COLORS['Yin'], edgecolor='k', s=15, alpha=0.8)
        
        if ext_data_opt in ["Sun", "Both"] and self.sun_data is not None:
            sun_x_col = ext_x_col_mapping.get('Sun', x_col)
            sun_y = self.sun_data['D_v'] / self.sun_data['ThroatDiameter_m']
            ax.scatter(self.sun_data[sun_x_col], sun_y, 
                      label="Sun et al. 2017", marker='s', 
                      color=self.pm.EXTERNAL_COLORS['Sun'], edgecolor='k', s=15, alpha=0.8)
        
        # Air injection reference line (optional)
        show_air_line = getattr(self, 'air_injection_var', tk.BooleanVar(value=True)).get()
        if show_air_line:
            # Air injection diameter is approximately 1 mm = 0.001 m, throat diameter is around 6 mm = 0.006 m
            air_injection_ratio = 0.001 / 0.006  # Approximate d_air/D_t ratio
            ax.axhline(air_injection_ratio, color='k', linestyle=':', linewidth=1.5, alpha=0.7, 
                      label=r'$D_{air}/D_t$ ≈ 1/6')
        
        self.pm.setup_plot_style(ax, xlabel=x_label, ylabel=r"$d_{30}/D_t$")
        
        if scale_opt == "Log":
            ax.set_xscale('log')
            
        # Use two-column legend for single plots with many items
        plot_type = self.plot_type.get()
        if plot_type in ['Reynolds', 'Weber', 'Capillary', 'PDFs Fixed Flow', 'Weber Fixed Fit']:
            self.pm.create_legend(ax, force_two_column=True, loc='upper right')
        else:
            self.pm.create_legend(ax, loc='upper right')
    
    def plot_reynolds(self):
        ext_mapping = {'Yin': 'Re_t', 'Sun': 'Re'}
        self.plot_dimensionless_number('Reynolds', 'Reynolds Number', ext_mapping)
    
    def plot_weber(self):
        ext_mapping = {'Yin': 'We', 'Sun': 'We'}
        self.plot_dimensionless_number('We_D', 'Weber Number', ext_mapping)
    
    def plot_capillary(self):
        ext_mapping = {'Yin': 'Ca', 'Sun': 'Ca'}
        self.plot_dimensionless_number('Ca', 'Capillary Number', ext_mapping)
    
    def plot_universal(self):
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
        
        ax = self.fig.add_subplot(111)
        self.pm.create_universal_scaling_plot(ax, df_fit, ['Reynolds', 'We_D'], 'D_v', "Re-We ", 
                                            self.yin_data, self.sun_data, show_external, include_external_fit,
                                            fluid_plot_include, fluid_fit_include, show_air_injection)
    
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
        temp_linestyle_map = {temp: self.pm.LINESTYLES[i % len(self.pm.LINESTYLES)] for i, temp in enumerate(unique_temps)}
        
        ax = self.fig.add_subplot(111)
        x_vals = np.linspace(1e-3, 600, 800)
        
        for _, row in avg_df.iterrows():
            mu_ln = row['LogMu']
            sigma_ln = row['LogSigma']
            d30 = row['D_v']
            temp = row['Temp']
            
            mu_mPas = row['mu'] * 1000
            gamma_mNm = row['Gamma'] * 1000
            
            color = self.pm.get_fluid_color(row['Viscosity_cSt'])
            linestyle = temp_linestyle_map[temp]
            
            label = f"{row['Viscosity_cSt']} {int(temp)}°F \n(μ={mu_mPas:.1f} mPa·s), σ={gamma_mNm:.1f} mN/m)"
            
            pdf_vals = lognorm.pdf(x_vals, s=sigma_ln, scale=np.exp(mu_ln))
            ax.plot(x_vals, pdf_vals, color=color, linestyle=linestyle, linewidth=0.75, label=label)
            ax.axvline(d30, color=color, linestyle=linestyle, linewidth=0.75)
        
        ax.set_xlabel(r'Diameter ($\mu$m)', fontsize=11)
        ax.set_ylabel('Probability Density', fontsize=11)
        ax.set_xlim(0, 500)
        ax.grid(True)
        
        # Use compact legend for PDF plot - longer lines to show line styles clearly
        ax.legend(fontsize=6, frameon=True, facecolor='white', edgecolor='black', 
                 loc='upper right', ncol=1, 
                 columnspacing=0.3, handletextpad=0.15, 
                 borderpad=0.2, handlelength=2.0, numpoints=1)
    
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
        color_marker_pairs = self.pm.get_color_marker_iterator(groups)
        
        # Plot individual data points and fits
        for j, ((visc, temp), (color, marker)) in enumerate(zip(groups, color_marker_pairs)):
            color = self.pm.FLUID_COLORS.get(visc, color)
            subset = avg_df[(avg_df['Viscosity_cSt'] == visc) & (avg_df['Temp'] == temp)]
            subset = subset.dropna(subset=['D_v', 'We_D', 'ThroatDiameter_m'])
            
            if len(subset) == 0:
                continue
            
            x = subset['We_D']
            y = subset['D_v'] * 1e-6 / subset['ThroatDiameter_m']
            label = f"{visc} {temp}°F"
            
            ax.scatter(x, y, marker=marker, s=16, color=color, label=label, 
                      alpha=0.8, edgecolors='white', linewidth=0.3)
            
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
                        
                        # Plot fit line
                        x_fit = np.logspace(np.log10(min(x_vals)), np.log10(max(x_vals)), 100)
                        y_fit = A * (x_fit ** fixed_b)
                        
                        # Use different linestyles for different series
                        linestyle = self.pm.LINESTYLES[j % len(self.pm.LINESTYLES)]
                        fit_label = f"{label}: $A = {A:.3f}$, $R^2 = {r_squared:.3f}$"
                        ax.plot(x_fit, y_fit, color=color, linestyle=linestyle, 
                               linewidth=1.5, label=fit_label)
                        
                except Exception as e:
                    print(f"Fitting error for {label}: {e}")
        
        # Add external data if requested (no fits for external data)
        if ext_data_opt in ["Yin", "Both"] and self.yin_data is not None:
            yin_y = self.yin_data['D_v'] / self.yin_data['ThroatDiameter_m']
            ax.scatter(self.yin_data['We'], yin_y, 
                      label="Yin et al. 2015", marker='s', 
                      color=self.pm.EXTERNAL_COLORS['Yin'], edgecolor='k', s=15, alpha=0.8)
        
        if ext_data_opt in ["Sun", "Both"] and self.sun_data is not None:
            sun_y = self.sun_data['D_v'] / self.sun_data['ThroatDiameter_m']
            ax.scatter(self.sun_data['We'], sun_y, 
                      label="Sun et al. 2017", marker='s', 
                      color=self.pm.EXTERNAL_COLORS['Sun'], edgecolor='k', s=15, alpha=0.8)
        
        # Add fixed exponent equation to plot title or text box
        ax.text(0.02, 0.98, f'Fixed Exponent: $b = {fixed_b:.2f}$\n$d/D_t = A \\cdot We^{{{fixed_b:.2f}}}$', 
               transform=ax.transAxes, fontsize=8, va='top', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.8))
        
        # Styling
        self.pm.setup_plot_style(ax, xlabel='Weber Number', ylabel=r'$d/D_t$')
        
        # Apply scaling based on user selection
        if scale_opt == "Log":
            ax.set_xscale('log')
            ax.set_yscale('log')
        else:  # Linear
            ax.set_xscale('linear')
            ax.set_yscale('linear')
        
        # Use two-column legend for this plot type
        self.pm.create_legend(ax, force_two_column=True, loc='upper right')
    
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
        
        df_fit = avg_df.dropna(subset=['D_v', 'ThroatDiameter_m', 'Reynolds', 'Ca']).copy()
        
        ax = self.fig.add_subplot(111)
        self.pm.create_universal_scaling_plot(ax, df_fit, ['Reynolds', 'Ca'], 'D_v', "Re-Ca ", 
                                            self.yin_data, self.sun_data, show_external, include_external_fit,
                                            fluid_plot_include, fluid_fit_include, show_air_injection)
    
    def get_publication_figure_size(self):
        """Get figure size based on publication format and plot type"""
        format_type = self.fig_format_var.get()
        plot_type = self.plot_type.get()
        
        # Determine if this is a subplot figure
        is_subplot = plot_type in ['Repeatability', 'Flow Rate', 'Temperature', 'Angle']
        
        if format_type == "Column Width":
            if is_subplot:
                return (4.2, 4.5)  # Wider for subplots
            else:
                return (3.5, 2.8)  # Standard single plot
        else:  # Full Width
            if is_subplot:
                return (8.0, 6.0)   # Wider for subplots
            else:
                return (7.0, 4.5)   # Standard single plot
    
    def setup_publication_style(self):
        """Configure matplotlib for publication-quality plots"""
        format_type = self.fig_format_var.get()
        
        if format_type == "Column Width":
            plt.rcParams.update({
                'font.size': 8,
                'axes.titlesize': 9,
                'axes.labelsize': 8,
                'xtick.labelsize': 7,
                'ytick.labelsize': 7,
                'legend.fontsize': 6,
                'lines.linewidth': 1.0,
                'axes.linewidth': 0.8,
                'patch.linewidth': 0.5,
            })
        else:  # Full Width
            plt.rcParams.update({
                'font.size': 10,
                'axes.titlesize': 12,
                'axes.labelsize': 11,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'legend.fontsize': 8,
                'lines.linewidth': 1.2,
                'axes.linewidth': 1.0,
                'patch.linewidth': 0.8,
            })
    
    def reset_matplotlib_style(self):
        """Reset matplotlib to default GUI settings"""
        plt.rcParams.update(plt.rcParamsDefault)
    
    def export_png(self):
        """Export current plot as high-DPI PNG"""
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
                # Store current figure size
                current_size = self.fig.get_size_inches()
                
                # Set publication figure size and style
                pub_size = self.get_publication_figure_size()
                self.setup_publication_style()
                self.fig.set_size_inches(pub_size)
                
                # Adjust layout for publication based on plot type
                plot_type = self.plot_type.get()
                if plot_type == 'Repeatability':
                    self.fig.tight_layout(pad=0.3)
                    self.fig.subplots_adjust(hspace=0.35, wspace=0.35)
                elif plot_type in ['Flow Rate', 'Temperature', 'Angle']:
                    self.fig.tight_layout(pad=0.3)
                    self.fig.subplots_adjust(hspace=0.4)
                else:
                    self.fig.tight_layout(pad=0.3)
                
                # Export at high DPI
                self.fig.savefig(filename, dpi=500, bbox_inches='tight', 
                               facecolor='white', edgecolor='none', 
                               format='png', pil_kwargs={'optimize': True})
                
                # Reset to original size and style
                self.fig.set_size_inches(current_size)
                self.reset_matplotlib_style()
                self.canvas.draw()
                
                messagebox.showinfo("Success", f"PNG exported to {filename}")
            except Exception as e:
                self.reset_matplotlib_style()  # Ensure we reset even on error
                messagebox.showerror("Error", f"Failed to export PNG: {e}")
    
    def export_pdf(self):
        """Export current plot as PDF"""
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
                # Store current figure size
                current_size = self.fig.get_size_inches()
                
                # Set publication figure size and style
                pub_size = self.get_publication_figure_size()
                self.setup_publication_style()
                self.fig.set_size_inches(pub_size)
                
                # Adjust layout for publication based on plot type
                plot_type = self.plot_type.get()
                if plot_type == 'Repeatability':
                    self.fig.tight_layout(pad=0.4)
                    self.fig.subplots_adjust(hspace=0.45, wspace=0.35)
                elif plot_type in ['Flow Rate', 'Temperature', 'Angle']:
                    self.fig.tight_layout(pad=0.3)
                    self.fig.subplots_adjust(hspace=0.4)
                else:
                    self.fig.tight_layout(pad=0.3)
                
                # Export as vector PDF
                self.fig.savefig(filename, bbox_inches='tight', 
                               facecolor='white', edgecolor='none', 
                               format='pdf')
                
                # Reset to original size and style
                self.fig.set_size_inches(current_size)
                self.reset_matplotlib_style()
                self.canvas.draw()
                
                messagebox.showinfo("Success", f"PDF exported to {filename}")
            except Exception as e:
                self.reset_matplotlib_style()  # Ensure we reset even on error
                messagebox.showerror("Error", f"Failed to export PDF: {e}")

    def export_data(self):
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
                if filename.endswith('.xlsx'):
                    self.filtered_df.to_excel(filename, index=False)
                else:
                    self.filtered_df.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Data exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export data: {e}")
    
    def compute_fluid_properties(self, tempK, fluid_key):
        props = fluid_properties[fluid_key]
        A, mu_ref, alpha, rho_ref = props["A"], props["mu_ref"], props["alpha"], props["rho_ref"]
        B = np.log10(mu_ref / 1000) - A / T_ref
        mu = 10 ** (A / tempK + B)
        rho = rho_ref * (1 - alpha * (tempK - T_ref))
        gamma = (props["surfaceTensionYIntercept"] + props["surfaceTensionSlope"] * (tempK - 273.15)) / 1000
        return mu, rho, gamma
    
    def load_all_experiment_data(self, base_dir):
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
                    
                    # LabVIEW data
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
                        except Exception:
                            pass
                    
                    # SAM data
                    sam_path = os.path.join(root, 'experiment_summary.csv')
                    try:
                        sam_df = pd.read_csv(sam_path)
                        record.update({
                            'LogMu': sam_df['log_mu'].iloc[0] + np.log(UM_PER_PIXEL),
                            'LogSigma': sam_df['log_sigma'].iloc[0],
                            'D32': sam_df['d32'].iloc[0] * UM_PER_PIXEL,
                            'D_v': sam_df['dv'].iloc[0] * UM_PER_PIXEL
                        })
                    except Exception:
                        pass
                    
                    records.append(record)
                except Exception:
                    continue
        
        df = pd.DataFrame.from_records(records)
        if df.empty:
            return df
        
        # Calculate derived properties
        df['deltaP'] = df.get('MeanP1', np.nan) - df.get('MeanP2', np.nan)
        df['deltaP_Pa'] = df['deltaP'] * 6894.75729
        tempF = df['MeanTemp'].combine_first(df['Temp'])
        df['tempK'] = (tempF - 32) / 1.8 + 273.15
        
        df['mu'] = np.nan
        df['rho'] = np.nan
        df['Gamma'] = np.nan
        
        # Compute fluid properties by viscosity
        for fluid_key in fluid_properties.keys():
            idx = df['Viscosity_cSt'] == fluid_key
            if idx.any():
                mu, rho, gamma = self.compute_fluid_properties(df.loc[idx, 'tempK'], fluid_key)
                df.loc[idx, 'mu'] = mu
                df.loc[idx, 'rho'] = rho
                df.loc[idx, 'Gamma'] = gamma
        
        # Calculate dimensionless numbers
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
    
    def import_yin_data(self):
        mu_water = 0.001
        sigma_water = 0.0728
        rho_water = 997
        D_t_yin = 0.023
        D_upstream = 0.053
        theta_yin = 8
        L_yin = (53 - 23) / 2 / np.tan(np.radians(theta_yin)) / 1000
        
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
        
        return pd.DataFrame({
            'Re_upstream': Re_upstream, 'Re_t': Re_throat, 'D_v': d_v_m,
            'Velocity_m_per_s': V_throat, 'Ca': Ca, 'We': We,
            'ThroatDiameter_m': D_t_yin, 'DivergingL_m': L_yin
        })
    
    def import_sun_data(self):
        mu_water = 0.001
        sigma_water = 0.0728
        rho_water = 997
        D_t = 0.025
        D_upstream = 0.05
        theta = 7.5
        L = (50 - 25) / 2 / np.tan(np.radians(theta)) / 1000
        
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
        
        return pd.DataFrame({
            'Re': Re_water, 'D_v': d_v_m, 'Velocity_m_per_s': V_throat,
            'Ca': Ca, 'We': We, 'ThroatDiameter_m': D_t, 'DivergingL_m': L
        })

def main():
    root = tk.Tk()
    app = MultiviscosityAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main()