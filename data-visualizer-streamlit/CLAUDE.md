# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the multiviscosity bubble analysis visualization system.

## Repository Overview

This directory contains a specialized Streamlit application for comprehensive analysis of bubble size distribution data across multiple fluid viscosities. The system processes experimental data from SAM2 segmentation results and provides advanced dimensional analysis capabilities.

## Core Application

### **`plotting_app_multiviscosity.py`** - Main Analysis Dashboard

A comprehensive Streamlit application with two main pages:

#### **Page 1: Filter Data**
- **Data Loading**: Automatically scans hierarchical experiment directories for two viscosity levels (10 cSt, 50 cSt)
- **Fluid Properties**: Temperature-dependent viscosity, density, and surface tension calculations using Andrade equation
- **Interactive Filtering**: Multi-column checkboxes for viscosity, angle, temperature, aeration, and flow rate selection
- **Export Capability**: CSV download of filtered datasets

#### **Page 2: Plot Results** - 11 Analysis Tabs

1. **Repeatability**: Trial-to-trial consistency analysis with R² correlation plots
2. **Flow Rate**: Trial-averaged bubble diameter vs flow rate by viscosity/temperature
3. **Temperature**: Experimental temperature effects on bubble properties  
4. **Angle**: Venturi angle influence on bubble formation
5. **Reynolds**: Reynolds number scaling with optional external data integration
6. **Weber**: Weber number analysis with power-law fitting
7. **Capillary**: Capillary number scaling analysis
8. **Universal**: Collapsed scaling using Re^a·We^b correlation
9. **PDFs Fixed Flow**: Log-normal probability density comparisons at selected flow rates
10. **Universal Capillary**: Re^a·Ca^b collapsed scaling
11. **Capillary w external**: Combined analysis with Yin et al. (2015) and Sun et al. (2017) literature data

## Data Structure Requirements

The application expects a specific hierarchical directory structure:

```
{base_dir}/
├── 10 cSt/                          # Low viscosity experiments
│   └── {angle} Degree/
│       └── {temp}F/
│           └── {aeration}_{percent} Percent Trial {trial}/
│               └── {flow_rate}/
│                   ├── experiment_summary.csv  # SAM2 results (log_mu, log_sigma, d32, dv)
│                   └── labview.txt             # Experimental conditions (Temp, Flow, P1, P2)
└── 50 cSt/                          # High viscosity experiments
    └── [same structure as 10 cSt]
```

## Key Constants and Parameters

### Physical Constants
- `UM_PER_PIXEL = 5.71` - Pixel to micron conversion factor
- `D_t = 6e-3` - Throat diameter (m) 
- `D_p = 15.8e-3` - Pipe diameter (m)
- `T_ref = 298.15` - Reference temperature (K)
- `GPM_to_m3_s = 1/(264.172053 * 60)` - Flow rate conversion

### Fluid Properties Dictionary
Temperature-dependent correlations for each fluid:
```python
fluid_properties = {
    "10 cSt": {"A": 687, "mu_ref": 10, "alpha": 0.00105, "rho_ref": 934, 
               "surfaceTensionYIntercept": 21.6, "surfaceTensionSlope": -0.06},
    "20 cSt": {"A": 752, "mu_ref": 19.7, "alpha": 0.00103, "rho_ref": 949,
               "surfaceTensionYIntercept": 22.1, "surfaceTensionSlope": -0.06},
    "50 cSt": {"A": 732, "mu_ref": 45, "alpha": 0.000994, "rho_ref": 959,
               "surfaceTensionYIntercept": 22.3, "surfaceTensionSlope": -0.06}
}
```

## Computed Variables

The system automatically calculates:

### **Fluid Properties**
- `mu`: Dynamic viscosity (Pa·s) using Andrade equation: μ = 10^(A/T + B)
- `rho`: Density (kg/m³) with thermal expansion: ρ = ρ_ref(1 - α(T - T_ref))
- `Gamma`: Surface tension (N/m) with temperature dependence
- `nu`: Kinematic viscosity (m²/s)

### **Flow Dynamics**
- `V_throat`: Throat velocity (m/s)
- `Reynolds`: Reynolds number (ρVD_t/μ)
- `dynamicPressure`: 0.5·ρ·V²
- `deltaP_normalized`: Pressure drop normalized by dynamic pressure

### **Dimensionless Groups**
- `Ca`: Capillary number (μV/γ)
- `We_D`: Weber number based on throat diameter (ρV²D_t/γ)  
- `We_L`: Weber number based on expansion length (ρV²L/γ)
- `L`: Expansion length calculated from geometry

### **Bubble Statistics**
- `D32`: Sauter mean diameter (μm)
- `D_v`: Volume-weighted mean diameter d₃₀ (μm)
- `LogMu`: Log-normal distribution location parameter
- `LogSigma`: Log-normal distribution scale parameter

## Analysis Features

### **Statistical Analysis**
- Trial averaging across duplicate experiments
- R² correlation analysis for repeatability assessment
- Log-normal distribution fitting and PDF generation
- Power-law correlation fitting with scipy.optimize.curve_fit

### **Dimensional Analysis** 
- Multi-parameter scaling relationships: d/D = A·Re^a·We^b·Ca^c
- Collapsed coordinate plotting for universal correlations
- External literature data integration (Yin et al. 2015, Sun et al. 2017)

### **Visualization**
- Consistent color scheme: 10 cSt (red), 50 cSt (black)
- Publication-ready formatting with Times New Roman fonts
- Interactive parameter selection and real-time plotting
- Exportable matplotlib figures

## Usage Instructions

### Running the Application
```bash
streamlit run plotting_app_multiviscosity.py
```

### Workflow
1. **Filter Data Page**: Set base directory path and select experimental conditions
2. **Plot Results Page**: Navigate through analysis tabs for different scaling relationships
3. **Export**: Download filtered data as CSV or save plots using Streamlit interface

### Directory Configuration
Update the default base directory in the text input field:
```python
base_dir = st.text_input("Base directory", value=r"G:\My Drive\Master's Data Processing\Both Viscosities")
```

## Data Integration

### **External Literature Data**
- **Yin et al. (2015)**: Water-based experiments with different geometry (D_t=23mm, θ=8°)
- **Sun et al. (2017)**: Water-based experiments (D_t=25mm, θ=7.5°)
- Automatic Reynolds, Weber, and Capillary number calculations for comparison

### **Internal Data Processing**
- Automatic trial averaging for duplicate experiments (Trial 1 & Trial 2)
- Missing data handling with pandas dropna() operations
- Error handling for malformed CSV files and directory structures

## Technical Notes

### **Performance Optimization**
- `@st.cache_data` decorators for expensive data loading operations
- Efficient pandas groupby operations for trial averaging
- Vectorized numpy operations for fluid property calculations

### **Error Handling**
- Debug mode toggle for detailed error reporting
- Graceful handling of missing files and directories
- Input validation for directory paths and data completeness

This system provides a comprehensive platform for analyzing bubble formation scaling relationships across different fluid viscosities, with integrated literature comparison and publication-ready visualization capabilities.