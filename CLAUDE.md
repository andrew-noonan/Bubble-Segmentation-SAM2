# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains tools for bubble segmentation analysis using SAM2 (Segment Anything Model 2), along with comprehensive data visualization capabilities for experimental bubble size distribution analysis. The codebase is split into two main components:

1. **Core SAM2 utilities** (`src/`) - Helper functions for image segmentation, analysis, and visualization
2. **Streamlit data visualizers** (`data-visualizer-streamlit/`) - Interactive web applications for experimental data analysis and comparison

## Core Architecture

### Main Modules (`src/`)

- **`analysis.py`** - Core statistical analysis functions for bubble properties
  - `compute_props()` - Calculate bubble properties (diameter, circularity, centroids) from segmentation masks
  - `summarize_props()` - Generate statistical summaries (D32, Dv, log-normal parameters) with circularity filtering  
  - `plot_diameter_histogram_from_summary()` - Generate publication-ready histograms from CSV data

- **`segmentation.py`** - Image processing and segmentation pipeline
  - `sobel_edge()` - Edge detection using Sobel filters
  - `generate_boxes_and_points()` - Generate bounding boxes and prompt points from edges
  - `multi_scale_box_masks()` - Multi-scale SAM2 prediction with different padding ratios
  - `watershed_split()` - Split overlapping regions using watershed algorithm
  - `filter_contained_masks()` - Remove masks contained within others based on overlap threshold

- **`visualization.py`** - Plotting and visualization utilities
  - `plot_detected_circles()` - Side-by-side comparison of original and annotated images
  - `plot_mask_stages()` - Visualize mask processing pipeline stages
  - `visualize_prompts_on_image()` - Overlay bounding boxes and prompt points on images

- **`app.py`** - Main Streamlit application for browsing experimental results
  - Directory scanning for hierarchical experiment folders
  - Frame-by-frame bubble overlay comparison (SAM2 vs MATLAB)
  - Histogram generation and R² analysis

### Data Visualization Suite (`data-visualizer-streamlit/`)

- **`plotting_app.py`** - Comprehensive analysis dashboard with:
  - Multi-parameter data loading (LabVIEW CSV, MATLAB results, SAM2 outputs)
  - Statistical analysis (repeatability, SAM vs MATLAB comparison)
  - Dimensional analysis (flow rate, temperature, angle effects)
  - Non-dimensional analysis (Reynolds, Weber, Capillary numbers)
  - External literature data integration (Yin et al., Sun et al.)

- **`plotting_app_multiviscosity.py`** - Extended version supporting multiple fluid viscosities
  - Fluid property calculations based on temperature-dependent correlations
  - Trial-averaged analysis across viscosity conditions
  - Collapsed scaling plots using dimensionless groups
  - Publication-ready PDF comparisons

- **`visualizer_app.py`** / **`visualizer_recirculation.py`** - Frame-by-frame experiment viewers
  - Interactive frame browsing with slider controls
  - Bubble overlay comparison (SAM2 vs MATLAB results)  
  - GIF export functionality
  - Summary statistics comparison tables

- **`histograms_non_streamlit.py`** - Standalone publication figure generation
  - High-resolution diameter distribution plots
  - Log-normal fitting with R² analysis
  - Customizable styling for publication

- **`utils.py`** - Minimal utility (appears to be empty placeholder)

## Key Data Flow

1. **Raw Images** → Sobel edge detection → Bounding box generation
2. **SAM2 Prediction** → Multi-scale processing → Watershed splitting → Containment filtering  
3. **Bubble Properties** → Statistical analysis → Log-normal fitting
4. **Visualization** → Interactive dashboards → Publication figures

## Common Development Tasks

### Running Streamlit Applications

```bash
# Main plotting dashboard
streamlit run data-visualizer-streamlit/plotting_app.py

# Multi-viscosity analysis  
streamlit run data-visualizer-streamlit/plotting_app_multiviscosity.py

# Frame viewer for specific experiment types
streamlit run data-visualizer-streamlit/visualizer_app.py
streamlit run data-visualizer-streamlit/visualizer_recirculation.py

# Main app for experiment browsing
streamlit run src/app.py
```

### Dependencies

The project requires two main dependency sets:
- **Core requirements** (`requirements.txt`): Basic SAM2 and analysis dependencies
- **Streamlit requirements** (`data-visualizer-streamlit/requirements.txt`): Additional visualization libraries

Install with:
```bash
pip install -r requirements.txt
pip install -r data-visualizer-streamlit/requirements.txt
```

## Data Structure Expectations

The codebase expects a hierarchical experimental data structure:
```
{base_dir}/
├── {angle} Degree/
│   ├── {temp}F/
│   │   ├── {aeration}_{percent} Percent Trial {trial}/
│   │   │   ├── {flow_rate}/
│   │   │   │   ├── experiment_summary.csv (SAM2 results)
│   │   │   │   ├── per_frame_props.json (per-frame bubble data)
│   │   │   │   ├── labview.txt (experimental conditions)
│   │   │   │   ├── 3 - Normalized/ (image frames)
│   │   │   │   └── MATLAB Results/ (comparison data)
```

For multi-viscosity experiments, an additional viscosity level is added:
```
{base_dir}/{viscosity} cSt/{angle} Degree/...
```

## Important Constants

- `UM_PER_PIXEL = 5.71` - Pixel to micron conversion factor
- `D_t = 6e-3` - Throat diameter (m)
- `CIRCULARITY_THRESH = 0.6` - Default circularity threshold for bubble filtering

## Testing and Validation

The codebase includes built-in validation through:
- Repeatability analysis comparing duplicate trials
- SAM2 vs MATLAB result comparison 
- R² analysis of log-normal fits
- Statistical summary validation

## Publication Figure Generation

For generating publication-quality figures, use:
- `histograms_non_streamlit.py` for standalone diameter distributions
- `plotting_app_multiviscosity.py` tabs for dimensional and non-dimensional analyses
- Built-in matplotlib styling with Times New Roman fonts and proper axis formatting