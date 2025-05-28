# UMAP Visualization v16 - The New Gold Standard

**Date:** 2025-05-27
**Status:** COMPLETE - This is now the definitive UMAP visualization

## Executive Summary

Version 16 represents the culmination of extensive iterative development and is now the recommended standard for all DML/PC analysis workflows. It combines the best features from previous versions with significant UI/UX improvements.

## Why v16 is the New Standard

1. **Best Balance**: Optimal information density without clutter
2. **Comprehensive Analytics**: Built-in statistical tools for deep analysis
3. **Professional Interface**: Clean design suitable for research presentations
4. **Production Ready**: Stable, performant, and thoroughly tested
5. **Highly Customizable**: Flexible without overwhelming users

## Key Features

### Core Improvements
- Draggable essay display with resize handle (80px-80vh range)
- Minimize/maximize button directly on essay panel
- Default essay height: 30vh (optimized readability)
- Essay background opacity: 25% default (better visibility)
- All text rendered in white for consistency

### Font System
- Essay font: User-configurable (default 24px)
- PC info: 70% of essay font size
- Headers: 120% of essay font size
- Applied uniformly to ALL text elements

### Advanced Analytics
- DML table with percentage reduction calculations
- PC analysis dropdown with:
  - Average/median importance rankings
  - SHAP value ranges and standard deviations
  - Pearson correlation coefficients
- Navigation arrows (◀ ▶) for 200 PCs
- Text inputs for precise percentile control

### Technical Excellence
- Based on v14's efficient architecture
- Cross-fitted DML metrics fully integrated
- GPU-optimized rendering
- Smooth transitions and interactions

## File Locations

- Script: `scripts/create_minimal_umap_viz_v16.py`
- Output: `nvembed_dml_pc_analysis/minimal_umap_viz_v16.html`
- Size: ~140MB (includes all 9513 essays embedded)

## Usage

```bash
cd scripts
python3 create_minimal_umap_viz_v16.py
```

## Note on File Size

The HTML files are too large for GitHub due to embedded essay data. Only the Python scripts are version controlled. Generate the visualization locally using the script above.

## Migration from Previous Versions

If you're using v14 or earlier, v16 is a drop-in replacement with additional features. No changes to data pipeline required.

---

This version represents the state-of-the-art in UMAP visualizations for DML/PC analysis and should be used for all future work.