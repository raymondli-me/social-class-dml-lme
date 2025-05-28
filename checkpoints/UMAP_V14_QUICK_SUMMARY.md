# UMAP v14 Quick Summary

## What We Built
Interactive 3D visualization showing how social class affects AI essay ratings, with text features as mediators.

## Key Results
- **Naive effect**: SC→AI θ=0.423 (R²=0.058) - raw correlation without text
- **DML with text**: θ=0.054 (cross-fitted) - true causal effect after controlling for text
- **Conclusion**: Text features explain ~87% of the SC→AI relationship

## Main Features
1. **Synchronized threshold controls** - percentiles and values update together
2. **Inline PC display** - Top 5 PCs with SHAP values in essay header  
3. **Simplified PC analysis** - Click PC → see probability impacts
4. **Gallery mode** - Navigate extreme cases systematically
5. **Adjustable display** - Font size, opacity, expand button (needs fix)

## Technical Stack
- NVEmbed (4096d) → PCA (200d) → Top 5 PCs → UMAP (3D)
- Double Machine Learning with 5-fold cross-fitting
- Three.js WebGL rendering
- Self-contained HTML (~15MB)

## To Run
```bash
cd /media/raymondli/Crucial\ X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme
python3 scripts/create_minimal_umap_viz_v14.py
# Output: nvembed_dml_pc_analysis/minimal_umap_viz_v14.html
```

## TODO
- Fix expand button positioning (currently overlaps)
- Consider moving to gallery controls or floating button