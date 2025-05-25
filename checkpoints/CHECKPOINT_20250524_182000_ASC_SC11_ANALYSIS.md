# CHECKPOINT: ASC Analysis with Actual Social Class (sc11) - 2025-05-24 18:20:00

## Summary
Successfully completed correlation analysis between AI-generated social class ratings and actual social class scores (sc11) for the ASC dataset.

## Key Findings

### 1. Dataset Information
- **ASC Essays**: 9,513 essays with essay IDs (TID)
- **Actual Social Class**: Variable `sc11` from external dataset
  - Location: `/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv`
  - Scale: 1-5 (1=lowest, 5=highest social class)
  - Distribution: Heavily skewed toward middle class (61.2% in class 3)

### 2. AI Performance Metrics

#### Correlation with Actual Social Class (sc11)
- **Standard Prompt**: r = 0.239 (p < 0.001)
- **Human-Style Prompt**: r = 0.242 (p < 0.001)
- **Average of Both Prompts**: r = 0.251 (p < 0.001)

#### Inter-prompt Agreement
- **Between Two AI Prompts**: r = 0.836 (p < 0.001)
- **Exact Agreement**: 65.5%
- **Within ±1 point**: 89.6%

### 3. Mean AI Ratings by Actual Social Class
```
Actual Class | Mean AI Rating | Std Dev | N
-------------|----------------|---------|------
1 (lowest)   | 3.81          | 1.18    | 863
2            | 4.07          | 1.25    | 1,162
3            | 4.34          | 1.34    | 5,825
4            | 4.94          | 1.54    | 1,241
5 (highest)  | 5.45          | 1.46    | 422
```

### 4. Key Insights
- AI ratings show clear monotonic increase with actual social class
- Moderate but significant correlation (~0.25) indicates AI captures approximately 6% of variance
- High inter-prompt agreement (0.836) suggests consistent measurement
- Both prompts perform similarly in predicting actual social class

## Files Created/Modified

### Analysis Scripts
- `scripts/analyze_asc_correlation.py` - Inter-prompt correlation analysis
- `scripts/analyze_asc_with_sc11.py` - Initial search for sc11 data
- `scripts/analyze_asc_vs_ai_correlation.py` - Main correlation analysis with sc11

### Output Files
- `asc_analysis_2prompts/asc_correlation_analysis.png` - Inter-prompt correlation visualization
- `asc_analysis_2prompts/asc_correlation_stats.csv` - Inter-prompt statistics
- `asc_analysis_2prompts/ai_vs_actual_correlation.png` - AI vs actual social class visualization
- `asc_analysis_2prompts/ai_vs_actual_correlations.csv` - Correlation results summary

### Data Files Used
- Original essays: `data/asc_9513_essays.csv`
- AI ratings: `asc_analysis_2prompts/run_20250524_162055/results_*.csv`
- Actual social class: External drive `/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv`

## Technical Notes
- All 9,513 essays successfully matched between datasets
- No missing values in correlations
- Used both Pearson and Spearman correlations for robustness
- Visualizations include scatter plots, regression lines, and box plots

## Implications
1. **Validity**: Moderate correlation with actual social class provides evidence of construct validity
2. **Reliability**: High inter-prompt agreement demonstrates measurement consistency
3. **Practical Use**: AI ratings can serve as a proxy for social class in large-scale text analysis
4. **Limitations**: ~25% correlation suggests AI captures social class signals but not complete picture

## Next Steps
- Compare with human annotator performance on same dataset
- Investigate which essay features drive AI predictions
- Test on additional datasets with known social class
- Explore ensemble methods to improve prediction accuracy