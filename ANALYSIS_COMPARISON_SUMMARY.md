# Analysis Comparison Summary

## 49-Prompt vs 50-Prompt Analysis

### Key Metrics Comparison

| Metric | 49 Prompts | 50 Prompts | Change |
|--------|------------|------------|---------|
| Overall Correlation | 0.987 | 0.990 | +0.003 |
| Mean Absolute Difference | 0.15 | 0.14 | -0.01 |
| Mean Within-Essay SD | 0.96 | 0.95 | -0.01 |
| Total Ratings | 490 | 500 | +10 |
| Best Prompt Correlation | 0.952 | 0.913 | -0.039 |

### New Prompt Performance
- **Added**: "peer_comparison" 
- **Expected**: High correlation based on design principles
- **Actual**: Ranked 50th/50 (r=0.710)
- **Lesson**: Theory doesn't always predict performance

### Top Performers (Both Analyses)
Consistent high performers:
- societal_position
- social_position  
- life_chances
- privilege_score
- community_status

### Key Insights
1. **Ensemble Robustness**: Even with a poor-performing addition, overall correlation improved
2. **Diminishing Returns**: 0.003 improvement suggests approaching optimal prompt count
3. **Unpredictability**: Well-designed prompts can still underperform
4. **Stability**: Core patterns remain consistent across analyses

### Recommendation
Use the 50-prompt set for maximum robustness, but consider weighted averaging to minimize impact of poor performers.