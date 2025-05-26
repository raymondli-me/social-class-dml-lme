# CHECKPOINT: Use Human MacArthur Ratings Only
**Date:** 2025-05-26
**Time:** Current session

## Critical Methodology Change
- **Previous approach:** Averaged two AI rating types (human MacArthur + ladder standard)
- **New approach:** Use ONLY human MacArthur ratings
- **Reason:** User specified to pick the human MacArthur rating type, not ladder standard

## Implementation Details
- In `prompts_100_complete.csv`, ratings are structured as:
  - Columns include variations of prompts with different rating types
  - We need to select only prompts that use "human macarthur" rating type
  - Do NOT average with "ladder standard" ratings

## Files Affected
- `scripts/nvembed_full_dims_analysis.py` - Updated to use single rating type
- All future analyses should follow this approach

## Key Code Change
```python
# OLD: Average all AI ratings
# ai_ratings_mean = ai_ratings_df.mean(axis=1)

# NEW: Use only human MacArthur ratings
human_macarthur_cols = [col for col in ai_ratings_df.columns 
                       if 'human macarthur' in col.lower()]
ai_ratings_mean = ai_ratings_df[human_macarthur_cols].mean(axis=1)
```