# CHECKPOINT: Full ASC Dataset Added
**Date:** 2025-05-24 12:50:00  
**Status:** Dataset ready for expanded analysis

## New Dataset Details
- **File:** `data/asc_9513_essays.csv`
- **Size:** 9,513 essays (18x larger than current 526)
- **Format:** TID, original (essay text)
- **Source:** ASC (Avon Siblings Cohort) study
- **Age:** All authors were 25 years old when writing

## Key Difference from Current Dataset
- Current (526 essays): Has human social class ratings
- New (9,513 essays): No human ratings, but actual social class data available elsewhere

## Next Steps
1. Run vLLM inference on all 9,513 essays with best-performing prompts
2. Compare AI ratings with actual social class measures
3. Scale up DML analysis with much larger sample size
4. Increased statistical power for causal inference

## Location
- Local: `/media/raymondli/Crucial X9/asc_essays/asc_9513_essays.csv`
- GitHub: `data/asc_9513_essays.csv`

This enables analysis at scale with ~18x more data points!