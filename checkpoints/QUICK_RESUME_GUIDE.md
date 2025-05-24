# Quick Resume Guide

## Where We Left Off
- Created blinded dataset (526 essays)
- Prepared scripts for ACTUAL vLLM processing
- Completed simulation (low correlation: 0.36)
- Ready to run REAL model inference

## To Resume:

### 1. Run Actual vLLM (30-60 mins)
```bash
cd /media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/analysis_20250523_full_526
python3 scripts/run_actual_vllm.py
```

### 2. Parse Results (2 mins)
```bash
python3 scripts/parse_vllm_results.py
```

### 3. Check Results
- Should see correlation ~0.8+ (not 0.36)
- Check: `cat reports/vllm_actual/*.txt | grep correlation`

## What Happens:
- 526 essays × 50 prompts = 26,300 REAL model calls
- Qwen-32B analyzes each essay text
- Rates social class 1-10 for each prompt
- BLIND to human judgments

## Next: 
- Binary prompts (100 features)
- DML-LME analysis
- Final paper

**Model is blind to labels!** ✓