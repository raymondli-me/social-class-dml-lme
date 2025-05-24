# Quick Start: Run vLLM Processing Manually

## The One Command You Need

```bash
cd /media/raymondli/Crucial\ X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/analysis_20250523_full_526 && python3 scripts/run_vllm_with_suppression.py
```

**That's it!** This will process all 526 essays with 50 prompts (~2-3 hours).

---

## What This Does
- Runs Qwen-32B model on your 526 essays
- Generates social class ratings (1-10) for each essay using 50 different prompts
- Saves results to `outputs/vllm_actual_suppressed/`
- Creates 26,300 total ratings

## Monitor Progress
In another terminal:
```bash
# Watch the output
tail -f /media/raymondli/Crucial\ X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/analysis_20250523_full_526/vllm_final_processing.log

# Check GPU usage
watch nvidia-smi
```

## Check Results
```bash
# See generated files
ls -la /media/raymondli/Crucial\ X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/analysis_20250523_full_526/outputs/vllm_actual_suppressed/

# View sample results
head /media/raymondli/Crucial\ X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/analysis_20250523_full_526/outputs/vllm_actual_suppressed/results_*.csv
```

## Run in Background (Optional)
If you want to close your terminal:
```bash
cd /media/raymondli/Crucial\ X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/analysis_20250523_full_526
nohup python3 -u scripts/run_vllm_with_suppression.py > vllm_processing.log 2>&1 &

# Get the process ID
echo $!

# Check if it's running
ps aux | grep run_vllm_with_suppression
```

---

**Expected Runtime**: 2-3 hours  
**GPU Memory**: Will use ~18GB across GPUs 0 and 1  
**Output**: CSV files with ratings in `outputs/vllm_actual_suppressed/`