# vLLM Processing Checkpoint
**Date:** 2025-05-24 07:35:00  
**Project:** Social Class Detection - Actual vLLM Inference  
**Current Status:** Working on vLLM model inference with compilation error workarounds

---

## Executive Summary

We're attempting to run actual vLLM inference on 526 essays with 50 prompts (26,300 total ratings). The model (Qwen-32B-AWQ) loads successfully but encounters compilation errors due to missing Python.h headers. We've implemented error suppression which allows the model to run in eager mode, bypassing the compilation issues.

---

## Current Situation

### ✅ What's Working
1. **Model Loading**: Qwen-32B-AWQ loads successfully across 2 GPUs
2. **Error Suppression**: Torch dynamo error suppression allows bypassing compilation
3. **Inference**: Model can generate responses when compilation errors are suppressed
4. **Data Preparation**: All 526 essays are properly blinded and ready

### ❌ Current Issues
1. **Missing Python Headers**: System lacks python3-dev package (requires sudo to install)
2. **Triton Compilation**: Cannot compile Triton CUDA kernels without Python.h
3. **Progress Bar Conflicts**: tqdm conflicts with nohup/background execution

### 🔧 Workarounds Applied
1. Set `torch._dynamo.config.suppress_errors = True`
2. Disabled tqdm progress bars for background compatibility
3. Added error handling for batch failures
4. Using eager execution mode instead of compiled mode

---

## Technical Details

### Error Encountered
```
fatal error: Python.h: No such file or directory
CalledProcessError: Command '['/usr/bin/gcc', '/tmp/xxx/main.c'...]' returned non-zero exit status 1.
```

### Model Configuration
- **Model**: Qwen/Qwen2.5-32B-Instruct-AWQ
- **GPUs**: 2 (tensor_parallel_size=2)
- **Memory**: ~9GB per GPU
- **Batch Size**: 50 essays per batch
- **Temperature**: 0.1 (for consistency)

---

## How to Run Manually

### Option 1: Install Python Headers (Recommended if you have sudo)
```bash
# Install the missing headers
sudo apt-get update
sudo apt-get install python3-dev

# Then run the original script
cd /media/raymondli/Crucial\ X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/analysis_20250523_full_526
python3 scripts/run_actual_vllm.py
```

### Option 2: Run with Error Suppression (Current Approach)
```bash
# Navigate to the analysis directory
cd /media/raymondli/Crucial\ X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/analysis_20250523_full_526

# Run the suppression script directly (foreground)
python3 scripts/run_vllm_with_suppression.py

# OR run in background with logging
nohup python3 -u scripts/run_vllm_with_suppression.py > vllm_output.log 2>&1 &

# Monitor progress
tail -f vllm_output.log
```

### Option 3: Simple Test Script
```bash
# Test if vLLM works at all
cd /media/raymondli/Crucial\ X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/analysis_20250523_full_526
python3 scripts/vllm_suppress_errors.py
```

---

## Script Locations

### Main Scripts
1. **With Suppression** (Currently Working):
   ```
   scripts/run_vllm_with_suppression.py
   ```
   - Implements error suppression
   - Disables tqdm for background compatibility
   - Saves results after each prompt

2. **Original Script**:
   ```
   scripts/run_actual_vllm.py
   ```
   - Requires python3-dev headers
   - Uses vllm-batch-processor

3. **Test Script**:
   ```
   scripts/vllm_suppress_errors.py
   ```
   - Simple test to verify vLLM works

### Data Files
- **Blinded Essays**: `data/essays_blinded_526.csv`
- **Prompts**: `data/ladder_variations_50_complete.csv`
- **Hidden Labels**: `data/labels_hidden_526.csv`

### Output Location
```
outputs/vllm_actual_suppressed/
├── results_[prompt_name]_[timestamp].csv  # Per-prompt results
├── checkpoint_[N]_prompts.csv             # Checkpoints every 10 prompts
└── all_results_526x50_[timestamp].csv     # Final combined results
```

---

## Expected Runtime

- **Per Prompt**: ~2-3 minutes (526 essays)
- **Total Time**: ~100-150 minutes for all 50 prompts
- **GPU Memory**: ~18GB total (9GB per GPU)
- **Output Size**: ~3-5MB of CSV data

---

## Next Steps After Completion

1. **Parse Results**:
   ```bash
   python3 scripts/parse_vllm_results.py
   ```

2. **Check Correlation**:
   - Should achieve ~0.8+ correlation with human judgments
   - Current simulation only achieved 0.36

3. **Process Binary Prompts**:
   - Run similar process with `prompts_100_complete.csv`
   - Generate binary features for DML analysis

4. **Run DML-LME Analysis**:
   - Combine continuous ratings and binary features
   - Estimate causal effect of education on perceived social class

---

## Troubleshooting

### If Script Fails
1. Check GPU memory: `nvidia-smi`
2. Check disk space: `df -h`
3. Verify model cache: `ls ~/.cache/huggingface/hub/models--Qwen*`
4. Try smaller batch size (change `batch_size = 50` to `batch_size = 10`)

### If Correlation is Low
1. Check raw responses for parsing errors
2. Verify prompt formatting
3. Compare a few manual ratings to ensure model understands task

### If GPU OOM
1. Reduce batch size
2. Use single GPU (set `tensor_parallel_size=1`)
3. Try smaller model (qwen-7b preset)

---

## Manual Command Summary

```bash
# Quick start (with error suppression)
cd /media/raymondli/Crucial\ X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/analysis_20250523_full_526
python3 scripts/run_vllm_with_suppression.py

# Check results
ls -la outputs/vllm_actual_suppressed/

# View a sample result
head outputs/vllm_actual_suppressed/results_*.csv
```

---

**Status**: Ready to run. Model works with error suppression. Just needs ~2 hours to complete all 26,300 inferences.