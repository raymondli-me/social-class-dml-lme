# NV-Embed-v2 Separate Environment Documentation

**Date:** May 26, 2025  
**Model:** nvidia/NV-Embed-v2  
**Status:** ✅ SUCCESSFUL with multi-GPU setup

## The Problem: Dependency Conflicts

NV-Embed-v2 required specific package versions that conflicted with our main environment:

```
Required: transformers==4.42.4, tokenizers==0.19.1
Main env: transformers==4.52.3, tokenizers==0.21.1
```

The newer transformers version was missing `MISTRAL_INPUTS_DOCSTRING` needed by NV-Embed-v2.

## The Solution: Temporary Environment Modification

### Step 1: Environment Preparation
```bash
# Working directory
cd "/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme"

# Created isolated folder
mkdir nvembed_isolated
cd nvembed_isolated
```

### Step 2: Package Downgrade
We downgraded packages in the main environment temporarily:

```bash
# Downgraded to compatible versions
pip install --user --force-reinstall transformers==4.42.4 tokenizers==0.19.1

# Added missing dependency
pip install accelerate
```

**⚠️ IMPORTANT:** This modified the main environment temporarily!

### Step 3: Hardware Configuration
- **GPUs:** 4x NVIDIA RTX 3090 (24GB each)
- **Model Distribution:** Automatic across GPUs using `device_map="auto"`
- **Memory:** Half precision (float16) to fit model
- **Batch Size:** 4 (conservative for memory)

Final device mapping:
```
GPU 0: latent_attention_model + layers 0-7
GPU 1: layers 8-20  
GPU 2: layers 21-31 + norm
GPU 3: (unused in final mapping)
```

## Scripts Created

### 1. `generate_nvembed_only.py` (Initial)
- Basic script with DataParallel
- **Failed:** CUDA OOM error

### 2. `generate_nvembed_fixed.py` (Second attempt)  
- Added `device_map="auto"` and `accelerate`
- **Failed:** Missing accelerate dependency

### 3. `generate_nvembed_final.py` (WORKING VERSION)
- ✅ Proper output format handling
- ✅ Multi-GPU distribution with accelerate  
- ✅ Robust error handling for different output keys
- ✅ Found correct key: "sentence_embeddings"

## Data Files Generated

### Location: `nvembed_isolated/`
```bash
nvembed_embeddings.npy       # 4,096-dim embeddings (9,513 x 4,096)
nvembed_essay_ids.npy        # Essay IDs for alignment
```

### Auto-copied to: `nvembed_checkpoints/`
```bash
../nvembed_checkpoints/nvembed_embeddings.npy
../nvembed_checkpoints/nvembed_essay_ids.npy
```

## Technical Specifications

### Model Details
- **Model:** nvidia/NV-Embed-v2 (7B parameters)
- **Embeddings:** 4,096 dimensions  
- **Context Length:** 2,048 tokens (reduced from 4,096 for memory)
- **Instruction:** "Instruct: Given a personal essay, analyze the social class indicators.\nQuery: "
- **Output Key:** "sentence_embeddings" ✅

### Performance
- **Processing Time:** ~20-30 minutes for 9,513 essays
- **Throughput:** ~300-500 essays/minute with 4 GPUs
- **File Size:** ~150MB for embeddings
- **Memory Usage:** ~22GB across 4 GPUs

## Environment States

### Current Environment (Modified)
```bash
transformers==4.42.4    # ⬇️ Downgraded from 4.52.3
tokenizers==0.19.1      # ⬇️ Downgraded from 0.21.1  
accelerate==1.7.0       # ➕ Added
torch==2.6.0           # ✅ Compatible
econml==0.15.1          # ✅ For DML analysis
```

### How to Restore Main Environment (If Needed)
```bash
# Restore to newer versions for main analysis
pip install --upgrade transformers tokenizers

# Verify restoration
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## Next Steps: Return to Main Analysis

### 1. Verify Data Availability
```bash
# Check embeddings exist
ls -la nvembed_checkpoints/nvembed_embeddings.npy
ls -la nvembed_checkpoints/nvembed_essay_ids.npy
```

### 2. Run DML Analysis  
Use existing analysis infrastructure:

```bash
# Create analysis script for pre-computed embeddings
python3 scripts/analyze_nvembed_precomputed.py
```

### 3. Expected Analysis Pipeline
1. ✅ Load `nvembed_embeddings.npy` (4,096-dim)
2. ✅ Reduce to 200 PCA components  
3. ✅ Run prediction evaluation (Linear + XGBoost)
4. ✅ Execute DML analysis (SC → AI ratings)
5. ✅ Compare with OpenAI and MPNet results

## Comparison Framework

### Results to Compare
| Model | Dims | Text→AI R² | Text→SC R² | Gap | DML θ | p-value |
|-------|------|------------|------------|-----|-------|---------|
| OpenAI | 3,072→200 | 0.923 | 0.537 | 38.6% | 0.0527 | <0.001 |
| MPNet | 768→200 | 0.451 | 0.050 | 40.2% | 0.0018 | 0.107 |
| **NV-Embed** | **4,096→200** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

Expected: NV-Embed should outperform both models significantly.

## Troubleshooting Record

### Issues Encountered & Solutions
1. **"MISTRAL_INPUTS_DOCSTRING" error** → Downgrade transformers
2. **"low_cpu_mem_usage requires Accelerate"** → Install accelerate  
3. **CUDA OOM with DataParallel** → Use device_map="auto"
4. **"dict has no last_hidden_state"** → Handle sentence_embeddings key

### Environment Dependencies
- ✅ transformers==4.42.4 (CRITICAL for NV-Embed-v2)
- ✅ accelerate (REQUIRED for device_map="auto")  
- ✅ torch (compatible with all versions tested)
- ✅ 16GB+ GPU memory per device

## Files to Keep/Delete

### Keep Permanently
```bash
nvembed_checkpoints/nvembed_embeddings.npy     # Main output
nvembed_checkpoints/nvembed_essay_ids.npy      # For verification
checkpoints/NVEMBED_SEPARATE_ENVIRONMENT_DOCUMENTATION.md  # This file
```

### Can Delete After Verification
```bash
nvembed_isolated/generate_nvembed_*.py          # Scripts
nvembed_isolated/nvembed_embeddings.npy        # Duplicate
nvembed_isolated/nvembed_essay_ids.npy         # Duplicate
```

## Success Metrics

✅ **Model Loaded:** 7B parameters across 4 GPUs  
✅ **Data Processed:** All 9,513 essays  
✅ **Output Format:** 4,096-dimensional embeddings  
✅ **Quality Check:** Used "sentence_embeddings" (correct key)  
✅ **File Size:** ~150MB (reasonable for 9513×4096 float32)  
✅ **Auto-copy:** Files saved to main checkpoint directory

## Contact & Restoration Info

- **Hardware:** 4x RTX 3090, CUDA 12.8, Driver 570.133.07
- **OS:** Ubuntu with Pop!_OS, Python 3.10
- **Main Analysis Environment:** `/2025_05_23_social_class_dml_lme/`
- **Isolated Environment:** `/nvembed_isolated/` (can be deleted)
- **Results Location:** `/nvembed_checkpoints/`

---

**✅ STATUS: COMPLETE - Ready for DML analysis with NV-Embed-v2 embeddings!**