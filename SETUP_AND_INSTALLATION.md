# Complete Setup and Installation Guide
**Date:** 2025-05-23 16:20:00  
**Purpose:** Reproducible environment setup for social class DML-LME analysis

## System Information
- **OS:** Linux 6.12.10-76061203-generic
- **Python:** Python 3.10.12 at /usr/bin/python3
- **Working Directory:** /media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme

## Prerequisites
1. Python 3.10+ installed
2. Access to vllm-batch-processor repository
3. GPU with sufficient memory for 32B model (recommended: 40GB+ VRAM)

## Step 1: Create Virtual Environment
```bash
cd "/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme"
python3 -m venv venv
source venv/bin/activate
```

## Step 2: Install Core Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install analysis packages
pip install pandas==2.0.3
pip install numpy==1.24.4
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install scikit-learn==1.3.2
pip install tqdm==4.66.1
pip install jupyter==1.0.0
pip install statsmodels==0.14.0
pip install scipy==1.11.4
pip install plotly==5.18.0
pip install tabulate==0.9.0
pip install jinja2==3.1.2
```

## Step 3: Install vLLM Batch Processor
```bash
# Install vllm-batch-processor from local path
pip install -e "/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/vllm-batch-processor"

# This will also install vLLM and its dependencies
```

## Step 4: Install DML-LME (when available)
```bash
# Currently not available, will use fallback regression
# pip install -e "/path/to/dml-lme"
```

## Step 5: Verify Installation
```bash
# Test imports
python3 -c "import pandas, numpy, matplotlib, seaborn, sklearn, vllm_batch_processor; print('All packages imported successfully')"

# Check vLLM installation
python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```

## Step 6: Environment Variables (if needed)
```bash
# Set CUDA visible devices if multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Set Hugging Face cache directory if needed
export HF_HOME="/path/to/model/cache"
```

## Package Versions for Reproducibility
Save this as requirements_exact.txt:
```
pandas==2.0.3
numpy==1.24.4
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.2
tqdm==4.66.1
jupyter==1.0.0
statsmodels==0.14.0
scipy==1.11.4
plotly==5.18.0
tabulate==0.9.0
jinja2==3.1.2
```

## Directory Structure Verification
```
2025_05_23_social_class_dml_lme/
├── data/
│   ├── essay_dataset.csv (526 essays)
│   ├── prompts_100_complete.csv (100 binary prompts)
│   └── ladder_variations_50_complete.csv (50 continuous prompts)
├── scripts/
│   ├── test_10_samples.py
│   ├── test_ladder_variations.py
│   └── full_analysis.py
├── outputs/
│   ├── test/
│   ├── ladder_test/
│   └── full/
├── visualizations/
│   └── ladder_test/
├── reports/
│   ├── METHODS_ladder_variations_test_20250523_160900.txt
│   └── INTERPRETATION_ladder_variations_20250523_161500.md
├── requirements.txt
├── SETUP_AND_INSTALLATION.md (this file)
└── README.md
```

## Running Tests
After setup, run tests in this order:
1. `python scripts/test_ladder_variations.py` - Test 50 social class prompts on 10 essays
2. `python scripts/test_10_samples.py` - Test binary prompts on 10 essays
3. `python scripts/full_analysis.py` - Full analysis on 526 essays

## Troubleshooting
1. **Import errors**: Ensure virtual environment is activated
2. **CUDA errors**: Check GPU availability with `nvidia-smi`
3. **Memory errors**: Reduce batch_size in scripts
4. **Model download**: Ensure internet connection for first run

## Reproducibility Checklist
- [ ] Python 3.10.12 installed
- [ ] Virtual environment created
- [ ] All packages installed with exact versions
- [ ] vllm-batch-processor accessible
- [ ] Data files present in data/
- [ ] GPU available with sufficient memory
- [ ] Scripts have execute permissions

---
*This setup guide ensures complete reproducibility of the analysis environment.*