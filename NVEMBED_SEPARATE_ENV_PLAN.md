# NV-Embed-v2 Separate Environment Plan

## Overview
Create an isolated environment with the exact dependencies needed for NV-Embed-v2, generate embeddings, and import them back to the main analysis environment.

## Steps to Execute

### 1. Create Separate Directory Structure
```bash
mkdir -p nvembed_isolated/
cd nvembed_isolated/
```

### 2. Create Virtual Environment
```bash
python3 -m venv nvembed_env
source nvembed_env/bin/activate
```

### 3. Install Specific Dependencies
```bash
pip install torch==2.2.0
pip install transformers==4.42.4
pip install sentence-transformers==2.7.0
pip install numpy pandas
```

### 4. Create Embedding Generation Script
Create a minimal script that:
- Loads the 9,513 essays
- Generates NV-Embed-v2 embeddings
- Saves them to a numpy file
- No analysis, just embedding generation

### 5. Run Embedding Generation
```bash
python generate_nvembed_only.py
```

### 6. Copy Embeddings Back
```bash
cp nvembed_embeddings.npy ../nvembed_checkpoints/
```

### 7. Return to Main Environment
```bash
deactivate
cd ..
```

### 8. Run Analysis with Pre-generated Embeddings
Use the main environment to load the embeddings and run the full DML analysis.

## File Structure
```
2025_05_23_social_class_dml_lme/
├── nvembed_isolated/          # Separate environment
│   ├── nvembed_env/          # Virtual environment
│   ├── generate_nvembed_only.py
│   └── nvembed_embeddings.npy
├── nvembed_checkpoints/       # Main analysis
│   └── nvembed_embeddings.npy # Copied from isolated env
└── scripts/
    └── analyze_nvembed_precomputed.py
```

## Benefits
1. No dependency conflicts
2. Clean separation of concerns
3. Reusable embeddings
4. Can delete isolated environment after generation

## Estimated Time
- Environment setup: 5 minutes
- Embedding generation: 20-30 minutes
- Analysis: 5 minutes
- Total: ~40 minutes