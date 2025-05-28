#!/usr/bin/env python3
import pickle
from pathlib import Path

CHECKPOINT_DIR = Path('/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/nvembed_checkpoints')

with open(CHECKPOINT_DIR / 'nvembed_pca_200_features.pkl', 'rb') as f:
    pca_data = pickle.load(f)
    print("Keys in pca_data:", list(pca_data.keys()))
    for key in pca_data.keys():
        value = pca_data[key]
        if hasattr(value, 'shape'):
            print(f"{key}: shape = {value.shape}")
        elif hasattr(value, '__len__'):
            print(f"{key}: length = {len(value)}")
        else:
            print(f"{key}: type = {type(value)}")
            
    if 'explained_variance_ratio' in pca_data:
        print("\nexplained_variance_ratio sample:", pca_data['explained_variance_ratio'][:5] if hasattr(pca_data['explained_variance_ratio'], '__getitem__') else pca_data['explained_variance_ratio'])