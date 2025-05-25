#!/usr/bin/env python3
"""
Analyze correlation between AI ratings and actual social class (sc11)
Note: Need to find the original ASC dataset with sc11 variable
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Searching for ASC dataset with social class variables...")

# Check what data files we have
data_dir = '/home/raymondli/social-class-dml-lme/data'
print(f"\nFiles in data directory:")
for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        print(f"  - {file}")
        # Check first few rows to see columns
        try:
            df_check = pd.read_csv(os.path.join(data_dir, file), nrows=1)
            print(f"    Columns: {list(df_check.columns)}")
        except:
            pass

# Note: The current ASC files only contain TID and essay text
# We need the original dataset with sc11 variable
print("\n⚠️  Current ASC files only contain essay text, not social class scores.")
print("Need to locate the original ASC dataset with 'sc11' variable.")

# Let's check if we can infer anything from the essay IDs
essays = pd.read_csv('/home/raymondli/social-class-dml-lme/data/asc_9513_essays.csv')
print(f"\nASC essays shape: {essays.shape}")
print(f"Sample essay IDs: {essays['TID'].head().tolist()}")

# Load our AI ratings
ladder_standard = pd.read_csv('/home/raymondli/social-class-dml-lme/asc_analysis_2prompts/run_20250524_162055/results_ladder_standard_improved_20250524_165833.csv')
human_ladder = pd.read_csv('/home/raymondli/social-class-dml-lme/asc_analysis_2prompts/run_20250524_162055/results_human_macarthur_ladder_improved_20250524_174149.csv')

print(f"\nAI ratings loaded:")
print(f"  - Standard prompt: {len(ladder_standard)} ratings")
print(f"  - Human-style prompt: {len(human_ladder)} ratings")

# Check if essay_id in our results matches TID format
print(f"\nSample AI rating IDs: {ladder_standard['essay_id'].head().tolist()}")
print("IDs match between datasets ✓" if ladder_standard['essay_id'].iloc[0] == essays['TID'].iloc[0] else "IDs don't match ✗")

print("\n" + "="*50)
print("CONCLUSION: Need the original ASC dataset file that contains:")
print("  - TID (essay identifier)")
print("  - sc11 (social class variable)")
print("  - Possibly other demographic variables")
print("\nWithout this file, we cannot correlate AI ratings with actual social class.")
print("="*50)