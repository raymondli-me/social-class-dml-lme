#!/usr/bin/env python3
"""
Analyze correlation between AI ratings and actual social class (sc11/asc)
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Loading ASC dataset with actual social class scores...")

# Try to find the file with sc11 or asc variable
sc11_file = "/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv"

# If the file doesn't exist, search for it
if not os.path.exists(sc11_file):
    print(f"File not found at {sc11_file}, searching for alternatives...")
    # Search for files with 'asc' in the name
    import glob
    possible_files = []
    search_dirs = [
        "/media/raymondli/Crucial X9/asc_essays/",
        "/media/raymondli/Crucial X9/2025_03_24_NCDS_longitudinal_analysis/",
        "/media/raymondli/Crucial X9/2025_05_22_VLLM_BACKUP/",
        "/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/"
    ]
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            files = glob.glob(f"{search_dir}**/*asc*.csv", recursive=True)
            files.extend(glob.glob(f"{search_dir}**/*9513*.csv", recursive=True))
            possible_files.extend(files)
    
    # Check each file for sc11 or asc column
    for file in possible_files[:20]:  # Limit to first 20 files
        try:
            df_check = pd.read_csv(file, nrows=1)
            cols = df_check.columns.tolist()
            if 'sc11' in cols or 'asc' in cols or 'ASC' in cols:
                print(f"Found potential file: {file}")
                print(f"Columns: {cols}")
                sc11_file = file
                break
        except:
            pass

# Load the actual social class data
try:
    asc_data = pd.read_csv(sc11_file)
    print(f"Loaded file: {sc11_file}")
    print(f"Shape: {asc_data.shape}")
    print(f"Columns: {asc_data.columns.tolist()}")
    
    # Find the social class column
    sc_col = None
    for col in ['sc11', 'asc', 'ASC', 'social_class']:
        if col in asc_data.columns:
            sc_col = col
            break
    
    if sc_col is None:
        print("ERROR: No social class column found!")
        print("Available columns:", asc_data.columns.tolist())
        exit(1)
    
    print(f"\nUsing social class column: '{sc_col}'")
    
except Exception as e:
    print(f"ERROR loading file: {e}")
    exit(1)

# Load AI ratings
print("\nLoading AI ratings...")
ladder_standard = pd.read_csv('/home/raymondli/social-class-dml-lme/asc_analysis_2prompts/run_20250524_162055/results_ladder_standard_improved_20250524_165833.csv')
human_ladder = pd.read_csv('/home/raymondli/social-class-dml-lme/asc_analysis_2prompts/run_20250524_162055/results_human_macarthur_ladder_improved_20250524_174149.csv')

# Merge all data
print("\nMerging datasets...")
# First merge AI ratings
ai_merged = pd.merge(ladder_standard, human_ladder, on='essay_id', suffixes=('_standard', '_human'))

# Then merge with actual social class
# Need to match on TID/essay_id
full_data = pd.merge(ai_merged, asc_data, left_on='essay_id', right_on='TID', how='inner')
print(f"Merged data shape: {full_data.shape}")

# Basic statistics
print(f"\n=== ACTUAL SOCIAL CLASS ('{sc_col}') STATISTICS ===")
print(f"Mean: {full_data[sc_col].mean():.3f}")
print(f"Std Dev: {full_data[sc_col].std():.3f}")
print(f"Min: {full_data[sc_col].min()}")
print(f"Max: {full_data[sc_col].max()}")
print(f"Unique values: {sorted(full_data[sc_col].unique())}")
print(f"Value counts:\n{full_data[sc_col].value_counts().sort_index()}")

# Calculate correlations
print("\n=== CORRELATION ANALYSIS ===")
print("\nStandard Prompt vs Actual Social Class:")
pearson_r1, pearson_p1 = stats.pearsonr(full_data['rating_standard'], full_data[sc_col])
spearman_r1, spearman_p1 = stats.spearmanr(full_data['rating_standard'], full_data[sc_col])
print(f"Pearson r = {pearson_r1:.4f} (p = {pearson_p1:.2e})")
print(f"Spearman ρ = {spearman_r1:.4f} (p = {spearman_p1:.2e})")

print("\nHuman-Style Prompt vs Actual Social Class:")
pearson_r2, pearson_p2 = stats.pearsonr(full_data['rating_human'], full_data[sc_col])
spearman_r2, spearman_p2 = stats.spearmanr(full_data['rating_human'], full_data[sc_col])
print(f"Pearson r = {pearson_r2:.4f} (p = {pearson_p2:.2e})")
print(f"Spearman ρ = {spearman_r2:.4f} (p = {spearman_p2:.2e})")

# Average of two prompts
full_data['rating_avg'] = (full_data['rating_standard'] + full_data['rating_human']) / 2
print("\nAverage of Both Prompts vs Actual Social Class:")
pearson_r3, pearson_p3 = stats.pearsonr(full_data['rating_avg'], full_data[sc_col])
spearman_r3, spearman_p3 = stats.spearmanr(full_data['rating_avg'], full_data[sc_col])
print(f"Pearson r = {pearson_r3:.4f} (p = {pearson_p3:.2e})")
print(f"Spearman ρ = {spearman_r3:.4f} (p = {spearman_p3:.2e})")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Standard prompt vs actual
ax1 = axes[0, 0]
ax1.scatter(full_data[sc_col], full_data['rating_standard'], alpha=0.3, s=10)
z1 = np.polyfit(full_data[sc_col], full_data['rating_standard'], 1)
p1 = np.poly1d(z1)
ax1.plot(sorted(full_data[sc_col].unique()), p1(sorted(full_data[sc_col].unique())), 'r-', linewidth=2)
ax1.set_xlabel(f'Actual Social Class ({sc_col})')
ax1.set_ylabel('AI Rating (Standard Prompt)')
ax1.set_title(f'Standard Prompt: r = {pearson_r1:.3f}')
ax1.grid(True, alpha=0.3)

# 2. Human-style prompt vs actual
ax2 = axes[0, 1]
ax2.scatter(full_data[sc_col], full_data['rating_human'], alpha=0.3, s=10)
z2 = np.polyfit(full_data[sc_col], full_data['rating_human'], 1)
p2 = np.poly1d(z2)
ax2.plot(sorted(full_data[sc_col].unique()), p2(sorted(full_data[sc_col].unique())), 'g-', linewidth=2)
ax2.set_xlabel(f'Actual Social Class ({sc_col})')
ax2.set_ylabel('AI Rating (Human-Style Prompt)')
ax2.set_title(f'Human-Style Prompt: r = {pearson_r2:.3f}')
ax2.grid(True, alpha=0.3)

# 3. Average rating vs actual
ax3 = axes[1, 0]
ax3.scatter(full_data[sc_col], full_data['rating_avg'], alpha=0.3, s=10)
z3 = np.polyfit(full_data[sc_col], full_data['rating_avg'], 1)
p3 = np.poly1d(z3)
ax3.plot(sorted(full_data[sc_col].unique()), p3(sorted(full_data[sc_col].unique())), 'b-', linewidth=2)
ax3.set_xlabel(f'Actual Social Class ({sc_col})')
ax3.set_ylabel('AI Rating (Average of Both)')
ax3.set_title(f'Average Rating: r = {pearson_r3:.3f}')
ax3.grid(True, alpha=0.3)

# 4. Box plots by actual class
ax4 = axes[1, 1]
data_for_box = []
labels_for_box = []
for sc_val in sorted(full_data[sc_col].unique()):
    data_for_box.append(full_data[full_data[sc_col] == sc_val]['rating_avg'].values)
    labels_for_box.append(str(sc_val))

ax4.boxplot(data_for_box, labels=labels_for_box)
ax4.set_xlabel(f'Actual Social Class ({sc_col})')
ax4.set_ylabel('AI Rating (Average)')
ax4.set_title('AI Rating Distribution by Actual Class')
ax4.grid(True, alpha=0.3)

plt.suptitle(f'ASC Dataset: AI Ratings vs Actual Social Class (n={len(full_data)})', fontsize=16)
plt.tight_layout()
plt.savefig('/home/raymondli/social-class-dml-lme/asc_analysis_2prompts/ai_vs_actual_correlation.png', dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to: asc_analysis_2prompts/ai_vs_actual_correlation.png")

# Save correlation results
results_df = pd.DataFrame({
    'Comparison': ['Standard vs Actual', 'Human-Style vs Actual', 'Average vs Actual'],
    'Pearson_r': [pearson_r1, pearson_r2, pearson_r3],
    'Pearson_p': [pearson_p1, pearson_p2, pearson_p3],
    'Spearman_r': [spearman_r1, spearman_r2, spearman_r3],
    'Spearman_p': [spearman_p1, spearman_p2, spearman_p3],
    'N': [len(full_data)] * 3
})
results_df.to_csv('/home/raymondli/social-class-dml-lme/asc_analysis_2prompts/ai_vs_actual_correlations.csv', index=False)
print("\nCorrelation results saved to: asc_analysis_2prompts/ai_vs_actual_correlations.csv")

# Additional analysis: Mean AI rating by actual class
print("\n=== MEAN AI RATINGS BY ACTUAL SOCIAL CLASS ===")
mean_by_class = full_data.groupby(sc_col).agg({
    'rating_standard': ['mean', 'std', 'count'],
    'rating_human': ['mean', 'std'],
    'rating_avg': ['mean', 'std']
}).round(3)
print(mean_by_class)