#!/usr/bin/env python3
"""
Analyze correlation between ASC human ratings and AI ratings
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the data
print("Loading ASC data...")
ladder_standard = pd.read_csv('/home/raymondli/social-class-dml-lme/asc_analysis_2prompts/run_20250524_162055/results_ladder_standard_improved_20250524_165833.csv')
human_ladder = pd.read_csv('/home/raymondli/social-class-dml-lme/asc_analysis_2prompts/run_20250524_162055/results_human_macarthur_ladder_improved_20250524_174149.csv')

print(f"Ladder standard shape: {ladder_standard.shape}")
print(f"Human ladder shape: {human_ladder.shape}")

# Merge on essay_id
merged = pd.merge(ladder_standard, human_ladder, on='essay_id', suffixes=('_standard', '_human'))
print(f"\nMerged data shape: {merged.shape}")

# Basic statistics for each prompt
print("\n=== BASIC STATISTICS ===")
print("\nLadder Standard (AI Standard Prompt):")
print(f"Mean: {ladder_standard['rating'].mean():.3f}")
print(f"Std Dev: {ladder_standard['rating'].std():.3f}")
print(f"Median: {ladder_standard['rating'].median():.3f}")
print(f"Min: {ladder_standard['rating'].min()}")
print(f"Max: {ladder_standard['rating'].max()}")
print(f"Skewness: {stats.skew(ladder_standard['rating']):.3f}")
print(f"Kurtosis: {stats.kurtosis(ladder_standard['rating']):.3f}")

print("\nHuman MacArthur Ladder (AI Human-Style Prompt):")
print(f"Mean: {human_ladder['rating'].mean():.3f}")
print(f"Std Dev: {human_ladder['rating'].std():.3f}")
print(f"Median: {human_ladder['rating'].median():.3f}")
print(f"Min: {human_ladder['rating'].min()}")
print(f"Max: {human_ladder['rating'].max()}")
print(f"Skewness: {stats.skew(human_ladder['rating']):.3f}")
print(f"Kurtosis: {stats.kurtosis(human_ladder['rating']):.3f}")

# Calculate correlations
print("\n=== CORRELATION ANALYSIS ===")
pearson_r, pearson_p = stats.pearsonr(merged['rating_standard'], merged['rating_human'])
spearman_r, spearman_p = stats.spearmanr(merged['rating_standard'], merged['rating_human'])

print(f"\nPearson correlation: r = {pearson_r:.4f} (p = {pearson_p:.2e})")
print(f"Spearman correlation: ρ = {spearman_r:.4f} (p = {spearman_p:.2e})")

# Calculate agreement statistics
print("\n=== AGREEMENT ANALYSIS ===")
mean_diff = (merged['rating_standard'] - merged['rating_human']).mean()
std_diff = (merged['rating_standard'] - merged['rating_human']).std()
print(f"Mean difference (standard - human): {mean_diff:.3f}")
print(f"Std dev of differences: {std_diff:.3f}")

# Exact agreement
exact_agreement = (merged['rating_standard'] == merged['rating_human']).mean()
within_1 = (abs(merged['rating_standard'] - merged['rating_human']) <= 1).mean()
within_2 = (abs(merged['rating_standard'] - merged['rating_human']) <= 2).mean()

print(f"\nExact agreement: {exact_agreement:.1%}")
print(f"Agreement within ±1: {within_1:.1%}")
print(f"Agreement within ±2: {within_2:.1%}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Scatter plot with regression line
ax1 = axes[0, 0]
ax1.scatter(merged['rating_standard'], merged['rating_human'], alpha=0.3, s=10)
ax1.plot([1, 10], [1, 10], 'r--', label='Perfect agreement')

# Add regression line
z = np.polyfit(merged['rating_standard'], merged['rating_human'], 1)
p = np.poly1d(z)
ax1.plot([1, 10], p([1, 10]), 'g-', label=f'Regression line')

ax1.set_xlabel('Ladder Standard Rating')
ax1.set_ylabel('Human MacArthur Ladder Rating')
ax1.set_title(f'Correlation: r = {pearson_r:.3f}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Distribution comparison
ax2 = axes[0, 1]
bins = np.arange(0.5, 11.5, 1)
ax2.hist(ladder_standard['rating'], bins=bins, alpha=0.5, label='Standard', density=True)
ax2.hist(human_ladder['rating'], bins=bins, alpha=0.5, label='Human-style', density=True)
ax2.set_xlabel('Rating')
ax2.set_ylabel('Density')
ax2.set_title('Rating Distributions')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Difference histogram
ax3 = axes[1, 0]
differences = merged['rating_standard'] - merged['rating_human']
ax3.hist(differences, bins=np.arange(-9.5, 10.5, 1), edgecolor='black')
ax3.set_xlabel('Rating Difference (Standard - Human)')
ax3.set_ylabel('Count')
ax3.set_title('Distribution of Rating Differences')
ax3.axvline(x=0, color='r', linestyle='--', label='No difference')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Heatmap of ratings
ax4 = axes[1, 1]
heatmap_data = pd.crosstab(merged['rating_standard'], merged['rating_human'])
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Blues', ax=ax4, cbar_kws={'label': 'Count'})
ax4.set_xlabel('Human MacArthur Ladder Rating')
ax4.set_ylabel('Ladder Standard Rating')
ax4.set_title('Rating Agreement Heatmap')

plt.suptitle('ASC Dataset: Comparison of Two Prompt Variations (n=9,513)', fontsize=16)
plt.tight_layout()
plt.savefig('/home/raymondli/social-class-dml-lme/asc_analysis_2prompts/asc_correlation_analysis.png', dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to: asc_analysis_2prompts/asc_correlation_analysis.png")

# Save detailed statistics
stats_dict = {
    'Dataset': ['ASC Essays'],
    'N_essays': [len(merged)],
    'Standard_mean': [ladder_standard['rating'].mean()],
    'Standard_std': [ladder_standard['rating'].std()],
    'Human_mean': [human_ladder['rating'].mean()],
    'Human_std': [human_ladder['rating'].std()],
    'Pearson_r': [pearson_r],
    'Pearson_p': [pearson_p],
    'Spearman_r': [spearman_r],
    'Spearman_p': [spearman_p],
    'Exact_agreement': [exact_agreement],
    'Within_1': [within_1],
    'Within_2': [within_2],
    'Mean_difference': [mean_diff],
    'Std_difference': [std_diff]
}

stats_df = pd.DataFrame(stats_dict)
stats_df.to_csv('/home/raymondli/social-class-dml-lme/asc_analysis_2prompts/asc_correlation_stats.csv', index=False)
print("\nStatistics saved to: asc_analysis_2prompts/asc_correlation_stats.csv")

# Distribution of confidence levels
print("\n=== CONFIDENCE ANALYSIS ===")
print("\nStandard prompt confidence distribution:")
print(ladder_standard['confidence'].value_counts(normalize=True).sort_index())
print("\nHuman-style prompt confidence distribution:")
print(human_ladder['confidence'].value_counts(normalize=True).sort_index())

# Cross-tabulation of confidence levels
confidence_crosstab = pd.crosstab(
    merged['confidence_standard'], 
    merged['confidence_human'],
    normalize='all'
)
print("\nConfidence level cross-tabulation (proportions):")
print(confidence_crosstab)