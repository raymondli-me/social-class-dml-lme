#!/usr/bin/env python3
"""Plot distributions of AI scores and actual social class"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Load data
print("Loading data...")
essays = pd.read_csv('data/asc_9513_essays.csv')
ai_ratings = pd.read_csv('asc_analysis_2prompts/run_20250524_162055/all_results_9513x2_20250524_174149.csv')
sc_data = pd.read_csv('/media/raymondli/Crucial X9/asc_essays/essay_data/asc_9513_sc11.csv')

# Pivot AI ratings
ai_pivot = ai_ratings.pivot_table(
    index='essay_id', 
    columns='prompt_name', 
    values='rating',
    aggfunc='first'
).reset_index()

# Merge data
data = essays.merge(ai_pivot, left_on='TID', right_on='essay_id', how='inner')
data = data.merge(sc_data, on='TID', how='inner')
data['ai_average'] = data[['ladder_standard_improved', 'human_macarthur_ladder_improved']].mean(axis=1)

print(f"Loaded {len(data)} essays with complete data")

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribution of Social Class Measures (N=9,513)', fontsize=16)

# 1. Actual Social Class (SC11)
ax = axes[0, 0]
ax.hist(data['sc11'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax.axvline(data['sc11'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data["sc11"].mean():.2f}')
ax.axvline(data['sc11'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {data["sc11"].median():.2f}')
ax.set_xlabel('Actual Social Class (SC11)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('A. Actual Social Class Distribution', fontsize=13)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 2. AI Average Rating
ax = axes[0, 1]
ax.hist(data['ai_average'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
ax.axvline(data['ai_average'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data["ai_average"].mean():.2f}')
ax.axvline(data['ai_average'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {data["ai_average"].median():.2f}')
ax.set_xlabel('AI Average Rating (1-10)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('B. AI Average Rating Distribution', fontsize=13)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 3. Comparison boxplot
ax = axes[0, 2]
df_melt = pd.DataFrame({
    'Actual SC11': data['sc11'],
    'AI Average': data['ai_average'],
    'Ladder Standard': data['ladder_standard_improved'],
    'Human MacArthur': data['human_macarthur_ladder_improved']
})
bp = ax.boxplot([df_melt[col].dropna() for col in df_melt.columns], 
                 labels=['Actual\nSC11', 'AI\nAverage', 'Ladder\nStandard', 'Human\nMacArthur'],
                 patch_artist=True, notch=True)
for patch, color in zip(bp['boxes'], ['skyblue', 'lightcoral', 'lightgreen', 'plum']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Score (1-10)', fontsize=12)
ax.set_title('C. Distribution Comparison', fontsize=13)
ax.grid(axis='y', alpha=0.3)

# 4. Scatter plot: AI vs Actual
ax = axes[1, 0]
scatter = ax.scatter(data['sc11'], data['ai_average'], alpha=0.3, s=10)
ax.plot([1, 10], [1, 10], 'r--', alpha=0.5, label='Perfect correlation')

# Add correlation
corr = data[['sc11', 'ai_average']].corr().iloc[0, 1]
ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel('Actual Social Class (SC11)', fontsize=12)
ax.set_ylabel('AI Average Rating', fontsize=12)
ax.set_title('D. AI vs Actual Social Class', fontsize=13)
ax.set_xlim(0, 11)
ax.set_ylim(0, 11)
ax.legend()
ax.grid(True, alpha=0.3)

# 5. Density plot comparison
ax = axes[1, 1]
# Normalize to same scale for comparison
sc11_norm = (data['sc11'] - data['sc11'].min()) / (data['sc11'].max() - data['sc11'].min()) * 9 + 1
ai_norm = data['ai_average']

ax.hist(sc11_norm, bins=50, alpha=0.5, density=True, label='Actual SC11 (normalized)', color='skyblue')
ax.hist(ai_norm, bins=50, alpha=0.5, density=True, label='AI Average', color='lightcoral')

# Add KDE
from scipy.stats import gaussian_kde
kde_sc11 = gaussian_kde(sc11_norm.dropna())
kde_ai = gaussian_kde(ai_norm.dropna())
x_range = np.linspace(1, 10, 100)
ax.plot(x_range, kde_sc11(x_range), 'b-', linewidth=2, label='SC11 KDE')
ax.plot(x_range, kde_ai(x_range), 'r-', linewidth=2, label='AI KDE')

ax.set_xlabel('Score (1-10)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('E. Density Comparison', fontsize=13)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 6. Distribution statistics
ax = axes[1, 2]
ax.axis('off')

# Calculate statistics
stats_text = "Distribution Statistics:\n" + "="*30 + "\n\n"
stats_text += "Actual SC11:\n"
stats_text += f"  Mean: {data['sc11'].mean():.3f}\n"
stats_text += f"  Std Dev: {data['sc11'].std():.3f}\n"
stats_text += f"  Skewness: {stats.skew(data['sc11'].dropna()):.3f}\n"
stats_text += f"  Kurtosis: {stats.kurtosis(data['sc11'].dropna()):.3f}\n\n"

stats_text += "AI Average:\n"
stats_text += f"  Mean: {data['ai_average'].mean():.3f}\n"
stats_text += f"  Std Dev: {data['ai_average'].std():.3f}\n"
stats_text += f"  Skewness: {stats.skew(data['ai_average'].dropna()):.3f}\n"
stats_text += f"  Kurtosis: {stats.kurtosis(data['ai_average'].dropna()):.3f}\n\n"

stats_text += f"Correlation: {corr:.3f}\n"
stats_text += f"N = {len(data):,}"

ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('ai_sc_distributions.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'ai_sc_distributions.png'")

# Additional analysis
print("\n" + "="*50)
print("DISTRIBUTION ANALYSIS")
print("="*50)

print("\nPercentile comparison:")
percentiles = [10, 25, 50, 75, 90]
for p in percentiles:
    sc_p = np.percentile(data['sc11'].dropna(), p)
    ai_p = np.percentile(data['ai_average'].dropna(), p)
    print(f"  {p}th percentile - SC11: {sc_p:.2f}, AI: {ai_p:.2f}")

print("\nRange analysis:")
print(f"  SC11 range: {data['sc11'].min():.1f} - {data['sc11'].max():.1f}")
print(f"  AI range: {data['ai_average'].min():.1f} - {data['ai_average'].max():.1f}")

# plt.show()  # Comment out to avoid GUI timeout