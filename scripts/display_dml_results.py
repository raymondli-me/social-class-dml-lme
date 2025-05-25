#!/usr/bin/env python3
"""Display saved DML results"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
with open('dml_checkpoints/dml_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Convert to DataFrame
df = pd.DataFrame(results)

# Display summary
print("\nDML Analysis Results with Cross-Validation")
print("=" * 60)
print(f"Total models run: {len(df)}")
print(f"Methods used: {df['method'].unique()}")
print(f"Targets analyzed: {df['target'].unique()}")

# Best result for each target
print("\nBest R² for each target:")
print("-" * 40)
for target in df['target'].unique():
    target_df = df[df['target'] == target]
    best = target_df.loc[target_df['r2'].idxmax()]
    print(f"{target:35} R² = {best['r2']:.3f} ({best['method']})")

# Create comprehensive visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Bar chart by method
pivot1 = df.pivot_table(index='method', columns='target', values='r2')
pivot1.plot(kind='bar', ax=ax1)
ax1.set_title('R² Scores by ML Method', fontsize=14)
ax1.set_xlabel('Method', fontsize=12)
ax1.set_ylabel('R² Score', fontsize=12)
ax1.legend(title='Target', bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Grouped bar chart by target
pivot2 = df.pivot(index='target', columns='method', values='r2')
pivot2.plot(kind='bar', ax=ax2)
ax2.set_title('R² Scores by Target Variable', fontsize=14)
ax2.set_xlabel('Target Variable', fontsize=12)
ax2.set_ylabel('R² Score', fontsize=12)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.legend(title='Method')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('dml_complete_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Print key findings
print("\n" + "="*60)
print("KEY FINDINGS:")
print("="*60)

# Compare AI vs actual SC
ai_methods = df[df['target'].str.contains('ai|ladder|human')]
sc_methods = df[df['target'] == 'sc11']

best_ai = ai_methods['r2'].max()
best_sc = sc_methods['r2'].max()
ratio = best_ai / best_sc

print(f"Best AI rating R²: {best_ai:.3f}")
print(f"Best actual SC R²: {best_sc:.3f}")
print(f"AI ratings predict essays {ratio:.1f}x better than actual social class")

# Method comparison
print("\nAverage R² by method (across all targets):")
method_avg = df.groupby('method')['r2'].mean().sort_values(ascending=False)
for method, avg_r2 in method_avg.items():
    print(f"  {method:10} {avg_r2:.3f}")