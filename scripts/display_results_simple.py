#!/usr/bin/env python3
import pickle
import pandas as pd

# Load results
with open('dml_checkpoints/dml_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Convert to DataFrame
df = pd.DataFrame(results)

print("\nDML Analysis Results with Cross-Validation (5-fold)")
print("=" * 70)
print(f"Total models: {len(df)}")
print(f"Methods: {sorted(df['method'].unique())}")

# Create a nice summary table
print("\nR² Scores by Target and Method:")
print("-" * 70)
pivot = df.pivot(index='target', columns='method', values='r2')
print(pivot.round(3))

print("\nBest R² for each target:")
print("-" * 50)
for target in sorted(df['target'].unique()):
    target_df = df[df['target'] == target]
    best = target_df.loc[target_df['r2'].idxmax()]
    print(f"{target:35} R² = {best['r2']:.3f} ({best['method']})")

# Key findings
print("\n" + "="*70)
print("KEY FINDINGS:")
print("="*70)

ai_methods = df[df['target'].str.contains('ai|ladder|human')]
sc_methods = df[df['target'] == 'sc11']

best_ai = ai_methods['r2'].max()
best_sc = sc_methods['r2'].max()
ratio = best_ai / best_sc

print(f"Best AI rating R²: {best_ai:.3f}")
print(f"Best actual SC R²: {best_sc:.3f}")
print(f"AI ratings predict essays {ratio:.1f}x better than actual social class")

# Method comparison
print("\nAverage R² across all targets:")
method_avg = df.groupby('method')['r2'].mean().sort_values(ascending=False)
for method, avg_r2 in method_avg.items():
    print(f"  {method:10} {avg_r2:.3f}")