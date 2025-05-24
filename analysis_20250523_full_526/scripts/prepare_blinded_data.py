#!/usr/bin/env python3
"""
Prepare blinded data for vLLM processing
Ensures LLM never sees human judgments or education labels
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_DIR = BASE_DIR / "data"

def prepare_blinded_data():
    """Create blinded dataset for vLLM processing"""
    print("=== Preparing Blinded Data for vLLM Processing ===")
    print(f"Timestamp: {datetime.now()}")
    
    # Load full dataset
    print("\nLoading original dataset...")
    essays_df = pd.read_csv(DATA_DIR / "essay_dataset.csv")
    print(f"Total essays: {len(essays_df)}")
    
    # Show what we're removing (for documentation)
    print("\nOriginal columns:", list(essays_df.columns))
    print("\nData being REMOVED for blinding:")
    print(f"- Human judgments (social class 1-10): mean = {essays_df['judgement'].mean():.2f}")
    print(f"- Education levels: {essays_df['criterion'].value_counts().to_dict()}")
    
    # Create blinded dataset - ONLY ID and TEXT
    blinded_df = essays_df[['TID', 'original']].copy()
    blinded_df = blinded_df.rename(columns={
        'TID': 'id',
        'original': 'text'
    })
    
    # Verify no leakage
    print("\nBlinded dataset columns:", list(blinded_df.columns))
    assert 'judgement' not in blinded_df.columns, "Human judgments leaked!"
    assert 'criterion' not in blinded_df.columns, "Education labels leaked!"
    
    # Save blinded data
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    blinded_file = OUTPUT_DIR / "essays_blinded_526.csv"
    blinded_df.to_csv(blinded_file, index=False)
    print(f"\nBlinded data saved to: {blinded_file}")
    
    # Also save the labels separately for later evaluation
    labels_df = essays_df[['TID', 'criterion', 'judgement']].copy()
    labels_df = labels_df.rename(columns={'TID': 'id'})
    labels_file = OUTPUT_DIR / "labels_hidden_526.csv"
    labels_df.to_csv(labels_file, index=False)
    print(f"Labels saved separately to: {labels_file}")
    
    # Create verification report
    report = f"""
BLINDING VERIFICATION REPORT
===========================
Generated: {datetime.now()}

Original Dataset:
- Total essays: {len(essays_df)}
- Columns: {', '.join(essays_df.columns)}

Blinded Dataset:
- Total essays: {len(blinded_df)}
- Columns: {', '.join(blinded_df.columns)}
- File: essays_blinded_526.csv

Hidden Labels:
- Human judgments (1-10 scale)
- Education criterion
- File: labels_hidden_526.csv

Verification:
✓ No human judgments in blinded data
✓ No education labels in blinded data
✓ Only ID and text provided to LLM
✓ Labels stored separately for post-hoc evaluation

This ensures the LLM analysis is completely blind to human assessments.
"""
    
    with open(OUTPUT_DIR / "blinding_verification.txt", 'w') as f:
        f.write(report)
    
    print("\n" + report)
    
    return blinded_file, labels_file

if __name__ == "__main__":
    blinded_file, labels_file = prepare_blinded_data()
    print(f"\n✅ Blinding complete!")
    print(f"Use this for vLLM: {blinded_file}")
    print(f"Use this for evaluation only: {labels_file}")