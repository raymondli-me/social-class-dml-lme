import pickle
import pandas as pd
import numpy as np

# Check the merged data structure
try:
    # Load essay dataset
    df = pd.read_csv('data/essay_dataset.csv')
    print("essay_dataset.csv columns:", df.columns.tolist())
    print("Shape:", df.shape)
    print("\nFirst few IDs:", df['id'].head())
except Exception as e:
    print("Error loading essay_dataset.csv:", e)

print("\n" + "="*50 + "\n")

# Check the 9513 dataset
try:
    df_9513 = pd.read_csv('data/asc_9513_essays.csv')
    print("asc_9513_essays.csv columns:", df_9513.columns.tolist())
    print("Shape:", df_9513.shape)
except Exception as e:
    print("Error loading asc_9513_essays.csv:", e)

print("\n" + "="*50 + "\n")

# Check AI ratings
try:
    ai_ratings = pd.read_csv('asc_analysis_2prompts/run_20250524_162055/all_results_9513x2_20250524_174149.csv')
    print("AI ratings columns:", ai_ratings.columns.tolist())
    print("Unique essay IDs:", ai_ratings['essay_id'].nunique())
    print("Sample IDs:", ai_ratings['essay_id'].head())
except Exception as e:
    print("Error loading AI ratings:", e)