# Social Class Detection with DML-LME Analysis

This repository contains the analysis pipeline for detecting social class indicators in essays using Large Language Models (vLLM) and Double Machine Learning (DML-LME) methods.

## Project Overview

This project analyzes 526 essays written by 25-year-olds describing their lives, using 100 different prompts to extract socioeconomic indicators. The analysis combines:

1. **vLLM Batch Processing**: Efficient parallel processing of essays through LLMs to extract binary features
2. **DML-LME Analysis**: Double Machine Learning with Linear Mixed Effects to identify causal relationships between education level and perceived social class

## Data

- **essay_dataset.csv**: 526 essays with education level labels and social class ratings
  - `TID`: Essay ID
  - `criterion`: Education level (Less than high school, High school, Some college, College, Graduate)
  - `judgement`: Social class rating (1-10 scale)
  - `original`: Essay text

- **prompts_100_complete.csv**: 100 binary classification prompts detecting:
  - Income mentions
  - Property ownership
  - Educational achievements
  - Luxury items/experiences
  - Professional indicators

## Repository Structure

```
2025_05_23_social_class_dml_lme/
├── data/                    # Input data files
│   ├── essay_dataset.csv
│   └── prompts_100_complete.csv
├── scripts/                 # Analysis scripts
│   ├── test_10_samples.py   # Quick test on 10 essays
│   └── full_analysis.py     # Complete analysis pipeline
├── outputs/                 # Processing outputs
│   ├── test/               # Test run outputs
│   └── full/               # Full analysis outputs
├── visualizations/          # Generated plots and figures
├── reports/                 # Analysis reports
└── requirements.txt         # Python dependencies
```

## Installation

1. Clone this repository:
```bash
git clone <repo-url>
cd 2025_05_23_social_class_dml_lme
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install local packages (update paths as needed):
```bash
# Install vllm-batch-processor
pip install -e /media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/vllm-batch-processor

# Install dml-lme (when available)
# pip install -e /path/to/dml-lme
```

## Usage

### Quick Test (10 samples)

Test the pipeline on 10 essays to verify setup:

```bash
python scripts/test_10_samples.py
```

This will:
- Process 10 essays with all 100 prompts
- Use a smaller model (Llama-3.2-3B) for faster testing
- Save outputs to `outputs/test/`

### Full Analysis

Run the complete analysis pipeline:

```bash
python scripts/full_analysis.py
```

This will:
1. Process all 526 essays with 100 prompts
2. Extract binary features for each socioeconomic indicator
3. Run DML analysis to measure treatment effects
4. Generate visualizations and reports

For parallel processing with multiple GPUs:
```python
# In full_analysis.py, set use_parallel=True
analyzer.run_full_pipeline(model_preset="qwen-72b", use_parallel=True)
```

## Key Features

### Efficient Batch Processing
- Processes all 100 prompts simultaneously (not sequentially) for each essay
- Supports parallel execution across multiple GPUs
- Automatic batch management and error recovery

### Analysis Pipeline
1. **Feature Extraction**: Convert essays to 100 binary features using LLM prompts
2. **DML Analysis**: Identify causal effect of education on social class perception
3. **Visualization**: Generate heatmaps, correlation matrices, and distribution plots
4. **Reporting**: Automated report generation with key findings

## Output Files

- **outputs/full/analysis_results.csv**: Complete dataset with extracted features
- **visualizations/**: 
  - `social_class_by_education.png`: Distribution of ratings by education level
  - `feature_heatmap.png`: Feature prevalence across education levels
  - `feature_correlations.png`: Correlation matrix of top features
- **reports/**: Timestamped analysis reports with statistics and findings

## Model Configuration

Default models (can be changed in scripts):
- Test: `meta-llama/Llama-3.2-3B-Instruct`
- Full analysis: `Qwen/Qwen2.5-72B-Instruct`

Supported presets:
- qwen-72b, qwen-32b, qwen-14b, qwen-7b, qwen-3b
- llama-70b, llama-8b, llama-3b

## Citation

If you use this analysis pipeline in your research, please cite:
```
[Citation information to be added]
```

## License

[License information to be added]