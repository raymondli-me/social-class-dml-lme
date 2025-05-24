# vLLM Implementation Methods Documentation
**Date:** 2025-05-24  
**Purpose:** Complete technical documentation for methods section of paper

---

## Overview

This document provides comprehensive technical details for implementing social class detection using Large Language Models (LLMs) via vLLM inference engine. We process 526 essays written by 11-year-old children (now 25 years old) through 51 different prompting strategies to measure perceived social class.

---

## Model Specifications

### Base Model
- **Model**: Qwen2.5-32B-Instruct-AWQ
- **Parameters**: 32 billion
- **Quantization**: AWQ (Activation-aware Weight Quantization)
- **Model Path**: `Qwen/Qwen2.5-32B-Instruct-AWQ`
- **Architecture**: Transformer-based autoregressive language model
- **Training**: Instruction-tuned variant optimized for following prompts

### Inference Configuration
- **Framework**: vLLM v0.8.5.post1
- **Tensor Parallelism**: 2 (distributed across 2 GPUs)
- **GPU Memory Utilization**: 0.8 (80% of available VRAM)
- **Maximum Context Length**: 2048 tokens
- **Data Type**: float16
- **Execution Mode**: Eager (with compilation error suppression)

### Hardware Setup
- **GPUs**: 2 × NVIDIA GeForce RTX 3090 (24GB VRAM each)
- **CUDA Devices**: 0, 1
- **Memory Usage**: ~9GB per GPU (18GB total)
- **Parallelization**: Tensor parallel across 2 GPUs

---

## Prompting Strategy

### Human-Equivalent Prompt (Primary)
The first prompt exactly replicates the MacArthur Scale of Subjective Social Status used with human raters:

```
System: You are rating the social class of families based on essays. Respond with just a number from 1 to 10.

User: Imagine that this ladder pictures how society is set up. At the top of the ladder (10) are the people who are the best off — they have the most money, the highest amount of schooling, and the jobs that bring the most respect. At the bottom (1) are people who are the worst off — they have the least money, little or no education, no job, or jobs that no one wants or respects. The following essay was written by an 11-year-old child.

Essay: [ESSAY TEXT]

Where do you think the family of the person who wrote this essay would be on this ladder? Please respond with a number from 1 to 10.
```

### Additional Prompts
50 additional prompts explore different framings of social class assessment, including:
- Economic position
- Social status
- Life opportunities
- Respect and prestige
- Material possessions
- Career success
- Family background
- Future prospects

---

## Generation Parameters

### Sampling Configuration
```python
SamplingParams(
    temperature=0.1,      # Near-deterministic for consistency
    max_tokens=200,       # Sufficient for complete responses
    top_p=1.0,           # No nucleus sampling restriction
    top_k=-1,            # No top-k restriction
    stop=None,           # No stop sequences (allow complete responses)
    presence_penalty=0.0, # No repetition penalties
    frequency_penalty=0.0,
    repetition_penalty=1.0
)
```

### Rationale for Parameters
- **Low Temperature (0.1)**: Ensures consistent, reproducible ratings across essays
- **200 Max Tokens**: Prevents truncation of responses while limiting computation
- **No Stop Sequences**: Allows model to complete thoughts naturally
- **No Penalties**: Avoids biasing the distribution of ratings

---

## Data Processing Pipeline

### 1. Data Preparation
- **Input**: 526 blinded essays (IDs anonymized, no demographic information)
- **Format**: CSV with columns: `id`, `text`
- **Blinding**: Essays contain no labels, education levels, or human ratings

### 2. Batch Processing
```python
batch_size = 50  # Process 50 essays per batch
total_prompts = 51
total_essays = 526
total_inferences = 26,826  # 51 × 526
```

### 3. Output Structure
```
outputs/vllm_actual_suppressed/run_YYYYMMDD_HHMMSS/
├── results_human_macarthur_ladder_[timestamp].csv
├── results_ladder_standard_[timestamp].csv
├── results_status_rating_[timestamp].csv
└── ... (51 prompt result files total)
```

### 4. Output Format
Each CSV contains:
- `essay_id`: Anonymized essay identifier
- `prompt_name`: Name of the prompting strategy
- `rating`: Parsed numeric rating (1-10) or None
- `raw_response`: Complete model output
- `timestamp`: ISO format timestamp

---

## Technical Implementation Details

### Error Handling
Due to missing Python development headers on the system, we implement:
```python
import torch._dynamo
torch._dynamo.config.suppress_errors = True
```
This allows the model to run in eager execution mode, bypassing Triton compilation.

### Memory Management
- Batch size optimized to prevent OOM errors
- GPU memory pre-allocated at 80% utilization
- Automatic garbage collection between prompts

### Progress Tracking
- Checkpoints saved every 10 prompts
- Each prompt's results saved immediately upon completion
- Timestamped folders prevent result mixing

---

## Reproducibility

### Software Versions
- Python: 3.10.12
- PyTorch: 2.6.0
- Transformers: 4.52.3
- vLLM: 0.8.5.post1
- CUDA: 12.6

### Random Seed
While temperature is near-zero (0.1), full determinism is not guaranteed due to:
- GPU floating-point non-determinism
- Parallel execution across GPUs
- Dynamic batching in vLLM

### Code Availability
All scripts are documented and available at:
```
analysis_20250523_full_526/scripts/run_vllm_with_suppression.py
```

---

## Expected Outcomes

### Performance Metrics
- **Processing Time**: ~2-3 minutes per prompt (all 526 essays)
- **Total Runtime**: ~2-3 hours for complete dataset
- **Throughput**: ~175 essays/minute with batching

### Quality Metrics
- **Expected Correlation**: 0.7-0.9 with human ratings
- **Missing Data**: <1% (due to parsing or generation failures)
- **Rating Distribution**: Approximately normal, centered around 5-6

---

## Limitations and Considerations

1. **Model Biases**: Qwen-32B may have training biases affecting social class perception
2. **Prompt Sensitivity**: Small wording changes can affect ratings
3. **Context Window**: Essays longer than ~1500 tokens are truncated
4. **Temporal Mismatch**: Model trained on recent data, essays from 1990s
5. **Cultural Context**: Model may not fully capture 1990s American social dynamics

---

## Analysis Plan

Post-processing steps:
1. Parse raw responses to extract numeric ratings
2. Calculate inter-prompt reliability (Cronbach's α)
3. Compare with human ratings (Pearson correlation)
4. Ensemble averaging across prompts
5. Use for downstream causal analysis (DML-LME)

---

## Ethical Considerations

- Essays are anonymized with no identifying information
- No attempt to re-identify individuals
- Results used only for academic research
- Model outputs not used for individual assessment
- Findings reported at aggregate level only

---

**This documentation provides complete technical specifications for replication and methods section writing.**