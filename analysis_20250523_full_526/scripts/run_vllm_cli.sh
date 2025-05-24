#!/bin/bash
# Run actual vLLM processing using the CLI

echo "=== ACTUAL vLLM PROCESSING USING CLI ==="
echo "Processing 526 essays with 50 prompts"
echo "This will take approximately 30-60 minutes..."

# Set paths
BASE_DIR="/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/analysis_20250523_full_526"
VLLM_DIR="/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/vllm-batch-processor"

# Change to vllm directory
cd "$VLLM_DIR"

# Run vLLM batch processor
# Using Qwen-32B for good quality/speed balance
# Adjust --preset to qwen-72b for highest quality (but slower)
vllm-batch process \
    "$BASE_DIR/data/essays_blinded_526.csv" \
    --preset qwen-32b \
    --prompt-config "$BASE_DIR/data/ladder_variations_50_complete.csv" \
    --output-dir "$BASE_DIR/outputs/vllm_actual" \
    --batch-size 10 \
    --max-new-tokens 50 \
    --temperature 0.1 \
    --save-every 50 \
    --run-name "full_526_essays_50_prompts"

echo ""
echo "✅ vLLM processing complete!"
echo "Results saved to: $BASE_DIR/outputs/vllm_actual"

# Parse results
echo ""
echo "Parsing results..."
cd "$BASE_DIR"
python3 scripts/parse_vllm_results.py