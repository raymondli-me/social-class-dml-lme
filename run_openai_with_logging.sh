#!/bin/bash
# Run OpenAI analysis with proper logging

cd "/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme"

# Source environment
source .env.openai

# Create log file with timestamp
LOG_FILE="openai_analysis_$(date +%Y%m%d_%H%M%S).log"

echo "Starting OpenAI Analysis with logging to: $LOG_FILE"
echo "Running in background with nohup..."

# Run with nohup and capture all output
nohup python3 -u scripts/openai_embedding_analysis.py > "$LOG_FILE" 2>&1 &

echo "Process started with PID: $!"
echo "Monitor progress with: tail -f $LOG_FILE"
echo "Or use: ./monitor_openai_progress.sh"