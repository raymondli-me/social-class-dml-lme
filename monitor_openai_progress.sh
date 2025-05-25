#!/bin/bash
# Monitor OpenAI embedding generation progress

CHECKPOINT_DIR="/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/openai_checkpoints"
OUTPUT_DIR="/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/openai_outputs"
VIZ_DIR="/media/raymondli/Crucial X9/2025_05_23_VLLM_IPAD_ANALYSES/2025_05_23_social_class_dml_lme/openai_visualizations"

echo "=== OpenAI Analysis Progress Monitor ==="
echo "Started at: $(date)"
echo

while true; do
    clear
    echo "=== OpenAI Analysis Progress Monitor ==="
    echo "Current time: $(date)"
    echo
    
    # Check if embeddings are done
    if [ -f "$CHECKPOINT_DIR/openai_embeddings.npy" ]; then
        echo "✓ Embeddings complete: $(ls -lh $CHECKPOINT_DIR/openai_embeddings.npy | awk '{print $5}')"
    else
        echo "⏳ Generating embeddings..."
        # Check if process is still running
        if pgrep -f "openai_embedding_analysis.py" > /dev/null; then
            echo "   Process is running (PID: $(pgrep -f openai_embedding_analysis.py))"
        else
            echo "   ⚠️  Process not found - may have completed or failed"
        fi
    fi
    
    # Check other checkpoints
    echo
    echo "Checkpoint files:"
    ls -la "$CHECKPOINT_DIR" 2>/dev/null | tail -n +4 | awk '{print "  " $9 " (" $5 ")"}'
    
    # Check outputs
    if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR 2>/dev/null)" ]; then
        echo
        echo "Output files:"
        ls -la "$OUTPUT_DIR" 2>/dev/null | tail -n +4 | awk '{print "  " $9 " (" $5 ")"}'
    fi
    
    # Check visualizations
    if [ -d "$VIZ_DIR" ] && [ "$(ls -A $VIZ_DIR 2>/dev/null)" ]; then
        echo
        echo "Visualization files:"
        ls -la "$VIZ_DIR" 2>/dev/null | tail -n +4 | awk '{print "  " $9 " (" $5 ")"}'
    fi
    
    # Check for completion
    if [ -f "$VIZ_DIR/umap_actual_sc_openai.html" ] && [ -f "$VIZ_DIR/umap_ai_rating_openai.html" ]; then
        echo
        echo "✅ ANALYSIS COMPLETE!"
        break
    fi
    
    sleep 30
done

echo
echo "Final output files:"
echo "1. Interactive UMAPs:"
echo "   - $VIZ_DIR/umap_actual_sc_openai.html"
echo "   - $VIZ_DIR/umap_ai_rating_openai.html"
echo "2. SHAP visualizations in: $VIZ_DIR/shap_analysis/"