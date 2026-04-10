#!/bin/bash
export TOGETHER_NO_BANNER=1
# Script to run GPT vs Annotator comparison for multiple models and tasks
# Usage: ./run_gpt_annotator_comparison.sh [model_name] [task_type] [judge_model]

# Default values
MODEL_NAME=${1:-openai_gpt_4.1} 
TASK_TYPE=${2:-manual} # manual, automatic
JUDGE_MODEL=${3:-gpt-5.1}
method=${4:-all_at_once_taxonomy}
consolidation=${5:-gpt-5.1}

# Determine model_type based on judge model name
if [[ "$JUDGE_MODEL" == *"claude"* ]]; then
    MODEL_TYPE="claude"
else
    MODEL_TYPE="openai"
fi

# Base paths
GPT_BASE_DIR="results/$MODEL_NAME/$method/unified_$consolidation/$TASK_TYPE"
ANNOTATOR_BASE_DIR="annotated/unified_$consolidation/$TASK_TYPE"

# Create log directory with judge model
mkdir -p "eval_results_$JUDGE_MODEL/$MODEL_NAME/$method/unified_$consolidation/$TASK_TYPE"

# Check if directories exist
if [ ! -d "$GPT_BASE_DIR" ]; then
    echo "Error: GPT results directory not found: $GPT_BASE_DIR"
    exit 1
fi

if [ ! -d "$ANNOTATOR_BASE_DIR" ]; then
    echo "Error: Annotator directory not found: $ANNOTATOR_BASE_DIR"
    exit 1
fi

echo "=========================================="
echo "GPT vs Annotator Comparison"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Task Type: $TASK_TYPE"
echo "Judge Model: $JUDGE_MODEL"
echo "Judge Model Type: $MODEL_TYPE"
echo "Output will be saved to: eval_results_$JUDGE_MODEL/$MODEL_NAME/$method/unified_$consolidation/$TASK_TYPE/"
echo "=========================================="
echo ""

# Count total files
total_files=$(ls -1 "$GPT_BASE_DIR"/*.json 2>/dev/null | wc -l)
processed=0
skipped=0
failed=0

# Process each JSON file
for gpt_file in "$GPT_BASE_DIR"/*.json; do
    if [ ! -f "$gpt_file" ]; then
        continue
    fi
    
    # Extract filename
    filename=$(basename "$gpt_file")
    annotator_file="$ANNOTATOR_BASE_DIR/$filename"
    
    # Check if annotator file exists
    if [ ! -f "$annotator_file" ]; then
        echo "[SKIP] $filename - Annotator file not found"
        skipped=$((skipped + 1))
        continue
    fi
    
    # Determine output path (Python script will save to eval_results_$JUDGE_MODEL)
    # Output will be: eval_results_$JUDGE_MODEL/.../MODEL_NAME/method/unified_consolidation/TASK_TYPE/eval_filename.json
    output_file="eval_results_$JUDGE_MODEL/$MODEL_NAME/$method/unified_$consolidation/$TASK_TYPE/$filename"
    
    # Check if output already exists
    if [ -f "$output_file" ]; then
        echo "[SKIP] $filename - Output already exists"
        skipped=$((skipped + 1))
        continue
    fi
    
    # Run comparison (let Python script handle output path generation)
    processed=$((processed + 1))
    echo "[$processed/$total_files] Processing $filename..."
    
    python evaluate_gpt_vs_annotator.py \
        --gpt_file "$gpt_file" \
        --annotator_file "$annotator_file" \
        --model_name "$JUDGE_MODEL" \
        --model_type "$MODEL_TYPE" \
        2>&1 | tee -a "eval_results_$JUDGE_MODEL/$MODEL_NAME/$method/unified_$consolidation/$TASK_TYPE/run.log"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Success"
    else
        echo "  ✗ Failed"
        failed=$((failed + 1))
    fi
    
    echo ""
done

echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Total files: $total_files"
echo "Processed: $processed"
echo "Skipped: $skipped"
echo "Failed: $failed"
echo "=========================================="

