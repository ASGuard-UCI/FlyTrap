#!/bin/bash

# Configuration
CONFIG=configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml
CHECKPOINT=pretrained_models/multi_domain_fast50_dcn_combined_256x192.pth
BASE_DIR=data/UCF-101
OUTPUT_DIR=output/ucf101_results

# Add debug output
echo "Starting script with the following configuration:"
echo "CONFIG: $CONFIG"
echo "CHECKPOINT: $CHECKPOINT"
echo "BASE_DIR: $BASE_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"

# Check if config and checkpoint files exist
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file $CONFIG not found!"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint file $CHECKPOINT not found!"
    exit 1
fi

# Check if base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "ERROR: Base directory $BASE_DIR not found!"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Get total number of videos and store them in an array
echo "Searching for .avi files in $BASE_DIR..."
mapfile -t videos < <(find -L ${BASE_DIR} -type f -name "*.avi")
total_videos=${#videos[@]}

echo "Found $total_videos videos."

# Randomly sample 1000 videos
echo "Randomly sampling 1000 videos from the dataset..."
mapfile -t sampled_videos < <(printf "%s\n" "${videos[@]}" | shuf -n 1000)
videos=("${sampled_videos[@]}")
total_videos=${#videos[@]}
echo "Selected $total_videos videos for processing."

# Exit if no videos were found
if [ $total_videos -eq 0 ]; then
    echo "ERROR: No .avi files found in $BASE_DIR or its subdirectories!"
    echo "Current directory: $(pwd)"
    echo "Listing contents of $BASE_DIR:"
    ls -la $BASE_DIR
    # List a few subdirectories to check
    if [ -d "$BASE_DIR/Archery" ]; then
        echo "Listing contents of $BASE_DIR/Archery:"
        ls -la $BASE_DIR/Archery | head -5
    fi
    exit 1
fi

current_video=0

# Function to process a single video
process_video() {
    local video_path=$1
    local output_dir=$2
    local video_name=$(basename "$video_path")
    local action_name=$(basename "$(dirname "$video_path")")
    
    # Create action-specific output directory
    local action_output_dir="${output_dir}/${action_name}"
    mkdir -p "${action_output_dir}"
    
    echo "Processing video: $video_path"
    echo "Output directory: $action_output_dir"
    
    # Process the video with a 3-minute timeout
    timeout 240 bash ./scripts/inference.sh ${CONFIG} ${CHECKPOINT} ${video_path} ${action_output_dir}
    
    # Check if the process was terminated due to timeout
    if [ $? -eq 124 ]; then
        echo "WARNING: Processing of $video_path timed out after 3 minutes. Moving to next video."
        # Clean up any partial output
        rm -f "${action_output_dir}/${video_name%.*}"*.json
    fi
}

# Process each video with progress bar
for video_path in "${videos[@]}"; do
    ((current_video++))
    # Calculate percentage
    percentage=$((current_video * 100 / total_videos))
    # Create progress bar
    bar_length=50
    filled=$((percentage * bar_length / 100))
    bar=$(printf '#%.0s' $(seq 1 $filled))
    empty=$(printf ' %.0s' $(seq 1 $((bar_length - filled))))
    
    # Print progress
    printf "\rProcessing: [%s%s] %d%% (%d/%d) - %s" "$bar" "$empty" "$percentage" "$current_video" "$total_videos" "$(basename "$video_path")"
    
    # Process the video
    process_video "$video_path" "$OUTPUT_DIR"
done

echo -e "\nAll videos have been processed!" 