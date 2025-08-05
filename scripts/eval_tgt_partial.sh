#!/bin/bash

# Load environment variables
. env.sh

config=$1

# Check if config parameter is provided
if [ -z "$config" ]; then
    echo "Error: Please provide a config file as argument"
    echo "Usage: $0 <config_file>"
    exit 1
fi

# Get all TGT files from ./tgt directory
tgt_dir="./tgt"
tgt_files=($(ls "$tgt_dir"/*.png))

# Iterate over all TGT files for the given config
for tgt_file in "${tgt_files[@]}"; do
    # Extract the TGT file name without extension for prefix
    tgt_name=$(basename "$tgt_file" .png)
    
    echo "Running evaluation with config: $config, TGT: $tgt_file"
    echo "Command: python tools/main.py $config --cfg-options patch_path=$tgt_file prefix=$tgt_name"
    
    python tools/main.py "$config" --cfg-options patch_path="$tgt_file" prefix="$tgt_name"
    
    echo "Completed: $config - $tgt_name"
    echo "----------------------------------------"
done

echo "All evaluations completed!"