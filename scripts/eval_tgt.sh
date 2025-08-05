#!/bin/bash

# Load environment variables
. env.sh

configs=(
    "config/final/mixformer_tgt.py"
    "config/final/siamrpn_alex_tgt.py"
    "config/final/siamrpn_mob_tgt.py"
    "config/final/siamrpn_resnet_tgt.py"
)

# Get all TGT files from ./tgt directory
tgt_dir="./tgt"
tgt_files=($(ls "$tgt_dir"/*.png))

# Iterate over all config and TGT file combinations
for config in "${configs[@]}"; do
    for tgt_file in "${tgt_files[@]}"; do
        # Extract the TGT file name without extension for prefix
        tgt_name=$(basename "$tgt_file" .png)
        
        echo "Running evaluation with config: $config, TGT: $tgt_file"
        echo "Command: python tools/main.py $config --cfg-options patch_path=$tgt_file prefix=$tgt_name"
        
        python tools/main.py "$config" --cfg-options patch_path="$tgt_file" prefix="$tgt_name"
        
        echo "Completed: $config - $tgt_name"
        echo "----------------------------------------"
    done
done

echo "All evaluations completed!"