#!/bin/bash

#!/bin/bash

# Check if both config file and patch are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <config_file> <patch_file>"
    echo "Example: $0 config/final/siamrpn_alex.py patches/person1_grass1_instance1_scale15.0.png"
    exit 1
fi

CONFIG=$1
PATCH=$2

videos=(
    "person3_bareground1_instance2"
    "person3_grass1_instance2"
    "person3_grass2_instance2"
    "person4_bareground1_instance2"
    "person4_grass1_instance2"
    "person4_grass2_instance2"
)

mkdir -p "exp_analysis/vogues"

. env.sh

for video in "${videos[@]}"; do
    # Extract base name from config path
    base_name=$(basename "$CONFIG" .py)
    
    # Construct patch path
    patch_path="work_dirs/${base_name}/patch.png"

    # attack case
    python analysis/vogues_defense.py \
        "$CONFIG" \
        "$PATCH" \
        --video "$video" \
        --attack \
        --save "exp_analysis/vogues/${base_name}_${video}_attack_vogues.json"

    # benign case
    python analysis/vogues_defense.py \
        "$CONFIG" \
        "$PATCH" \
        --video "$video" \
        --save "exp_analysis/vogues/${base_name}_${video}_benign_vogues.json"

done