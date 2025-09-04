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

# Set up the environment
. env.sh

# Run attack case
python analysis/vogues_defense.py \
    "$CONFIG" \
    "$PATCH" \
    --videos "${videos[@]}" \
    --attack

# Run benign case
python analysis/vogues_defense.py \
    "$CONFIG" \
    "$PATCH" \
    --videos "${videos[@]}"