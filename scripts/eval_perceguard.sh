#!/bin/bash

CONFIG=$1
PATCH=$2

videos=(
    "person3_grass1_instance2"
    "person3_grass2_instance2"
    "person4_bareground1_instance2"
    "person4_grass1_instance2"
    "person4_grass2_instance2"
)

for video in "${videos[@]}"; do
    # Extract base name from config path
    base_name=$(basename "$CONFIG" .py)
    
    # Run attack case
    python analysis/percepguard_defense.py \
        --config "$CONFIG" \
        --patch "$PATCH" \
        --video "$video" \
        --attack

    # Run benign case
    python analysis/percepguard_defense.py \
        --config "$CONFIG" \
        --patch "$PATCH" \
        --video "$video"
done