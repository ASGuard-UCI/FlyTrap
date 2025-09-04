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

# Set up the environment
. env.sh

# Run attack case
python analysis/percepguard_defense.py \
    --config "$CONFIG" \
    --patch "$PATCH" \
    --videos "${videos[@]}" \
    --attack