#!/bin/bash

docker run -it --rm \
  --gpus all \
  --shm-size 16g \
  -v $(pwd):/workspace/flytrap \
  -v $(pwd)/data:/workspace/flytrap/data \
  -w /workspace/flytrap \
  flytrap
