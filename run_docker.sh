#!/bin/bash

docker run -it --rm \
  --gpus all \
  --shm-size 16g \
  -v $(pwd):/workspace/flytrap \
  -v /home/shaoyux/data:/workspace/flytrap/data \
  flytrap
