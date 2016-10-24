#!/bin/bash
MSCOCO_DIR="/data/im2txt/mscoco"
MODEL_DIR="${HOME}/im2txt/model"

# Ignore GPU devices (only necessary if your GPU is currently memory
# constrained, for example, by running the training script).
export CUDA_VISIBLE_DEVICES=""

# Run the evaluation script. This will run in a loop, periodically loading the
# latest model checkpoint file and computing evaluation metrics.
bazel-bin/im2txt/evaluate \
      --input_file_pattern="${MSCOCO_DIR}/val-?????-of-00004" \
        --checkpoint_dir="${MODEL_DIR}/train" \
          --eval_dir="${MODEL_DIR}/eval"
