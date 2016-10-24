#!/bin/bash

# Build the inference binary.
bazel build -c opt im2txt/server

# Run inference to generate captions.
bazel-bin/im2txt/server
