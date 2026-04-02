#!/bin/bash
# Inference on COLMAP data with infer_colmap.py.
# Usage: bash scripts/infer_colmap.sh
#
# Expected COLMAP data format:
#   <scene_dir>/
#   ├── images/                  # Input images (customizable via --images_dir)
#   │   ├── frame_00000.png
#   │   ├── frame_00001.png
#   │   └── ...
#   └── sparse/0/               # COLMAP reconstruction (customizable via --sparse_dir)
#       ├── cameras.bin          # Camera intrinsics (binary or .txt format)
#       └── images.bin           # Camera extrinsics (binary or .txt format)
#
# Notes:
#   - Only PINHOLE and SIMPLE_PINHOLE camera models are supported.
#     Images must be undistorted (e.g., using COLMAP's image_undistorter).
#   - Both binary (.bin) and text (.txt) COLMAP formats are supported.
#   - With --data_dir + --scene_name, the scene path is <data_dir>/<scene_name>/.

# 8-view, 512x960
SCENE=02267acf6fb98de36173bf4e7db9734c8c421dcb00267e42964dc15134cbb1be
MODEL_PRESET=dl3dv_8v_512x960

DATA_DIR=datasets/dl3dv-colmap-demo
OUTPUT_DIR=results/colmap-dl3dv-demo/${MODEL_PRESET}

python scripts/infer_colmap.py \
    --model_preset $MODEL_PRESET \
    --data_dir $DATA_DIR \
    --scene_name $SCENE \
    --output_dir $OUTPUT_DIR \
    --save_images \
    --save_video \
    --save_ply


# 16-view, 540x960
SCENE=02267acf6fb98de36173bf4e7db9734c8c421dcb00267e42964dc15134cbb1be
MODEL_PRESET=dl3dv_16v_540x960

DATA_DIR=datasets/dl3dv-colmap-demo
OUTPUT_DIR=results/colmap-dl3dv-demo/${MODEL_PRESET}

python scripts/infer_colmap.py \
    --model_preset $MODEL_PRESET \
    --data_dir $DATA_DIR \
    --scene_name $SCENE \
    --output_dir $OUTPUT_DIR \
    --frame_distance 120 \
    --save_images \
    --save_video \
    --save_ply

