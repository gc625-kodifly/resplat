#!/usr/bin/env bash
# To resume training, add: checkpointing.resume=true output_dir=<original_output_dir> wandb.id=<wandb_run_id>
# To disable wandb logging, add: wandb.mode=disabled

# Initial model training (4x nodes, 4 gpus per node, batch size 2 per gpu)
# can also be trained with 1 node and 4 gpus, but with 4x more steps (200k instead of 50k)
python -m src.main +experiment=dl3dv \
    data_loader.train.batch_size=2 \
    dataset.view_sampler.num_context_views=8 \
    dataset.view_sampler.num_target_views=6 \
    dataset.view_sampler.min_distance_between_context_views=24 \
    dataset.view_sampler.max_distance_between_context_views=45 \
    dataset.view_sampler.initial_min_distance_between_context_views=20 \
    dataset.view_sampler.initial_max_distance_between_context_views=30 \
    dataset.image_shape=[256,448] \
    trainer.max_steps=50000 \
    trainer.num_nodes=4 \
    checkpointing.pretrained_depth=pretrained/resplat-depth-base-352x640-60be7abf.pth \
    wandb.project=dl3dv-view8-256x448 \
    output_dir=checkpoints/resplat/dl3dv-view8-256x448/base-init


# Evaluation on DL3DV
# NOTE: You can use the released pretrained model (which includes both init and refine weights)
# with checkpointing.no_strict_load=true, or use your own init-only checkpoint without no_strict_load.
# psnr 27.365
# ssim 0.877
# lpips 0.130
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
    mode=test \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.num_context_views=8 \
    dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_40_ctx_8v_tgt_8v.json \
    dataset.image_shape=[256,448] \
    checkpointing.pretrained_model=pretrained/resplat-base-dl3dv-256x448-view8-1934a04c.pth \
    checkpointing.no_strict_load=true

