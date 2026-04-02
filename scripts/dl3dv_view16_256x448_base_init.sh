#!/usr/bin/env bash
# To resume training, add: checkpointing.resume=true output_dir=<original_output_dir> wandb.id=<wandb_run_id>
# To disable wandb logging, add: wandb.mode=disabled

# Initial model training (16 views, fine-tuned from 8-view init) (4x nodes, 4 gpus per node, batch size 2 per gpu)
# can also be trained with 1 node and 4 gpus, but with 4x more steps (200k instead of 50k)
python -m src.main +experiment=dl3dv \
    data_loader.train.batch_size=2 \
    dataset.view_sampler.num_context_views=16 \
    dataset.view_sampler.num_target_views=6 \
    dataset.view_sampler.min_distance_between_context_views=50 \
    dataset.view_sampler.max_distance_between_context_views=80 \
    dataset.view_sampler.initial_min_distance_between_context_views=40 \
    dataset.view_sampler.initial_max_distance_between_context_views=60 \
    dataset.image_shape=[256,448] \
    trainer.max_steps=50000 \
    trainer.num_nodes=4 \
    optimizer.lr=1e-4 \
    optimizer.lr_monodepth=1e-6 \
    checkpointing.pretrained_model=<view8_init_checkpoint> \
    wandb.project=dl3dv-view16-256x448 \
    output_dir=checkpoints/resplat/dl3dv-view16-256x448/base-init


# Evaluation on DL3DV (16 views)
# NOTE: You can use the released pretrained model (which includes both init and refine weights)
# with checkpointing.no_strict_load=true, or use your own init-only checkpoint without no_strict_load.
# psnr 26.949
# ssim 0.869
# lpips 0.135
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
    mode=test \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.num_context_views=16 \
    dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_80_ctx_16v_tgt_16v.json \
    dataset.image_shape=[256,448] \
    checkpointing.pretrained_model=pretrained/resplat-base-dl3dv-256x448-view16-f38bf984.pth \
    checkpointing.no_strict_load=true
