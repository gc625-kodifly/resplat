#!/usr/bin/env bash
# To resume training, add: checkpointing.resume=true output_dir=<original_output_dir> wandb.id=<wandb_run_id>
# To disable wandb logging, add: wandb.mode=disabled

# Refinement training (32 views) (4x nodes, 4 gpus per node, batch size 1 per gpu)
# can also be trained with 1 node and 4 gpus, but with 4x more steps (120k instead of 30k)
python -m src.main +experiment=dl3dv \
    data_loader.train.batch_size=1 \
    dataset.view_sampler.num_context_views=32 \
    dataset.view_sampler.num_target_views=8 \
    dataset.view_sampler.min_distance_between_context_views=120 \
    dataset.view_sampler.max_distance_between_context_views=170 \
    dataset.view_sampler.initial_min_distance_between_context_views=80 \
    dataset.view_sampler.initial_max_distance_between_context_views=120 \
    dataset.image_shape=[256,448] \
    dataset.pose_align_middle_view=true \
    trainer.max_steps=30000 \
    trainer.num_nodes=4 \
    train.depth_smooth_loss_weight=0. \
    model.encoder.num_refine=4 \
    model.encoder.train_min_refine=1 \
    model.encoder.train_max_refine=4 \
    model.encoder.recurrent_use_checkpointing=true \
    optimizer.lr=1e-4 \
    optimizer.lr_monodepth=0. \
    checkpointing.pretrained_model=<view32_init_checkpoint> \
    checkpointing.no_strict_load=true \
    checkpointing.resume_update_module=<view16_refine_checkpoint> \
    wandb.project=dl3dv-view32-256x448 \
    output_dir=checkpoints/resplat/dl3dv-view32-256x448/base-refine


# Evaluation on DL3DV (32 views)
# refine 4
# psnr 28.301
# ssim 0.891
# lpips 0.114

# 1/10 subset, refine 4
# psnr 28.476
# ssim 0.900
# lpips 0.109

CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
    mode=test \
    dataset.test_chunk_interval=10 \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.num_context_views=32 \
    dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_160_ctx_32v_tgt_24v.json \
    dataset.pose_align_middle_view=true \
    dataset.image_shape=[256,448] \
    model.encoder.num_refine=4 \
    checkpointing.pretrained_model=pretrained/resplat-base-dl3dv-256x448-view32-439b63a6.pth
