#!/usr/bin/env bash
# To resume training, add: checkpointing.resume=true output_dir=<original_output_dir> wandb.id=<wandb_run_id>
# To disable wandb logging, add: wandb.mode=disabled

# training (4x nodes, 4 gpus per node, batch size 1 per gpu)
# can also be trained with 1 node and 4 gpus, but with 4x more steps (200k instead of 50k)
python -m src.main +experiment=dl3dv \
    data_loader.train.batch_size=1 \
    dataset.roots=[datasets/dl3dv_960p_benchmark_torch] \
    trainer.eval_index=assets/dl3dv_evaluation_longlrm/dl3dv_longlrm_ctx16_tgt_every8th.json \
    dataset.view_sampler.num_context_views=16 \
    dataset.view_sampler.num_target_views=8 \
    dataset.view_sampler.min_distance_between_context_views=150 \
    dataset.view_sampler.max_distance_between_context_views=350 \
    dataset.view_sampler.initial_min_distance_between_context_views=100 \
    dataset.view_sampler.initial_max_distance_between_context_views=200 \
    dataset.image_shape=[512,960] \
    dataset.ori_image_shape=[540,960] \
    trainer.max_steps=50000 \
    trainer.num_nodes=4 \
    train.half_res_lpips_loss=true \
    model.encoder.gaussian_adapter.gaussian_scale_max=3. \
    model.encoder.depth_pred_half_res=true \
    model.encoder.init_use_checkpointing=true \
    optimizer.lr=1e-4 \
    optimizer.lr_monodepth=1e-6 \
    checkpointing.pretrained_model=<view8_512x960_init_checkpoint> \
    wandb.project=dl3dv-view16-540x960 \
    output_dir=checkpoints/resplat/dl3dv-view16-540x960/base-init

# NOTE: You can use the released pretrained model (which includes both init and refine weights)
# with checkpointing.no_strict_load=true, or use your own init-only checkpoint without no_strict_load.

# eval on dl3dv longlrm split
# full 140 scene, 540x960, no crop
# psnr 22.689
# ssim 0.742
# lpips 0.307
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
    mode=test \
    dataset.roots=[datasets/dl3dv_960p_benchmark_torch] \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.num_context_views=16 \
    dataset.view_sampler.index_path=assets/dl3dv_evaluation_longlrm/dl3dv_longlrm_ctx16_tgt_every8th.json \
    dataset.image_shape=[540,960] \
    dataset.ori_image_shape=[540,960] \
    train.half_res_lpips_loss=true \
    model.encoder.gaussian_adapter.gaussian_scale_max=3. \
    model.encoder.depth_pred_half_res=true \
    model.encoder.init_use_checkpointing=true \
    checkpointing.pretrained_model=pretrained/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth \
    checkpointing.no_strict_load=true \
    model.encoder.no_crop_image=true

