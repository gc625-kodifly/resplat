#!/usr/bin/env bash
# To resume training, add: checkpointing.resume=true output_dir=<original_output_dir> wandb.id=<wandb_run_id>
# To disable wandb logging, add: wandb.mode=disabled

# small model (1 node, 4 gpus)
python -m src.main +experiment=dl3dv \
    data_loader.train.batch_size=2 \
    dataset.view_sampler.num_context_views=8 \
    dataset.view_sampler.num_target_views=6 \
    dataset.view_sampler.min_distance_between_context_views=24 \
    dataset.view_sampler.max_distance_between_context_views=45 \
    dataset.view_sampler.initial_min_distance_between_context_views=20 \
    dataset.view_sampler.initial_max_distance_between_context_views=30 \
    dataset.image_shape=[256,448] \
    trainer.max_steps=100000 \
    model.encoder.monodepth_vit_type=vits \
    model.encoder.gaussian_regressor_channels=256 \
    checkpointing.pretrained_depth=pretrained/resplat-depth-small-352x640-b0ebc084.pth \
    wandb.project=dl3dv-view8-256x448 \
    output_dir=checkpoints/resplat/dl3dv-view8-256x448/small-init


# eval
# NOTE: You can use the released pretrained model (which includes both init and refine weights)
# with checkpointing.no_strict_load=true, or use your own init-only checkpoint without no_strict_load.
# dl3dv 8 view
# psnr 26.767
# ssim 0.865
# lpips 0.142

CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
    mode=test \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.num_context_views=8 \
    dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_40_ctx_8v_tgt_8v.json \
    dataset.image_shape=[256,448] \
    model.encoder.monodepth_vit_type=vits \
    model.encoder.gaussian_regressor_channels=256 \
    checkpointing.pretrained_model=pretrained/resplat-small-dl3dv-256x448-view8-548993fe.pth \
    checkpointing.no_strict_load=true

# to save the rendered images and gt images, add:
    # test.save_image=true \
    # test.save_gt_image=true \
    # output_dir=results/resplat-dl3dv-view8-256x448-small-init
