#!/usr/bin/env bash
# To resume training, add: checkpointing.resume=true output_dir=<original_output_dir> wandb.id=<wandb_run_id>
# To disable wandb logging, add: wandb.mode=disabled

# large model (8x nodes, 4 gpus per node, batch size 2 per gpu)
# can also be trained with 1 node and 4 gpus, but with 8x more steps (400k instead of 50k)
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
    trainer.num_nodes=8 \
    model.encoder.monodepth_vit_type=vitl \
    model.encoder.gaussian_regressor_channels=768 \
    checkpointing.pretrained_depth=pretrained/resplat-depth-large-352x640-05f9beac.pth \
    wandb.project=dl3dv-view8-256x448 \
    output_dir=checkpoints/resplat/dl3dv-view8-256x448/large-init


# Evaluation on DL3DV
# psnr 27.861
# ssim 0.886
# lpips 0.121
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
    mode=test \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.num_context_views=8 \
    dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_40_ctx_8v_tgt_8v.json \
    dataset.image_shape=[256,448] \
    model.encoder.monodepth_vit_type=vitl \
    model.encoder.gaussian_regressor_channels=768 \
    checkpointing.pretrained_model=pretrained/resplat-large-dl3dv-256x448-view8-62f1703a.pth 

    