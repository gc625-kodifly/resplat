#!/usr/bin/env bash
# To resume training, add: checkpointing.resume=true output_dir=<original_output_dir> wandb.id=<wandb_run_id>
# To disable wandb logging, add: wandb.mode=disabled

# Refinement training (4x nodes, 4 gpus per node, batch size 2 per gpu)
# can also be trained with 1 node and 4 gpus, but with 4x more steps (120k instead of 30k)
python -m src.main +experiment=dl3dv \
    data_loader.train.batch_size=1 \
    dataset.view_sampler.num_context_views=8 \
    dataset.view_sampler.num_target_views=6 \
    dataset.view_sampler.min_distance_between_context_views=24 \
    dataset.view_sampler.max_distance_between_context_views=45 \
    dataset.view_sampler.initial_min_distance_between_context_views=20 \
    dataset.view_sampler.initial_max_distance_between_context_views=30 \
    dataset.image_shape=[256,448] \
    dataset.pose_align_middle_view=true \
    trainer.max_steps=30000 \
    trainer.num_nodes=4 \
    train.depth_smooth_loss_weight=0. \
    model.encoder.num_refine=4 \
    model.encoder.train_min_refine=1 \
    model.encoder.train_max_refine=4 \
    optimizer.lr=1e-4 \
    optimizer.lr_monodepth=0. \
    checkpointing.pretrained_model=<init_checkpoint> \
    checkpointing.no_strict_load=true \
    wandb.project=dl3dv-view8-256x448 \
    output_dir=checkpoints/resplat/dl3dv-view8-256x448/base-refine


# Evaluation on DL3DV

# refine 1
# psnr 28.523
# ssim 0.896
# lpips 0.113

# refine 2
# psnr 28.959
# ssim 0.902
# lpips 0.107

# refine 3
# psnr 29.131
# ssim 0.904
# lpips 0.104

# refine 4
# psnr 29.198
# ssim 0.904
# lpips 0.104


CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
    mode=test \
    dataset.test_chunk_interval=1 \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.num_context_views=8 \
    dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_40_ctx_8v_tgt_8v.json \
    dataset.pose_align_middle_view=true \
    dataset.image_shape=[256,448] \
    model.encoder.num_refine=2 \
    checkpointing.pretrained_model=pretrained/resplat-base-dl3dv-256x448-view8-1934a04c.pth

