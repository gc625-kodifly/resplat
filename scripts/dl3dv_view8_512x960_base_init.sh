#!/usr/bin/env bash
# To resume training, add: checkpointing.resume=true output_dir=<original_output_dir> wandb.id=<wandb_run_id>
# To disable wandb logging, add: wandb.mode=disabled

# init, 8 views, 512x960 (finetune from 256x448 init) (4x nodes, 4 gpus per node, batch size 1 per gpu)
# can also be trained with 1 node and 4 gpus, but with 4x more steps (200k instead of 50k)
python -m src.main +experiment=dl3dv \
    data_loader.train.batch_size=1 \
    trainer.eval_index=assets/dl3dv_evaluation/dl3dv_start_0_distance_60_ctx_8v_tgt_8v.json \
    dataset.roots=[datasets/dl3dv_960p] \
    dataset.view_sampler.num_context_views=8 \
    dataset.view_sampler.num_target_views=6 \
    dataset.view_sampler.min_distance_between_context_views=30 \
    dataset.view_sampler.max_distance_between_context_views=60 \
    dataset.view_sampler.initial_min_distance_between_context_views=24 \
    dataset.view_sampler.initial_max_distance_between_context_views=40 \
    dataset.image_shape=[512,960] \
    dataset.ori_image_shape=[540,960] \
    trainer.max_steps=50000 \
    trainer.num_nodes=4 \
    optimizer.lr=1e-4 \
    optimizer.lr_monodepth=1e-6 \
    checkpointing.pretrained_model=<view8_256x448_init_checkpoint> \
    wandb.project=dl3dv-view8-512x960 \
    output_dir=checkpoints/resplat/dl3dv-view8-512x960/base-init-4node


# eval
# NOTE: You can use the released pretrained model (which includes both init and refine weights)
# with checkpointing.no_strict_load=true, or use your own init-only checkpoint without no_strict_load.

# dl3dv 8 view
# psnr 26.214
# ssim 0.842
# lpips 0.185
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
    mode=test \
    dataset.roots=[datasets/dl3dv_960p_benchmark_torch] \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.num_context_views=8 \
    dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_60_ctx_8v_tgt_8v.json \
    dataset.image_shape=[512,960] \
    dataset.ori_image_shape=[540,960] \
    checkpointing.pretrained_model=pretrained/resplat-base-dl3dv-512x960-view8-8179ed87.pth \
    checkpointing.no_strict_load=true 


# generalization to different number of views
# 8 views
# dataset.view_sampler.num_context_views=8 \
# dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_60_ctx_8v_tgt_8v.json \
# psnr 26.214
# ssim 0.842
# lpips 0.185

# 10 views
# dataset.view_sampler.num_context_views=10 \
# dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_60_ctx_8v_tgt_8v_add2views.json \
# psnr 26.610
# ssim 0.852
# lpips 0.176

# 12 views
# dataset.view_sampler.num_context_views=12 \
# dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_60_ctx_8v_tgt_8v_add4views.json \
# psnr 26.770
# ssim 0.857
# lpips 0.171

# 14 views
# dataset.view_sampler.num_context_views=14 \
# dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_60_ctx_8v_tgt_8v_add6views.json \
# psnr 26.884
# ssim 0.862
# lpips 0.166


# 16 views
# dataset.view_sampler.num_context_views=16 \
# dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_60_ctx_8v_tgt_8v_add8views.json \
# psnr 26.898
# ssim 0.865
# lpips 0.164


CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
    mode=test \
    dataset.roots=[datasets/dl3dv_960p_benchmark_torch] \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.num_context_views=16 \
    dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_60_ctx_8v_tgt_8v_add8views.json \
    dataset.image_shape=[512,960] \
    dataset.ori_image_shape=[540,960] \
    checkpointing.pretrained_model=pretrained/resplat-base-dl3dv-512x960-view8-8179ed87.pth \
    checkpointing.no_strict_load=true



# eval
# dl3dv generalization to different resolutions
# dl3dv 8 view
# 448x832
# psnr 25.468
# ssim 0.833
# lpips 0.187

# 416x768
# psnr 24.136
# ssim 0.808
# lpips 0.207

# 320x640
# psnr 20.896
# ssim 0.726
# lpips 0.276


CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
    mode=test \
    dataset.roots=[datasets/dl3dv_960p_benchmark_torch] \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.num_context_views=8 \
    dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_60_ctx_8v_tgt_8v.json \
    dataset.image_shape=[416,768] \
    dataset.ori_image_shape=[540,960] \
    checkpointing.pretrained_model=pretrained/resplat-base-dl3dv-512x960-view8-8179ed87.pth \
    checkpointing.no_strict_load=true



# generalization to different number of views
# 6 view
# assets/dl3dv_evaluation/dl3dv_start_0_distance_60_ctx_6v_tgt_8v.json
# psnr 25.223
# ssim 0.819
# lpips 0.208

# 4 view
# assets/dl3dv_evaluation/dl3dv_start_0_distance_50_ctx_4v_tgt_every8th.json
# psnr 24.056
# ssim 0.786
# lpips 0.240


CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
    mode=test \
    dataset.roots=[datasets/dl3dv_960p_benchmark_torch] \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.num_context_views=6 \
    dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_60_ctx_6v_tgt_8v.json \
    dataset.image_shape=[512,960] \
    dataset.ori_image_shape=[540,960] \
    checkpointing.pretrained_model=pretrained/resplat-base-dl3dv-512x960-view8-8179ed87.pth \
    checkpointing.no_strict_load=true


