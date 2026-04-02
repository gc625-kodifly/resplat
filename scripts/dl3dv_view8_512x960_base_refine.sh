#!/usr/bin/env bash
# To resume training, add: checkpointing.resume=true output_dir=<original_output_dir> wandb.id=<wandb_run_id>
# To disable wandb logging, add: wandb.mode=disabled

# refine, 8 views, 512x960 (4x nodes, 4 gpus per node, batch size 1 per gpu)
# can also be trained with 1 node and 4 gpus, but with 4x more steps (120k instead of 30k)
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
    dataset.pose_align_middle_view=true \
    dataset.image_shape=[512,960] \
    dataset.ori_image_shape=[540,960] \
    trainer.max_steps=30000 \
    trainer.num_nodes=4 \
    train.depth_smooth_loss_weight=0. \
    model.encoder.num_refine=4 \
    model.encoder.train_min_refine=1 \
    model.encoder.train_max_refine=4 \
    model.encoder.recurrent_use_checkpointing=true \
    optimizer.lr=1e-4 \
    optimizer.lr_monodepth=0. \
    checkpointing.pretrained_model=<view8_512x960_init_checkpoint> \
    checkpointing.no_strict_load=true \
    checkpointing.resume_update_module=<view8_256x448_refine_checkpoint> \
    wandb.project=dl3dv-view8-512x960 \
    output_dir=checkpoints/resplat/dl3dv-view8-512x960/base-refine


# eval

# dl3dv 8 view

# refine 1
# psnr 27.154
# ssim 0.859
# lpips 0.169

# refine 2
# psnr 27.513
# ssim 0.865
# lpips 0.163

# refine 3
# psnr 27.647
# ssim 0.867
# lpips 0.161

# refine 4
# psnr 27.698
# ssim 0.868
# lpips 0.160

# refine 5
# psnr 27.712
# ssim 0.868
# lpips 0.160


# 1/10 subset, refine 2
# psnr 29.348
# ssim 0.912
# lpips 0.128

CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
    mode=test \
    dataset.roots=[datasets/dl3dv_960p_benchmark_torch] \
    dataset.test_chunk_interval=1 \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.num_context_views=8 \
    dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_60_ctx_8v_tgt_8v.json \
    dataset.pose_align_middle_view=true \
    dataset.image_shape=[512,960] \
    dataset.ori_image_shape=[540,960] \
    model.encoder.num_refine=2 \
    checkpointing.pretrained_model=pretrained/resplat-base-dl3dv-512x960-view8-8179ed87.pth 


# generalization to different number of views
# 8 views
# dataset.view_sampler.num_context_views=8 \
# dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_60_ctx_8v_tgt_8v.json \
# psnr 27.698
# ssim 0.868
# lpips 0.160

# 10 views
# dataset.view_sampler.num_context_views=10 \
# dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_60_ctx_8v_tgt_8v_add2views.json \
# psnr 28.184
# ssim 0.877
# lpips 0.151

# 12 views
# dataset.view_sampler.num_context_views=12 \
# dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_60_ctx_8v_tgt_8v_add4views.json \
# psnr 28.472
# ssim 0.883
# lpips 0.144

# 14 views
# dataset.view_sampler.num_context_views=14 \
# dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_60_ctx_8v_tgt_8v_add6views.json \
# psnr 28.747
# ssim 0.888
# lpips 0.139

# 16 views
# dataset.view_sampler.num_context_views=16 \
# dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_60_ctx_8v_tgt_8v_add8views.json \
# psnr 28.918
# ssim 0.892
# lpips 0.135


CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
    mode=test \
    dataset.roots=[datasets/dl3dv_960p_benchmark_torch] \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.num_context_views=16 \
    dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_60_ctx_8v_tgt_8v_add8views.json \
    dataset.pose_align_middle_view=true \
    dataset.image_shape=[512,960] \
    dataset.ori_image_shape=[540,960] \
    model.encoder.num_refine=4 \
    checkpointing.pretrained_model=pretrained/resplat-base-dl3dv-512x960-view8-8179ed87.pth



# dl3dv generalization to different resolutions
# full set
# 448x832
# refine 1
# psnr 26.895
# ssim 0.856
# lpips 0.166

# refine 2
# psnr 27.350
# ssim 0.864
# lpips 0.159

# refine 3
# psnr 27.511
# ssim 0.866
# lpips 0.157

# refine 4
# psnr 27.574
# ssim 0.867
# lpips 0.156


# 416x768
# refine 1
# psnr 26.201
# ssim 0.844
# lpips 0.173

# refine 2
# psnr 26.868
# ssim 0.854
# lpips 0.164

# refine 3
# psnr 27.085
# ssim 0.858
# lpips 0.160

# refine 4
# psnr 27.171
# ssim 0.860
# lpips 0.159


# 320x640
# refine 1
# psnr 23.898
# ssim 0.798
# lpips 0.215

# refine 2
# psnr 25.393
# ssim 0.824
# lpips 0.191

# refine 3
# psnr 25.891
# ssim 0.834
# lpips 0.181

# refine 4
# psnr 26.075
# ssim 0.837
# lpips 0.178


CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
    mode=test \
    dataset.roots=[datasets/dl3dv_960p_benchmark_torch] \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.num_context_views=8 \
    dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_60_ctx_8v_tgt_8v.json \
    dataset.pose_align_middle_view=true \
    dataset.image_shape=[320,640] \
    dataset.ori_image_shape=[540,960] \
    model.encoder.num_refine=4 \
    checkpointing.pretrained_model=pretrained/resplat-base-dl3dv-512x960-view8-8179ed87.pth



# generalization to different number of views
# 6 view
# assets/dl3dv_evaluation/dl3dv_start_0_distance_60_ctx_6v_tgt_8v.json
# refine 1
# psnr 25.698
# ssim 0.826
# lpips 0.198

# refine 2
# psnr 25.795
# ssim 0.827
# lpips 0.196

# refine 3
# psnr 25.816
# ssim 0.827
# lpips 0.196


# 4 view
# assets/dl3dv_evaluation/dl3dv_start_0_distance_50_ctx_4v_tgt_every8th.json
# refine 1
# psnr 24.503
# ssim 0.792
# lpips 0.230

# refine 2
# psnr 24.587
# ssim 0.794
# lpips 0.229

# refine 3
# psnr 24.605
# ssim 0.794
# lpips 0.228


CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
    mode=test \
    dataset.roots=[datasets/dl3dv_960p_benchmark_torch] \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.num_context_views=4 \
    dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_50_ctx_4v_tgt_every8th.json \
    dataset.image_shape=[512,960] \
    dataset.ori_image_shape=[540,960] \
    model.encoder.num_refine=3 \
    checkpointing.pretrained_model=pretrained/resplat-base-dl3dv-512x960-view8-8179ed87.pth



# render video
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \
    mode=test \
    dataset.roots=[datasets/dl3dv_960p_benchmark_torch] \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.num_context_views=8 \
    dataset.view_sampler.index_path=assets/dl3dv_evaluation/dl3dv_start_0_distance_80_ctx_8v_video.json \
    dataset.image_shape=[540,960] \
    dataset.ori_image_shape=[540,960] \
    model.encoder.num_refine=4 \
    checkpointing.pretrained_model=pretrained/resplat-base-dl3dv-512x960-view8-8179ed87.pth \
    model.encoder.no_crop_image=true \
    test.save_video=true \
    test.compute_scores=false \
    test.render_chunk_size=10 \
    test.stablize_camera=true \
    output_dir=results/resplat-dl3dv-view8-512x960-base-refine-rendervideo

