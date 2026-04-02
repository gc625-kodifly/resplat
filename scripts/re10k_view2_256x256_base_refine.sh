#!/usr/bin/env bash
# To resume training, add: checkpointing.resume=true output_dir=<original_output_dir> wandb.id=<wandb_run_id>
# To disable wandb logging, add: wandb.mode=disabled

# refine model (4x nodes, 4 gpus per node, batch size 2 per gpu)
# can also be trained with 1 node and 4 gpus, but with 4x more steps (400k instead of 100k)
python -m src.main +experiment=re10k \
    data_loader.train.batch_size=2 \
    dataset.test_chunk_interval=10 \
    dataset.pose_align_middle_view=true \
    trainer.max_steps=100000 \
    trainer.num_nodes=4 \
    train.depth_smooth_loss_weight=0. \
    model.encoder.latent_downsample=2 \
    model.encoder.fixed_latent_size=false \
    model.encoder.init_gaussian_multiple=4 \
    model.encoder.num_refine=2 \
    model.encoder.refine_same_num_points=true \
    optimizer.lr=1e-4 \
    optimizer.lr_monodepth=0. \
    checkpointing.pretrained_model=<init_checkpoint> \
    checkpointing.no_strict_load=true \
    wandb.project=re10k-view2-256x256 \
    output_dir=checkpoints/resplat/re10k-view2-256x256/base-refine


# eval on re10k
# psnr 29.751
# ssim 0.912
# lpips 0.100

# 1/100 subset
# psnr 29.218
# ssim 0.907
# lpips 0.103

CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
    mode=test \
    dataset.test_chunk_interval=100 \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.num_context_views=2 \
    dataset.pose_align_middle_view=true \
    model.encoder.latent_downsample=2 \
    model.encoder.fixed_latent_size=false \
    model.encoder.init_gaussian_multiple=4 \
    model.encoder.num_refine=2 \
    model.encoder.refine_same_num_points=true \
    checkpointing.pretrained_model=pretrained/resplat-base-re10k-256x256-view2-b90d1b53.pth 



# generalization to acid
# psnr 29.870
# ssim 0.864
# lpips 0.135
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
    mode=test \
    dataset.test_chunk_interval=1 \
    dataset.roots=[datasets/acid] \
    dataset.view_sampler.index_path=assets/evaluation_index_acid.json \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.num_context_views=2 \
    model.encoder.latent_downsample=2 \
    model.encoder.fixed_latent_size=false \
    model.encoder.init_gaussian_multiple=4 \
    model.encoder.num_refine=2 \
    model.encoder.refine_same_num_points=true \
    checkpointing.pretrained_model=pretrained/resplat-base-re10k-256x256-view2-b90d1b53.pth

