#!/usr/bin/env bash
# To resume training, add: checkpointing.resume=true output_dir=<original_output_dir> wandb.id=<wandb_run_id>
# To disable wandb logging, add: wandb.mode=disabled

# base model (4x nodes, 4 gpus per node, batch size 8 per gpu)
# can also be trained with 1 node and 4 gpus, but with 4x more steps (800k instead of 200k)
python -m src.main +experiment=re10k \
    data_loader.train.batch_size=8 \
    dataset.test_chunk_interval=10 \
    trainer.max_steps=200000 \
    trainer.num_nodes=4 \
    model.encoder.init_gaussian_multiple=16 \
    checkpointing.pretrained_depth=pretrained/resplat-depth-base-352x640-60be7abf.pth \
    wandb.project=re10k-view2-256x256 \
    output_dir=checkpoints/resplat/re10k-view2-256x256/base-init



# eval on re10k
# NOTE: You can use the released pretrained model (which includes both init and refine weights)
# with checkpointing.no_strict_load=true, or use your own init-only checkpoint without no_strict_load.
# psnr 29.055
# ssim 0.905
# lpips 0.107
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
    mode=test \
    dataset.test_chunk_interval=1 \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.num_context_views=2 \
    model.encoder.init_gaussian_multiple=16 \
    checkpointing.pretrained_model=pretrained/resplat-base-re10k-256x256-view2-b90d1b53.pth \
    checkpointing.no_strict_load=true



# generalization to acid
# psnr 28.848
# ssim 0.854
# lpips 0.148
CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=re10k \
    mode=test \
    dataset.test_chunk_interval=1 \
    dataset.roots=[datasets/acid] \
    dataset.view_sampler.index_path=assets/evaluation_index_acid.json \
    dataset/view_sampler=evaluation \
    dataset.view_sampler.num_context_views=2 \
    model.encoder.init_gaussian_multiple=16 \
    checkpointing.pretrained_model=pretrained/resplat-base-re10k-256x256-view2-b90d1b53.pth \
    checkpointing.no_strict_load=true

