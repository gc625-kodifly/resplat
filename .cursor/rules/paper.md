# ReSplat: Learning Recurrent Gaussian Splats

**Paper**: [arXiv:2510.08575](https://arxiv.org/abs/2510.08575)
**Authors**: Haofei Xu, Daniel Barath, Andreas Geiger, Marc Pollefeys (ETH Zurich, University of Tübingen, Microsoft)

## Core Idea

ReSplat is a **feed-forward recurrent model** for 3D Gaussian splatting that iteratively refines Gaussians using the **rendering error as a gradient-free feedback signal**. Unlike prior methods that perform a single forward pass, ReSplat decomposes reconstruction into:

1. **Compact initialization** — predict Gaussians in a 16x subsampled space (16x fewer Gaussians than per-pixel methods)
2. **Recurrent refinement** — a weight-sharing recurrent module predicts per-Gaussian parameter deltas guided by rendering error

The rendering error feedback naturally adapts to unseen data distributions at test time, enabling robust generalization across datasets, view counts, and resolutions.

## Pipeline Overview

```
N posed images → Depth prediction (1/4 res) → Unproject to 3D point cloud (M = N×H×W/16)
→ kNN + global attention (Point Transformer) → Gaussian head (MLP) → Initial Gaussians G⁰

For t = 0..T-1:
  Render input views with current Gaussians Gᵗ
  Compute rendering error: pixel-space (RGB diff) + feature-space (ResNet-18 features diff)
  Global attention on error → propagate to Gaussians
  Update module (kNN attention) → predict Δgᵗ, Δzᵗ
  Gᵗ⁺¹ = Gᵗ + ΔGᵗ (additive updates)

Final Gaussians → gsplat rasterization → novel views
```

## Architecture Details

### Initial Gaussian Reconstruction (Section 3.1)

- **Depth prediction**: `MultiViewUniMatch` — combines DINOv2-style monocular features with multi-view cost volume matching. Produces per-view depth at 1/4 resolution (for 16x Gaussian compression).
- **Point cloud construction**: Unproject depth to 3D via camera parameters → M = N × H/4 × W/4 points, each with feature vector fⱼ.
- **3D context aggregation**: 6 alternating blocks of kNN attention (k=16, Point Transformer) and global attention (with pixel unshuffle to N×H/16×W/16 for efficiency, then pixel shuffle back).
- **Gaussian head**: 2-layer MLP decodes per-point Gaussian parameters (position offset, quaternion rotation, scale, opacity, spherical harmonics).
- **SH initialization**: DC component initialized from input RGB values (`RGB2SH`).
- **Hidden state**: z⁰ⱼ = f*ⱼ (context-aggregated features serve as initial hidden state for recurrence).

### Recurrent Gaussian Update (Section 3.2)

The update module is only active when `num_refine > 0`. Converges after ~4 iterations.

**Per iteration t:**

1. **Render context views** using current Gaussians via the gsplat decoder.
2. **Compute rendering error**:
   - Pixel-space: `I_rendered - I_gt`, pixel-unshuffled 4x to match Gaussian resolution.
   - Feature-space: frozen ResNet-18 extracts 3-scale features (1/2, 1/4, 1/8 res) from both rendered and GT images, resized to 1/4 and concatenated. Difference computed by subtraction.
   - Combined via element-wise addition (pixel error projected via linear+LayerNorm to match feature dims).
3. **Error propagation**: Global attention on the N×H/4×W/4 error features so each Gaussian receives info from all rendered pixels, not just its local pixel.
4. **Update prediction**: Concatenate (detached Gaussian params gᵗ, hidden state zᵗ, error eᵗ) → `PointLinearWrapper` + 4 kNN attention blocks → 4-layer MLP update head (zero-initialized to start as identity).
5. **Additive update**: gᵗ⁺¹ = gᵗ + Δgᵗ, zᵗ⁺¹ = zᵗ + Δzᵗ.

Key design: previous Gaussians are **detached** (no gradient through past rendering) — the error signal guides updates in a gradient-free, feed-forward manner.

### Decoder

- **gsplat** library (`rasterization` from gsplat v1.5.3) with Mip-Splatting for antialiased rendering.
- Renders both RGB and depth.
- Supports packed mode for variable Gaussian counts.

## Coordinate System

- **Camera intrinsics**: normalized (row 1 / width, row 2 / height).
- **Camera extrinsics**: OpenCV convention, camera-to-world (+X right, +Y down, +Z into screen).
- **Global reference frame**: aligned to the **middle input view** (empirically best — balances spatial distribution of Gaussians).

## Training

Two-stage training:

### Stage 1: Initial Gaussian Prediction
- **Loss**: L1 rendering loss + perceptual loss (VGG, λ=0.5) on rendered target views + edge-aware depth smoothness (α=0.01) on input view depth maps.
- Trains the full encoder pipeline (depth predictor + Point Transformer + Gaussian head).

### Stage 2: Recurrent Refinement
- **Freeze** the initialization model; train only the update module.
- **Loss**: Rendering loss on all intermediate predictions with exponentially increasing weights (γ=0.9).
- Training randomly samples T ∈ [1, 4] iterations per step.
- Progressive training: 256×448 (8 views) → 512×960 (8 views) → 512×960 (16 views).
- Each stage: 50K steps init + 30K steps refine on 16 GH200 GPUs.

### Optimizer
- AdamW with cosine learning rate schedule.

## Key Results

| Setup | PSNR | SSIM | LPIPS | #Gaussians |
|-------|------|------|-------|------------|
| DL3DV 8v 512×960, iter=0 (init only) | 26.21 | 0.842 | 0.185 | 246K |
| DL3DV 8v 512×960, iter=4 | 27.70 | 0.868 | 0.160 | 246K |
| DL3DV 16v 540×960, iter=2 | 23.51 | 0.766 | 0.284 | 518K |
| RE10K 2v 256×256 | 29.75 | 0.912 | 0.100 | — |

- **100x faster** than optimization-based 3DGS.
- **16x fewer Gaussians** than per-pixel methods (MVSplat, DepthSplat).
- **4x faster rendering** than per-pixel methods.
- Recurrent refinement adds +1.5 to +3.5 dB PSNR over initialization.

## Generalization Properties

The rendering error feedback enables robust generalization to:
- **Unseen datasets** (train DL3DV → test RealEstate10K): recurrent model improves more than init-only.
- **Different view counts** (train 8v → test 16, 32 views): recurrent model exploits extra views better.
- **Different resolutions** (train 512×960 → test 320×640): up to +5 dB improvement via recurrence.

## Model Variants

| Variant | Backbone | Total Params | Init Params | Refine Params |
|---------|----------|-------------|-------------|---------------|
| Small | ViT-S | 76M | 62M | 14M |
| Base | ViT-B | 223M | 209M | 14M |
| Large | ViT-L | 559M | — | 14M |

The recurrent module is always 14M parameters regardless of backbone size.

## Code-to-Paper Mapping

| Paper Concept | Code Location |
|---------------|---------------|
| Full pipeline | `src/model/model_wrapper.py` (ModelWrapper) |
| Encoder (init) | `src/model/encoder/encoder_resplat.py` (EncoderReSplat) |
| Depth prediction | `src/model/encoder/unimatch/mv_unimatch.py` (MultiViewUniMatch) |
| Point Transformer (kNN + global attn) | `src/model/encoder/encoder_resplat.py` → `PlainPointTransformer` |
| Gaussian head | `src/model/encoder/encoder_resplat.py` → `gaussian_head` (MLP) |
| Gaussian adapter (unproject, covariance) | `src/model/encoder/encoder_resplat.py` → inline in forward() |
| Recurrent update | `src/model/encoder/encoder_resplat.py` → `forward_update()` |
| Error features (ResNet-18) | `src/model/encoder/encoder_resplat.py` → `ResNetFeatureWrapper` |
| Error global attention | `src/model/encoder/encoder_resplat.py` → `MultViewLowresAttn` |
| Update module | `src/model/encoder/encoder_resplat.py` → `update_module` (PointLinearWrapper + PT) |
| Update head | `src/model/encoder/encoder_resplat.py` → `update_head` (MLP, zero-init) |
| Decoder (gsplat) | `src/model/decoder/gsplat_decoder_splatting_cuda.py` |
| Gaussian types | `src/model/types.py` (Gaussians dataclass) |
| Losses | `src/loss/` (MSE, LPIPS, depth smoothness) |
| Config system | `src/config.py` + `config/` (Hydra YAML) |
| COLMAP inference | `scripts/infer_colmap.py` (standalone, MODEL_PRESETS) |
| Training entry | `src/main.py` (Hydra + PyTorch Lightning) |
| Custom CUDA ops (kNN) | `src/model/encoder/pointops/` |

## Ablation Insights

- **Rendering error is critical**: removing it drops -1.9 dB. Feature-space error > RGB error. Combining via addition is best.
- **kNN attention in init model**: most important component for maintaining quality with 16x compression.
- **Hidden state in recurrent model**: pivotal — encodes high-level features, not just raw Gaussian params.
- **Weight-sharing recurrence** outperforms non-weight-sharing stacked networks (even with 4x more params).
- **ResNet-18** features outperform DINOv2 for error computation (better spatial fidelity from conv architecture).
- **Middle-view coordinate system** is +0.9 dB over COLMAP's default global frame.
