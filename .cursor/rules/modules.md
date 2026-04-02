# ReSplat Tensor Flow — Every Function, Every Shape

Captured from an actual inference run: preset `dl3dv_16v_540x960`, DL3DV COLMAP scene.
Notation: B=1, V=16 context views, H=512, W=896, Vt=44 target views.

---

## Stage 0: Data Loading & Batch Construction

### 0.1 COLMAP loading

```
load_colmap_scene()                                → scene_data dict
  read_intrinsics_binary("sparse/0/cameras.bin")   → cam_intrinsics: {id: ColmapCamera(width, height, params)}
  read_extrinsics_binary("sparse/0/images.bin")    → cam_extrinsics: {id: ColmapImage(qvec, tvec, name)}
  qvec2rotmat(img.qvec)                            → R: np.ndarray [3, 3]
  np.linalg.inv(w2c)                               → c2w: np.ndarray [4, 4]  (per image)
  np.stack(c2w_list)                                → scene_data["c2w"]: [60, 4, 4]
  np.stack(intrinsics_list)                         → scene_data["intrinsics"]: [60, 3, 3] (normalized: fx/W, fy/H, cx/W, cy/H)
```

### 0.2 Frame subsetting

```
subset_scene_data(scene_data, start_frame=0, frame_distance=60)
  → scene_data with 60 images (indices 0..59)
```

### 0.3 Resolution computation

```
compute_target_shape(orig_h=537, orig_w=954, max_resolution=960)
  scale = 960 / max(537,954) = 1.006...
  scale >= 1.0 → h,w = 537,954
  h = (537 // 64) * 64 = 512
  w = (954 // 64) * 64 = 896
  → (512, 896)
```

### 0.4 View selection

```
select_context_views(c2w=[60,4,4], num_context=16, strategy="fps")
  positions = c2w[:, :3, 3]                        → torch.Tensor [1, 60, 3]
  farthest_point_sample(positions, 16)              → centroids: [1, 16]  (indices into 60 views)
  np.sort(indices)                                  → context_indices: np.ndarray [16]

select_target_views(60, context_indices, "remaining")
  np.setdiff1d(np.arange(60), context_indices)     → target_indices: np.ndarray [44]
```

### 0.5 Image loading

```
load_and_preprocess_images(context_paths, 512, 896)
  PIL.Image.open(path).convert("RGB")              → PIL Image (537×954)
  img.resize((896, 512), Image.LANCZOS)             → PIL Image (512×896)
  torchvision.transforms.ToTensor()(img)            → [3, 512, 896] float32 [0,1]
  torch.stack(images, dim=0)                        → context_images: [16, 3, 512, 896]

load_and_preprocess_images(target_paths, 512, 896)  → target_images: [44, 3, 512, 896]
```

### 0.6 Batch construction

```
build_batch(context_images, target_images, context_c2w, target_c2w, ...)
  torch.cat([context_c2w, target_c2w], dim=0)       → all_c2w: [60, 4, 4]
  mid_idx = 16 // 2 = 8

  camera_normalization(context_c2w[8:9], all_c2w)
    torch.inverse(pivotal_pose)                     → camera_norm_matrix: [1, 4, 4]
    torch.bmm(camera_norm_matrix.repeat(60,1,1), all_c2w)  → all_c2w_aligned: [60, 4, 4]

  context_c2w_aligned = all_c2w_aligned[:16]        → [16, 4, 4]
  target_c2w_aligned = all_c2w_aligned[16:]         → [44, 4, 4]

  context_images.unsqueeze(0)                       → batch["context"]["image"]:      [1, 16, 3, 512, 896]
  context_c2w_aligned.unsqueeze(0)                  → batch["context"]["extrinsics"]:  [1, 16, 4, 4]
  context_K.unsqueeze(0)                            → batch["context"]["intrinsics"]:  [1, 16, 3, 3]
  torch.full((1, 16), 0.01)                         → batch["context"]["near"]:        [1, 16]
  torch.full((1, 16), 200.0)                        → batch["context"]["far"]:         [1, 16]
  target_images.unsqueeze(0)                        → batch["target"]["image"]:        [1, 44, 3, 512, 896]
  target_c2w_aligned.unsqueeze(0)                   → batch["target"]["extrinsics"]:   [1, 44, 4, 4]
  target_K.unsqueeze(0)                             → batch["target"]["intrinsics"]:   [1, 44, 3, 3]

  move_to_device(batch, "cuda:0")                   → all tensors moved to GPU
```

### 0.7 Data shim

```
data_shim(batch)                                    (from apply_patch_shim)
  → batch unchanged (patch_size=16*4=64 since no_crop_image=false; 512 and 896 already divisible by 64)
```

---

## Stage 1: Encoder Forward — `EncoderReSplat.forward()` (encoder_resplat.py:343)

**Input:** `context` dict from batch. `b, v, _, h, w = context["image"].shape` → b=1, v=16, h=512, w=896

### 1.1 Camera distance matrix (for kNN neighbor selection)

```
context["extrinsics"][:, :, :3, -1].detach()        → xyzs: [1, 16, 3]
torch.cdist(xyzs, xyzs, p=2)                        → cameras_dist_matrix: [1, 16, 16]
torch.argsort(cameras_dist_matrix)                   → cameras_dist_index_full: [1, 16, 16]
cameras_dist_index_full[:, :, :(local_mv_match+1)]   → cameras_dist_index: [1, 16, local_mv_match+1]
```

### 1.2 Depth prediction (half-resolution path)

```
rearrange(context["image"], "b v c h w -> (b v) c h w")  → [16, 3, 512, 896]
F.interpolate(..., scale_factor=0.5, mode='bilinear')     → [16, 3, 256, 448]
rearrange(..., "(b v) c h w -> b v c h w", b=1, v=16)    → half_img: [1, 16, 3, 256, 448]

self.depth_predictor(half_img, ...)                       ← MultiViewUniMatch.forward()
  (internally: DINOv2 ViT-B backbone + cost volume + refinement)
  → results_dict:
      "depth_preds":         list of 1× [1, 16, 256, 448]   (depth at half-res, NOT upsampled)
      "raw_mono_features":   4× [16, 768, 18, 32]           (ViT-B/14 features: 252/14=18, 448/14=32)
      "features_cnn_all_scales": [
                                   [16, 128, 32, 56],        (1/8 of half-res 256×448)
                                   [16, 96, 64, 112],        (1/4 of half-res)
                                   [16, 64, 128, 224],       (1/2 of half-res)
                                 ]
      "features_mv":         1× [16, 128, 32, 56]           (multi-view matching features, 1/8 of half-res)
      "match_probs":         list of 1× [16, D, H', W']     (cost volume softmax)
```

Since `depth_pred_half_res=true`, non-depth features are upsampled 2×:

```
F.interpolate(results_dict[key][i], scale_factor=2, mode='bilinear')
  raw_mono_features:     4× [16, 768, 18, 32]  → 4× [16, 768, 36, 64]
  features_cnn_all_scales:
    [16, 128, 32, 56]   → [16, 128, 64, 112]
    [16, 96, 64, 112]   → [16, 96, 128, 224]
    [16, 64, 128, 224]  → [16, 64, 256, 448]
  features_mv:           [16, 128, 32, 56]   → [16, 128, 64, 112]
  match_probs:           upsampled 2× too

depth = depth_preds[-1]                          → [1, 16, 256, 448]  (stays at half-res)
```

### 1.3 Mono feature alignment (to latent_downsample=4 → 128×224)

```
# Resize each raw_mono_feature to 1/16 of full-res = 32×56
F.interpolate(raw_mono_features[i], size=(512//16, 896//16))
  → 4× [16, 768, 32, 56]

# pixel_shuffle with factor=4: channels /= 16, spatial ×= 4
F.pixel_shuffle(x, upscale_factor=4)
  → 4× [16, 768/16=48, 128, 224]

# concat along channel dim
torch.cat(mono_features, dim=1)
  → mono_features: [16, 48*4=192, 128, 224]
```

### 1.4 CNN feature alignment (to 128×224)

```
cnn_features = results_dict["features_cnn_all_scales"][::-1]
  → [ [16, 64, 256, 448],       (1/2 half-res, upsampled 2×)
       [16, 96, 128, 224],       (1/4 half-res, upsampled 2×)
       [16, 128, 64, 112] ]      (1/8 half-res, upsampled 2×)

target_h, target_w = 512 // 4, 896 // 4 = 128, 224

F.interpolate(cnn_features[0], size=(128, 224))  → [16, 64, 128, 224]
F.interpolate(cnn_features[1], size=(128, 224))  → [16, 96, 128, 224]   (no-op, already 128×224)
F.interpolate(cnn_features[2], size=(128, 224))  → [16, 128, 128, 224]

torch.cat(cnn_features, dim=1)                   → cnn_features: [16, 64+96+128=288, 128, 224]
```

### 1.5 Multi-view feature alignment

```
mv_features = results_dict["features_mv"][0]     → [16, 128, 64, 112]
F.interpolate(mv_features, size=(128, 224))      → mv_features: [16, 128, 128, 224]
```

### 1.6 All-features concat

```
torch.cat((mono_features, cnn_features, mv_features), dim=1)
  → features: [16, 192+288+128=608, 128, 224]
```

### 1.7 Match probability extraction

```
match_prob = results_dict['match_probs'][-1]     → [16, D, H', W']
torch.max(match_prob, dim=1, keepdim=True)[0]    → match_prob: [16, 1, H', W']
F.interpolate(match_prob, size=(128, 224))        → match_prob: [16, 1, 128, 224]
```

### 1.8 Image unshuffle (pixel_unshuffle to latent res)

```
rearrange(context["image"], "b v c h w -> (b v) c h w")  → [16, 3, 512, 896]
F.pixel_unshuffle(..., downscale_factor=4)                → img_unshuffle: [16, 3*4²=48, 128, 224]
```

### 1.9 Latent depth (downsample depth to latent res)

```
# depth is [1, 16, 256, 448] (half-res). Scale to latent = 1/(4//2) = 1/2
F.interpolate(depth, scale_factor=0.5, mode='bilinear')   → latent_depth: [1, 16, 128, 224]
```

### 1.10 Full concat for gaussian_regressor

```
rearrange(latent_depth, "b v h w -> (b v) () h w")        → [16, 1, 128, 224]

torch.cat((img_unshuffle, latent_depth, match_prob, features), dim=1)
  → concat: [16, 48 + 1 + 1 + 608 = 658, 128, 224]
```

### 1.11 Gaussian regressor CNN

```
self.gaussian_regressor(concat)                   ← nn.Sequential:
  nn.Conv2d(658, 512, 3, 1, 1)                    → [16, 512, 128, 224]
  nn.GELU()                                       → [16, 512, 128, 224]
  nn.Conv2d(512, 512, 3, 1, 1)                    → out: [16, 512, 128, 224]
```

### 1.12 Concat for Point Transformer input

```
torch.cat([out, img_unshuffle, features, match_prob], dim=1)
  → out: [16, 512 + 48 + 608 + 1 = 1169, 128, 224]
```

### 1.13 Save condition_features (for refinement later)

```
condition_features = out                          → [16, 1169, 128, 224]
  (note: later overwritten at line 556 with PT output, final shape [16, 512, 128, 224])
```

### 1.14 Linear projection to PT input dimension

```
h, w = latent_depth.shape[-2:] = 128, 224

rearrange(out, "bv c h w -> (bv h w) c")         → [16*128*224 = 458752, 1169]

self.proj(...)                                    ← nn.Linear(1169, 512)
  → tmp_feature: [458752, 512]
```

### 1.15 Build 3D point cloud from depth

```
sample_image_grid((128, 224), device)             ← src/geometry/projection.py
  torch.meshgrid(torch.arange(128), torch.arange(224), indexing="ij")
  coordinates reversed to (x, y) normalized [0,1]
  → xy_ray: [128, 224, 2]

rearrange(xy_ray, "h w xy -> (h w) () xy")       → xy_ray: [28672, 1, 2]

xy_ray.unsqueeze(0).unsqueeze(0).repeat(1, 16, 1, 1, 1)  → tmp_coords: [1, 16, 28672, 1, 2]

rearrange(latent_depth, "b v h w -> b v (h w) () ()")     → tmp_depth: [1, 16, 28672, 1, 1]

context["extrinsics"].unsqueeze(2).unsqueeze(2)           → tmp_extrinsics: [1, 16, 1, 1, 4, 4]
context["intrinsics"].unsqueeze(2).unsqueeze(2)           → tmp_intrinsics: [1, 16, 1, 1, 3, 3]

get_world_rays(tmp_coords, tmp_extrinsics, tmp_intrinsics)  ← src/geometry/projection.py
  unproject(coordinates, ones, intrinsics)
    homogenize_points(coordinates)                → [..., 3]  (append 1)
    einsum(intrinsics.inverse(), coords)          → ray_directions
  directions / directions[..., -1:]               → normalized directions
  homogenize_vectors(directions)                  → [..., 4]  (append 0)
  transform_cam2world(directions, extrinsics)     → world-space directions [..., 4]
  → origins: [1, 16, 28672, 1, 3], directions: [1, 16, 28672, 1, 3]

origins + directions * tmp_depth                  → point_cloud: [1, 16, 28672, 1, 3]

rearrange(point_cloud, "b v h w c -> (b v h w) c")  → point_cloud: [458752, 3]
```

### 1.16 Point Transformer (initial)

```
offset = torch.tensor([16*128*224])               → offset: [1]  (batch boundary marker)

self.pt((point_cloud, tmp_feature, offset), b=1, v=16, h=128, w=224)
  ← PlainPointTransformer(512, knn_samples, num_blocks=6)
  (internally: 6 alternating blocks of kNN attention + multi-view global attention)
  → pt_output: [458752, 512]

out = tmp_feature + pt_output                     → out: [458752, 512]  (residual connection)

rearrange(out, "(bv h w) c -> bv c h w", h=128, w=224)
  → condition_features: [16, 512, 128, 224]      (this OVERWRITES the earlier condition_features)
```

### 1.17 Gaussian head MLP

```
self.gaussian_head(out)                           ← nn.Sequential:
  nn.Linear(512, 58)                              → [458752, 58]
  nn.GELU()                                       → [458752, 58]
  nn.Linear(58, 58)                               → out: [458752, 58]

  58 = 4(quat) + 3(scale) + 1(opacity) + 2(offset) + 48(SH: 3 colors × 16 basis)
```

### 1.18 Reshape Gaussian parameters

```
rearrange(out, "(b v h w) c -> (b v) c h w", b=1, h=128, w=224)  → [16, 58, 128, 224]
gaussians.float()                                                  → [16, 58, 128, 224]
```

### 1.19 Prepare depths for unprojection (init_gaussian_multiple=1 path)

```
rearrange(latent_depth, "b v h w -> b v (h w) () ()")  → depths: [1, 16, 28672, 1, 1]
```

### 1.20 Split raw Gaussian parameters

```
rearrange(gaussians, "(b v) c h w -> b v c h w", b=1, v=16)   → [1, 16, 58, 128, 224]
rearrange(..., "b v c h w -> b v (h w) c")                      → raw_gaussians: [1, 16, 28672, 58]

raw_gaussians.split([4, 3, 1, 2, 48], dim=-1)
  → rotations_unnorm: [1, 16, 28672, 4]
  → scales:           [1, 16, 28672, 3]
  → opacities_raw:    [1, 16, 28672, 1]
  → offset:           [1, 16, 28672, 2]
  → sh:               [1, 16, 28672, 48]
```

### 1.21 Process opacities

```
opacities_raw.sigmoid()                           → opacities: [1, 16, 28672, 1]  (range 0.0003–0.9999)
```

### 1.22 Process scales

```
F.softplus(scales - exp_scale_bias)               → softplus output
torch.clamp(..., min=1e-6, max=3.0)               → scales: [1, 16, 28672, 3]  (range 1e-6 to 3.0)
```

### 1.23 Normalize rotations

```
rotations_unnorm / (rotations_unnorm.norm(dim=-1, keepdim=True) + 1e-8)
  → rotations: [1, 16, 28672, 4]  (unit quaternions)
```

### 1.24 Build covariance matrices

```
build_covariance(scales, rotations)               ← src/model/encoder/common/gaussians.py
  scale.diag_embed()                              → S: [1, 16, 28672, 3, 3]  (diagonal scale matrix)
  quaternion_to_matrix(rotations)                 → R: [1, 16, 28672, 3, 3]  (rotation matrix)
  R @ S @ Sᵀ @ Rᵀ                                → covariances_local: [1, 16, 28672, 3, 3]

context["extrinsics"][..., :3, :3].unsqueeze(2)   → c2w_rotations: [1, 16, 1, 3, 3]

c2w_rotations @ covariances_local @ c2w_rotations.transpose(-1, -2)
  → covariances: [1, 16, 28672, 3, 3]  (world-space)
```

### 1.25 Compute Gaussian means (unproject with sub-pixel offset)

```
# scale_factor=1 for init_gaussian_multiple=1, latent_downsample=4
h, w = 128, 224

sample_image_grid((128, 224), device)             → xy_ray: [128, 224, 2]
rearrange(xy_ray, "h w xy -> (h w) () xy")       → xy_ray: [28672, 1, 2]

offset.sigmoid().unsqueeze(-2)                    → offset_xy: [1, 16, 28672, 1, 2]
pixel_size = 1 / tensor([224, 128])               → [1/224, 1/128]
xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size → xy_ray: [1, 16, 28672, 1, 2]  (sub-pixel adjusted)

get_world_rays(xy_ray, extrinsics.unsqueeze(2).unsqueeze(2), intrinsics.unsqueeze(2).unsqueeze(2))
  → origins: [1, 16, 28672, 1, 3], directions: [1, 16, 28672, 1, 3]

origins + directions * depths                     → means: [1, 16, 28672, 1, 3]
```

### 1.26 Process spherical harmonics

```
rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)  → sh: [1, 16, 28672, 3, 16]

# Downsample input images for SH initialization
rearrange(context["image"], "b v c h w -> (b v) c h w")  → [16, 3, 512, 896]
F.interpolate(..., scale_factor=1/4, mode='area')          → [16, 3, 128, 224]
rearrange(..., "(b v) c h w -> b v c h w", b=1, v=16)     → [1, 16, 3, 128, 224]
rearrange(..., "b v c h w -> b v (h w) c")                 → sh_input_images: [1, 16, 28672, 3]

RGB2SH(sh_input_images)                           ← (rgb - 0.5) / 0.2821
  → [1, 16, 28672, 3]

sh[..., 0] = sh[..., 0] + RGB2SH(sh_input_images)  (add to DC component)
```

### 1.27 Construct Gaussians dataclass (flatten V into N)

```
rearrange(means, "b v r spp xyz -> b (v r spp) xyz")      → means:       [1, 458752, 3]
rearrange(covariances, "b v r i j -> b (v r) i j")        → covariances: [1, 458752, 3, 3]
rearrange(sh, "b v r c d_sh -> b (v r) c d_sh")           → harmonics:   [1, 458752, 3, 16]
rearrange(opacities, "b v r spp -> b (v r spp)")           → opacities:   [1, 458752]
rearrange(scales, "b v r xyz -> b (v r) xyz")              → scales:      [1, 458752, 3]
rearrange(rotations, "b v r wxyz -> b (v r) wxyz")         → rotations:   [1, 458752, 4]
rearrange(rotations_unnorm, ...)                            → rot_unnorm:  [1, 458752, 4]

Gaussians(means, covariances, harmonics, opacities, scales, rotations, rotations_unnorm)
```

### 1.28 Return dict

```
results = {
    "gaussians": Gaussians(...),                  (shapes above)
    "depths": depth_preds[-1],                    → [1, 16, 512, 896]   (upsampled from 256×448 since return_depth=true)
    "condition_features": condition_features,     → [16, 512, 128, 224]
}
```

---

## Stage 2: Recurrent Refinement — `EncoderReSplat.forward_update()` (encoder_resplat.py:767)

Called from `infer_colmap.py:run_inference()` with `num_refine=2`.

### 2.1 Detach initial Gaussians

```
prev_means = init_gaussians.means.detach()                    → [1, 458752, 3]
prev_scales = init_gaussians.scales.detach()                  → [1, 458752, 3]
prev_rotations_unnorm = init_gaussians.rotations_unnorm.detach()  → [1, 458752, 4]
torch.logit(init_gaussians.opacities.detach(), eps=1e-6)      → prev_opacities_raw: [1, 458752]
prev_opacities_raw.unsqueeze(-1)                              → [1, 458752, 1]
init_gaussians.harmonics.detach()                             → prev_shs: [1, 458752, 3, 16]
rearrange(prev_shs, "b n c x -> b n (c x)")                  → prev_shs: [1, 458752, 48]
```

### 2.2 Project condition_features to hidden state

```
self.update_proj(condition_features.detach())      ← nn.Conv2d(512, 512, 1)
  → state: [16, 512, 128, 224]

rearrange(state, "(b v) c h w -> b (v h w) c", b=1, v=16)
  → state: [1, 458752, 512]

rearrange(state, "b n c -> (b n) c")              → tmp_state: [458752, 512]
```

### 2.3 Initial context-view render (before loop)

```
renderer.forward(prev_gaussians, context_extrinsics, context_intrinsics, near, far, (512, 896))
  ← GSplatDecoderSplattingCUDA.forward()
    gaussians.harmonics.permute(0, 1, 3, 2)       → colors: [1, 458752, 16, 3]
    int(sqrt(16)) - 1 = 3                         → sh_degree: 3
    extrinsics.inverse()                           → viewmats: [1, 16, 4, 4]
    intrinsics * [896, 512]                        → Ks: [1, 16, 3, 3]  (pixel-scale intrinsics)

    gsplat.rendering.rasterization(
      means=[1,458752,3], quats=[1,458752,4], scales=[1,458752,3],
      opacities=[1,458752], colors=[1,458752,16,3], sh_degree=3,
      viewmats=[1,16,4,4], Ks=[1,16,3,3], width=896, height=512,
      covars=[1,458752,3,3], render_mode="RGB+ED", rasterize_mode="antialiased", packed=True
    )
    → render_colors: [1, 16, 512, 896, 4]  (RGBD)
    → render_alphas: [1, 16, 512, 896, 1]

    render_colors[..., :3].permute(0, 1, 4, 2, 3) → color: [1, 16, 3, 512, 896]
    render_colors[..., -1]                         → depth: [1, 16, 512, 896]

  → input_render.color: [1, 16, 3, 512, 896]
```

---

### 2.4 Refinement Iteration 0

#### 2.4a Extract ResNet-18 features from rendered + GT images

```
rearrange(input_render.color, "b v c h w -> (b v) c h w")  → input0: [16, 3, 512, 896]
rearrange(context["image"], "b v c h w -> (b v) c h w")    → input1: [16, 3, 512, 896]

torch.cat((input0, input1), dim=0)                          → concat: [32, 3, 512, 896]
T.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225])(concat) → input_tensor: [32, 3, 512, 896]

self.update_feature(input_tensor)                           ← ResNetFeatureWarpper.forward()
  resnet.conv1(x)                                           → [32, 64, 256, 448]  (7×7 conv, stride 2)
  out.append(x)                                              feat[0]: [32, 64, 256, 448]
  resnet.bn1 → relu → maxpool                               → [32, 64, 128, 224]
  resnet.layer1(x)                                           → [32, 64, 128, 224]
  out.append(x)                                              feat[1]: [32, 64, 128, 224]
  resnet.layer2(x)                                           → [32, 128, 64, 112]
  out.append(x)                                              feat[2]: [32, 128, 64, 112]

latent_h, latent_w = 512 // 4, 896 // 4 = 128, 224

# Resize each scale to (128, 224):
F.interpolate(feat[0], (128, 224))                → [32, 64, 128, 224]
feat[1] already (128, 224)                        → [32, 64, 128, 224]
F.interpolate(feat[2], (128, 224))                → [32, 128, 128, 224]

torch.cat(out, dim=1)                             → all_features: [32, 64+64+128=256, 128, 224]

render_view_features = all_features[:16]          → [16, 256, 128, 224]
input_view_features = all_features[16:]           → [16, 256, 128, 224]  (cached, reused in iter 1)
```

#### 2.4b Compute feature error

```
corr = render_view_features - input_view_features → [16, 256, 128, 224]

rearrange(corr, "(b v) c h w -> b (v h w) c", b=1, v=16)
  → input_render_error: [1, 458752, 256]
```

#### 2.4c Compute RGB error

```
input_render.color - context["image"]             → [1, 16, 3, 512, 896]
rearrange(..., "b v c h w -> (b v) c h w")       → [16, 3, 512, 896]

F.pixel_unshuffle(..., downscale_factor=4)        → [16, 3*16=48, 128, 224]

rearrange(..., "(b v) c h w -> b (v h w) c", b=1, v=16)  → [1, 458752, 48]

self.update_rgb_error_proj(...)                   ← nn.Sequential:
  nn.Linear(48, 256)                              → [1, 458752, 256]
  nn.LayerNorm(256)                               → rgb_render_error: [1, 458752, 256]
```

#### 2.4d Combine errors

```
input_render_error + rgb_render_error             → input_render_error: [1, 458752, 256]
```

#### 2.4e Prepare previous Gaussians as flat tensor

```
torch.cat((prev_means, prev_scales, prev_rotations_unnorm, prev_opacities_raw, prev_shs), dim=-1)
  → prev_gaussians_concat: [1, 458752, 3+3+4+1+48=59]
```

#### 2.4f Prepare point cloud for kNN

```
rearrange(prev_means, "b n c -> (b n) c")        → point_cloud: [458752, 3]
tmp_batch_size = 458752
offset = torch.tensor([458752])                   → [1]
```

#### 2.4g Flatten Gaussian concat

```
rearrange(prev_gaussians_concat, "b n c -> (b n) c")  → tmp_gaussian: [458752, 59]
```

#### 2.4h Global attention on error

```
for blk in self.update_error_attn:                ← MultViewLowresAttn (1 block)
  blk(input_render_error, v=16, h=128, w=224)     → input_render_error: [1, 458752, 256]

rearrange(input_render_error, "b n c -> (b n) c") → tmp_render_error: [458752, 256]
```

#### 2.4i Concat for update module

```
torch.cat((tmp_gaussian, tmp_state, tmp_render_error), dim=-1)
  → concat: [458752, 59 + 512 + 256 = 827]
```

#### 2.4j Update module (PointLinearWrapper + PlainPointTransformer)

```
self.update_module[0]([point_cloud, concat, offset])  ← PointLinearWrapper(827, 512)
  → pxo: (point_cloud, features=[458752, 512], offset)

self.update_module[1](pxo, b=1, v=16, h=128, w=224)  ← PlainPointTransformer(512, knn_samples, num_blocks=4)
  (4 blocks of kNN attention)
  → [458752, 512]

tmp_state = update_module_output + tmp_state      → tmp_state: [458752, 512]  (residual)
```

#### 2.4k Delta Gaussian head

```
self.update_head(tmp_state)                       ← nn.Sequential:
  nn.Linear(512, 512) → GELU                     → [458752, 512]
  nn.Linear(512, 512) → GELU                     → [458752, 512]
  nn.Linear(512, 512) → GELU                     → [458752, 512]
  nn.Linear(512, 59)                              → [458752, 59]  (zero-initialized at init)

rearrange(..., "(b n) c -> b n c", b=1)           → delta_gaussians: [1, 458752, 59]
```

#### 2.4l Split deltas

```
delta_gaussians.split((3, 3, 4, 1, 48), dim=-1)
  → delta_means:      [1, 458752, 3]   (abs_mean=0.024)
  → delta_scales:     [1, 458752, 3]   (abs_mean=0.021)
  → delta_rotations:  [1, 458752, 4]   (abs_mean=0.037)
  → delta_opacities:  [1, 458752, 1]   (abs_mean=0.346)
  → delta_shs:        [1, 458752, 48]  (abs_mean=0.086)
```

#### 2.4m Apply deltas

```
prev_means + delta_means                          → prev_means: [1, 458752, 3]
(prev_scales + delta_scales).clamp(min=1e-6)      → prev_scales: [1, 458752, 3]
prev_rotations_unnorm + delta_rotations           → prev_rotations_unnorm: [1, 458752, 4]
rotations_unnorm / (norm + 1e-8)                  → prev_rotations: [1, 458752, 4]

build_covariance(prev_scales, prev_rotations)     → [1, 458752, 3, 3]
rearrange(..., "b (v hw) x y -> b v hw x y", v=16)  → [1, 16, 28672, 3, 3]
c2w_rot @ covariances @ c2w_rot.T                → [1, 16, 28672, 3, 3]
rearrange(..., "b v hw x y -> b (v hw) x y")     → covariances: [1, 458752, 3, 3]

prev_opacities_raw + delta_opacities              → [1, 458752, 1]
prev_shs + delta_shs                              → [1, 458752, 48]
```

#### 2.4n Construct refined Gaussians

```
Gaussians(
  prev_means,                                     → [1, 458752, 3]
  covariances,                                    → [1, 458752, 3, 3]
  rearrange(prev_shs, "b n (x y) -> b n x y", x=3),  → [1, 458752, 3, 16]
  prev_opacities_raw.squeeze(-1).sigmoid(),       → [1, 458752]
  prev_scales,                                    → [1, 458752, 3]
  rotations=prev_rotations,                       → [1, 458752, 4]
  rotations_unnorm=prev_rotations_unnorm,         → [1, 458752, 4]
)
```

#### 2.4o Render target views (44 views)

```
renderer.forward(prev_gaussians, target_extrinsics, ...)
  gsplat.rendering.rasterization(
    means=[1,458752,3], ..., viewmats=[1,44,4,4], Ks=[1,44,3,3], width=896, height=512
  )
  → color: [1, 44, 3, 512, 896]
  → depth: [1, 44, 512, 896]
```

#### 2.4p Render context views (for next iteration)

```
renderer.forward(prev_gaussians, context_extrinsics, ...)
  gsplat.rendering.rasterization(
    viewmats=[1,16,4,4], Ks=[1,16,3,3]
  )
  → input_render.color: [1, 16, 3, 512, 896]
```

---

### 2.5 Refinement Iteration 1

Same flow as 2.4, but:
- `input_view_features` is **cached** from iter 0 (not recomputed)
- Only `render_view_features` is recomputed via `self.update_feature(transform(input0))`
- Deltas are smaller (converging):

```
delta_means:      abs_mean=0.016  (was 0.024, -33%)
delta_scales:     abs_mean=0.018  (was 0.021, -17%)
delta_rotations:  abs_mean=0.029  (was 0.037, -22%)
delta_opacities:  abs_mean=0.197  (was 0.346, -43%)
delta_shs:        abs_mean=0.090  (was 0.086, ~stable)
```

Final render target views (44) + context views (16) same as 2.4o/p.

---

## Stage 3: Final Rendering — `infer_colmap.py:run_inference()`

### 3.1 Extract final Gaussians

```
refine_output["gaussian"][-1]                     → gaussians (from iter 1)
  means: [1, 458752, 3], covariances: [1, 458752, 3, 3], harmonics: [1, 458752, 3, 16],
  opacities: [1, 458752], scales: [1, 458752, 3], rotations: [1, 458752, 4]
```

### 3.2 Render target views in chunks

```
# Chunk 1: views 0-9
decoder.forward(gaussians, target_extrinsics[:, 0:10], ..., (512, 896))
  gsplat.rendering.rasterization(viewmats=[1,10,4,4], Ks=[1,10,3,3])
  → output.color: [1, 10, 3, 512, 896]
  output.color[0]                                 → [10, 3, 512, 896]

# Chunk 2: views 10-19 → [10, 3, 512, 896]
# Chunk 3: views 20-29 → [10, 3, 512, 896]
# Chunk 4: views 30-39 → [10, 3, 512, 896]
# Chunk 5: views 40-43 → [4, 3, 512, 896]

torch.cat(all_colors, dim=0)                      → rendered: [44, 3, 512, 896]
torch.cat(all_depths, dim=0)                      → rendered_depth: [44, 512, 896]
```

### 3.3 Smooth video rendering (if --save_video)

```
render_smooth_video(gaussians, decoder, all_c2w_np=[60,4,4], ...)
  render_stabilization_path(poses_3x4, k_size=45)  → smoothed_list: 60× [3, 4]
  camera_normalization(context_c2w[8:9], smoothed_c2w_t)  → [60, 4, 4]

  # 6 chunks of 10 views:
  decoder.forward(gaussians, extrinsics=[1,10,4,4], ...) → [1, 10, 3, 512, 896]
  → 60 frames → imageio.mimwrite("video.mp4", ...)
```

### 3.4 Metrics computation

```
compute_psnr(target_images, rendered)              → psnr_vals: [44]
compute_ssim(target_images, rendered)              → ssim_vals: [44]
compute_lpips(target_images[i:end], rendered[i:end])  → lpips chunk: [chunk_size]
torch.cat(lpips_chunks)                            → lpips_vals: [44]
```

---

## Decoder Detail: `GSplatDecoderSplattingCUDA.forward()` (every call)

```
gaussians.means                                   → means: [1, 458752, 3]
gaussians.rotations_unnorm                        → quats: [1, 458752, 4]
gaussians.scales                                  → scales: [1, 458752, 3]
gaussians.covariances                             → covars: [1, 458752, 3, 3]
gaussians.opacities                               → opacities: [1, 458752]
gaussians.harmonics.permute(0, 1, 3, 2)           → colors: [1, 458752, 16, 3]
int(math.sqrt(16)) - 1                            → sh_degree: 3
extrinsics.inverse()                              → viewmats: [1, V, 4, 4]
intrinsics.clone()                                → Ks: [1, V, 3, 3]
Ks[:, :, 0] *= width                              (scale fx, cx to pixels)
Ks[:, :, 1] *= height                             (scale fy, cy to pixels)

gsplat.rendering.rasterization(...)               → render_colors: [1, V, H, W, 4]
                                                    render_alphas: [1, V, H, W, 1]

render_colors[..., :3].permute(0,1,4,2,3)        → color: [1, V, 3, H, W]
render_colors[..., -1]                            → depth: [1, V, H, W]
render_alphas.squeeze(-1)                         → accumulated_alpha: [1, V, H, W]
```
