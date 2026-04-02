# Model Zoo

- We provide pre-trained models for view synthesis with 3D Gaussian splatting using ReSplat, as well as depth models for initialization.

- We assume that the downloaded weights are stored in the `pretrained` directory. It's recommended to create a symbolic link from `YOUR_MODEL_PATH` to `pretrained` using
```
ln -s YOUR_MODEL_PATH pretrained
```

- To verify the integrity of downloaded files, each model on this page includes its [sha256sum](https://sha256sum.com/) prefix in the file name (where applicable), which can be checked using the command `sha256sum filename`.


## Gaussian Splatting

- The models are trained on RealEstate10K (re10k) and/or DL3DV (dl3dv) datasets at various resolutions. The number of training views ranges from 2 to 32.

- All models are trained in two stages: initial Gaussian prediction followed by recurrent refinement.


| Model                                |  Training Data   |  Training Resolution  | Training Views | Params (M) |                           Download                           |
| ------------------------------------ | :--------------: | :-------------------: | :------------: | :--------: | :----------------------------------------------------------: |
| resplat-small-dl3dv-256x448-view8-548993fe.pth    |      dl3dv       |        256x448        |       8        |     76     | [download](https://huggingface.co/haofeixu/resplat/resolve/main/resplat-small-dl3dv-256x448-view8-548993fe.pth) |
| resplat-base-dl3dv-256x448-view8-1934a04c.pth     |      dl3dv       |        256x448        |       8        |    223     | [download](https://huggingface.co/haofeixu/resplat/resolve/main/resplat-base-dl3dv-256x448-view8-1934a04c.pth) |
| resplat-large-dl3dv-256x448-view8-62f1703a.pth    |      dl3dv       |        256x448        |       8        |    559     | [download](https://huggingface.co/haofeixu/resplat/resolve/main/resplat-large-dl3dv-256x448-view8-62f1703a.pth) |
| resplat-base-dl3dv-256x448-view16-f38bf984.pth    |      dl3dv       |        256x448        |      16        |    223     | [download](https://huggingface.co/haofeixu/resplat/resolve/main/resplat-base-dl3dv-256x448-view16-f38bf984.pth) |
| resplat-base-dl3dv-256x448-view32-439b63a6.pth    |      dl3dv       |        256x448        |      32        |    223     | [download](https://huggingface.co/haofeixu/resplat/resolve/main/resplat-base-dl3dv-256x448-view32-439b63a6.pth) |
| resplat-base-dl3dv-512x960-view8-8179ed87.pth     |      dl3dv       |        512x960        |       8        |    223     | [download](https://huggingface.co/haofeixu/resplat/resolve/main/resplat-base-dl3dv-512x960-view8-8179ed87.pth) |
| resplat-base-dl3dv-540x960-view16-a72dc6d0.pth    |      dl3dv       |        540x960        |      16        |    223     | [download](https://huggingface.co/haofeixu/resplat/resolve/main/resplat-base-dl3dv-540x960-view16-a72dc6d0.pth) |
| resplat-base-re10k-256x256-view2-b90d1b53.pth     |      re10k       |        256x256        |       2        |    223     | [download](https://huggingface.co/haofeixu/resplat/resolve/main/resplat-base-re10k-256x256-view2-b90d1b53.pth) |



## Depth Prediction

- The depth models are used for initializing the ReSplat Gaussian splatting models. They are fine-tuned from [pre-trained DepthSplat depth models](https://github.com/cvg/depthsplat/blob/main/MODEL_ZOO.md#depth-prediction), but using log-depth sampling instead of inverse-depth sampling (for plane sweep stereo with near and far depth ranges).

- The scale of the predicted depth is aligned with the scale of the camera pose translation.

| Model                                       |                  Training Data                   |  Training Resolution   | Training Views | Params (M) |                           Download                           |
| ------------------------------------------- | :----------------------------------------------: | :--------------------: | :------------: | :--------: | :----------------------------------------------------------: |
| resplat-depth-small-352x640-b0ebc084.pth    | (re10k+dl3dv) &rarr; (scannet+tartanair+vkitti2) | 448x768 &rarr; 352x640 |      2       |     36     | [download](https://huggingface.co/haofeixu/resplat/resolve/main/resplat-depth-small-352x640-b0ebc084.pth) |
| resplat-depth-base-352x640-60be7abf.pth     | (re10k+dl3dv) &rarr; (scannet+tartanair+vkitti2) | 448x768 &rarr; 352x640 |      2       |    111     | [download](https://huggingface.co/haofeixu/resplat/resolve/main/resplat-depth-base-352x640-60be7abf.pth) |
| resplat-depth-large-352x640-05f9beac.pth    | (re10k+dl3dv) &rarr; (scannet+tartanair+vkitti2) | 448x768 &rarr; 352x640 |      2       |    330     | [download](https://huggingface.co/haofeixu/resplat/resolve/main/resplat-depth-large-352x640-05f9beac.pth) |

