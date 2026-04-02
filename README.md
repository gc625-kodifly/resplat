<p align="center">
  <h1 align="center">ReSplat: Learning Recurrent Gaussian Splatting</h1>
  <p align="center">
    <a href="https://haofeixu.github.io/">Haofei Xu</a>
    &middot;
    <a href="https://scholar.google.com/citations?user=U9-D8DYAAAAJ">Daniel Barath</a>
    &middot;
    <a href="http://www.cvlibs.net/">Andreas Geiger</a>
    &middot;
    <a href="https://people.inf.ethz.ch/marc.pollefeys/">Marc Pollefeys</a>
  </p>
  <h3 align="center">
    <a href="https://arxiv.org/abs/2510.08575">Paper</a> | <a href="https://haofeixu.github.io/resplat/">Project Page</a> | <a href="MODEL_ZOO.md">Models</a>
  </h3>
</p>

<p align="center">
  <img src="https://haofeixu.github.io/resplat/assets/teaser.png" alt="ReSplat teaser" width="100%">
</p>

ReSplat is a feed-forward recurrent model for 3D Gaussian splatting that iteratively refines Gaussians using the rendering error as a gradient-free feedback signal for test-time adaptation.

**Key features:**
- **Compact initialization**: Predicts Gaussians in a subsampled space (16× fewer Gaussians than prior per-pixel methods)
- **Recurrent refinement**: Weight-sharing recurrent module that uses rendering error to predict per-Gaussian parameter updates

## Installation

This codebase is developed with Python 3.12, PyTorch 2.7.0, and CUDA 12.8.

We recommend setting up a virtual environment (e.g., [conda](https://docs.anaconda.com/miniconda/) or [venv](https://docs.python.org/3/library/venv.html)) before installation:

```bash
# conda
conda create -y -n resplat python=3.12
conda activate resplat

# or venv
# python -m venv /path/to/venv/resplat
# source /path/to/venv/resplat/bin/activate

# torch 2.7.0, cuda 12.8
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt

# Install gsplat 1.5.3
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3

# Install pointops (kNN)
cd src/model/encoder/pointops && python setup.py install && cd ../../../..
```

## Model Zoo

Pre-trained models are available in the [Model Zoo](MODEL_ZOO.md).

Download the weights and place (or symlink) them in the `pretrained` directory:

```bash
ln -s YOUR_MODEL_PATH pretrained
```

## Camera Conventions

The camera intrinsic matrices are normalized, with the first row divided by the image width and the second row divided by the image height.

The camera extrinsic matrices follow the OpenCV convention for camera-to-world transformation (+X right, +Y down, +Z pointing into the screen).

## Dataset Preparation

See [DATASETS.md](DATASETS.md) for detailed instructions on preparing RealEstate10K, DL3DV and ACID datasets.

Symlink the downloaded datasets to the `datasets` directory:

```bash
ln -s YOUR_DATASET_PATH datasets
```

## Demo

Check [scripts/infer_colmap.sh](scripts/infer_colmap.sh) for running our pre-trained models on COLMAP datasets.

A demo scene can be downloaded [here](https://huggingface.co/datasets/haofeixu/depthsplat/resolve/main/dl3dv-colmap-demo.zip) to quickly try our method.


## Evaluation

Evaluation scripts are also provided in [scripts/](scripts) for reproducing the results in our paper.

## Training

ReSplat is trained in two stages: (1) initial Gaussian prediction and (2) recurrent refinement.

The training scripts in [scripts/](scripts) contain the exact commands and hyperparameters used for the experiments in our paper. Please refer to them for detailed configurations.

Before training, you need to download the pre-trained [depth model](MODEL_ZOO.md), and set up your [wandb account](config/main.yaml) (in particular, by setting `wandb.entity=YOUR_ACCOUNT`) for logging.


## Citation

If you find this work useful, please consider citing:

```bibtex
@article{xu2025resplat,
  title={ReSplat: Learning Recurrent Gaussian Splatting},
  author={Xu, Haofei and Barath, Daniel and Geiger, Andreas and Pollefeys, Marc},
  journal={arXiv preprint arXiv:2510.08575},
  year={2025}
}
```

## Acknowledgements

Our codebase builds upon several excellent open-source projects: [pixelSplat](https://github.com/dcharatan/pixelsplat), [MVSplat](https://github.com/donydchen/mvsplat), [MVSplat360](https://github.com/donydchen/mvsplat360), [UniMatch](https://github.com/autonomousvision/unimatch), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), [DepthSplat](https://github.com/cvg/depthsplat), [Pointcept](https://github.com/Pointcept/Pointcept), [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [gsplat](https://github.com/nerfstudio-project/gsplat), and [DL3DV](https://github.com/DL3DV-10K/Dataset). We thank all the authors for their great work.
