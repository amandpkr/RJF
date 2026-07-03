# Learning on the Manifold: Unlocking Standard Diffusion Transformers with Representation Encoders<br><sub>Official PyTorch Implementation</sub>

### [Paper](https://arxiv.org/abs/2602.10099) | ECCV 2026

> [**Learning on the Manifold: Unlocking Standard Diffusion Transformers with Representation Encoders**](https://arxiv.org/abs/2602.10099)<br>
> [Amandeep Kumar](https://amandpkr.github.io/), [Vishal M. Patel](https://scholar.google.com/citations?user=AkEXTbIAAAAJ&hl=en)<br>
> Johns Hopkins University<br>

Standard Diffusion Transformers fail to converge when trained directly on high-dimensional feature spaces of pretrained representation encoders (e.g., DINOv2). While prior work attributes this to a capacity bottleneck, we demonstrate that the failure is fundamentally **geometric**. We identify **Geometric Interference** as the root cause: standard Euclidean flow matching forces probability paths through the low-density interior of the hyperspherical feature manifold rather than following the manifold surface. To resolve this, we propose **Riemannian Flow Matching with Jacobi Regularization (RJF)**, which constrains the generative process to manifold geodesics and corrects for curvature-induced error propagation — enabling standard DiT architectures to converge without width scaling.

This repository contains:

* A PyTorch implementation of RJF built on top of the [RAE](https://github.com/bytetriper/RAE) codebase.
* A PyTorch implementation of LightningDiT trained with our Spherical Flow Matching objective.
* Training and sampling scripts for the two-stage RAE + RJF pipeline.

## Environment

### Dependency Setup

1. Create environment and install via `uv`:
   ```bash
   conda create -n rjf python=3.10 -y
   conda activate rjf
   pip install uv

   # Install PyTorch (adjust CUDA version as needed)
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

   # Install other dependencies
   uv pip install -r requirements.txt
   ```

## Data & Model Preparation

### Download Pre-trained RAE Models

We use the pretrained Stage 1 RAE decoder from the upstream RAE release. To download all models at once:

```bash
cd RJF
pip install huggingface_hub
hf download nyu-visionx/RAE-collections \
  --local-dir models
```

To download specific models, run:
```bash
hf download nyu-visionx/RAE-collections \
  <remote_model_path> \
  --local-dir models
```

### Download Pretrained RJF Weights

Pretrained Stage 2 RJF checkpoints (`DiT-B` and `DiT-XL`) are available on Hugging Face:

```bash
hf download Aman015/RJF --local-dir ckpts_imagenet
```

| Model | File | Config |
|-------|------|--------|
| LightningDiT-B + RJF | `DiT_B.pt` | `configs/stage2/sampling/ImageNet256/DiT-B_DINOv2-B.yaml` |
| LightningDiT-XL + RJF | `DiT_XL.pt` | `configs/stage2/sampling/ImageNet256/DiT-XL_DINOv2-B.yaml` |

After downloading, set the `stage_2.ckpt` field in the corresponding sampling config to point to the downloaded `.pt` file.

### Prepare Dataset

1. Download ImageNet-1k.
2. Point Stage 2 scripts to the training split via `--data-path`.

## Config-based Initialization

All training and sampling entrypoints are driven by OmegaConf YAML files. A single config describes the Stage 1 autoencoder, the Stage 2 diffusion model, and the solver used during training or inference. A minimal example looks like:

```yaml
stage_1:
  target: stage1.RAE
  params: { ... }

stage_2:
  target: stage2.models.lightningDiT.LightningDiT
  params: { ... }
  ckpt: <path_to_ckpt>            # for sampling only; omit during training

transport:
  params:
    path_type: 'Spherical'        # RJF: geodesic SLERP paths on S^{d-1}
    prediction: velocity
    loss_weight: null
    time_dist_type: 'logit-normal_0_1'

sampler:
  mode: ODE
  params:
    sampling_method: euler        # geodesic Euler (exponential map) integrator
    num_steps: 50
    ...

guidance:
  method: cfg
  scale: 1.0
  ...

misc:
  latent_size: [768, 16, 16]
  num_classes: 1000
  time_dist_shift_dim: 196608     # 768 * 16 * 16
  time_dist_shift_base: 4096

training:
  ...

eval:
  ...
```

- `stage_1` instantiates the frozen DINOv2 encoder and pretrained ViT-XL decoder. Set `noise_tau: 0.0` at inference.
- `stage_2` defines the LightningDiT diffusion transformer. During sampling you must provide `ckpt`; during training omit it so weights initialise randomly.
- `transport`, `sampler`, and `guidance` select the flow path, integrator, and guidance schedule. Set `path_type: 'Spherical'` to enable RJF; `'Linear'` falls back to standard Euclidean flow matching.
- `misc` collects shapes, class counts, and time-warp scaling constants used by both stages.
- `training` contains defaults that the training scripts consume (epochs, learning rate, EMA decay, gradient accumulation, etc.).
- `eval` contains settings for online gFID evaluation during training.

### Provided Configs

#### Stage 1 (RAE — adopted from upstream, see [bytetriper/RAE](https://github.com/bytetriper/RAE))

| Config | Description |
|--------|-------------|
| `configs/stage1/pretrained/DINOv2-B.yaml` | DINOv2-B encoder + ViT-XL decoder (with latent normalization stats) |
| `configs/stage1/pretrained/MAE.yaml` | MAE-B encoder variant |
| `configs/stage1/pretrained/SigLIP2.yaml` | SigLIP2-B encoder variant |
| `configs/stage1/training/DINOv2-B_decXL.yaml` | Training config for the ViT-XL decoder |

#### Stage 2 (RJF)

| Config | Description |
|--------|-------------|
| `configs/stage2/training/ImageNet256/DiT-B_DINOv2-B.yaml` | LightningDiT-B training with RJF on ImageNet-256 |
| `configs/stage2/training/ImageNet256/DiT-XL_DINOv2-B.yaml` | LightningDiT-XL training with RJF on ImageNet-256 |
| `configs/stage2/sampling/ImageNet256/DiT-B_DINOv2-B.yaml` | Sampling config for DiT-B |
| `configs/stage2/sampling/ImageNet256/DiT-XL_DINOv2-B.yaml` | Sampling config for DiT-XL |

## Stage 2: RJF Diffusion Transformer

### Training

Set the required environment variables and launch via `train.sh`:

```bash
export WANDB_KEY="<your_key>"
export ENTITY="<your_entity>"
export PROJECT="<your_project>"
export EXPERIMENT_NAME="<your_experiment_name>"

bash train.sh
```

This runs:

```bash
NCCL_P2P_DISABLE=1 torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/train.py \
  --config configs/stage2/training/ImageNet256/DiT-B_DINOv2-B.yaml \
  --data-path <imagenet_train_split> \
  --results-dir ckpts_imagenet \
  --precision fp32 \
  --compile
```

Checkpoints and logs are saved under `results-dir/$EXPERIMENT_NAME/`.

**Resuming.** If the checkpoint folder already exists (`results-dir/$EXPERIMENT_NAME/`), the script will automatically resume from the latest checkpoint.

**Logging.** To enable `wandb`, set `WANDB_KEY`, `ENTITY`, and `PROJECT` as environment variables (as above), then add the `--wandb` flag to the training command.

**Online Eval.** The script supports online gFID evaluation during training. Paste the following block into the training config:

```yaml
eval:
  eval_interval: 25000
  eval_model: true
  data_path: 'data/imagenet/val/'
  reference_npz_path: 'data/imagenet/VIRTUAL_imagenet256_labeled.npz'
```

**torch.compile.** Use `--compile` flag to enable `torch.compile` for potentially faster training.

### Sampling

`src/riemannian_sampling.py` uses the same config schema to draw a small batch of images on a single device and saves them to `sample.png`:

```bash
python src/riemannian_sampling.py \
  --config configs/stage2/sampling/ImageNet256/DiT-B_DINOv2-B.yaml \
  --seed 42
```

The config's `stage_2.ckpt` field must point to a valid checkpoint.

### Distributed Sampling for Evaluation

`src/riemannian_sampling_ddp.py` parallelises sampling across GPUs, producing per-image PNGs and a packed FID-ready `.npz`. Launch it via `src/sample_ddp.sh`:

```bash
NCCL_P2P_DISABLE=1 torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/riemannian_sampling_ddp.py \
  --config configs/stage2/sampling/ImageNet256/DiT-B_DINOv2-B.yaml \
  --sample-dir samples \
  --precision fp32 \
  --label-sampling equal \
  --default-radius 45.10
```

## Stage 1: Representation Autoencoder

The Stage 1 RAE is adopted directly from the upstream RAE codebase. Please refer to [bytetriper/RAE](https://github.com/bytetriper/RAE) for full Stage 1 training, sampling, and statistic-calculation instructions. Pretrained weights are available from the `nyu-visionx/RAE-collections` HuggingFace collection.

## Evaluation

### ADM Suite FID setup

Use the ADM evaluation suite to score generated samples:

1. Clone the repo:
   ```bash
   git clone https://github.com/openai/guided-diffusion.git
   cd guided-diffusion/evaluation
   ```

2. Create an environment and install dependencies:
   ```bash
   conda create -n adm-fid python=3.10
   conda activate adm-fid
   pip install 'tensorflow[and-cuda]'==2.19 scipy requests tqdm
   ```

3. Download ImageNet statistics (256×256 shown here):
   ```bash
   wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
   ```

4. Evaluate:
   ```bash
   python evaluator.py VIRTUAL_imagenet256_labeled.npz /path/to/samples.npz
   ```

## Citation

```bibtex
@inproceedings{kumar2026learning,
  title={Learning on the Manifold: Unlocking Standard Diffusion Transformers with Representation Encoders},
  author={Kumar, Amandeep and Patel, Vishal M.},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## Acknowledgement

This code is built upon the following repositories:

* [RAE](https://github.com/bytetriper/RAE) — for the Stage 1 RAE architecture, pretrained weights, and training codebase.
* [SiT](https://github.com/willisma/SiT) — for the diffusion/transport implementation.
* [LightningDiT](https://github.com/hustvl/LightningDiT/) — for the LightningDiT architecture.
* [MAE](https://github.com/facebookresearch/mae) — for the ViT decoder architecture.
