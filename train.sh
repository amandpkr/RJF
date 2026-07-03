export WANDB_KEY=""
export ENTITY=""
export PROJECT=""
export EXPERIMENT_NAME=""
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/train.py \
  --config configs/stage2/training/ImageNet256/DiT-B_DINOv2-B.yaml\
  --data-path /mnt/store/akumar99/data/imagenet/train \
  --results-dir ckpts_imagenet \
  --precision fp32 \
  --compile \