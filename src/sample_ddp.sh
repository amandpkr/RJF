torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  src/riemannian_sampling_ddp.py \
  --config configs/stage2/sampling/ImageNet256/DiT-XL_DINOv2-B.yaml \
  --sample-dir samples \
  --precision fp32 \
  --label-sampling equal \
  --default-radius 45.1