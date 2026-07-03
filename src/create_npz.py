import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.cuda.amp import autocast
from tqdm import tqdm
from pathlib import Path


sample_dir = "/mnt/store/akumar99/r_learning/experiment_on_small_architecture/riemannian_fm_384_384_8_4_6_6_jacobian/samples/DiTwDDTHead-ep-0000030-cfg-1.00-bs125-ODE-50-euler-fp32"

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


a = create_npz_from_sample_folder(
    sample_dir=sample_dir,
    num=50000,
)   