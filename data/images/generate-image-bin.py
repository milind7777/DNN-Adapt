import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from pathlib import Path

# --- CONFIG ---
output_bin_file = "batch_input_nchw.bin"
dataset_name = "CIFAR100"
target_size = (224, 224)
num_images = 500

# --- TRANSFORM PIPELINE ---
transform = T.Compose([
    T.Resize(target_size),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# --- LOAD DATASET ---
dataset = torchvision.datasets.CIFAR100(
    root="./data",
    train=False,
    transform=transform,
    download=True,
)

print(f"Loaded dataset with {len(dataset)} images.")

# --- PROCESS IMAGES ---
batch = []

for i in range(num_images):
    img, label = dataset[i]
    batch.append(img)

batch_tensor = torch.stack(batch)  # Shape: (N, C, H, W)
print("Batch tensor shape:", batch_tensor.shape)

# --- SAVE TO SINGLE .bin FILE ---
batch_numpy = batch_tensor.numpy().astype(np.float32)  # (N, C, H, W)
batch_numpy.tofile(output_bin_file)

print(f"Saved full batch to {output_bin_file}")
