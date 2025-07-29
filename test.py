import torch

print("Is CUDA available:", torch.cuda.is_available())
print("Torch built with CUDA:", torch.version.cuda)
print("Torch CUDA version (runtime):", torch.backends.cudnn.version())
