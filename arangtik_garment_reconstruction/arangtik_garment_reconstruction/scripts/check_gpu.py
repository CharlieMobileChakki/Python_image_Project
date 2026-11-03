import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("VRAM (GB):", round(props.total_memory / (1024 ** 3), 2))
else:
    print("No CUDA GPU is visible to PyTorch.")
    print("This project is configured to require CUDA by default.")
