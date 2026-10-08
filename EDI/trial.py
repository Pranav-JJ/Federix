# # import torch

# # # Check if CUDA is available
# # if torch.cuda.is_available():
# #     print("CUDA is available!")
# #     device = torch.device("cuda")
# # else:
# #     print("CUDA is not available. Using CPU instead.")
# #     device = torch.device("cpu")

# # # Print the device being used
# # print(f"Running on device: {device}")


# import torch

# # Check CUDA version
# print(f"CUDA version: {torch.version.cuda}")


import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU instead.")
    device = torch.device("cpu")

# Print the device being used
print(f"Running on device: {device}")

# Create a tensor on the selected device
a = torch.tensor([1, 2, 3], device=device)
b = torch.tensor([4, 5, 6], device=device)

# Perform a simple operation on the GPU
c = a + b

# Print the result
print(f"Result: {c}")

