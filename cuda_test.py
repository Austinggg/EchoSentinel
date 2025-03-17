import torch

# 检查 CUDA 是否可用
print("CUDA Available:", torch.cuda.is_available())

# 输出 PyTorch 版本
print("PyTorch Version:", torch.__version__)

# 创建一个简单的张量
x = torch.rand(5, 3)
print("Random Tensor:", x)