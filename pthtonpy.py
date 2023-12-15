import torch
import os
import numpy as np

# 加载PTH文件
tensor = torch.load("output/test_generation/2023-12-13-18-01-48/syn/samples.pth")

# 设置保存路径
file_path = "output/test_generation/2023-12-13-18-01-48/syn/samples.npy"

# 保存为npy文件
np.save(file_path, tensor.numpy())

# 加载npy文件
data = np.load(file_path)

# 拆解三维矩阵为n个二维矩阵
n_matrices = []
for i in range(data.shape[0]):
    n_matrices.append(data[i, :, :])

# 保存每个二维矩阵为npy文件
for idx, matrix in enumerate(n_matrices):
    np.savetxt(
        f"output/test_generation/2023-12-13-18-01-48/syn/samples_{idx}.txt",
        matrix,
    )
