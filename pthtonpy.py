import torch
import os
import numpy as np

# 设置保存路径
file_path = "output/test_generation/2023-12-16-11-08-55/syn/samples"

# 加载PTH文件
tensor = torch.load("%s.pth" % (file_path))

# 保存为npy文件
np.save("%s.npy" % file_path, tensor.numpy())

# 加载npy文件
data = np.load("%s.npy" % file_path)

# 拆解三维矩阵为n个二维矩阵
n_matrices = []
for i in range(data.shape[0]):
    n_matrices.append(data[i, :, :])

# 保存每个二维矩阵为txt文件
for idx, matrix in enumerate(n_matrices):
    np.savetxt(
        "%s_%d.txt" % (file_path, idx),
        matrix,
        fmt="%s",
    )
