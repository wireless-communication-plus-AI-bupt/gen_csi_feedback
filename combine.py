# 注意此处的CSI_featuresA是从getSample中再一次抽取的样本用于测试
# 此处的CSI_featuresA和训练时候所用的数据集是两个不同数据集，但都可以用getSample来提取
import h5py
import numpy as np

# 1. 读取 CSI_featuresA.mat 数据（维度正确，无需转置）
with h5py.File('CSI_featuresA.mat', 'r') as f:
    data_a = f['CSI_featuresA'][:]
    shape_a = data_a.shape
    print(f"CSI_featuresA 维度: {shape_a}")  # 应输出 (2, 32, 12, 1000)

# 2. 读取 new_generated_data.mat 数据并转置前三维
with h5py.File('new_generated_data.mat', 'r') as f:
    data_new = f['new_generated_CSI_features'][:]
    print(f"new_generated 原始维度: {data_new.shape}")  # 应输出 (12, 2, 32, 9000)

    # 转置前三维：(12, 2, 32) → (2, 32, 12)，转置轴为 (1, 2, 0, 3)
    data_new_transposed = np.transpose(data_new, axes=(1, 2, 0, 3))
    shape_new = data_new_transposed.shape
    print(f"转置后维度: {shape_new}")  # 应输出 (2, 32, 12, 9000)

# 3. 验证前三维一致性（现在应该一致）
if shape_a[:3] != shape_new[:3]:
    raise ValueError(
        f"转置后前三维仍不一致！\n"
        f"CSI_featuresA 前三维: {shape_a[:3]}\n"
        f"new_generated 转置后前三维: {shape_new[:3]}"
    )

# 4. 沿最后一维拼接（axis=3），顺序：转置后的new_data在前，data_a在后
combined_data = np.concatenate([data_new_transposed, data_a], axis=3)

# 5. 保存合并后的文件
with h5py.File('CSI_featuresC.mat', 'w') as f:
    f.create_dataset('CSI_featuresA', data=combined_data)

print(f"合并成功！最终维度: {combined_data.shape}")  # 应输出 (2, 32, 12, 10000)
