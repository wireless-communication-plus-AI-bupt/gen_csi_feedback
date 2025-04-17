# 从'CSI_featuresB.mat'中随机抽取一定数量的数据保存在'CSI_featuresA.mat'中
# 用这个'CSI_featuresA.mat'去训练WGAN
import h5py
import numpy as np

# 输入输出文件路径
input_file = 'CSI_featuresB.mat'
output_file = 'CSI_featuresA.mat'

# 读取原始MAT文件
with h5py.File(input_file, 'r') as f:
    csi_data = f['CSI_featuresA'][:]  # 读取全部数据

    # 获取样本维度大小（假设最后一个维度是样本维度）
    n_samples = csi_data.shape[-1]
    # 生成1000个不重复的随机索引（范围0到n_samples-1）
    random_indices = np.random.choice(n_samples, size=1000, replace=False)

    # 按随机索引抽取数据（保持其他维度不变，最后一维选取随机索引）
    selected_data = csi_data[..., random_indices]

# 创建新的MAT文件并保存数据
with h5py.File(output_file, 'w') as f:
    f.create_dataset('CSI_featuresA', data=selected_data)

print(f"成功随机抽取1000个数据并保存到 {output_file}")
print(f"新数据集维度：{selected_data.shape}")
