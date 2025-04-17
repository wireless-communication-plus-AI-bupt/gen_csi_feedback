import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import matplotlib.pyplot as plt
import torch.nn.functional as F
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr_G", type=float, default=0.00003)
parser.add_argument("--lr_D", type=float, default=0.00001)
parser.add_argument("--b1", type=float, default=0.5)
parser.add_argument("--b2", type=float, default=0.999)
parser.add_argument("--latent_dim", type=int, default=100)
parser.add_argument("--sample_interval", type=int, default=400)
parser.add_argument("--clip_value", type=float, default=0.01)
opt = parser.parse_args()

# 重新定义生成器类，与训练时保持一致
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.img_shape = (2, 32, 12)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

# 加载生成器模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
generator = Generator().to(device);
generator.load_state_dict(torch.load(r'D:\PyTorch-GAN-master\PyTorch-GAN-master\implementations\gan1\generator.pth'));
generator.eval();

# 生成1000个数据
latent_dim = 100;
num_samples = 9000;#生成数据数目
z = torch.randn(num_samples, latent_dim, device=device);
gen_imgs = generator(z);

# 获取原数据的最大值和最小值，这里假设已获取到
max_val = 0.5235208717230841;
min_val = -0.5380426822852186;

# 反归一化操作
gen_imgs = (gen_imgs + 1) / 2 * (max_val - min_val) + min_val;

# 将生成的数据转换为numpy数组并调整维度
gen_imgs = gen_imgs.cpu().detach().numpy();
gen_imgs = np.transpose(gen_imgs, (3, 1, 2, 0));

# 保存数据到新的.mat文件
with h5py.File('new_generated_data.mat', 'w') as f:
    f.create_dataset('new_generated_CSI_features', data=gen_imgs);
