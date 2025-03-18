import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import matplotlib.pyplot as plt

# 创建保存目录
os.makedirs(r'D:\PyTorch-GAN-master\PyTorch-GAN-master\implementations\gan1', exist_ok=True)

# 参数配置
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr_G", type=float, default=0.0002)
parser.add_argument("--lr_D", type=float, default=0.0001)
parser.add_argument("--b1", type=float, default=0.5)
parser.add_argument("--b2", type=float, default=0.999)
parser.add_argument("--latent_dim", type=int, default=100)
parser.add_argument("--sample_interval", type=int, default=400)
parser.add_argument("--clip_value", type=float, default=0.01)  # 添加权重裁剪参数
opt = parser.parse_args()

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 自定义数据集类
class CSIDataset(Dataset):
    def __init__(self, file_path):
        with h5py.File(file_path, 'r') as f:
            data = f['CSI_featuresA'][()]
        data = np.transpose(data, (3, 0, 1, 2))
        max_val = data.max()
        min_val = data.min()
        self.data = 2 * ((data - min_val) / (max_val - min_val)) - 1
        self.data = torch.FloatTensor(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], 0


dataset = CSIDataset('CSI_featuresB.mat')
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)


# 生成器
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


# 判别器
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.img_shape = (2, 32, 12)
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(int(np.prod(self.img_shape)), 512)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Linear(256, 1))  # 移除Sigmoid
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


# 初始化模型
generator = Generator().to(device)
critic = Critic().to(device)

# 优化器（添加权重裁剪）
optimizer_G = optim.Adam(generator.parameters(), lr=opt.lr_G, betas=(opt.b1, opt.b2))
optimizer_C = optim.Adam(critic.parameters(), lr=opt.lr_D, betas=(opt.b1, opt.b2))

# 训练统计
g_losses = []
c_losses = []

# 训练循环（修改损失计算和添加权重裁剪）
for epoch in range(opt.n_epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        # 训练Critic
        optimizer_C.zero_grad()

        # 真实样本的损失
        real_validity = critic(real_imgs)

        # 生成假样本
        z = torch.randn(batch_size, opt.latent_dim, device=device)
        gen_imgs = generator(z).detach()
        fake_validity = critic(gen_imgs)

        # Wasserstein距离损失
        c_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
        c_loss.backward()
        optimizer_C.step()

        # 权重裁剪（可选）
        for p in critic.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # 训练生成器（每5次critic训练训练1次生成器）
        if i % 5 == 0:
            optimizer_G.zero_grad()

            gen_imgs = generator(z)
            g_loss = -torch.mean(critic(gen_imgs))
            g_loss.backward()
            optimizer_G.step()

        # 记录损失
        g_losses.append(g_loss.item())
        c_losses.append(c_loss.item())

        if i % 50 == 0:
            print(f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] "
                  f"[C loss: {c_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

# 保存模型和绘制损失曲线
torch.save(generator.state_dict(), r'D:\PyTorch-GAN-master\PyTorch-GAN-master\implementations\gan1\generator.pth')
torch.save(critic.state_dict(), r'D:\PyTorch-GAN-master\PyTorch-GAN-master\implementations\gan1\critic.pth')

plt.figure(figsize=(10, 5))
plt.plot(g_losses, label="Generator Loss")
plt.plot(c_losses, label="Critic Loss")
plt.title("Training Loss History")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(r'D:\PyTorch-GAN-master\PyTorch-GAN-master\implementations\gan1\loss_curve.png')
plt.close()
