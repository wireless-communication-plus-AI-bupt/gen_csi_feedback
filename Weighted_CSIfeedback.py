#注意fb——design那个文件的cal_score_tf需要做一点修改：
#在最后两行加上如下代码
#def cal_score_tf(w_true, w_pre, reduction='mean'):
#    elif reduction == 'none':
#        return sample_loss

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import math

# 强制所有函数即时执行（包括模型训练）
tf.config.run_functions_eagerly(True)

# 强制tf.data管道即时执行
tf.data.experimental.enable_debug_mode()

# 设置日志级别
tf.get_logger().setLevel('INFO')

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from fb_design import *
import h5py
from matplotlib import pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")  # 使用非交互模式
import argparse

# 创建解析器并添加新参数
parser = argparse.ArgumentParser(description='datasets_file')
parser.add_argument('--dataset', type=str)
parser.add_argument('--gpu_id', type=str, default='0', help='gpu_id')
# 添加新参数用于控制样本数量和权重
real1_num = 5000  # 数据集中前5000个样本为真样本
fake_num = 13000  # 数据集中真样本之后为假样本
real2_num = 2000  # 数据集最后两千个样本是真样本，用于验证。
parser.add_argument('--real_weight', type=float, default=1.5, help='Weight for real samples')
parser.add_argument('--fake_weight', type=float, default=1, help='Weight for fake samples')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

file_path_train_dataset = 'CSI_featuresC.mat'
print(file_path_train_dataset)
'''=============== You need to configure here: ====================================='''
# Set feedback_bits
feedback_bits = 132
# Build model
EMBEDDING_DIM = 64 * 8
NUM_QUAN_BITS = 4
NUM_HEAD = 8  # 8 16
LAYER = 10  # 5，10
learning_rate = 0.001  # 初始学习率

'''================Read and Prepare Data ============================================='''
# 计算总样本数
total_samples = real1_num + fake_num + real2_num

# 创建样本权重数组，根据样本的分布
# 确保权重为float32类型
sample_weights = np.concatenate([
    np.ones(real1_num, dtype=np.float32) * args.real_weight,
    np.ones(fake_num, dtype=np.float32) * args.fake_weight,
    np.ones(real2_num, dtype=np.float32) * args.real_weight
])

# 加载数据集
data_mat = h5py.File(file_path_train_dataset, 'r')
string = 'CSI_featuresA'
print(string)
data = data_mat[string]
print(data.__class__)
print(data.shape)

# 确保数据集足够大
if data.shape[3] < total_samples:
    raise ValueError(f"Dataset has only {data.shape[3]} samples, but requested {total_samples} samples")

# 选择指定数量的样本
selected_data = data[:total_samples, :, :, :]
selected_data = selected_data.transpose([3, 2, 1, 0])
data_train = selected_data.reshape(-1, 12 * 32 * 2).astype(np.float32)  # 转换为 float32

# 创建全局索引数组
indices = np.arange(data_train.shape[0])

# 随机打乱索引
np.random.shuffle(indices)

# 使用打乱的索引重新排序数据
shuffled_data = data_train[indices]
shuffled_weights = sample_weights[indices]  # 与数据同步洗牌

'''================Data Splitting ==================================================='''
# 使用相同的权重划分策略
Samplenum = shuffled_data.shape[0]
split_idx = int(Samplenum * 0.9)

# 训练集
X_train = shuffled_data[:split_idx, :]
Y_train = shuffled_data[:split_idx, :]  # 自编码器，输入=目标
train_weights = shuffled_weights[:split_idx]  # 训练集权重（向量）

# 验证集
x_val = shuffled_data[split_idx:, :]
y_val = shuffled_data[split_idx:, :]
val_weights = shuffled_weights[split_idx:]  # 验证集权重（向量）

INPUT_SHAPE = X_train.shape[-1]  # 768

print(f"Total samples: {total_samples}")
print(f"Real samples part1: {real1_num}, Fake samples: {fake_num}, Real samples part2: {real2_num}")
print(f"Real weight: {args.real_weight}, Fake weight: {args.fake_weight}")
print(f"Training set size: {X_train.shape[0]}, Validation set size: {x_val.shape[0]}")

'''================================================================================'''

# 创建验证数据集（不需要洗牌）
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(512).prefetch(tf.data.AUTOTUNE)


# 创建编码器
def create_encoder():
    encoderInput = keras.Input(shape=(INPUT_SHAPE))
    encoderOutput = Encoder(encoderInput, feedback_bits, NUM_QUAN_BITS, EMBEDDING_DIM, NUM_HEAD)
    return keras.Model(inputs=encoderInput, outputs=encoderOutput, name='Encoder')


# 创建解码器
def create_decoder():
    decoderInput = keras.Input(shape=(INPUT_SHAPE))
    decoderOutput = Decoder(decoderInput, feedback_bits, NUM_QUAN_BITS, EMBEDDING_DIM, NUM_HEAD)
    return keras.Model(inputs=decoderInput, outputs=decoderOutput, name="Decoder")


# 创建量化器模型
def create_quantizer():
    vqInput = keras.Input(shape=(INPUT_SHAPE))
    x = layers.Dense(units=2 * int(feedback_bits / NUM_QUAN_BITS), activation='sigmoid')(vqInput)
    codes_before = layers.Reshape((int(feedback_bits / NUM_QUAN_BITS), 2))(x)
    codes, code_vecs = VectorQuantizer(2 ** NUM_QUAN_BITS)(codes_before)
    ze = Lambda(lambda x: x[0] + tf.stop_gradient(x[1] - x[0]))([codes_before, code_vecs])
    x = tf.reshape(ze, (-1, 2 * int(feedback_bits / NUM_QUAN_BITS)))
    x = layers.Dense(768)(x)  # 768
    return keras.Model(vqInput, [code_vecs, codes_before, x], name="Quantizer")


# 创建完整的自动编码器模型
def create_autoencoder():
    encoder = create_encoder()
    decoder = create_decoder()
    quantizer = create_quantizer()

    inputs = keras.Input(shape=(INPUT_SHAPE))

    # 编码器输出
    enc_output = encoder(inputs)

    # 量化器输出
    e, z, quant_output = quantizer(enc_output)

    # 解码器输出
    dec_output = decoder(quant_output)

    # 创建主模型
    autoencoder = keras.Model(inputs=inputs, outputs=dec_output, name='Autoencoder')

    # 创建中间模型，用于获取中间层输出
    intermediate_model = keras.Model(
        inputs=inputs,
        outputs=[dec_output, e, z, enc_output, quant_output],
        name='Intermediate_Model'
    )

    return autoencoder, intermediate_model, encoder, quantizer, decoder


# 创建模型
autoencoderModel, intermediateModel, encoderModel, quantizerModel, decoderModel = create_autoencoder()


# 自定义损失函数（接受权重作为输入）
def weighted_loss(inputs, outputs, model_outputs, weights_batch):
    # 从模型输出中提取中间值
    _, e, z, _, _ = model_outputs

    # 计算基础损失
    base_loss = cal_score_tf(inputs, outputs, reduction='none')

    # 确保权重为float32类型
    weights_batch = tf.cast(weights_batch, tf.float32)

    # 应用权重（每个样本一个权重值）
    weighted_loss_value = base_loss * tf.reshape(weights_batch, (-1, 1))

    # 添加矢量量化损失项 - 使用实际计算的张量
    mse_e = tf.reduce_mean((tf.stop_gradient(z) - e) ** 2)  # 使码本空间趋向于待量化的向量
    mse_z = tf.reduce_mean((tf.stop_gradient(e) - z) ** 2)  # 使待量化的向量趋向于码本空间

    # 返回加权损失
    return tf.reduce_mean(weighted_loss_value) + mse_e + 0.25 * mse_z


# 验证损失函数（不使用权重）
def val_loss(inputs, outputs, model_outputs):
    # 从模型输出中提取中间值
    _, e, z, _, _ = model_outputs

    # 计算基础损失
    base_loss = cal_score_tf(inputs, outputs, reduction='none')

    # 添加矢量量化损失项
    mse_e = tf.reduce_mean((tf.stop_gradient(z) - e) ** 2)
    mse_z = tf.reduce_mean((tf.stop_gradient(e) - z) ** 2)

    return tf.reduce_mean(base_loss) + mse_e + 0.25 * mse_z


# 自定义训练步（添加梯度裁剪）
def train_step(inputs, targets, weights_batch):
    with tf.GradientTape() as tape:
        # 获取模型输出（包括中间值）
        model_outputs = intermediateModel(inputs, training=True)
        predictions = model_outputs[0]  # 主输出

        # 计算损失
        loss = weighted_loss(inputs, predictions, model_outputs, weights_batch)

    # 计算梯度
    gradients = tape.gradient(loss, autoencoderModel.trainable_variables)

    # 梯度裁剪：防止梯度爆炸
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, 0.5)  # 阈值降低到0.5

    # 更新权重（使用裁剪后的梯度）
    optimizer.apply_gradients(zip(clipped_gradients, autoencoderModel.trainable_variables))

    # 返回损失值
    return loss


# 验证函数
def validate_step(inputs, targets):
    model_outputs = intermediateModel(inputs, training=False)
    predictions = model_outputs[0]
    loss = val_loss(inputs, predictions, model_outputs)
    return loss


# 训练模型
# 重新设计的三阶段学习率调度器
class ThreePhaseLRScheduler:
    def __init__(self, warmup_epochs, stable_epochs, decay_epochs, initial_lr):
        self.warmup_epochs = warmup_epochs
        self.stable_epochs = stable_epochs
        self.decay_epochs = decay_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch):
        # 阶段1: 预热阶段（线性增长）
        if epoch < self.warmup_epochs:
            return self.initial_lr * (epoch / self.warmup_epochs)

        # 阶段2: 稳定阶段（保持初始学习率）
        elif epoch < self.warmup_epochs + self.stable_epochs:
            return self.initial_lr

        # 阶段3: 衰减阶段（余弦退火）
        else:
            decay_epoch = epoch - self.warmup_epochs - self.stable_epochs
            decay_ratio = decay_epoch / self.decay_epochs
            return 0.5 * self.initial_lr * (1 + math.cos(math.pi * decay_ratio))


# 根据训练配置计算调度参数
num_epochs = 600
total_samples_per_epoch = 18000  # 每个epoch的训练样本数
batch_size = 512

# 计算每个epoch的批次数
steps_per_epoch = math.ceil(total_samples_per_epoch / batch_size)  # 36步/epoch

# 设置三阶段调度参数
warmup_epochs = 30  # 前30个epoch预热
stable_epochs = 240  # 中间240个epoch保持稳定
decay_epochs = num_epochs - warmup_epochs - stable_epochs  # 最后330个epoch衰减

# 创建学习率调度器
lr_scheduler = ThreePhaseLRScheduler(
    warmup_epochs=warmup_epochs,
    stable_epochs=stable_epochs,
    decay_epochs=decay_epochs,
    initial_lr=learning_rate
)

# 创建优化器（初始学习率将被调度器覆盖）
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# 创建列表记录训练历史
train_loss_history = []
val_loss_history = []
epoch_learning_rates = []  # 记录每个epoch的学习率

# 训练循环
for epoch in range(num_epochs):
    # 设置当前epoch的学习率
    current_lr = lr_scheduler(epoch)
    optimizer.learning_rate.assign(current_lr)
    epoch_learning_rates.append(current_lr)

    print(f"\nEpoch {epoch + 1}/{num_epochs}, Learning Rate: {current_lr:.6f}")

    # 在每个epoch开始时洗牌训练数据
    # 创建新的随机索引
    shuffle_indices = np.arange(X_train.shape[0])
    np.random.shuffle(shuffle_indices)

    # 应用洗牌索引
    X_train_shuffled = X_train[shuffle_indices]
    Y_train_shuffled = Y_train[shuffle_indices]
    train_weights_shuffled = train_weights[shuffle_indices]

    # 创建新的洗牌后的数据集
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train_shuffled, Y_train_shuffled, train_weights_shuffled)
    )
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # 训练阶段
    epoch_train_loss = []
    for batch_idx, (inputs, targets, weights_batch) in enumerate(train_dataset):
        # 执行训练步
        loss = train_step(inputs, targets, weights_batch)
        epoch_train_loss.append(loss.numpy())

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}: Loss = {loss:.4f}")

    # 计算平均训练损失
    avg_train_loss = np.mean(epoch_train_loss)
    train_loss_history.append(avg_train_loss)
    print(f"Epoch {epoch + 1} Train Loss: {avg_train_loss:.4f}")

    # 验证阶段
    epoch_val_loss = []
    for inputs, targets in val_dataset:
        loss = validate_step(inputs, targets)
        epoch_val_loss.append(loss.numpy())

    # 计算平均验证损失
    avg_val_loss = np.mean(epoch_val_loss)
    val_loss_history.append(avg_val_loss)
    print(f"Epoch {epoch + 1} Val Loss: {avg_val_loss:.4f}")


####################  储存loss，绘制loss曲线
plt.figure(figsize=(15, 10))

# 损失曲线
plt.subplot(2, 1, 1)
plt.plot(train_loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# 学习率曲线
plt.subplot(2, 1, 2)
plt.plot(epoch_learning_rates)
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')

plt.tight_layout()
plt.savefig("custom_training_loss_with_lr.png")  # 将图形保存为文件

'''================测试阶段================'''

# 测试编码器
print("Starting encoder prediction...")
encode_feature1 = encoderModel.predict(x_val, batch_size=512)
print('encode_feature1 shape:', encode_feature1.shape)

# 测试量化模型
print("Starting quantizer prediction...")
code_vecs, codes_before, decode_feature1 = quantizerModel.predict(encode_feature1, batch_size=512)
print('decode_feature1 shape:', decode_feature1.shape)

# 测试解码器
print("Starting decoder prediction...")
W_pre1 = decoderModel.predict(decode_feature1, batch_size=512)
print('W_pre1 shape:', W_pre1.shape)

# 计算评估指标
NUM_SAMPLES = x_val.shape[0]
NUM_SUBBAND = 12  # 子载波数
gcs, sgcs = cal_score(x_val, W_pre1, NUM_SAMPLES, NUM_SUBBAND)
print('GCS: %f' % gcs)
print('SGCS: %f' % sgcs)

# 保存模型权重
autoencoderModel.save_weights('autoencoder_model_weights.h5')
print("Model weights saved successfully.")
