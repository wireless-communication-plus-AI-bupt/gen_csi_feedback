import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

from tensorflow import keras
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
parser = argparse.ArgumentParser(description='datasets_file')
parser.add_argument('--dataset', type = str)
parser.add_argument('--gpu_id', type = str, default = '0', help = 'gpu_id')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

file_path_train_dataset = args.train_dataset
print(file_path_train_dataset)

'''=============== You need to configure here: ====================================='''
# Set feedback_bits
feedback_bits = 132
# Build model
EMBEDDING_DIM = 64 * 8
NUM_QUAN_BITS = 4
NUM_HEAD = 8  # 8 16
LAYER = 10  # 5，10
learning_rate = 0.001
criterion = cal_score_tf

'''================Read Data  ======================================================='''
data_mat = h5py.File(file_path_train_dataset, 'r')
string = list(data_mat.keys())[0]
print(string)
data = data_mat[string]
print(data.__class__)
print(data.shape)

data_train = np.array(data[:,:,:,:])
# (114000, 12, 32, 2)
data_train = data_train.transpose([3,2,1,0])
print(data_train.__class__)
print(data_train.shape)

x = data_train[:, :, :, :]  
x_data = x.reshape(-1, 12 * 32 * 2)
y_data = x_data
print(x_data.shape)
'''================Data Splating ======================================================='''
SAmplenum = x_data.shape[0]
X_train = x_data[0:int(SAmplenum * 0.9), :]

SAMplenum = y_data.shape[0]
Y_train = y_data[0:int(SAMplenum * 0.9), :]

Samplenum = x_data.shape[0]
x_train_pre = x_data[int(Samplenum * 0.9):Samplenum, :]

SAMPlenum = y_data.shape[0]
y_train_pre = y_data[int(SAMPlenum * 0.9):SAMPlenum, :]

INPUT_SHAPE = X_train.shape[-1]  # 768 1536

'''================================================================================'''

#   encoder model
encoderInput = keras.Input(shape=(INPUT_SHAPE))
encoderOutput = Encoder(encoderInput, feedback_bits, NUM_QUAN_BITS, EMBEDDING_DIM, NUM_HEAD)
encoder = keras.Model(inputs=encoderInput, outputs=[encoderOutput], name='Encoder')

#   decoder model
decoderInput = keras.Input(shape=(INPUT_SHAPE))
decoderOutput = Decoder(decoderInput, feedback_bits, NUM_QUAN_BITS, EMBEDDING_DIM, NUM_HEAD)
decoder = keras.Model(inputs=decoderInput, outputs=[decoderOutput], name="Decoder")

#   Quantizer model
vqInput = keras.Input(shape=(INPUT_SHAPE))
vq_layer = VectorQuantizer(2**NUM_QUAN_BITS)
x = layers.Dense(units=2*int(feedback_bits / NUM_QUAN_BITS), activation='sigmoid')(vqInput)
codes_before = layers.Reshape((int(feedback_bits/NUM_QUAN_BITS), 2))(x)
codes, code_vecs = vq_layer(codes_before)
# 
ze = Lambda(lambda x: x[0] + K.stop_gradient(x[1] - x[0]))([codes_before, code_vecs])

x = tf.reshape(ze, (-1, 2*int(feedback_bits / NUM_QUAN_BITS)))
x = layers.Dense(768)(x)  # 768


q_model = keras.Model(vqInput, [code_vecs, codes_before, x])

#   autoencoder model
autoencoderInput = keras.Input(shape=(INPUT_SHAPE))
x = autoencoderInput

x = encoder(x)
print(x.shape)
e,z,x = q_model(x)  # e是矢量量化的码本空间，z是待量化的向量（encoder的输出）
print(x.shape)
x = decoder(x)
print(x.shape)

autoencoderModel = keras.Model(inputs=autoencoderInput, outputs=x, name='Autoencoder')
loss_gcs = cal_score_tf(autoencoderInput, x)
mse_e = K.mean((K.stop_gradient(z) - e)**2)  # 使码本空间趋向于待量化的向量，停止z的梯度下降就是不优化z，只优化e
mse_z = K.mean((K.stop_gradient(e) - z)**2)  # 使待量化的向量趋向于码本空间
loss = loss_gcs+mse_e+0.25*mse_z  # 损失函数由：loss_sgcs即(1-sgcs)，和后两项均方误差共同构成（mse代表均方误差）
autoencoderModel.add_loss(loss)
autoencoderModel.summary()

val_gcs = MyMAE()

# sgd = tensorflow.keras.optimizers.SGD(learning_rate=0.00001, decay=1e-4, momentum=0.9)
adam = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate)

# autoencoderModel.compile(optimizer='sgd', metrics=val_gcs)

# Model train
autoencoderModel.compile(optimizer='adam', metrics=val_gcs)  # metrics作为另一个评价标准，不参与训练，history中val_前缀代表验证集上的表现
history = autoencoderModel.fit(x=X_train, y=Y_train, batch_size=512, epochs=300, verbose=2, validation_split=0.1)

####################  储存loss，绘制loss曲线
loss = history.history['loss']
val_loss = history.history['val_loss']
loss = np.array(loss)
val_loss = np.array(val_loss)

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("BS_overfitting.png")  # 将图形保存为文件
EN_BATCH_SIZE = 512

encode_feature1 = encoder.predict(x_data, EN_BATCH_SIZE)
print('encode_feature1:')
print(encode_feature1.shape)
code_vecs, codes_before, decode_feature1 = q_model.predict(encode_feature1, EN_BATCH_SIZE)
    
print('decode_feature1:')
print(decode_feature1.shape)
W_pre1 = decoder.predict(decode_feature1, EN_BATCH_SIZE)
print('W_pre1:')
print(W_pre1.shape)

NUM_SAMPLES = x_data.shape[0]
NUM_SUBBAND = 12  # 子载波数
gcs,sgcs = cal_score(x_data, W_pre1, NUM_SAMPLES, NUM_SUBBAND)  # 处理时隙任务，计算SGCS时取x_train的最后一个时隙
print('GCS: %f' %gcs)
print('SGCS: %f' %sgcs)
