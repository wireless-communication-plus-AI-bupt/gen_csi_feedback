import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import *
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import layers as layers1
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
# from tensorflow.python.framework import ops
# from tensorflow.python.ops import math_ops
from tensorflow.python.ops.init_ops import truncated_normal_initializer
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.python.keras.layers.convolutional import Conv2D as cnn1
from keras.layers.normalization import layer_normalization as bn
# from tensorflow.python.keras.layers.normalization.batch_normalization import BatchNormalization as bn
# # from tensorflow.python.keras.layers.normalization import BatchNormalization as bn
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
# from tensorflow.keras.layers.pooling import AveragePooling2D as pool
from tensorflow.python.keras.layers.pooling import MaxPooling2D as pool_Max

# Number to Bit Function Defining
def Num2Bit(Num, B):
    Num_ = Num.numpy()
    bit = (np.unpackbits(np.array(Num_, np.uint8), axis=1).reshape(-1, Num_.shape[1], 8)[:, :, (8 - B):]).reshape(-1,Num_.shape[1] * B)
    bit.astype(np.float32)
    return tf.convert_to_tensor(bit, dtype=tf.float32)


# Bit to Number Function Defining
def Bit2Num(Bit, B):
    Bit_ = Bit.numpy()
    Bit_.astype(np.float32)
    Bit_ = np.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = np.zeros(shape=np.shape(Bit_[:, :, 1]))
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return tf.cast(num, dtype=tf.float32)

# Quantization and Dequantization Layers Defining
# @tf.custom_gradient
def QuantizationOp(x, B):
    step = tf.cast((2 ** B), dtype=tf.float32)
    result = tf.cast((tf.round(x * step - 0.5)), dtype=tf.float32)
    result = tf.py_function(func=Num2Bit, inp=[result, B], Tout=tf.float32)

    def custom_grad(dy):
        grad = dy
        return (grad, grad)

    return result, custom_grad


class QuantizationLayer(keras.layers.Layer):
    def __init__(self, B, **kwargs):
        self.B = B
        super(QuantizationLayer, self).__init__()

    def call(self, x):
        return QuantizationOp(x, self.B)

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(QuantizationLayer, self).get_config()
        base_config['B'] = self.B
        return base_config


# @tf.custom_gradient
def DequantizationOp(x, B):
    x = tf.py_function(func=Bit2Num, inp=[x, B], Tout=tf.float32)
    step = tf.cast((2 ** B), dtype=tf.float32)
    result = tf.cast((x + 0.5) / step, dtype=tf.float32)

    def custom_grad(dy):
        grad = dy * 1
        return (grad, grad)

    return result, custom_grad


class DeuantizationLayer(keras.layers.Layer):
    def __init__(self, B, **kwargs):
        self.B = B
        super(DeuantizationLayer, self).__init__()

    def call(self, x):
        return DequantizationOp(x, self.B)

    def get_config(self):
        base_config = super(DeuantizationLayer, self).get_config()
        base_config['B'] = self.B
        return base_config


class ScaledDotProductAttention(keras.layers.Layer):
    r"""The attention layer that takes three inputs representing queries, keys and values.

    \text{Attention}(Q, K, V) = \text{softmax}(\frac{Q K^T}{\sqrt{d_k}}) V

    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 return_attention=False,
                 history_only=False,
                 **kwargs):
        """Initialize the layer.

        :param return_attention: Whether to return attention weights.
        :param history_only: Whether to only use history data.
        :param kwargs: Arguments for parent class.
        """
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.return_attention = return_attention
        self.history_only = history_only
        self.intensity = self.attention = None

    def get_config(self):
        config = {
            'return_attention': self.return_attention,
            'history_only': self.history_only,
        }
        base_config = super(ScaledDotProductAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            query_shape, key_shape, value_shape = input_shape
        else:
            query_shape = key_shape = value_shape = input_shape
        output_shape = query_shape[:-1] + value_shape[-1:]
        if self.return_attention:
            attention_shape = query_shape[:2] + (key_shape[1],)
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            mask = mask[0]
        if self.return_attention:
            return [mask, None]
        return mask

    def call(self, inputs, mask=None, **kwargs):
        if isinstance(inputs, list):
            query, key, value = inputs
        else:
            query = key = value = inputs
        if isinstance(mask, list):
            mask = mask[1]
        feature_dim = K.shape(query)[-1]
        e = K.batch_dot(query, key, axes=2) / K.sqrt(K.cast(feature_dim, dtype=K.floatx()))
        if self.history_only:
            query_len, key_len = K.shape(query)[1], K.shape(key)[1]
            indices = K.expand_dims(K.arange(0, key_len), axis=0)
            upper = K.expand_dims(K.arange(0, query_len), axis=-1)
            e -= 10000.0 * K.expand_dims(K.cast(indices > upper, K.floatx()), axis=0)
        if mask is not None:
            e -= 10000.0 * (1.0 - K.cast(K.expand_dims(mask, axis=-2), K.floatx()))
        self.intensity = e
        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        self.attention = e / K.sum(e, axis=-1, keepdims=True)
        v = K.batch_dot(self.attention, value)
        if self.return_attention:
            return [v, self.attention]
        return v


class MultiHead(keras.layers.Layer):
    """Multi-head attention layer.

    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 head_num,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 history_only=False,
                 **kwargs):
        """Initialize the layer.

        :param head_num: Number of heads.
        :param activation: Activations for linear mappings.
        :param use_bias: Whether to use bias term.
        :param kernel_initializer: Initializer for linear mappings.
        :param bias_initializer: Initializer for linear mappings.
        :param kernel_regularizer: Regularizer for linear mappings.
        :param bias_regularizer: Regularizer for linear mappings.
        :param kernel_constraint: Constraints for linear mappings.
        :param bias_constraint: Constraints for linear mappings.
        :param history_only: Whether to only use history in attention layer.
        """
        self.supports_masking = True
        self.head_num = head_num
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.history_only = history_only

        self.Wq = self.Wk = self.Wv = self.Wo = None
        self.bq = self.bk = self.bv = self.bo = None

        self.intensity = self.attention = None
        super(MultiHead, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'head_num': self.head_num,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'history_only': self.history_only,
        }
        base_config = super(MultiHead, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def gelu(self, features, approximate=False, name=None):
        with ops.name_scope(name, "Gelu", [features]):
            features = ops.convert_to_tensor(features, name="features")
            if approximate:
                coeff = math_ops.cast(0.044715, features.dtype)
                return 0.5 * features * (
                        1.0 + math_ops.tanh(0.7978845608028654 *
                                            (features + coeff * math_ops.pow(features, 3))))
            else:
                return 0.5 * features * (1.0 + math_ops.erf(
                    features / math_ops.cast(1.4142135623730951, features.dtype)))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            q, k, v = input_shape
            return q[:-1] + (v[-1],)
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        if isinstance(input_mask, list):
            return input_mask[0]
        return input_mask

    def build(self, input_shape):
        if isinstance(input_shape, list):
            q, k, v = input_shape
        else:
            q = k = v = input_shape
        feature_dim = int(v[-1])
        if feature_dim % self.head_num != 0:
            raise IndexError('Invalid head number %d with the given input dim %d' % (self.head_num, feature_dim))
        self.Wq = self.add_weight(
            shape=(int(q[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wq' % self.name,
        )
        if self.use_bias:
            self.bq = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bq' % self.name,
            )
        self.Wk = self.add_weight(
            shape=(int(k[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wk' % self.name,
        )
        if self.use_bias:
            self.bk = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bk' % self.name,
            )
        self.Wv = self.add_weight(
            shape=(int(v[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wv' % self.name,
        )
        if self.use_bias:
            self.bv = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bv' % self.name,
            )
        self.Wo = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wo' % self.name,
        )
        if self.use_bias:
            self.bo = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bo' % self.name,
            )
        super(MultiHead, self).build(input_shape)

    @staticmethod
    def _reshape_to_batches(x, head_num):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        head_dim = feature_dim // head_num
        x = K.reshape(x, (batch_size, seq_len, head_num, head_dim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size * head_num, seq_len, head_dim))

    @staticmethod
    def _reshape_attention_from_batches(x, head_num):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = K.reshape(x, (batch_size // head_num, head_num, seq_len, feature_dim))
        return K.permute_dimensions(x, [0, 2, 1, 3])

    @staticmethod
    def _reshape_from_batches(x, head_num):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = K.reshape(x, (batch_size // head_num, head_num, seq_len, feature_dim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size // head_num, seq_len, feature_dim * head_num))

    @staticmethod
    def _reshape_mask(mask, head_num):
        if mask is None:
            return mask
        seq_len = K.shape(mask)[1]
        mask = K.expand_dims(mask, axis=1)
        mask = K.tile(mask, [1, head_num, 1])
        return K.reshape(mask, (-1, seq_len))

    def call(self, inputs, mask=None):
        if isinstance(inputs, list):
            q, k, v = inputs
        else:
            q = k = v = inputs
        if isinstance(mask, list):
            q_mask, k_mask, v_mask = mask
        else:
            q_mask = k_mask = v_mask = mask
        q = K.dot(q, self.Wq)
        k = K.dot(k, self.Wk)
        v = K.dot(v, self.Wv)
        if self.use_bias:
            q += self.bq
            k += self.bk
            v += self.bv
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        scaled_dot_product_attention = ScaledDotProductAttention(
            history_only=self.history_only,
            name='%s-Attention' % self.name,
        )
        y = scaled_dot_product_attention(
            inputs=[
                self._reshape_to_batches(q, self.head_num),
                self._reshape_to_batches(k, self.head_num),
                self._reshape_to_batches(v, self.head_num),
            ],
            mask=[
                self._reshape_mask(q_mask, self.head_num),
                self._reshape_mask(k_mask, self.head_num),
                self._reshape_mask(v_mask, self.head_num),
            ],
        )
        self.intensity = self._reshape_attention_from_batches(scaled_dot_product_attention.intensity, self.head_num)
        self.attention = self._reshape_attention_from_batches(scaled_dot_product_attention.attention, self.head_num)
        y = self._reshape_from_batches(y, self.head_num)
        y = K.dot(y, self.Wo)
        if self.use_bias:
            y += self.bo
        if self.activation is not None:
            y = self.activation(y)

        # if TF_KERAS:
        # Add shape information to tensor when using `tf.keras`
        input_shape = [K.int_shape(q), K.int_shape(k), K.int_shape(v)]
        output_shape = self.compute_output_shape(input_shape)
        if output_shape[1] is not None:
            output_shape = (-1,) + output_shape[1:]
            y = K.reshape(y, output_shape)
        return y, v



class VectorQuantizer(keras.layers.Layer):
    """量化层
    """
    def __init__(self, num_codes, **kwargs):
        super(VectorQuantizer, self).__init__(**kwargs)
        self.num_codes = num_codes
    def build(self, input_shape):
        super(VectorQuantizer, self).build(input_shape)
        self.dim = input_shape[-1]
        self.lookup_table = self.add_weight('lookup_table',
                                    dtype=tf.float32,
                                    shape=[self.num_codes, self.dim],
                                    initializer=truncated_normal_initializer(mean=0.5, stddev=0))
        # self.embeddings = self.add_weight(
        #     name='embeddings',
        #     shape=(self.num_codes, dim),
        #     initializer='uniform'
        # )
    def call(self, inputs):
        """inputs.shape=[None, m, m, dim]
        """
        z = tf.expand_dims(inputs, -2)  # (B, t, 1, D)
        lookup_table_ = tf.reshape(self.lookup_table, [1, 1, self.num_codes, self.dim])  # (1, 1, K, D)
        dist = tf.norm(z - lookup_table_, axis=-1)  # Broadcasting -> (B, T', K)
        k = tf.argmin(dist, axis=-1)  # (B, t)
        # result = tf.py_function(func=Num2Bit, inp=[k, 4], Tout=tf.float32)
        z_q = tf.gather(self.lookup_table, k)  # (B, t, D)

        return [k, z_q]
    def compute_output_shape(self, input_shape):
        return [input_shape[:-1], input_shape]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_codes': self.num_codes,
        })
        return config



def Encoder(x, NUM_FEEDBACK_BITS, NUM_QUAN_BITS, EMBEDDING_DIM, NUM_HEAD):
    def liner_embedding(y, embedding_dim):
        y = layers.Dense(embedding_dim)(y)
        return y * tf.math.sqrt(tf.cast(embedding_dim, dtype=tf.float32))

    def get_angles(pos, i, embedding_dim):
        get_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embedding_dim))
        return pos * get_rates

    def positional_encoding(sequence_length, embedding_dim):
        angle_rads = get_angles(np.arange(sequence_length)[:, np.newaxis],
                                np.arange(embedding_dim)[np.newaxis, :],
                                embedding_dim)
        # 第2i项使用sin
        sines = np.sin(angle_rads[:, 0::2])
        # 第2i+1项使用cos
        coses = np.cos(angle_rads[:, 1::2])

        pos_encoding = np.concatenate([sines, coses], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)
        return pos_encoding

    def att_block(y, num_head):
        short_cut = y
        y = layers.LayerNormalization()(y)
        y, v = MultiHead(head_num=num_head)(y)

        y = layers.Dropout(0.1)(y)
        y = layers.Add()([short_cut, y])
        return y

    def fc_block(y, inner_dim, embedding_dim):
        short_cut = y
        y = layers.LayerNormalization()(y)

        y = layers.Dense(inner_dim)(y)
        y = layers.ReLU()(y)
        y = layers.Dropout(0.1)(y)
        y = layers.Dense(embedding_dim)(y)

        y = layers.Dropout(0.1)(y)
        y = layers.Add()([short_cut, y])
        return y
    
    x = layers.Reshape((12, 64))(x)  # 这是初始的维度 12*32*2
    # x = layers.Reshape((24, 64))(x)  # 这是初始的维度 12*32*2
    # x = layers.Reshape((60, 64))(x)  # 这是为了做时隙任务后改的维度 5*12*32*2，相当于整组数据都反馈，五个时隙都输出
    # src_embedding
    x = liner_embedding(x, EMBEDDING_DIM)  # embedding只对于64的那个维度做，相当于初始数据的最后两个维度（32*2）
    # src_pos_embedding
    x = x + positional_encoding(x.shape[1], EMBEDDING_DIM)[:, :x.shape[1], :]
    x = layers.Dropout(0.1)(x)
    # (B, 12, 512)
    # trans_encoder
    for i in range(5):  # 5,10
        x = att_block(x, NUM_HEAD)
        x = fc_block(x, int(EMBEDDING_DIM * 2), EMBEDDING_DIM)
    # (B, 12, 512)
    x = layers.LayerNormalization()(x)
    # (B, 12, 512)
    x = layers.Dense(64)(x)
    # (B, 12, 64)
    x = layers.Flatten()(x)
    # (B, 768)

    #x = vq(x_before_quan)
    #x = x + tf.stop_gradient(VQ()(x) - x)
    #x = QuantizationLayer(NUM_QUAN_BITS)(x)
    return x


def Decoder(x, NUM_FEEDBACK_BITS, NUM_QUAN_BITS, EMBEDDING_DIM, NUM_HEAD):
    def liner_embedding(y, embedding_dim):
        y = layers.Dense(embedding_dim)(y)
        return y * tf.math.sqrt(tf.cast(embedding_dim, dtype=tf.float32))

    def get_angles(pos, i, embedding_dim):
        get_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embedding_dim))
        return pos * get_rates

    def positional_encoding(sequence_length, embedding_dim):
        angle_rads = get_angles(np.arange(sequence_length)[:, np.newaxis],
                                np.arange(embedding_dim)[np.newaxis, :],
                                embedding_dim)
        # 第2i项使用sin
        sines = np.sin(angle_rads[:, 0::2])
        # 第2i+1项使用cos
        coses = np.cos(angle_rads[:, 1::2])

        pos_encoding = np.concatenate([sines, coses], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)
        return pos_encoding

    def att_block(y, num_head):
        short_cut = y
        y = layers.LayerNormalization()(y)

        y, v = MultiHead(head_num=num_head)(y)

        y = layers.Dropout(0.1)(y)
        y = layers.Add()([short_cut, y])
        return y

    def fc_block(y, inner_dim, embedding_dim):
        short_cut = y
        y = layers.LayerNormalization()(y)

        y = layers.Dense(inner_dim)(y)
        y = layers.ReLU()(y)
        y = layers.Dropout(0.1)(y)
        y = layers.Dense(embedding_dim)(y)

        y = layers.Dropout(0.1)(y)
        y = layers.Add()([short_cut, y])
        return y
    # forward
    # x = DeuantizationLayer(NUM_QUAN_BITS)(x)
    # x = tf.reshape(x,(-1, int(NUM_FEEDBACK_BITS / NUM_QUAN_BITS))) - 0.5

    # x = layers.Dense(units=6144, activation="gelu")(x_afterquan)

    x = layers.Reshape((12, 64))(x)  # 这是初始的维度 12*32*2
    # x = layers.Reshape((24, 64))(x)  # 这是初始的维度 12*32*2
    # x = layers.Reshape((60, 64))(x)  # 这是为了做时隙任务后改的维度 5*12*32*2，相当于整组数据都反馈，五个时隙都输出

    # trg_embedding
    x = liner_embedding(x, EMBEDDING_DIM)  # embedding只对于64的那个维度做，相当于初始数据的最后两个维度（32*2）
    # src_pos_embedding

    x += positional_encoding(x.shape[1], EMBEDDING_DIM)[:, :x.shape[1], :]
    x = layers.Dropout(0.1)(x)

    for i in range(5):  # 5,10
        x = att_block(x, NUM_HEAD)
        x = fc_block(x, int(EMBEDDING_DIM * 2), EMBEDDING_DIM)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(64)(x)
    x = layers.Flatten()(x)
    return x

def Encoder_LSTM(x, NUM_FEEDBACK_BITS, NUM_QUAN_BITS, EMBEDDING_DIM, NUM_HEAD):
    def liner_embedding(y, embedding_dim):
        y = layers.Dense(embedding_dim)(y)
        return y * tf.math.sqrt(tf.cast(embedding_dim, dtype=tf.float32))

    def get_angles(pos, i, embedding_dim):
        get_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embedding_dim))
        return pos * get_rates

    def positional_encoding(sequence_length, embedding_dim):
        angle_rads = get_angles(np.arange(sequence_length)[:, np.newaxis],
                                np.arange(embedding_dim)[np.newaxis, :],
                                embedding_dim)
        # 第2i项使用sin
        sines = np.sin(angle_rads[:, 0::2])
        # 第2i+1项使用cos
        coses = np.cos(angle_rads[:, 1::2])

        pos_encoding = np.concatenate([sines, coses], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)
        return pos_encoding

    def att_block(y, num_head):
        short_cut = y
        y = layers.LayerNormalization()(y)
        y, v = MultiHead(head_num=num_head)(y)

        y = layers.Dropout(0.1)(y)
        y = layers.Add()([short_cut, y])
        return y

    def fc_block(y, inner_dim, embedding_dim):
        short_cut = y
        y = layers.LayerNormalization()(y)

        y = layers.Dense(inner_dim)(y)
        y = layers.ReLU()(y)
        y = layers.Dropout(0.1)(y)
        y = layers.Dense(embedding_dim)(y)

        y = layers.Dropout(0.1)(y)
        y = layers.Add()([short_cut, y])
        return y

    # # 引入LSTM层,使用ConvLSTM提取特征后拍平也可以
    # print('LSTM layer:')
    # x = layers.Reshape((4, 768))(x)  # 引入LSTM层
    # print(x.shape)
    # x = LSTM(units=768)(x)
    # print(x.shape)
    # x = layers.Dense(768)(x)
    # print(x.shape)
    # print('End of LSTM')

    # 引入ConvLSTM层
    print('ConvLSTM layer:')
    x = layers.Reshape((5, 12, 64, 1))(x)  # 引入ConvLSTM层，发射天线*实部虚部作为一个维度（和上述embedding想法相同），最后1作为单通道
    # x = layers.Reshape((3, 12, 64, 1))(x)  # 引入ConvLSTM层，发射天线*实部虚部作为一个维度（和上述embedding想法相同），最后1作为单通道
    print(x.shape)
    # x = ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', data_format='channels_last', return_sequences=True)(x)
    # x = bn()(x)  # 加一个bn防止过拟合
    # print(x.shape)
    # x = ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', data_format='channels_last', return_sequences=True)(x)
    # x = bn()(x)  # 加一个bn防止过拟合
    # print(x.shape)
    x = ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', data_format='channels_last')(x)
    x = bn()(x)  # 加一个bn防止过拟合
    print(x.shape)
    x = layers.Flatten()(x)  # 把网络拍平
    print(x.shape)
    # x = layers.Dense(3840)(x)  # 5*12*32*2 五个时隙都作为输出，相当于整组数据都反馈，五个时隙都输出
    x = layers.Dense(768)(x)  # 12*32*2 一组内只有最后一个时隙作为输出，跟没有参考历史信息的反馈方案做对比，应该有性能提升
    # x = layers.Dropout(0.1)(x)  # 加一个dropout防止过拟合
    print(x.shape)
    print('End of ConvLSTM')

    # 这是初始的Encoder
    x = layers.Reshape((12, 64))(x)  # 这是初始的维度 12*32*2
    # x = layers.Reshape((60, 64))(x)  # 这是为了做时隙任务后改的维度 5*12*32*2
    print(x.shape)
    # src_embedding
    x = liner_embedding(x, EMBEDDING_DIM)
    print('Embedding:')
    print(x.shape)
    # src_pos_embedding
    x = x + positional_encoding(x.shape[1], EMBEDDING_DIM)[:, :x.shape[1], :]
    print(x.shape)
    x = layers.Dropout(0.1)(x)
    print(x.shape)
    # trans_encoder
    for i in range(5):  # 5,10
        x = att_block(x, NUM_HEAD)
        x = fc_block(x, int(EMBEDDING_DIM * 2), EMBEDDING_DIM)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(64)(x)
    x = layers.Flatten()(x)

    #x = vq(x_before_quan)
    #x = x + tf.stop_gradient(VQ()(x) - x)
    #x = QuantizationLayer(NUM_QUAN_BITS)(x)
    return x

def Encoder_TSF(x, NUM_FEEDBACK_BITS, NUM_QUAN_BITS, EMBEDDING_DIM, NUM_HEAD):
    def liner_embedding(y, embedding_dim):
        y = layers.Dense(embedding_dim)(y)
        return y * tf.math.sqrt(tf.cast(embedding_dim, dtype=tf.float32))

    def get_angles(pos, i, embedding_dim):
        get_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embedding_dim))
        return pos * get_rates

    def positional_encoding(sequence_length, embedding_dim):
        angle_rads = get_angles(np.arange(sequence_length)[:, np.newaxis],
                                np.arange(embedding_dim)[np.newaxis, :],
                                embedding_dim)
        # 第2i项使用sin
        sines = np.sin(angle_rads[:, 0::2])
        # 第2i+1项使用cos
        coses = np.cos(angle_rads[:, 1::2])

        pos_encoding = np.concatenate([sines, coses], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)
        return pos_encoding

    def att_block(y, num_head):
        short_cut = y
        y = layers.LayerNormalization()(y)
        y, v = MultiHead(head_num=num_head)(y)

        y = layers.Dropout(0.1)(y)
        y = layers.Add()([short_cut, y])
        return y

    def fc_block(y, inner_dim, embedding_dim):
        short_cut = y
        y = layers.LayerNormalization()(y)

        y = layers.Dense(inner_dim)(y)
        y = layers.ReLU()(y)
        y = layers.Dropout(0.1)(y)
        y = layers.Dense(embedding_dim)(y)

        y = layers.Dropout(0.1)(y)
        y = layers.Add()([short_cut, y])
        return y

    x = layers.Dense(768)(x)  # 恢复768维度
    x = layers.Reshape((12, 64))(x)  # 这是初始的维度 12*32*2
    # x = layers.Reshape((24, 64))(x)  # 这是初始的维度 12*32*2
    # x = layers.Reshape((60, 64))(x)  # 这是为了做时隙任务后改的维度 5*12*32*2，相当于整组数据都反馈，五个时隙都输出
    # src_embedding
    x = liner_embedding(x, EMBEDDING_DIM)  # embedding只对于64的那个维度做，相当于初始数据的最后两个维度（32*2）
    # src_pos_embedding
    x = x + positional_encoding(x.shape[1], EMBEDDING_DIM)[:, :x.shape[1], :]
    x = layers.Dropout(0.1)(x)
    # trans_encoder
    for i in range(5):  # 5,10
        x = att_block(x, NUM_HEAD)
        x = fc_block(x, int(EMBEDDING_DIM * 2), EMBEDDING_DIM)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(64)(x)
    x = layers.Flatten()(x)

    #x = vq(x_before_quan)
    #x = x + tf.stop_gradient(VQ()(x) - x)
    #x = QuantizationLayer(NUM_QUAN_BITS)(x)
    return x

def Encoder_LSTM_2(x, NUM_FEEDBACK_BITS, NUM_QUAN_BITS, EMBEDDING_DIM, NUM_HEAD, loaded_encoder_states):
    def liner_embedding(y, embedding_dim):
        y = layers.Dense(embedding_dim)(y)
        return y * tf.math.sqrt(tf.cast(embedding_dim, dtype=tf.float32))

    def get_angles(pos, i, embedding_dim):
        get_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embedding_dim))
        return pos * get_rates

    def positional_encoding(sequence_length, embedding_dim):
        angle_rads = get_angles(np.arange(sequence_length)[:, np.newaxis],
                                np.arange(embedding_dim)[np.newaxis, :],
                                embedding_dim)
        # 第2i项使用sin
        sines = np.sin(angle_rads[:, 0::2])
        # 第2i+1项使用cos
        coses = np.cos(angle_rads[:, 1::2])

        pos_encoding = np.concatenate([sines, coses], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)
        return pos_encoding

    def att_block(y, num_head):
        short_cut = y
        y = layers.LayerNormalization()(y)
        y, v = MultiHead(head_num=num_head)(y)

        y = layers.Dropout(0.1)(y)
        y = layers.Add()([short_cut, y])
        return y

    def fc_block(y, inner_dim, embedding_dim):
        short_cut = y
        y = layers.LayerNormalization()(y)

        y = layers.Dense(inner_dim)(y)
        y = layers.ReLU()(y)
        y = layers.Dropout(0.1)(y)
        y = layers.Dense(embedding_dim)(y)

        y = layers.Dropout(0.1)(y)
        y = layers.Add()([short_cut, y])
        return y

    # 引入ConvLSTM层
    print('ConvLSTM layer:')
    x = layers.Reshape((1, 12, 64, 1))(x)  # 引入ConvLSTM层，发射天线*实部虚部作为一个维度（和上述embedding想法相同），最后1作为单通道
    # x = layers.Reshape((3, 12, 64, 1))(x)  # 引入ConvLSTM层，发射天线*实部虚部作为一个维度（和上述embedding想法相同），最后1作为单通道
    print(x.shape)

    x, en_state_h, en_state_c = ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', data_format='channels_last', return_state=True)(x)
    # x, state_h, state_c = ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', data_format='channels_last', return_state=True, initial_state=loaded_encoder_states)(x)
    encoder_states = [en_state_h, en_state_c]  # 保存网络状态
    x = bn()(x)  # 加一个bn防止过拟合
    print(x.shape)
    x = Flatten()(x)  # 把网络拍平
    print(x.shape)
    # x = layers.Dense(3840)(x)  # 5*12*32*2 五个时隙都作为输出，相当于整组数据都反馈，五个时隙都输出
    x = layers.Dense(768)(x)  # 12*32*2 一组内只有最后一个时隙作为输出，跟没有参考历史信息的反馈方案做对比，应该有性能提升
    # x = layers.Dropout(0.1)(x)  # 加一个dropout防止过拟合
    print(x.shape)
    print('End of Encoder ConvLSTM')

    # 这是初始的Encoder
    x = layers.Reshape((12, 64))(x)  # 这是初始的维度 12*32*2
    # x = layers.Reshape((60, 64))(x)  # 这是为了做时隙任务后改的维度 5*12*32*2
    print(x.shape)
    # src_embedding
    x = liner_embedding(x, EMBEDDING_DIM)
    print('Embedding:')
    print(x.shape)
    # src_pos_embedding
    x = x + positional_encoding(x.shape[1], EMBEDDING_DIM)[:, :x.shape[1], :]
    print(x.shape)
    x = layers.Dropout(0.1)(x)
    print(x.shape)
    # trans_encoder
    for i in range(5):  # 5,10
        x = att_block(x, NUM_HEAD)
        x = fc_block(x, int(EMBEDDING_DIM * 2), EMBEDDING_DIM)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(64)(x)
    x = layers.Flatten()(x)

    return x, encoder_states

def Decoder_LSTM_2(x, NUM_FEEDBACK_BITS, NUM_QUAN_BITS, EMBEDDING_DIM, NUM_HEAD, loaded_decoder_states):
    def liner_embedding(y, embedding_dim):
        y = layers.Dense(embedding_dim)(y)
        return y * tf.math.sqrt(tf.cast(embedding_dim, dtype=tf.float32))

    def get_angles(pos, i, embedding_dim):
        get_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embedding_dim))
        return pos * get_rates

    def positional_encoding(sequence_length, embedding_dim):
        angle_rads = get_angles(np.arange(sequence_length)[:, np.newaxis],
                                np.arange(embedding_dim)[np.newaxis, :],
                                embedding_dim)
        # 第2i项使用sin
        sines = np.sin(angle_rads[:, 0::2])
        # 第2i+1项使用cos
        coses = np.cos(angle_rads[:, 1::2])

        pos_encoding = np.concatenate([sines, coses], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)
        return pos_encoding

    def att_block(y, num_head):
        short_cut = y
        y = layers.LayerNormalization()(y)

        y, v = MultiHead(head_num=num_head)(y)

        y = layers.Dropout(0.1)(y)
        y = layers.Add()([short_cut, y])
        return y

    def fc_block(y, inner_dim, embedding_dim):
        short_cut = y
        y = layers.LayerNormalization()(y)

        y = layers.Dense(inner_dim)(y)
        y = layers.ReLU()(y)
        y = layers.Dropout(0.1)(y)
        y = layers.Dense(embedding_dim)(y)

        y = layers.Dropout(0.1)(y)
        y = layers.Add()([short_cut, y])
        return y
    # forward
    # x = DeuantizationLayer(NUM_QUAN_BITS)(x)
    # x = tf.reshape(x,(-1, int(NUM_FEEDBACK_BITS / NUM_QUAN_BITS))) - 0.5

    # x = layers.Dense(units=6144, activation="gelu")(x_afterquan)

    x = layers.Reshape((12, 64))(x)  # 这是初始的维度 12*32*2
    # x = layers.Reshape((60, 64))(x)  # 这是为了做时隙任务后改的维度 5*12*32*2，相当于整组数据都反馈，五个时隙都输出

    # trg_embedding
    x = liner_embedding(x, EMBEDDING_DIM)  # embedding只对于64的那个维度做，相当于初始数据的最后两个维度（32*2）
    # src_pos_embedding

    x += positional_encoding(x.shape[1], EMBEDDING_DIM)[:, :x.shape[1], :]
    x = layers.Dropout(0.1)(x)

    for i in range(5):  # 5,10
        x = att_block(x, NUM_HEAD)
        x = fc_block(x, int(EMBEDDING_DIM * 2), EMBEDDING_DIM)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(64)(x)
    x = layers.Flatten()(x)

    # 引入ConvLSTM层
    print('ConvLSTM layer:')
    x = layers.Reshape((1, 12, 64, 1))(x)  # 引入ConvLSTM层，发射天线*实部虚部作为一个维度（和上述embedding想法相同），最后1作为单通道
    # x = layers.Reshape((3, 12, 64, 1))(x)  # 引入ConvLSTM层，发射天线*实部虚部作为一个维度（和上述embedding想法相同），最后1作为单通道
    print(x.shape)

    x, de_state_h, de_state_c = ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', data_format='channels_last', return_state=True)(x)
    # x, de_state_h, de_state_c = ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', data_format='channels_last', return_state=True, initial_state=loaded_decoder_states)(x)
    decoder_states = [de_state_h, de_state_c]  # 保存网络状态
    x = bn()(x)  # 加一个bn防止过拟合
    print(x.shape)
    x = Flatten()(x)  # 把网络拍平
    print(x.shape)
    # x = layers.Dense(3840)(x)  # 5*12*32*2 五个时隙都作为输出，相当于整组数据都反馈，五个时隙都输出
    x = layers.Dense(768)(x)  # 12*32*2 一组内只有最后一个时隙作为输出，跟没有参考历史信息的反馈方案做对比，应该有性能提升
    # x = layers.Dropout(0.1)(x)  # 加一个dropout防止过拟合
    print(x.shape)
    print('End of Decoder ConvLSTM')

    return x, decoder_states

def Decoder_2(x, NUM_FEEDBACK_BITS, NUM_QUAN_BITS, EMBEDDING_DIM, NUM_HEAD):
    x = layers.Reshape((12, 32, 2))(x)
    x = cnn1(filters=512, kernel_size=(3, 3), strides=1, padding='same', data_format='channels_last')(x)
    x = bn()(x)
    x = Activation('relu')(x)
    x = cnn1(filters=128, kernel_size=(5, 5), strides=1, padding='same', data_format='channels_last')(x)
    x = bn()(x)
    x = Activation('relu')(x)
    x = pool_Max(pool_size=(2, 2), strides=1, padding='same')(x)
    x = cnn1(filters=64, kernel_size=(5, 5), strides=1, padding='same', data_format='channels_last')(x)
    x = bn()(x)
    x = Activation('relu')(x)
    x = Flatten()(x)  # 把网络拍平
    print(x.shape)
    # x = Dropout(0.1)(x)
    # model.add(Flatten())
    x = Dense(768)(x)
    return x


def cos_sim(vector_a, vector_b):
    """
    vector_a, vector_b:预编码向量, (B, Tx) B是样本个数, Tx是天线个数, 每一个元素是复数
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = vector_a * vector_b.H
    num1 = np.sqrt(vector_a * vector_a.H)
    num2 = np.sqrt(vector_b * vector_b.H)
    cos = (num / (num1 * num2))
    return cos


def cal_score(w_true, w_pre, NUM_SAMPLES, NUM_SUBBAND):
    img_total = 64
    num_sample_subband = NUM_SAMPLES * NUM_SUBBAND
    W_true = np.reshape(w_true, [num_sample_subband, img_total])
    W_pre = np.reshape(w_pre, [num_sample_subband, img_total])
    W_true2 = W_true[0:num_sample_subband, 0:int(img_total):2] + 1j * W_true[0:num_sample_subband, 1:int(img_total):2]
    W_pre2 = W_pre[0:num_sample_subband, 0:int(img_total):2] + 1j * W_pre[0:num_sample_subband, 1:int(img_total):2]
    score_cos_gcs = 0
    score_cos_sgcs = 0
    for i in range(num_sample_subband):
        W_true2_sample = W_true2[i:i + 1, ]
        W_pre2_sample = W_pre2[i:i + 1, ]
        score_tmp = cos_sim(W_true2_sample, W_pre2_sample)
        if np.isnan(score_tmp):
            raise ValueError("SGCS computation Error")
        score_cos_gcs = score_cos_gcs + abs(score_tmp)
        score_cos_sgcs = score_cos_sgcs + abs(score_tmp)*abs(score_tmp)
    score_cos_gcs = score_cos_gcs / num_sample_subband
    score_cos_sgcs = score_cos_sgcs / num_sample_subband
    return score_cos_gcs,score_cos_sgcs

def cal_score_LSTM(w_true, w_pre, NUM_SAMPLES, NUM_SLOT, NUM_SUBBAND):
    img_total = 64
    num_sample_slot_subband = NUM_SAMPLES * NUM_SLOT * NUM_SUBBAND
    W_true = np.reshape(w_true, [num_sample_slot_subband, img_total])
    W_pre = np.reshape(w_pre, [num_sample_slot_subband, img_total])
    W_true2 = W_true[0:num_sample_slot_subband, 0:int(img_total):2] + 1j * W_true[0:num_sample_slot_subband, 1:int(img_total):2]
    W_pre2 = W_pre[0:num_sample_slot_subband, 0:int(img_total):2] + 1j * W_pre[0:num_sample_slot_subband, 1:int(img_total):2]
    score_cos_gcs = 0
    score_cos_sgcs = 0
    for i in range(num_sample_slot_subband):
        W_true2_sample = W_true2[i:i + 1, ]
        W_pre2_sample = W_pre2[i:i + 1, ]
        score_tmp = cos_sim(W_true2_sample, W_pre2_sample)
        score_cos_gcs = score_cos_gcs + abs(score_tmp)
        score_cos_sgcs = score_cos_sgcs + abs(score_tmp)*abs(score_tmp)
    score_cos_gcs = score_cos_gcs / num_sample_slot_subband
    score_cos_sgcs = score_cos_sgcs / num_sample_slot_subband
    return score_cos_gcs,score_cos_sgcs

def cal_score_12_6_2(w_true, w_pre, NUM_SAMPLES, NUM_SUBBAND):
    img_total = 12  # 最后两个维度的乘积6*2
    num_sample_subband = NUM_SAMPLES * NUM_SUBBAND
    W_true = np.reshape(w_true, [num_sample_subband, img_total])
    W_pre = np.reshape(w_pre, [num_sample_subband, img_total])
    W_true2 = W_true[0:num_sample_subband, 0:int(img_total):2] + 1j * W_true[0:num_sample_subband, 1:int(img_total):2]
    W_pre2 = W_pre[0:num_sample_subband, 0:int(img_total):2] + 1j * W_pre[0:num_sample_subband, 1:int(img_total):2]
    score_cos_gcs = 0
    score_cos_sgcs = 0
    for i in range(num_sample_subband):
        W_true2_sample = W_true2[i:i + 1, ]
        W_pre2_sample = W_pre2[i:i + 1, ]
        score_tmp = cos_sim(W_true2_sample, W_pre2_sample)
        score_cos_gcs = score_cos_gcs + abs(score_tmp)
        score_cos_sgcs = score_cos_sgcs + abs(score_tmp)*abs(score_tmp)
    score_cos_gcs = score_cos_gcs / num_sample_subband
    score_cos_sgcs = score_cos_sgcs / num_sample_subband
    return score_cos_gcs,score_cos_sgcs

def cal_score_8_4_2(w_true, w_pre, NUM_SAMPLES, NUM_SUBBAND):
    img_total = 8  # 最后两个维度的乘积4*2
    num_sample_subband = NUM_SAMPLES * NUM_SUBBAND
    W_true = np.reshape(w_true, [num_sample_subband, img_total])
    W_pre = np.reshape(w_pre, [num_sample_subband, img_total])
    W_true2 = W_true[0:num_sample_subband, 0:int(img_total):2] + 1j * W_true[0:num_sample_subband, 1:int(img_total):2]
    W_pre2 = W_pre[0:num_sample_subband, 0:int(img_total):2] + 1j * W_pre[0:num_sample_subband, 1:int(img_total):2]
    score_cos_gcs = 0
    score_cos_sgcs = 0
    for i in range(num_sample_subband):
        W_true2_sample = W_true2[i:i + 1, ]
        W_pre2_sample = W_pre2[i:i + 1, ]
        score_tmp = cos_sim(W_true2_sample, W_pre2_sample)
        score_cos_gcs = score_cos_gcs + abs(score_tmp)
        score_cos_sgcs = score_cos_sgcs + abs(score_tmp)*abs(score_tmp)
    score_cos_gcs = score_cos_gcs / num_sample_subband
    score_cos_sgcs = score_cos_sgcs / num_sample_subband
    return score_cos_gcs,score_cos_sgcs


def cos_sim_gcs(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = np.linalg.norm(vector_a * vector_b.H,2)
    num1 = np.linalg.norm(vector_a,2)
    num2 = np.linalg.norm(vector_b,2)
    cos = (num / (num1 * num2))
    return cos



class MyMAE(keras.metrics.Metric):
    def __init__(self, name="mymae", **kwargs):
        super(MyMAE,self).__init__(name=name, **kwargs)
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')
        self.img_total = 64

    def update_state(self, y_true, y_pred, sample_weight=None):


        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        t = cal_score_tf(y_true, y_pred, reduction='mean')

        # y_true = tf.cast(y_true,dtype=tf.float32)
        # y_pred = tf.cast(y_pred, dtype=tf.float32)
        # t = tf.reduce_mean(tf.abs(y_true-y_pred),axis=-1)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, dtype=tf.float32)
            ndim = t.ndim
            weight_ndim = sample_weight.ndim
            t = tf.reduce_mean(t, axis=list(range(weight_ndim, ndim)))
            t = tf.multiply(t, sample_weight)
        t_sum = tf.reduce_sum(t)
        self.total.assign_add(t_sum)
        if sample_weight is not None:
            num = tf.reduce_sum(sample_weight)
        else:
            num = tf.cast(tf.size(t),dtype=tf.float32)
        self.count.assign_add(num)

    def result(self):
        return self.total / self.count



def cal_score_tf(w_true, w_pre, reduction='mean'):
    num_batch, num_sc, num_ant = w_true.shape[0], 12, 32
    w_true = tf.reshape(w_true, [-1, num_ant, 2])
    w_pre = tf.reshape(w_pre, [-1, num_ant, 2])

    w_true_re, w_true_im = w_true[..., 0], w_true[..., 1]
    w_pre_re, w_pre_im = w_pre[..., 0], w_pre[..., 1]

    numerator_re = tf.reduce_sum((w_true_re * w_pre_re + w_true_im * w_pre_im), -1)
    numerator_im = tf.reduce_sum((w_true_im * w_pre_re - w_true_re * w_pre_im), -1)
    denominator_0 = tf.reduce_sum((tf.square(w_true_re) + tf.square(w_true_im)), -1)
    denominator_1 = tf.reduce_sum((tf.square(w_pre_re) + tf.square(w_pre_im)), -1)
    cos_similarity = tf.sqrt(tf.square(numerator_re) + tf.square(numerator_im)) / (
                tf.sqrt(denominator_0) * tf.sqrt(denominator_1))
    #cos_similarity = tf.square(cos_similarity)
    cos_similarity = tf.square(cos_similarity)

    if reduction == 'mean':
        cos_similarity_scalar = tf.reduce_mean(cos_similarity)
    elif reduction == 'sum':
        cos_similarity_scalar = tf.reduce_sum(cos_similarity)

    return 1 - cos_similarity_scalar

def cal_score_tf_LSTM(w_true, w_pre, reduction='mean'):
    # w_true = w_true[:, 2, :]  # 取原始输入的最后一个时隙和反馈输出x进行比较,注意最前面还有一个维度是None，应用于3时隙
    w_true = w_true[:, 4, :]  # 取原始输入的最后一个时隙和反馈输出x进行比较,注意最前面还有一个维度是None，应用于5时隙
    print('cal_score_tf_LSTM中的维度，切片后的w_true，以及w_pre')
    print(w_true.shape)
    print(w_pre.shape)

    num_batch, num_sc, num_ant = w_true.shape[0], 12, 32
    w_true = tf.reshape(w_true, [-1, num_ant, 2])
    w_pre = tf.reshape(w_pre, [-1, num_ant, 2])

    w_true_re, w_true_im = w_true[..., 0], w_true[..., 1]
    w_pre_re, w_pre_im = w_pre[..., 0], w_pre[..., 1]

    numerator_re = tf.reduce_sum((w_true_re * w_pre_re + w_true_im * w_pre_im), -1)
    numerator_im = tf.reduce_sum((w_true_im * w_pre_re - w_true_re * w_pre_im), -1)
    denominator_0 = tf.reduce_sum((tf.square(w_true_re) + tf.square(w_true_im)), -1)
    denominator_1 = tf.reduce_sum((tf.square(w_pre_re) + tf.square(w_pre_im)), -1)
    cos_similarity = tf.sqrt(tf.square(numerator_re) + tf.square(numerator_im)) / (
                tf.sqrt(denominator_0) * tf.sqrt(denominator_1))
    #cos_similarity = tf.square(cos_similarity)
    cos_similarity = tf.square(cos_similarity)

    if reduction == 'mean':
        cos_similarity_scalar = tf.reduce_mean(cos_similarity)
    elif reduction == 'sum':
        cos_similarity_scalar = tf.reduce_sum(cos_similarity)

    return 1 - cos_similarity_scalar

def cal_score_tf_TSF(w_true, w_pre, reduction='mean'):
    w_true = w_true[:, 768:1536]  # 取原始输入的第二部分
    num_batch, num_sc, num_ant = w_true.shape[0], 12, 32
    w_true = tf.reshape(w_true, [-1, num_ant, 2])
    w_pre = tf.reshape(w_pre, [-1, num_ant, 2])

    w_true_re, w_true_im = w_true[..., 0], w_true[..., 1]
    w_pre_re, w_pre_im = w_pre[..., 0], w_pre[..., 1]

    numerator_re = tf.reduce_sum((w_true_re * w_pre_re + w_true_im * w_pre_im), -1)
    numerator_im = tf.reduce_sum((w_true_im * w_pre_re - w_true_re * w_pre_im), -1)
    denominator_0 = tf.reduce_sum((tf.square(w_true_re) + tf.square(w_true_im)), -1)
    denominator_1 = tf.reduce_sum((tf.square(w_pre_re) + tf.square(w_pre_im)), -1)
    cos_similarity = tf.sqrt(tf.square(numerator_re) + tf.square(numerator_im)) / (
                tf.sqrt(denominator_0) * tf.sqrt(denominator_1))
    #cos_similarity = tf.square(cos_similarity)
    cos_similarity = tf.square(cos_similarity)

    if reduction == 'mean':
        cos_similarity_scalar = tf.reduce_mean(cos_similarity)
    elif reduction == 'sum':
        cos_similarity_scalar = tf.reduce_sum(cos_similarity)

    return 1 - cos_similarity_scalar


def average_weights(weights1, weights2, alpha):
    weightsAve = []
    for i in range(len(weights1)):
        weights_merge = (1 - alpha) * weights1[i] + alpha * weights2[i]
        weightsAve.append(weights_merge)
    return weightsAve


def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def get_custom_objects():
    return {"QuantizationLayer": QuantizationLayer, "DeuantizationLayer": DeuantizationLayer, "MultiHead": MultiHead
            ,"VectorQuantizer":VectorQuantizer}
# =======================================================================================================================
# =======================================================================================================================



def cosine_schedule_with_warmup(curr_epoch):
    global epochs
    lr_start = 1e-8
    lr_max = 1e-4
    lr_min = 1e-8
    num_warmup_steps = 20
    num_keep_lr_max_steps = 0
    num_keep_lr_min_steps = 0
    epochs = 200
    # linear warmup
    if curr_epoch < num_warmup_steps:
        lr = (lr_max - lr_min) / num_warmup_steps * curr_epoch + lr_start

    # keep lr_max
    elif num_warmup_steps <= curr_epoch < num_warmup_steps + num_keep_lr_max_steps:
        lr = lr_max

    # start cosine decay
    # after decay_steps, keep lr_min
    else:
        decay_steps = epochs - num_warmup_steps - num_keep_lr_max_steps - num_keep_lr_min_steps
        cosine_decay = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=lr_max,
                                                                 decay_steps=decay_steps,
                                                                 alpha=lr_min / lr_max,
                                                                 )
        lr = cosine_decay(curr_epoch - num_warmup_steps - num_keep_lr_max_steps).numpy()

    return lr