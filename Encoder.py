import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K



class Encoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, d_model, num_heads, middle_units,
                max_seq_len, epsilon=1e-6, dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.n_layers = n_layers
        self.d_model = d_model
        self.pos_embedding = PositionalEncoding(sequence_len=max_seq_len, embedding_dim=d_model)

        self.encode_layer = [EncoderLayer(d_model=d_model, num_heads=num_heads, 
                                          middle_units=middle_units, 
                                          epsilon=epsilon, dropout_rate=dropout_rate)
                            for _ in range(n_layers)]
        
    def call(self, inputs, mask, training):
        emb = inputs
        emb = self.pos_embedding(emb)
        
        for i in range(self.n_layers):
            emb = self.encode_layer[i](emb, mask, training)

        return emb





class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, middle_units, epsilon=1e-6, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, middle_units)
        
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, inputs, mask, training):
        # 多头注意力网络
        att_output = self.mha([inputs, inputs, inputs, mask])
        att_output = self.dropout1(att_output, training=training)
        out1 = self.layernorm1(inputs + att_output)  # (batch_size, input_seq_len, d_model)
        
        # 前向网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)   # (batch_size, input_seq_len, d_model)
        
        return out2




class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        self.eps = epsilon
        super(LayerNormalization, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=tf.ones_initializer(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=tf.zeros_initializer(), trainable=True)
        super(LayerNormalization, self).build(input_shape)
        
    def call(self, x):
        mean = tf.keras.backend.mean(x, axis=-1, keepdims=True)
        std = tf.keras.backend.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    
    def compute_output_shape(self, input_shape):
        return input_shape





def point_wise_feed_forward_network(d_model, middle_units):
    
    return tf.keras.Sequential([
        tf.keras.layers.Dense(middle_units, activation='relu'),
        tf.keras.layers.Dense(d_model)])





# 构造 multi head attention 层

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        # d_model 必须可以正确分为各个头
        assert d_model % num_heads == 0
        
        # 分头后的维度
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        # 分头, 将头个数的维度 放到 seq_len 前面
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        q, k, v, mask = inputs
        batch_size = tf.shape(q)[0]

        # 分头前的前向网络，获取q、k、v语义
        q = self.wq(q) # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)

        # 分头
        q = self.split_heads(q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size) # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size) # (batch_size, num_heads, seq_len_v, depth)
        
        # 通过缩放点积注意力层
        scaled_attention = scaled_dot_product_attention(q, k, v, mask) # (batch_size, num_heads, seq_len_q, depth)
        
        # “多头维度” 后移
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3]) # (batch_size, seq_len_q, num_heads, depth)

        # 合并 “多头维度”
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # 全连接层
        output = self.dense(concat_attention)
        
        return output




def scaled_dot_product_attention(q, k, v, mask):
    
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dim_k = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dim_k)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)

    return output




def padding_mask(seq):
    
    # 获取为 0的padding项
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # 扩充维度用于attention矩阵
    return seq[:, np.newaxis, np.newaxis, :] # (batch_size, 1, 1, seq_len)





class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_len=None, embedding_dim=None, mode='sum', **kwargs):
        self.sequence_len = sequence_len
        self.embedding_dim = embedding_dim
        self.mode = mode
        super(PositionalEncoding, self).__init__(**kwargs)

    def call(self, x):
        if (self.embedding_dim == None) or (self.mode == 'sum'):
            self.embedding_dim = int(x.shape[-1])
        
        position_embedding = np.array([
            [pos / np.power(10000, 2. * i / self.embedding_dim) for i in range(self.embedding_dim)]
            for pos in range(self.sequence_len)])

        position_embedding[:, 0::2] = np.sin(position_embedding[:, 0::2])  # dim 2i
        position_embedding[:, 1::2] = np.cos(position_embedding[:, 1::2])  # dim 2i+1
        
        position_embedding = tf.cast(position_embedding, dtype=tf.float32)
        
        if self.mode == 'sum':
            return position_embedding + x
        
        elif self.mode == 'concat':
            position_embedding = tf.reshape(
                tf.tile(position_embedding, (int(x.shape[0]), 1)), 
                (-1, self.sequence_len, self.embedding_dim)
                )

            return tf.concat([position_embedding, x], 2)
        
    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2]+self.embedding_dim)






if __name__ == "__main__":

    sample_encoder = Encoder(2, 512, 8, 1024, 60)
    sample_encoder_output = sample_encoder(tf.random.uniform((15, 60, 512)),
                            None, False)

    print(sample_encoder_output.shape)