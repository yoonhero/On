import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Encoding Position Infomation to give a positional information
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    """
    Get Angle of Sin or Cos Function

    pos / (10000**((i/2)/d_model))
    """    
    def get_angles(self, position, i, d_model):
        angles = 1/tf.pow(10000, (2*(i//2))/ tf.cast(d_model, tf.float32)) 
        return position * angles
    

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )

        # Even Index(2i) -> Sine
        sines = tf.math.sin(angle_rads[:, 0::2])

        # Odd Index(2i+1) -> Cos
        cosines = tf.math.cos(angle_rads[:, 1::2])


        angle_rads = np.zeros(angle_rads.shape)

        # start :: step 
        # example [0 ~ 10][0::2] -> [0. 2, 4, 6, 8, 10]
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines

        pos_encoding = tf.constant(angle_rads)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        print(pos_encoding.shape)
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
    


# Scaled Dot-Product Attention
def scaled_dot_product_attention(query, key, value, mask):
    # query Dimension : d_model/num_heads 
    # key Dimension : d_model/num_heads
    # value Dimension : d_model/num_heads
    # padding_mask : (batch_size, 1, 1, key Dimension)

    # Q와 K의 곱. Attention Score Matrix 
    # transpose is 전치행렬
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # Scailing
    # divide root(dk=depth)
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # Masking. Insert very small negative number in Attention Score Matrix to mask
    # Very Small -> Softmax and 0
    if mask is not None:
        logits += (mask * -1e9)
        
    
    # Softmax perform as last dimension of Key
    # Attention Weight: (batch_size, num_heads, last query Dimension, key last Dimension)
    attention_weights = tf.nn.softmax(logits, axis=-1)


    # output : (batch_size, num_heads, query last Dimension, d_model/num_heads)
    output = tf.matmul(attention_weights, value)

    return output, attention_weights



if __name__ == "__main__":
    # 문장의 길이 50, 임베딩 벡터의 차원 128
    sample_pos_encoding = PositionalEncoding(50, 128)

    plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 128))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()


    # 임의의 Query, Key, Value인 Q, K, V 행렬 생성
    np.set_printoptions(suppress=True)
    temp_k = tf.constant([[10,0,0],
                        [0,10,0],
                        [0,0,10],
                        [0,0,10]], dtype=tf.float32)  # (4, 3)

    temp_v = tf.constant([[   1,0],
                        [  10,0],
                        [ 100,5],
                        [1000,6]], dtype=tf.float32)  # (4, 2)
    temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)

    # 함수 실행
    temp_out, temp_attn = scaled_dot_product_attention(temp_q, temp_k, temp_v, None)
    print(temp_attn) # 어텐션 분포(어텐션 가중치의 나열)
    print(temp_out) # 어텐션 값 