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
    

if __name__ == "__main__":
    # 문장의 길이 50, 임베딩 벡터의 차원 128
    sample_pos_encoding = PositionalEncoding(50, 128)

    plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 128))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()
