import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras 
from keras.layers import Layer, Dense, Dropout, LayerNormalization, Embedding, Lambda
import os

# Encoding Position Infomation to give a positional information
class PositionalEncoding(Layer):
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



# MultiHeadAttention (Scaled Dot-Product Attention * num_heads)
class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model
    
        assert d_model % self.num_heads == 0


        # d_model divide into num_heads
        # Principle of Paper = 64
        self.depth = d_model // self.num_heads

        # WQ, WK, WV에 해당하는 밀집층 정의
        self.query_dense = Dense(units=d_model)
        self.key_dense = Dense(units=d_model)
        self.value_dense = Dense(units=d_model)

        # WO에 해당하는 밀집층 정의
        self.dense = Dense(units=d_model)

    
    # num_heads 개수만큼 q, k, v를 split하는 함수
    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth)
        )
    
        return tf.transpose(inputs, perm=[0, 2, 1, 3])
    

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # 1. WQ, WK, WV에 해당하는 밀집층 지나기
        # q : (batch_size, query의 문장 길이, d_model)
        # k : (batch_size, key의 문장 길이, d_model)
        # v : (batch_size, value의 문장 길이, d_model)
        # 참고) 인코더(k, v)-디코더(q) 어텐션에서는 query 길이와 key, value의 길이는 다를 수 있다.
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # 2. 헤드 나누기
        # q : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        # k : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
        # v : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # 3. Scaled Dot-Product Attention
        # (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        # (batch_size, query의 문장 길이, num_heads, d_model/num_heads)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # 4. Connect Head (Concatenate)
        # (batch_size, query의 문장 길이, d_model)
        concat_attention = tf.reshape(scaled_attention, shape=(batch_size, -1, self.d_model))


        # 5. WO에 해당하는 밀집층 지나기
        # (batch_size, query의 문장 길이, d_model)
        outputs = self.dense(concat_attention)


        return outputs



# Create Padding Mask 
# Integer Sequence => 0 else 1
def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)

    # (batch_size, 1, 1, key의 문장 길이)
    return mask[:, tf.newaxis, tf.newaxis, :]
  


def encoder_layer(dff, d_model, num_heads, dropout, name='encoder_layer'):
    inputs = keras.Input(shape=(None, d_model), name="inputs")

    # Encoder using PaddingMask
    padding_mask = keras.Input(shape=(1, 1, None), name="padding_mask")


    # Multi-Head Attention (First Sub Layer / Self Attention)
    attention = MultiHeadAttention(
        d_model, num_heads, name="attention"
    )({
        'query': inputs, 'key': inputs, 'value': inputs, # Q = K = V
        'mask': padding_mask # Using Padding Mast
    })


    # Dropout + Residual Connection and Layer Normalizaion
    attention = Dropout(rate=dropout)(attention)
    attention = LayerNormalization(epsilon=1e-6)(inputs+attention)


    # Positional Wise FFNN (Second Sublayer)
    outputs = Dense(units=dff, activation="relu")(attention)
    outputs = Dense(units=d_model)(outputs)


    # Dropout + Residual Connection and Layer Normalizaion
    outputs = Dropout(rate=dropout)(outputs)
    outputs = LayerNormalization(epsilon=1e-6)(attention + outputs)

    return keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


def encoder(vocab_size, num_layers, dff, d_model, num_heads, dropout, name="encoder"):
    inputs = keras.Input(shape=(None,), name="inputs")

    # Encoder using Padding Mask
    padding_mask = keras.Input(shape=(1, 1, None), name="padding_mask")

    # Positional Encoding + Dropout
    embeddings = Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = Dropout(rate=dropout)(embeddings)

    # Encoder Stack as many as num_layers
    for i in range(num_layers):
        outputs = encoder_layer(dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout, name="encoder_layer_{}".format(i))([outputs, padding_mask])

    return keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)




# Decoder's First Sublayer Masking Future Token
def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x) # Include Padding mask
    return tf.maximum(look_ahead_mask, padding_mask)




def decoder_layer(dff, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = keras.Input(shape=(None, d_model), name="encoder_outputs")

    # Look Ahead Mask (First Layer)
    look_ahead_mask = keras.Input(
        shape=(1, None, None), name="look_ahead_mask"
    )

    # Padding Mask (Second Layer)
    padding_mask = keras.Input(shape=(1, 1, None), name="padding_mask")


    # Multi-Head Attention (First Sublayer / Masked Self Attention)
    attention1 = MultiHeadAttention(
        d_model, num_heads, name="attention1"
    )(inputs={
        'query': inputs, 'key': inputs, 'value': inputs, # Q = K = V
        'mask': look_ahead_mask # Look ahead mask
    })


    # Residual Connection and Normalizaion
    attention1 = LayerNormalization(epsilon=1e-6)(attention1+inputs)


    # Multi-Head Attention (Second Sublayer / Decoder-Encoder Attention)
    attention2 = MultiHeadAttention(
        d_model, num_heads, name="attention2"
    )(
        inputs={
            'query': attention1, 'key': enc_outputs, 'value': enc_outputs, # Q != K = V
            'mask': padding_mask # Padding mask
        }
    )


    # Dropout + Residual Connection and Layer Normalizaion
    attention2 = Dropout(rate=dropout)(attention2)
    attention2 = LayerNormalization(epsilon=1e-6)(attention2+attention1)


    # Positional Wise FFNN (Third Sublayer)
    outputs = Dense(units=dff, activation="relu")(attention2)
    outputs = Dense(units=d_model)(outputs)

    # Droptout + Residual Connection and LayerNormalization
    outputs = Dropout(rate=dropout)(outputs)
    outputs = LayerNormalization(epsilon=1e-6)(outputs+attention2)

    return keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name
    )



def decoder(vocab_size, num_layers, dff, d_model, num_heads, dropout, name="decoder"):
    inputs = keras.Input(shape=(None, ), name="inputs")
    enc_outputs = keras.Input(shape=(None, d_model), name="encoder_outputs")

    # Decoder Using Both Look-Ahead Mask and Padding Mask 
    look_ahead_mask = keras.Input(shape=(1, None, None), name="look_ahead_mask")
    padding_mask = keras.Input(shape=(1, 1, None), name="padding_mask")


    # Positional Encoding + Dropout
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = Dropout(rate=dropout)(embeddings)


    # Stak Decoder as many as num_layers
    for i in range(num_layers):
        outputs = decoder_layer(dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout, name="decoder_layer_{}".format(i))(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name
    )



def transformer(vocab_size, num_layers, dff, d_model, num_heads, dropout, name="transformer"):
    # Encoder's Input
    inputs = keras.Input(shape=(None, ), name="inputs")


    # Decoder's Input
    dec_inputs = keras.Input(shape=(None, ), name="dec_inputs")


    # Encoder's Padding mask
    enc_padding_mask = Lambda(create_look_ahead_mask, output_shape=(1, None, None), name="enc_padding_mask")(inputs)

    # Decoder's Look-ahead Mask
    look_ahead_mask = Lambda(create_look_ahead_mask, output_shape=(1, None, None), name="look_ahead_mask")(dec_inputs)

    # Decoder's Padding Mask
    dec_padding_mask = Lambda(create_padding_mask, output_shape=(1, 1, None), name="dec_padding_mask")(inputs)

    
    # Encoder's Output enc_outputs. Transfer to Decoder
    enc_outputs = encoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout)(inputs=[inputs, enc_padding_mask])

    # Decoder's Output dec_outputs. Transfer to Final Layer
    dec_outputs = decoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout)(inputs=[inputs, enc_outputs, look_ahead_mask, dec_padding_mask])


    # Final Layer to Predict Next Word
    outputs = Dense(units=vocab_size, name="outputs")(dec_outputs)

    return keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)



# Using CrossEntropy Function as Loss Function Becuase of multi-class classification problem
def loss_function(y_true, y_pred, MAX_LENGTH):
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))

  loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')(y_true, y_pred)

  mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
  loss = tf.multiply(loss, mask)

  return tf.reduce_mean(loss)



# Calculate Learning Rate
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
  
  def get_config(self):
    config = {
      'd_model': self.d_model,
      'warmup_steps': self.warmup_steps,
    }
    
    return config



if __name__ == "__main__":
    os.putenv('TF_GPU_ALLOCATOR', 'cuda_malloc_async')
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

    print(create_padding_mask(tf.constant([[1, 21, 777, 0, 0]])))

    print(create_look_ahead_mask(tf.constant([[1, 2, 0, 4, 5]])))


    big_transformer = transformer(
    vocab_size = 100000,
    num_layers = 4,
    dff = 1024,
    d_model = 512,
    num_heads = 8,
    dropout = 0.1,
    name="big_transformer")

    tf.keras.utils.plot_model(
    big_transformer, to_file='big_transformer.png', show_shapes=True)

    big_transformer.summary()


    sample_learning_rate = CustomSchedule(d_model=128)

    plt.plot(sample_learning_rate(tf.range(200000, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.show()