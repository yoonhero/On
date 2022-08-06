import tensorflow as tf
from keras.models import Model, load_model
from keras import preprocessing
from torch import load

features = ["헤어짐 ㅠ ㅋㅋ"]
corpus = [preprocessing.text.text_to_word_sequence(text) for text in features]
tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)
print(sequences, corpus)
MAX_SEQ_LEN = 15
padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding="post")


model = load_model("cnn_sentiment.h5")
model.summary()

predict = model.predict(padded_seqs[[0]])
predict_class = tf.math.argmax(predict, axis=1)
print(f"감정 예측 클래스:{predict_class.numpy()}")
