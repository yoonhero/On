from cgi import test
from tkinter.tix import MAX
import pandas as pd
import tensorflow as tf
from keras import preprocessing
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPool1D, concatenate


train_file = "./ChatbotData.csv"
data = pd.read_csv(train_file, delimiter=",")
features = data['Q'].tolist()
labels = data['label'].tolist()

# word index sequence vector
corpus = [preprocessing.text.text_to_word_sequence(text) for text in features]

tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)
word_index = tokenizer.word_index

MAX_SEQ_LEN = 15
padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding="post")


# train, validation, test dataset create 7:2:1
ds = tf.data.Dataset.from_tensor_slices((padded_seqs, labels))
ds = ds.shuffle(len(features))

train_size = int(len(padded_seqs) * 0.7)
val_size = int(len(padded_seqs) * 0.2)
test_size = int(len(padded_seqs) * 0.1)

train_ds = ds.take(train_size).batch(20)
val_ds = ds.skip(train_size).take(val_size).batch(20)
test_ds = ds.skip(train_size+val_size).take(test_size).batch(20)



# HyperParameters
dropout_prob = 0.5
EMB_SIZE = 128
EPOCH = 5
VOCAB_SIZE = len(word_index) + 1


# CNN MODEL
input_layer = Input(shape=(MAX_SEQ_LEN,))
embedding_layer = Embedding(VOCAB_SIZE, EMB_SIZE, input_length=MAX_SEQ_LEN)(input_layer)
dropout_emb = Dropout(rate=dropout_prob)(embedding_layer)


conv1 = Conv1D(filters=128, kernel_size=3, padding="valid", activation=tf.nn.relu)(dropout_emb)
pool1 = GlobalMaxPool1D()(conv1)

conv2 = Conv1D(filters=128, kernel_size=4, padding="valid", activation=tf.nn.relu)(dropout_emb)
pool2 = GlobalMaxPool1D()(conv2)

conv3 = Conv1D(filters=128, kernel_size=5, padding="valid", activation=tf.nn.relu)(dropout_emb)
pool3 = GlobalMaxPool1D()(conv3)

# 3, 4, 5-gram 이후 합치기
concat = concatenate([pool1, pool2, pool3])

hidden = Dense(128, activation=tf.nn.relu)(concat)
dropout_hidden = Dropout(rate=dropout_prob)(hidden)
logits = Dense(3, name="logits")(dropout_hidden)
predictions = Dense(3, activation=tf.nn.softmax)(logits)


# CREATE MODEL
model = Model(inputs=input_layer, outputs=predictions)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])


# model training
model.fit(train_ds, validation_data=val_ds, epochs=EPOCH, verbose=1)

# model evaluation
loss, accuracy = model.evaluate(test_ds, verbose=1)
print("Accuracy: " + str(accuracy * 100))
print("loss: %f" % (loss))


# save the model
model.save('cnn_sentiment.h5')
