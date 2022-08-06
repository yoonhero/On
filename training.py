import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.preprocessing.sequence import pad_sequences
from nlp_utils import preprocess_sentence, TextTokenizing
from transformer import transformer, CustomSchedule, loss_function
from utils import make_checkpoint, accuracy, load_csv_and_processing
from hyperparameters import NUM_LAYERS, D_MODEL, NUM_HEADS, DFF, DROPOUT, MAX_LENGTH
import os


questions, answers = load_csv_and_processing("./small_dataset.csv")

questions = questions[15000:]
answers = answers[15000:]


textTokenizing = TextTokenizing()
# tokenizer = textTokenizing.create_tokenizer(questions, answers, target_vocab_size=2**15)
# textTokenizing.save_tokenizer("super_super_small_vocab")
textTokenizing.load_tokenizer("super_super_small_vocab")

VOCAB_SIZE, START_TOKEN, END_TOKEN = textTokenizing.tokens()

VOCAB_SIZE, START_TOKEN, END_TOKEN


questions, answers = textTokenizing.tokenize_and_filter(questions, answers)

print(f'질문 데이터의 크기:{questions.shape}')
print(f'답변 데이터의 크기:{answers.shape}')



BATCH_SIZE = 64
BUFFER_SIZE = 20000

dataset = textTokenizing.make_dataset(BATCH_SIZE, BUFFER_SIZE)


model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    dff=DFF,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)


model.summary()



cp_callback = make_checkpoint("training_super_small_2/cp-{epoch:04d}.ckpt")


learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])


EPOCHS = 40
model.fit(dataset, epochs=EPOCHS, callbacks=[cp_callback])