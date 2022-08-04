import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import re
import tensorflow_datasets as tfds
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

# Space Between word and mark(.,!?)
def preprocess_sentence( sentence):
     # ex) 12시 땡! -> 12시 땡 ! 
    sentence = re.sub(r"([?.!,])", r" \1", sentence)
    sentence = sentence.strip()
    return sentence


class TextTokenizing():
    def __init__(self, inputs, outputs, MAX_LENGTH=50):
        # 단어 집합을 생성
    # return Tokenizer, START_TOKEN, END_TOKEN
        self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            inputs+outputs, target_vocab_size=2**13
        )

        self.START_TOKEN = [self.tokenizer.vocab_size]
        self.END_TOKEN = [self.tokenizer.vocab_size+1]
        self.VOCAB_SIZE = self.tokenizer.vocab_size + 2
        
        self.MAX_LENGTH = MAX_LENGTH

    def tokenize_and_filter(self):
        tokenized_inputs, tokenized_outputs = [], []

        for (input, output) in zip(self.inputs, self.outputs):
            input = self.START_TOKEN + self.tokenizer.encode(input) + self.END_TOKEN
            output = self.START_TOKEN + self.tokenizer.encode(output) + self.END_TOKEN

            tokenized_inputs.append(input)
            tokenized_outputs.append(output)

        
        # Padding
        tokenized_inputs = pad_sequences(tokenized_inputs, maxlen=self.MAX_LENGTH, padding="post")
        tokenized_outputs = pad_sequences(tokenized_outputs, maxlen=self.MAX_LENGTH, padding="post")


        return tokenized_inputs, tokenized_outputs

    def call(self):
        return self.tokenizer, self.START_TOKEN, self.END_TOKEN