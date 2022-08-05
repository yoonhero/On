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
    def __init__(self, MAX_LENGTH:int=50):
        self.tokenizer = None

        self.START_TOKEN = None
        self.END_TOKEN = None
        self.VOCAB_SIZE = None
        
        self.loading_file = False

        self.MAX_LENGTH = MAX_LENGTH
        
        self.tokenized_inputs = []
        self.tokenized_outputs = []

    def _init_parameters(self):
        self.START_TOKEN = [self.tokenizer.vocab_size]
        self.END_TOKEN = [self.tokenizer.vocab_size + 1]
        self.VOCAB_SIZE =self.tokenizer.vocab_size + 2 


    def create_tokenizer(self, inputs:list[str], outputs:list[str], target_vocab_size:int=2**13):
        self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            inputs+outputs, target_vocab_size=target_vocab_size
        )
        self._init_parameters()

    def load_tokenizer(self, filename:str):
        self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(filename)

        self._init_parameters()

        self.loading_file = True

        return self.tokenizer
    
    def save_tokenizer(self, filename:str):
        self.tokenizer.save_to_file(filename)

    def tokenize_and_filter(self, inputs:list[str], outputs:list[str]):
        self.tokenized_inputs, self.tokenized_outputs = [], []

        for (input, output) in zip(inputs, outputs):
            input = self.START_TOKEN + self.tokenizer.encode(input) + self.END_TOKEN
            output = self.START_TOKEN + self.tokenizer.encode(output) + self.END_TOKEN

            self.tokenized_inputs.append(input)
            self.tokenized_outputs.append(output)
        
        # Padding
        self.tokenized_inputs = pad_sequences(self.tokenized_inputs, maxlen=self.MAX_LENGTH, padding="post")
        self.tokenized_outputs = pad_sequences(self.tokenized_outputs, maxlen=self.MAX_LENGTH, padding="post")

        return self.tokenized_inputs, self.tokenized_outputs


    # 텐서플로우 dataset을 이용하여 셔플(shuffle)을 수행하되, 배치 크기로 데이터를 묶는다.
    # 또한 이 과정에서 교사 강요(teacher forcing)을 사용하기 위해서 디코더의 입력과 실제값 시퀀스를 구성한다.
    def make_dataset(self, batch_size, buffer_size):
        # Decoder real sequence has to remove <SOS> toke
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'inputs': self.tokenized_inputs,
                'dec_inputs': self.tokenized_outputs[:, :-1], # decoder input. Last Padding Token removed
            },
            {
                'outputs': self.tokenized_outputs[:, 1:] # First Token removed. <sos> token gone
            }
        ))

        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset
        

    def tokens(self):
        return self.VOCAB_SIZE, self.START_TOKEN, self.END_TOKEN 



