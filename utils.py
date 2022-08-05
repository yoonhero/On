import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import re
import tensorflow_datasets as tfds
import tensorflow as tf
from nlp_utils import preprocess_sentence
import json
from os import times
import torch
import openpyxl
import time
import datetime


# Predict Module 
# use it like this Predict("안녕") 
class use_model():
    def __init__(self, model, tokenizer, START_TOKEN:list[int], END_TOKEN:list[int], MAX_LENGTH:int):
        self.model = model
        
        self.tokenizer = tokenizer
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN

        self.MAX_LENGTH = MAX_LENGTH


    def _evaluate(self, sentence:str):
        # 입력 문장에 대한 전처리
        sentence = preprocess_sentence(sentence)

        # 입력 문장에 시작 토큰과 종료 토큰을 추가
        sentence = tf.expand_dims(
            self.START_TOKEN + self.tokenizer.encode(sentence) + self.END_TOKEN, axis=0)

        output = tf.expand_dims(self.START_TOKEN, 0)

        # 디코더의 예측 시작
        for i in range(self.MAX_LENGTH):
            predictions = self.model(inputs=[sentence, output], training=False)

            # 현재 시점의 예측 단어를 받아온다.
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            # 만약 현재 시점의 예측 단어가 종료 토큰이라면 예측을 중단
            if tf.equal(predicted_id, self.END_TOKEN[0]):
                break

            # 현재 시점의 예측 단어를 output(출력)에 연결한다.
            # output은 for문의 다음 루프에서 디코더의 입력이 된다.
            output = tf.concat([output, predicted_id], axis=-1)

        # 단어 예측이 모두 끝났다면 output을 리턴.
        return tf.squeeze(output, axis=0)

    def predict(self, sentence:str=None):
        if sentence == None or len(sentence) == 0:
            return 

        print(sentence)

        prediction = self._evaluate(sentence)

        # prediction == 디코더가 리턴한 챗봇의 대답에 해당하는 정수 시퀀스
        # tokenizer.decode()를 통해 정수 시퀀스를 문자열로 디코딩.
        predicted_sentence = self.tokenizer.decode(
            [i for i in prediction if i < self.tokenizer.vocab_size])

        return predicted_sentence



# Load Json File
def load_json(filename: str) -> dict:
    with open(filename, "r") as f:
        return json.load(f)


# Get Device Training Environment
def get_device(isMac: bool = False) -> str:
    if isMac:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Read Excel(.xlsx) File
#         # 셀 주소로 값 출력
#         print(load_ws['B2'].value)
#         # 셀 좌표로 값 출력
#         print(load_ws.cell(3, 2).value)
def load_xlsx(filename: str, sheet_name: str = "Sheet1"):
    wb = openpyxl.load_workbook(filename, data_only=True)
    sheet = wb[sheet_name]

    all_values = [s.value for s in sheet.rows]

    return all_values


# Logging Time Package Class
class TimeLogger:
    def __init__(self, func):
        def logger(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            print(f"Calling {func.__name__}: {time.time() - start:.5f}s")
            return result
        self._logger = logger

    def __call__(self, *args, **kwargs):
        return self._logger(*args, **kwargs)


# Logging Running Function
def LoggingResult(func):
    import logging
    filename = '{}.log'.format(func.__name__)
    logging.basicConfig(handlers=[logging.FileHandler(
        filename, 'a', 'utf-8')], level=logging.INFO)

    def wrapper(*args, **kwargs):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        logging.info(
            '[{}] Running Result args - {}, kwargs - {}'.format(timestamp, args, kwargs))
        return func(*args, **kwargs)

    return wrapper


if __name__ == '__main__':
    @TimeLogger
    def calculate_sum_n_cls(n):
        return sum(range(n))

    calculate_sum_n_cls(200000)



def load_latest_checkpoint(checkpoint_directory: str):
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_directory)

    return latest_checkpoint

    