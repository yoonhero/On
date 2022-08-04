import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import re
import tensorflow_datasets as tfds
import tensorflow as tf

from nlp_utils import preprocess_sentence


def evaluate(model, sentence, tokenizer, START_TOKEN, END_TOKEN, MAX_LENGTH):
    # 입력 문장에 대한 전처리
    sentence = preprocess_sentence(sentence)

    # 입력 문장에 시작 토큰과 종료 토큰을 추가
    sentence = tf.expand_dims(
        START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

    output = tf.expand_dims(START_TOKEN, 0)

    # 디코더의 예측 시작
    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

        # 현재 시점의 예측 단어를 받아온다.
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # 만약 현재 시점의 예측 단어가 종료 토큰이라면 예측을 중단
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        # 현재 시점의 예측 단어를 output(출력)에 연결한다.
        # output은 for문의 다음 루프에서 디코더의 입력이 된다.
        output = tf.concat([output, predicted_id], axis=-1)

    # 단어 예측이 모두 끝났다면 output을 리턴.
    return tf.squeeze(output, axis=0)


def predict(model, sentence, tokenizer, START_TOKEN, END_TOKEN, MAX_LENGTH):
  prediction = evaluate(model=model, sentence=sentence, tokenizer=tokenizer, START_TOKEN=START_TOKEN, END_TOKEN=END_TOKEN, MAX_LENGTH=MAX_LENGTH)

  # prediction == 디코더가 리턴한 챗봇의 대답에 해당하는 정수 시퀀스
  # tokenizer.decode()를 통해 정수 시퀀스를 문자열로 디코딩.
  predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])

  return predicted_sentence