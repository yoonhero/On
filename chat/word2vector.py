from gensim.models import Word2Vec
from konlpy.tag import Komoran
import time

# Naver movie review


def read_review_data(filename):
    with open(filename, 'r') as f:
        data = [line.split("\t") for line in f.read().splitlines()]
        data = data[1:]
    return data


start = time.time()

# reading review file
review_data = read_review_data("./ratings.txt")
print(len(review_data))
print(f"T {time.time() - start}")


# executing nouns
komoran = Komoran()
docs = [komoran.nouns(sentence[1] for sentence in review_data)]
print(f"T {time.time() - start}")


# Word2Vec Model
model =
