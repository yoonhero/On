from gensim.models import Word2Vec
from konlpy.tag import Komoran
import time


# Naver movie review
def read_review_data(filename):
    with open(filename, 'r') as f:
        data = [line.split("\t") for line in f.read().splitlines()]
        data = data[1:]
    return data


def train_words2vec():
    start = time.time()
    # reading review file
    review_data = read_review_data("./ratings.txt")
    print(len(review_data))
    print(f"T {time.time() - start}")

    # executing nouns
    komoran = Komoran()
    print(review_data[0][1])
    docs = [komoran.nouns(sentence[1]) for sentence in review_data]
    print(f"T {time.time() - start}")

    # Word2Vec Model
    model = Word2Vec(sentences=docs, window=4, hs=1, min_count=2, sg=1)
    print(f"T {time.time() - start}")

    model.save("nvmc.model")
    print(f"corpus_count : {model.corpus_count}")
    print(f"corpus_total_words : {model.corpus_total_words}")


if __name__ == "__main__":
    # train_words2vec()
    # loading model
    model = Word2Vec.load('nvmc.model')
    print(f"corpus_total_words : ", model.corpus_total_words)

    # word '사랑' embedding vector
    print(f"사랑: {model.wv['사랑']}")

    # word similarity
    print("일요일 = 월요일", model.wv.similarity(w1='일요일', w2="월요일"))

    # execute most similar word
    print(model.wv.most_similar("시리즈", topn=5))
