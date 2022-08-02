from konlpy.tag import Hannanum
from konlpy.tag import Okt
import numpy as np

hannanum = Hannanum()
okt = Okt()
# print(hannanum.morphs("사랑햌ㅋㅋ"))


def ignore_words(words):
    ignore_patterns = ["?", "!", ",", "."]
    clean_words = [normalize(w) for w in words if w not in ignore_patterns]
    return clean_words


def normalize(sentence):
    return okt.normalize(sentence)


def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return hannanum.morphs(sentence)


def bag_of_words(tokenized_sentence, all_words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag


if __name__ == "__main__":
    a = "사랑햌ㅋㅋ"
    print(a)
    a = tokenize(a)
    print(a)
    a = normalize(a[0])
    print(a)