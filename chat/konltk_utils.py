from konlpy.tag import Hannanum
import numpy as np

hannanum = Hannanum()


def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    sentence = sentence.replace(".", "")
    return hannanum.morphs(sentence)


# def stem(word):
#     """
#     stemming = find the root form of the word
#     examples:
#     words = ["organize", "organizes", "organizing"]
#     words = [stem(w) for w in words]
#     -> ["organ", "organ", "organ"]
#     """
#     pass


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
    a = "롯데마트의 흑마늘 양념 치킨이 논란이 되고 있다."
    print(a)
    a = tokenize(a)

