from chat.utils import get_device, load_json
import random
import torch
from model import NeuralNet
from konltk_utils import bag_of_words, tokenize
from pyconstantt import constant

device = get_device(True)

constant.FileName = "kointents.json"
intents = load_json(constant.FileName)

FILE = "model2.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Otterlee"

print("Let's chat!")

while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.5:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['response'])}")
    else:
        print(f'{bot_name}: I do not understand...')
