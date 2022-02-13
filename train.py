import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json
from nltk_utils import tokenize, stemming, bag_of_words
import numpy as np
from data import ChatDataset
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words, tags, xy = [], [], []

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stemming(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))  # Unique lables

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

BATCH_SIZE = 8
LR = 2e-3
EPOCHS = 1000
INPUT_SIZE = len(X_train[0])
HIDDEN_SIZE = 8
OUTPUT_SIZE = len(tags)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = ChatDataset(X_train=X_train, y_train=y_train)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

model = NeuralNet(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_classes=OUTPUT_SIZE)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

for epoch in range(EPOCHS):
    for (words, lables) in train_loader:
        words = words.to(DEVICE)
        lables = lables.to(DEVICE)
        out = model(words)
        loss = criterion(out, lables)
        optim.zero_grad()
        loss.backward()
        optim.step()

    if (epoch + 1) % 100 == 0:
        print(f'epoch: {epoch + 1}/{EPOCHS}, loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": INPUT_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "output_size": OUTPUT_SIZE,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)
print(f'training complete, model state saved to {FILE}')