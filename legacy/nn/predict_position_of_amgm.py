# Heavily brought from Sean Robertson's pytorch tutorial on seq2seq

from __future__ import unicode_literals, print_function, division
from io import open
import sys
import os

sys.path.insert(0, os.path.abspath('../..'))
from legacy.nn.autoencoder import prepareData
from legacy.utils import ExpressionDataset

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as data_handler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "../data/first_or_second_addition_amgm_2_4.txt"
ENCODER_PATH = "../pt_models/autoencoder_encoder{}.pt".format(device)
MODEL_PATH = "../pt_models/MLPClassfier{}.pt".format(device)
MAX_LENGTH = 100


def get_code_from_sentence(lang, sentence, encoder, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = lang.tensorFromExpression(sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        return encoder_hidden


def build_dataset(lang, encoder_path=ENCODER_PATH, data_path=DATA_PATH):
    encoder = torch.load(encoder_path)
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(data_path, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs and normalize
    data_points = [[s for s in l.split('\t')] for l in lines]
    tensor_datapoints = list()
    for dp in data_points:
        input_tensor = torch.cat((get_code_from_sentence(lang, dp[0], encoder),
                                  get_code_from_sentence(lang, dp[1], encoder)), 2)
        target_tensor = torch.Tensor([int(dp[2])], device=device)
        tensor_datapoints.append((input_tensor, target_tensor))
    training_tensors = tensor_datapoints[:int(len(tensor_datapoints) * 0.9)]
    validation_tensors = tensor_datapoints[int(len(tensor_datapoints) * 0.9):]

    training_set = ExpressionDataset(training_tensors)
    validation_set = ExpressionDataset(validation_tensors)
    return training_set, validation_set


class MLPClassifier(nn.Module):

    def __init__(self, hidden_size):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(2 * hidden_size, 4 * hidden_size)
        self.fc2 = nn.Linear(4 * hidden_size, int(hidden_size / 2))
        self.fc3 = nn.Linear(int(hidden_size / 2), 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(classifier, optimizer, dataset, criterion, epoch):
    sampler = data_handler.sampler.RandomSampler(dataset)
    dloader = data_handler.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                                      drop_last=False)
    for e in range(epoch):
        total_loss = 0.0
        for datapoint in dloader:
            optimizer.zero_grad()
            input_tensor = datapoint[0].view(-1, 2 * HIDDEN_SIZE)
            # print(input_tensor.shape)
            target_tensor = datapoint[1].long().view(-1)
            output_tensor = classifier(input_tensor)
            # print(output_tensor.shape, target_tensor.shape)
            loss = criterion(output_tensor, target_tensor)
            loss.backward()
            optimizer.step()
            running_loss = loss.item()
            total_loss += running_loss
            # print("Running loss: {}".format(running_loss))
        print("Epoch loss: {}".format(total_loss / (int(len(dataset) / BATCH_SIZE) + 1)))


def evaluate(classifier, dataset):
    length = len(dataset)
    accurate_counting = 0
    for dp in dataset:
        output = classifier.encode_entity(dp[0])
        label = dp[1]
        if output[0][0][0] >= output[0][0][1] and label == 0.0:
            accurate_counting += 1
        elif output[0][0][0] < output[0][0][1] and label == 1.0:
            accurate_counting += 1
        print(output, label)
    print("Validation accuracy: {}".format(accurate_counting / length))


SOS_token = 0
EOS_token = 1
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.05
HIDDEN_SIZE = 64
EPOCH_SIZE = 100
BATCH_SIZE = 32

if __name__ == "__main__":
    lang, pairs = prepareData('Math')
    training_set, validation_set = build_dataset(lang)
    if os.path.isfile(MODEL_PATH) and False:
        mlp_classifier = torch.load(MODEL_PATH)
    else:
        mlp_classifier = MLPClassifier(hidden_size=HIDDEN_SIZE)
        optimizer = optim.Adam(mlp_classifier.parameters(), lr=LEARNING_RATE,
                               weight_decay=WEIGHT_DECAY, amsgrad=True)
        criterion = nn.CrossEntropyLoss()
        train(classifier=mlp_classifier, optimizer=optimizer, dataset=training_set,
              criterion=criterion, epoch=EPOCH_SIZE)
        torch.save(mlp_classifier, MODEL_PATH)
    evaluate(classifier=mlp_classifier, dataset=validation_set)
