import pickle
import sys
import random
from random import shuffle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_handler
import torch.nn.functional as F

from legacy.pseq.graph import *
from legacy.pseq.element_categories import *
from legacy.utils import ComputationGraphDataset

random.seed(1)
torch.manual_seed(1)


class RecursiveNN(nn.Module):
    def __init__(self, input_size, embed_size, one_input_ops=one_input_ops, two_input_ops=two_input_ops):
        """

        :param input_size: The input capacity needed, shall be larger than or equal to the graph input capacity
        :param embed_size: A design choice: the dimensionality of input_entity_embedding
        :param one_input_ops: A list of operations with single inputs
        :param two_input_ops: A list of operations with binary inputs
        """
        super(RecursiveNN, self).__init__()
        self.input_size = input_size
        self.embedding = nn.Embedding(num_embeddings=self.input_size, embedding_dim=embed_size)
        self.one_input_ops = one_input_ops
        self.two_input_ops = two_input_ops
        self.oio_linears = nn.ModuleList([nn.Linear(embed_size, embed_size) for _ in range(len(one_input_ops))])
        self.tio_linears = nn.ModuleList([nn.Linear(2 * embed_size, embed_size) for _ in range(len(two_input_ops))])
        self.operation_counter_final_layer = nn.Linear(embed_size, 1)
        self.activation = F.relu

    def forward(self, operation):
        iounits = operation.inputs

        if operation.operation_type in self.one_input_ops and len(iounits) == 1:
            h = self.oio_linears[self.one_input_ops.index(operation.operation_type)](iounits[0].tensor)

        elif operation.operation_type in self.two_input_ops and len(iounits) == 2:
            x = torch.cat((iounits[0].tensor, iounits[1].tensor), 1)
            h = self.tio_linears[self.two_input_ops.index(operation.operation_type)](x)

        else:
            raise InputError("The length of the symbols is {}. The operation used is {}"
                             .format(len(iounits), operation.operation_type),
                             "These are not valid or coherent.")

        operation.get_output()[0].tensor = self.activation(h)
        return operation.get_output()[0].tensor

    def traverse(self, graph):
        if ONE_HOT is False:
            # Initialize input embeddings
            for i in range(len(graph.inputs)):
                graph.inputs[i].tensor = self.embedding(torch.tensor([i]))

            for op in graph.operations:
                self.forward(op)
        else:
            for i in range(len(graph.inputs)):
                tensor_i = np.zeros(shape=(1, EMBEDDING_SIZE))
                tensor_i[0][i] = 1
                graph.inputs[i].tensor = torch.FloatTensor(tensor_i)

            for op in graph.operations:
                self.forward(op)

        return graph.get_iounits()[-1].tensor

    def operation_counter(self, graph):
        return self.operation_counter_final_layer(self.traverse(graph))


def counter_validation_loss(NN, validation_set, criterion):
    validation_loss = 0.0
    labels = [np.array(len(vgraph.operations)).reshape((1, 1)) for vgraph in validation_set]
    for i in range(len(validation_set)):
        outputs = NN.operation_counter(validation_set[i])
        loss = criterion(outputs, torch.FloatTensor(labels[i]))
        validation_loss += loss.item()
    return validation_loss / len(validation_set)


def counter_validation_accuracy(NN, validation_set):
    outputs = []
    labels = [len(vgraph.operations) for vgraph in validation_set]
    for _ in validation_set:
        y = NN.operation_counter(_)
        outputs.append(int(y + (0.5 if y > 0 else -0.5)))
    return len([i for i, j in zip(outputs, labels) if i == j]) / len(outputs)


BATCH_SIZE = 32
EPOCHS = 30
EPOCHS_TO_CHANGE_LEARNING_RATE = 5
WEIGHT_DECAY = 0.05
DATASET_SIZE = 4096
EMBEDDING_SIZE = 10
LEARNING_RATE = 5e-5
ONE_HOT = True
CUDA = False
if len(sys.argv) > 1:
    if sys.argv[1].lower() == "cuda":
        CUDA = True

if __name__ == "__main__":
    # Set up recursive neural network
    rnn = RecursiveNN(input_size=2, embed_size=EMBEDDING_SIZE)
    best_rnn = rnn
    best_accuracy = 0.

    # Get dataset ready
    all_graphs = pickle.load(open("../data/2input4degree.pkl", "rb"))
    shuffle(all_graphs)
    graphs = all_graphs[:int(len(all_graphs) * 0.8)]
    validation_graphs = all_graphs[int(len(all_graphs) * 0.8):]
    dataset = ComputationGraphDataset(graphs)

    # Unbalanced dataset, use WeightedRandomSampler to augument the data
    weights = [1.] * len(graphs)
    for training_sample_idx in range(len(graphs)):
        if len(graphs[training_sample_idx].operations) == 4:
            weights[training_sample_idx] = 1. / 4112
        elif len(graphs[training_sample_idx].operations) == 3:
            weights[training_sample_idx] = 1. / 248
        elif len(graphs[training_sample_idx].operations) == 2:
            weights[training_sample_idx] = 1. / 20
        elif len(graphs[training_sample_idx].operations) == 1:
            weights[training_sample_idx] = 1. / 2
    sampler = data_handler.sampler.WeightedRandomSampler(weights, DATASET_SIZE)
    batch_sampler = data_handler.sampler.BatchSampler(sampler, batch_size=BATCH_SIZE, drop_last=False)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(rnn.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, amsgrad=True)
    labels = [np.array([len(graph.operations)]).reshape((1, 1)) for graph in graphs]

    # A working example: the function is optimized to find how many operations are used in the graph
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch in batch_sampler:
            running_loss = 0.0
            for i in batch:
                optimizer.zero_grad()
                outputs = rnn.operation_counter(graphs[i])
                loss = criterion(outputs, torch.FloatTensor(labels[i]))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            total_loss += running_loss
            print(running_loss / len(batch))
            print("Validation loss: " + str(counter_validation_loss(rnn, validation_graphs, criterion)))

            va = counter_validation_accuracy(rnn, validation_graphs)
            if va > best_accuracy:
                best_rnn = deepcopy(rnn)
                best_accuracy = va
            print("Validation accuracy: " + str(va))
        print("Epoch loss: " + str(total_loss / len(graphs)))

    torch.save(best_rnn.state_dict(), "../data/mytraining.pt")
    print("Best validation accuracy: {}".format(best_accuracy))
