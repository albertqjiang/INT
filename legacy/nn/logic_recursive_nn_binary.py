import json
import time
import sys
import os

sys.path.insert(0, os.path.abspath('../..'))

from operator import itemgetter

from logic.utils import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_handler

set_random_seed(3)


class EntityPairDataset(data_handler.Dataset):
    def __init__(self, entity_pair_list):
        self.entity_pair_list = entity_pair_list

    def __len__(self):
        return len(self.entity_pair_list)

    def __getitem__(self, ind):
        return self.entity_pair_list[ind]


class LogicRecursiveNN(nn.Module):
    def __init__(self, theorems, numerical_functions, input_entities, entity_embedding_size, theorem_embedding_size):
        super(LogicRecursiveNN, self).__init__()
        # Taking class arguments
        self.theorems = theorems
        self.numerical_functions = numerical_functions
        self.input_entities = input_entities
        self.entity_embedding_size = entity_embedding_size
        self.theorem_embedding_size = theorem_embedding_size

        # Define activations
        self.activation = nn.ReLU()
        self.entity_activation = nn.Tanh()
        self.final_activation = nn.Sigmoid()

        # Define embeddings
        self.theorem_embedding = nn.Embedding(num_embeddings=len(theorems), embedding_dim=theorem_embedding_size)
        self.entity_embedding = nn.Embedding(num_embeddings=len(input_entities), embedding_dim=entity_embedding_size)

        # Memorize tensors corresponding to entity strings
        self.entity_string2index = {ie.to_string(): torch.LongTensor([ind])
                                    for ind, ie in enumerate(input_entities)}
        self.theorem_string2index = {theorem.name: torch.LongTensor([ind])
                                     for ind, theorem in enumerate(theorems)}

        # Other trainable layers
        self.two2one_layers = nn.ModuleDict({
            nf.name: nn.Sequential(
                nn.Linear(2 * entity_embedding_size, 4 * entity_embedding_size),
                self.activation,
                nn.Linear(4 * entity_embedding_size, entity_embedding_size),
                self.activation
            )
            for nf in self.numerical_functions if nf.input_no == 2
        })
        self.one2one_layers = nn.ModuleDict({
            nf.name: nn.Sequential(
                nn.Linear(entity_embedding_size, 4 * entity_embedding_size),
                self.activation,
                nn.Linear(4 * entity_embedding_size, entity_embedding_size),
                self.entity_activation
            )
            for nf in numerical_functions if nf.input_no == 1
        })
        self.theorem_layers = nn.ModuleDict({
            theorem.name: nn.Sequential(
                nn.Linear(theorem_embedding_size + theorem.input_no * entity_embedding_size, entity_embedding_size),
                self.activation,
                nn.Linear(entity_embedding_size, int(entity_embedding_size / 2)),
                self.activation,
                nn.Linear(int(entity_embedding_size / 2), int(entity_embedding_size / 4)),
                self.activation,
                nn.Linear(int(entity_embedding_size / 4), 1),
                self.final_activation
            )
            for theorem in theorems
        })

    def encode_entity(self, entity):
        if not entity.recent_numerical_function:
            return self.entity_embedding(self.entity_string2index[entity.name])
        else:
            if entity.recent_numerical_function.name in self.two2one_layers:
                operand_tensors = [self.encode_entity(op_ent) for op_ent in entity.rnc_operands]
                concatenated = torch.cat(operand_tensors, dim=1)
                entity_tensor = self.two2one_layers[entity.recent_numerical_function.name](concatenated)
                self.entity_string2index[entity.name] = entity_tensor
                return entity_tensor
            elif entity.recent_numerical_function.name in self.one2one_layers:
                operand_tensor = self.encode_entity(entity.rnc_operands[0])
                entity_tensor = self.one2one_layers[entity.recent_numerical_function.name](operand_tensor)
                self.entity_string2index[entity.name] = entity_tensor
                return entity_tensor

    def forward(self, theorem, entities):
        final_tensor = self.theorem_layers[theorem.name](
            torch.cat(
                [self.theorem_embedding(self.theorem_string2index[theorem.name])] +
                [self.encode_entity(entity) for entity in entities],
                dim=1
            )
        )
        # print(final_tensor)
        return final_tensor


def train(lrNN, entity_pairs, optimizer, losses, batch_size, print_every, counter):
    optimizer.zero_grad()
    # pprint([(entity0.name, entity1.name, target) for entity0, entity1, target in entity_pairs])
    outputs = [(lrNN(theorem=pos_is_non_neg, entities=[entity0, entity1]), target)
               for entity0, entity1, target in entity_pairs]
    output_tensor = torch.cat([output[0] for output in outputs], dim=0)
    target_tensor = torch.FloatTensor([output[1] for output in outputs]).view(-1, 1)

    # print(output_tensor[0], target_tensor[0])
    loss = criterion(output_tensor, target_tensor)
    loss.backward(retain_graph=True)
    optimizer.step()
    if counter % print_every == 0:
        losses.append(loss.item())
    print(loss.item())


if __name__ == "__main__":
    # Parameters
    ENTITY_DATA_PATH = "../data/balanced_entities.pkl"

    # NN parameters
    EPOCHS = 3000
    HIDDEN_SIZE = 128
    THEOREM_HIDDEN_SIZE = 8
    LEARNING_RATE = 1e-6
    MOMENTUM = 0.9
    BATCH_SIZE = 128
    WEIGHT_DECAY = 0
    PRINT_EVERY = 100
    TRAINING_TEST_SPLIT = 0.8

    # Numerical functions
    add = NumericalFunction("add", 2)
    sub = NumericalFunction("sub", 2)
    mul = NumericalFunction("mul", 2)
    sqr = NumericalFunction("sqr", 1)
    inv = NumericalFunction("inv", 1)
    numeric_collection = [add, sub, mul, sqr, inv]

    # Logic functions
    r = LogicFunction("Real")
    nonNeg = LogicFunction("NonNegative")
    pos = LogicFunction("Positive")
    equals = LogicFunction("Equals")

    # Theorems
    real_subtraction_closed = Theorem(name="real_subtraction_closed", input_no=3,
                                      input_constraints=[(sub, (0, 1), (2,))], assumptions=[(r, (0,)), (r, (1,))],
                                      conclusions=[(r, (2,))])
    real_sqr_non_neg = Theorem(name="real_sqr_non_neg", input_no=2, input_constraints=[(sqr, (0,), (1,))],
                               assumptions=[(r, (0,))], conclusions=[(nonNeg, (1,))])

    # Only for testing, remove afterwards
    pos_is_non_neg = Theorem(name="pos_is_non_neg", input_no=2, input_constraints=[(equals, (0,), (1,))],
                             assumptions=[(nonNeg, (0,))], conclusions=[(pos, (1,))])

    # Entities
    x = Entity("input0")
    y = Entity("input1")
    x_and_y = add.execute_nf([x, y])
    x_inv = inv.execute_nf([x])
    y_sqr = sqr.execute_nf([y])
    x_inv_inv = inv.execute_nf([x_inv])
    combined = mul.execute_nf([x_inv, y_sqr])

    # Numerical evaluator
    ne = NumericalEvaluator()

    # Neural net
    lrNN = LogicRecursiveNN(theorems=[pos_is_non_neg],
                            numerical_functions=numeric_collection, input_entities=[x, y],
                            entity_embedding_size=HIDDEN_SIZE, theorem_embedding_size=THEOREM_HIDDEN_SIZE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(lrNN.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Get dataset
    entity_pair_list = pickle.load(open(ENTITY_DATA_PATH, "rb"))
    dataset_size = len(entity_pair_list)
    training_entity_pairs = entity_pair_list[:int(TRAINING_TEST_SPLIT * dataset_size)]
    test_entity_pairs = entity_pair_list[int(TRAINING_TEST_SPLIT * dataset_size):]

    entity_pair_dataset = EntityPairDataset(entity_pair_list)
    random_sampler = data_handler.RandomSampler(entity_pair_dataset)
    random_batch_sampler = data_handler.BatchSampler(random_sampler, batch_size=BATCH_SIZE, drop_last=False)

    losses = list()
    accuracies = list()
    validation_accuracies = list()
    start_time = time.time()
    epoch = 1

    if os.path.isfile("../pt_models/logic_recur_nn/checkpoint.pt"):
        state = torch.load("../pt_models/logic_recur_nn/checkpoint.pt")
        epoch = state['epoch']
        lrNN = state['model']
        optimizer = state['optimizer']
        validation_accuracies = state['validation_accuracies']
        losses = state['losses']
        print(epoch)

    while epoch < EPOCHS:
        print(epoch)
        try:
            for batch_indices in random_batch_sampler:
                entity_pairs = itemgetter(*batch_indices)(entity_pair_dataset)
                train(lrNN=lrNN, entity_pairs=entity_pairs, optimizer=optimizer, losses=losses,
                      batch_size=BATCH_SIZE, print_every=PRINT_EVERY, counter=counter)
            epoch += 1
            state = {
                'epoch': epoch, 'model': lrNN, 'optimizer': optimizer,
                'validation_accuracies': validation_accuracies, 'losses': losses
            }
            torch.save(state, "../pt_models/logic_recur_nn/checkpoint.pt")
        except RuntimeError:
            state = torch.load("../pt_models/logic_recur_nn/checkpoint.pt")
            epoch = state['epoch']
            lrNN = state['model']
            optimizer = state['optimizer']
        validation_accuracies.append(entity_pair_evaluate_model(lrNN, pos_is_non_neg, test_entity_pairs))

    torch.save(lrNN, "../pt_models/logic_recur_nn/logic_recursive_nn.pt")
    json.dump(losses, open('../pt_models/logic_recur_nn/logic_recursive_loss.json', 'w'))
    json.dump(validation_accuracies, open('../pt_models/logic_recur_nn/logic_recursive_validation_accuracy.json', 'w'))
