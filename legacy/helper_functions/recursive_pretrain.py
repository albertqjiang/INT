import pickle
import json
import time
import sys
import os

sys.path.insert(0, os.path.abspath('../..'))

from operator import itemgetter

from logic.utils import NumericalEvaluator, set_random_seed
from legacy.helper_functions.RecursiveAttentiveEncoder import RecursiveAttentiveEncoder
from legacy.logic.logicRL import LogicBasedProver
from legacy.helper_functions.utils import load_config, HingeLoss

import torch
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


def train(recursive_encoder, entity_pairs, optimizer, criterion):
    output_tensors = list()
    target_tensors = list()

    for entity_pair in entity_pairs:
        print(entity_pair[0].name, entity_pair[1].name)
        output1 = recursive_encoder.encode_entity(entity_pair[0])
        output2 = recursive_encoder.encode_entity(entity_pair[1])
        output = output1 - output2
        target = 1 if entity_pair[2] else -1
        target = torch.FloatTensor([target]).to(recursive_encoder.device)
        output_tensors.append(output)
        target_tensors.append(target)

    output_tensor = torch.cat(output_tensors, dim=0)
    target_tensor = torch.cat(target_tensors, dim=0).unsqueeze(1)
    loss = criterion(output_tensor, target_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


if __name__ == "__main__":
    # Parameters
    ENTITY_DATA_PATH = "../data/balanced_entities.pkl"
    config = load_config()["recur"]
    HINGE_MIN = config["HINGE_MIN"]
    HINGE_MAX = float(config["HINGE_MAX"])

    # NN parameters
    EPOCHS = 5000
    HIDDEN_SIZE = 128
    THEOREM_HIDDEN_SIZE = 8
    LEARNING_RATE = 1e-6
    MOMENTUM = 0.9
    BATCH_SIZE = 32
    WEIGHT_DECAY = 0
    PRINT_EVERY = 100
    TRAINING_TEST_SPLIT = 0.9

    # Prover
    proof = pickle.load(open("../data/simple_proof.p", "rb"))
    prover = LogicBasedProver(
        proof=proof,
        ent_maxsize=config["ent_maxsize"],
        gt_maxsize=config["gt_maxsize"],
        lemma_maxsize=config["lemma_maxsize"],
        lemma_embedding_size=config["lemma_embedding_size"],
        lemma_operand_size=config["lemma_operand_size"],
        objective_maxsize=config["objective_maxsize"]
    )

    # Numerical evaluator
    ne = NumericalEvaluator()

    # Neural net
    criterion = HingeLoss(min_val=HINGE_MIN, max_val=HINGE_MAX)
    recursive_encoder = RecursiveAttentiveEncoder(prover)
    optimizer = optim.Adam(recursive_encoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, amsgrad=True)

    # Get dataset
    entity_pair_list = pickle.load(open(ENTITY_DATA_PATH, "rb"))
    dataset_size = len(entity_pair_list)
    training_entity_pairs = entity_pair_list[:int(TRAINING_TEST_SPLIT * dataset_size)]
    test_entity_pairs = entity_pair_list[int(TRAINING_TEST_SPLIT * dataset_size):]

    entity_pair_dataset = EntityPairDataset(training_entity_pairs)
    random_sampler = data_handler.RandomSampler(entity_pair_dataset)
    random_batch_sampler = data_handler.BatchSampler(random_sampler, batch_size=BATCH_SIZE, drop_last=False)

    losses = list()
    accuracies = list()
    validation_accuracies = list()
    start_time = time.time()
    epoch = 1
    counter = 0

    if os.path.isfile("../pt_models/logic_recur_nn/checkpoint.pt"):
        state = torch.load("../pt_models/logic_recur_nn/checkpoint.pt")
        epoch = state['epoch']
        recursive_encoder = state['model']
        optimizer = state['optimizer']
        losses = state['losses']

    while epoch < EPOCHS:
        print(epoch)
        running_loss = 0.
        running_counter = 0
        for batch_indices in random_batch_sampler:
            entity_pairs = itemgetter(*batch_indices)(entity_pair_dataset)
            running_loss += train(recursive_encoder=recursive_encoder, entity_pairs=entity_pairs,
                                  optimizer=optimizer, criterion=criterion).item() * len(batch_indices)
            running_counter += len(batch_indices)

        print(running_loss / running_counter)
        losses.append(running_loss / running_counter)
        epoch += 1

        state = {'epoch': epoch, 'model': recursive_encoder, 'optimizer': optimizer, 'losses': losses}
        torch.save(state, "../pt_models/logic_recur_nn/checkpoint.pt")
        json.dump(losses, open('../pt_models/logic_recur_nn/logic_recursive_loss.json', 'w'))

    torch.save(recursive_encoder, "../pt_models/logic_recur_nn/logic_recursive_nn.pt")
    json.dump(losses, open('../pt_models/logic_recur_nn/logic_recursive_loss.json', 'w'))
