import random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.utils.data as data_handler
import numpy as np


class ReplayBuffer:
    def __init__(self, name, capacity, cuda=True):
        if cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.name = name
        self.capacity = capacity
        self.buffer = list()
        self.dict_keys = ["state", "action", "reward", "next_state", "done"]

    def cache(self, new_transitions):
        self.buffer.extend(new_transitions)
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity:]

    def push(self, state, action, reward, next_state, done):
        self.buffer.append({"state": state, "action": action, "reward": reward,
                            "next_state": next_state, "done": done})
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity:]

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        batch_copy = deepcopy(batch)
        batch_dict = {key: list() for key in self.dict_keys}
        for transition in batch_copy:
            for key in self.dict_keys:
                batch_dict[key].append(transition[key])
        return batch_dict

    def __len__(self):
        return len(self.buffer)


class List2Dataset(data_handler.Dataset):
    def __init__(self, d_list):
        self.dataset = d_list

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class HingeLoss(nn.Module):
    r"""Creates a criterion that forces equivalent entities to have close representations and non-equivalent ones to be
    far away from each other
    Math: hinge loss = {
        two_norm(x) * y
    }
    """

    def __init__(self, min_val, max_val):
        super(HingeLoss, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        # Device to operate on
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, difference, label):
        """
        :param difference: capacity(N, *)
        :param label: capacity(N, 1)
        :return: hinge loss of the difference and label
        """
        diff_norm = torch.mul(torch.norm(difference, p=2, dim=1), label)

        clamped_singed_diff_norm = torch.mean(torch.clamp(diff_norm, self.min_val, self.max_val))

        return clamped_singed_diff_norm


def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_config(config_path="../../rl_algos/config.json"):
    # Get arguments from config files
    import json
    configuration = json.load(open(config_path, "r"))
    return configuration


def prepare_batches(replay_buffer, batch_size, drop_last=False):
    """

    :param replay_buffer: replay buffer to use
    :param batch_size: batch capacity
    :param drop_last: whether to drop the last batch
    :return: a list of batches of transitions from the replay buffer
    """
    transition_dataset = List2Dataset(replay_buffer.buffer)
    r_sampler = data_handler.RandomSampler(transition_dataset)

    current_batch = list()
    batch_index_counter = 1
    for index in r_sampler:
        transition = transition_dataset[index]
        current_batch.append(transition)
        batch_index_counter += 1

        if batch_index_counter % batch_size == 1:
            yielded_batch = deepcopy(current_batch)
            current_batch = list()
            batch_index_counter = 1
            yield yielded_batch


def convert_batch_dict_to_dict_batch(batch):
    """
    :param batch: [{"state", "next_state", etc.}]
    :return: {"state": [], "next_state": [], etc.}
    """
    dict_batch = {name: list() for name in ["state", "action", "next_state", "reward", "done"]}
    for datapoint in batch:
        for key, value in datapoint.items():
            dict_batch[key].append(value)
    return dict_batch


def normalize_rewards(reward_vector, normalization_value):
    """
    Normalize a reward: divide by the largest value column-wise
    :param reward_vector: of the shape batch_size * 1
    :param normalization_value: normalization constant
    :return: normalized reward
    """
    reward_normed = reward_vector / normalization_value
    return reward_normed


if __name__ == "__main__":
    criterion = HingeLoss(-10, 10)
    a = torch.rand(10, 10)
    b = torch.ones(10, 1)

    print(criterion(a, b))
