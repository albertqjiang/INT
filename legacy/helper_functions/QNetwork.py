import random
import unittest
from operator import itemgetter

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetworkActionOut(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, cuda=True):
        super(QNetworkActionOut, self).__init__()
        if cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.general_q_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

        self.to(self.device)

    def forward(self, state):
        """
        Give q value of the state-action pair given
        :param state: batch_length * state_dim
        :return:
        """
        return self.general_q_layer(state)

    def find_maximum_q_value(self, state):
        """
        Give maximum value for q for action tensors in the action_tensor_space
        :param state: batch_length * state_dim
        :param action_tensor_space: M * action_dim
        :return: batch_size * 1
        """
        max_q_value, max_q_position = torch.max(self.forward(state), dim=1)
        max_q_value = max_q_value.unsqueeze(1)
        return max_q_value

    def find_maximum_q(self, state, action_space):
        """
        :param state: batch_length * state_dim
        :param action_space: M * action
        :param action_tensor_space: M * action_dim
        :return: maximum q action: batch_length * action_dim, maximum q value: batch_length * 1
        """
        action_q_vector = self.forward(state)
        max_q, max_position = torch.max(action_q_vector, 1)
        max_q = max_q.unsqueeze(-1)
        max_position = max_position.cpu().numpy().tolist()
        max_action = itemgetter(*max_position)(action_space)
        return max_action, max_q

    def sample_action(self, state, action_space, action_tensor_space, temperature=1.):
        action_q_vector = self.forward(state)
        action_q_prob = F.softmax((action_q_vector - torch.max(action_q_vector)) * temperature,
                                  dim=-1).detach().cpu().numpy().tolist()
        action_index = list(range(action_tensor_space.size(0)))
        action_index_choices = [random.choices(action_index, weights=action_prob) for action_prob in action_q_prob]

        action_choices = [itemgetter(*index_list)(action_space) for index_list in action_index_choices]
        return action_choices, action_q_vector

    def sample_action_with_logits(self, state, action_space, action_tensor_space, temperature=1.):
        action_q_vector = self.forward(state=state)
        action_q_prob = F.softmax((action_q_vector - torch.max(action_q_vector)) * temperature, dim=-1)
        action_q_prob_list = torch.tensor(action_q_prob).detach().cpu().numpy().tolist()
        action_index = list(range(action_tensor_space.size(0)))
        action_index_choices = [random.choices(action_index, weights=action_prob) for action_prob in action_q_prob_list]

        action_choices = [itemgetter(*index_list)(action_space) for index_list in action_index_choices]
        return action_choices, action_q_prob

    def q_loss(self, state, action, reward, next_state, discount_factor, done,
               criterion=nn.MSELoss()):
        q_values = self.forward(state)
        import pdb;
        pdb.set_trace()
        next_max_q_values = self.find_maximum_q_value(next_state)
        next_max_q_values = (1 - done) * next_max_q_values.detach()
        target = reward + discount_factor * next_max_q_values
        return criterion(q_values, target)


class QNetworkActionIn(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, cuda=True):
        super(QNetworkActionIn, self).__init__()
        if cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.general_q_layer = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.to(self.device)

    def forward(self, *args):
        """
        Give q value of the state-action pair given
        :param state: batch_length * state_dim
        :param action: batch_length * action_dim
        :return:
        """
        state = args[0]
        action = args[1]
        assert state.size(1) == self.state_dim
        assert action.size(1) == self.action_dim
        state_action = torch.cat([state, action], dim=1)
        return self.general_q_layer(state_action)

    def action_space_q(self, state, action_tensor_space):
        """
        Give q value for state-action* pair for any action* in the action_tensor_space
        :param state: batch_length * state_dim
        :param action_tensor_space: M * action_dim
        :return: batch_length * M
        """
        action_q_vector = list()
        batch_length = state.size(0)
        for action in action_tensor_space:
            batch_action = action.repeat(batch_length, 1)
            action_q_vector.append(self.forward(state, batch_action))
        action_q_vector = torch.cat(action_q_vector, dim=1)
        return action_q_vector

    def find_maximum_q_value(self, state, action_tensor_space):
        """
        Give maximum value for q for action tensors in the action_tensor_space
        :param state: batch_length * state_dim
        :param action_tensor_space: M * action_dim
        :return: batch_size * 1
        """
        max_q_value, max_q_position = torch.max(self.action_space_q(state, action_tensor_space), dim=1)
        max_q_value = max_q_value.unsqueeze(1)
        return max_q_value

    def find_maximum_q(self, state, action_space, action_tensor_space):
        """
        :param state: batch_length * state_dim
        :param action_space: M * action
        :param action_tensor_space: M * action_dim
        :return: maximum q action: batch_length * action_dim, maximum q value: batch_length * 1
        """
        action_q_vector = self.action_space_q(state=state, action_tensor_space=action_tensor_space)
        max_q, max_position = torch.max(action_q_vector, 1)
        max_q = max_q.unsqueeze(-1)
        max_position = max_position.cpu().numpy().tolist()
        max_action = itemgetter(*max_position)(action_space)
        return max_action, max_q

    def sample_action(self, state, action_space, action_tensor_space, temperature=1.):
        action_q_vector = self.action_space_q(state=state, action_tensor_space=action_tensor_space)
        action_q_prob = F.softmax((action_q_vector - torch.max(action_q_vector)) * temperature,
                                  dim=-1).detach().cpu().numpy().tolist()
        action_index = list(range(action_tensor_space.size(0)))
        action_index_choices = [random.choices(action_index, weights=action_prob) for action_prob in action_q_prob]

        action_choices = [itemgetter(*index_list)(action_space) for index_list in action_index_choices]
        return action_choices, action_q_vector

    def sample_action_with_logits(self, state, action_space, action_tensor_space, temperature=1.):
        action_q_vector = self.action_space_q(state=state, action_tensor_space=action_tensor_space)
        action_q_prob = F.softmax((action_q_vector - torch.max(action_q_vector)) * temperature, dim=-1)
        action_q_prob_list = torch.tensor(action_q_prob).detach().cpu().numpy().tolist()
        action_index = list(range(action_tensor_space.size(0)))
        action_index_choices = [random.choices(action_index, weights=action_prob) for action_prob in action_q_prob_list]

        action_choices = [itemgetter(*index_list)(action_space) for index_list in action_index_choices]
        return action_choices, action_q_prob

    def q_loss(self, state, action, reward, next_state, action_tensor_space, discount_factor, done,
               criterion=nn.MSELoss()):
        q_values = self.forward(state, action)
        next_max_q_values = self.find_maximum_q_value(next_state, action_tensor_space)
        next_max_q_values = (1 - done) * next_max_q_values.detach()
        target = reward + discount_factor * next_max_q_values
        return criterion(q_values, target)
