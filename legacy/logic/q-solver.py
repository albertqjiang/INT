from logic.utils import *

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_handler


class QNetwork(nn.Module):
    def __init__(self, prover):
        super(QNetwork, self).__init__()
        input_size = (prover.ent_maxsize + prover.gt_maxsize + prover.lemma_maxsize +
                      prover.obj_maxsize + 1 + prover.lemma_operand_maxsize, prover.lemma_embedding_size)
        concatenated_size = input_size[0] * input_size[1]
        self.fc1 = nn.Linear(concatenated_size, int(concatenated_size / 2))
        self.fc2 = nn.Linear(int(concatenated_size / 2), int(concatenated_size / 4))
        self.fc3 = nn.Linear(int(concatenated_size / 4), 1)

    def forward(self, obs_tensor, act_tensor):
        x = torch.cat((obs_tensor, act_tensor), dim=1).view(obs_tensor.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TransitionDataset(data_handler.Dataset):
    def __init__(self, replay_buffer, prover):
        self.prover = prover
        self.replay_buffer = replay_buffer

    def __getitem__(self, index):
        transition = self.replay_buffer.buffer[index]
        return {"st": self.prover.raw_observation2tensor(transition[0]).squeeze(),
                "action": self.prover.action2tensor(transition[1]).squeeze(),
                "reward": transition[2],
                "st+1": self.prover.raw_observation2tensor(transition[3]).squeeze(),
                "max_q": transition[4]}

    def __len__(self):
        return len(self.replay_buffer.buffer)


def max_action_and_q(prover, q_nn):
    max_action, max_q = None, 0.
    for action in exhaust_actions(prover=prover):
        predicted_q = q_nn(prover.observe(), prover.action2tensor(action))
        if predicted_q > max_q:
            max_action = action
            max_q = predicted_q
    return max_action, max_q


def run_one_episode(prover, q_nn, depth=10, randomized=True):
    prover_copy = deepcopy(prover)
    sequence_in_episode = list()

    if randomized:
        for _ in range(depth):
            action = random.choice(exhaust_actions(prover_copy))
            s, r, done, new_s = prover_copy.step(action)
            _, max_q = max_action_and_q(prover_copy, q_nn)
            sequence_in_episode.append((s, action, r, new_s, max_q))
    return sequence_in_episode


if __name__ == "__main__":
    BATCH_SIZE = 10
    EPISODES = 5
    EPISODE_DEPTH = 3

    lgProver = simple_prover()
    q_nn = QNetwork(lgProver)
    optimizer = optim.Adam(q_nn.parameters())
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer("Q Learning Buffer")
    # action = random.choice(exhaust_actions(lgProver))
    # obs = lgProver.observe()
    # act = lgProver.action2tensor(action)
    # print(q_nn.encode_entity(obs_tensor=obs, act_tensor=act))
    # pprint(max_action_and_q(lgProver, q_nn))

    # Sample 10 episodes of data randomly
    episodes = list()
    for i in range(EPISODES):
        print("Episode: {}".format(i))
        for single_transition in run_one_episode(lgProver, q_nn, depth=EPISODE_DEPTH):
            episodes.append(single_transition)
    replay_buffer.cache(episodes)

    # Preparing dataset
    training_dataset = TransitionDataset(replay_buffer, lgProver)
    data_loader = data_handler.DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # One training step
    for i in data_loader:
        optimizer.zero_grad()
        target = (i["reward"].float() + i["max_q"].float())
        prediction = q_nn(i["st"], i["action"])
        loss = criterion(prediction.squeeze(), target)
        loss.backward()
        optimizer.step()
        break
