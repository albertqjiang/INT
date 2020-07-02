import os
import sys

sys.path.insert(0, os.path.abspath('../..'))
from logic.utils import simple_prover, exhaust_actions, set_random_seed, end_to_end_max_q_and_action, \
    ReplayBuffer, what_is_proved
from logic.logic import LogicFunction, NumericalFunction
from legacy.logic.logicRL import REWARD_THEOREM_PROCEEDED, REWARD_PROOF_COMPLETE
from copy import deepcopy
import random
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_handler

ENTITY_EMBEDDING_SIZE = 128
THEOREM_EMBEDDING_SIZE = 128
BATCH_SIZE = 10
EPISODES = 10
EPISODE_DEPTH = 10
EPOCHS = 1
LEARNING_RATE = 1e-4
BUFFER_MAX_SIZE = 1000
ITERATION = 10


class End2EndQNetwork(nn.Module):
    def __init__(self, prover, numerical_functions, logic_functions,
                 entity_embedding_size=ENTITY_EMBEDDING_SIZE, theorem_embedding_size=THEOREM_EMBEDDING_SIZE):
        super(End2EndQNetwork, self).__init__()

        # Device to operate on
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dimensions of the spaces used
        # Entity(E): R^128
        # Theorem(THEOREM): R^8
        # Logic statements(LG): (-1, 1)^128
        # A_TENSORIZED = (-1, 1)^128

        # Getting parameters from prover and other inputs
        self.theorems = prover.proof.lemmas
        self.numerical_functions = numerical_functions
        self.logic_functions = logic_functions
        self.input_entities = prover.proof.entities
        self.entity_embedding_size = entity_embedding_size
        self.theorem_embedding_size = theorem_embedding_size

        # Maximum sizes of properties to fix dimensions
        self.ground_truth_maxsize = prover.gt_maxsize
        self.entity_maxsize = prover.ent_maxsize
        self.theorem_maxsize = prover.lemma_maxsize
        self.theorem_operand_maxsize = prover.lemma_operand_maxsize
        self.objective_maxsize = prover.obj_maxsize

        # Define embeddings
        self.input_entity_embedding = nn.Embedding(num_embeddings=len(self.input_entities),
                                                   embedding_dim=self.entity_embedding_size)
        self.theorem_embedding = nn.Embedding(num_embeddings=len(self.theorems),
                                              embedding_dim=self.theorem_embedding_size)

        # Memorize tensors corresponding to entity strings
        self.input_entity_string2index = {ie.to_string(): torch.LongTensor([ind], device=self.device)
                                          for ind, ie in enumerate(self.input_entities)}
        self.theorem_string2index = {theorem.name: torch.LongTensor([ind], device=self.device)
                                     for ind, theorem in enumerate(self.theorems)}

        # Define activations
        self.activation = nn.ReLU()
        self.entity_activation = nn.Tanh()
        self.final_activation = nn.Sigmoid()

        # Numerical function layers: E^n -> E
        self.nf_two2one_layers = nn.ModuleDict({
            nf.name: nn.Sequential(
                nn.Linear(2 * self.entity_embedding_size, 4 * self.entity_embedding_size),
                nn.ReLU(),
                nn.Linear(4 * self.entity_embedding_size, self.entity_embedding_size),
            )
            for nf in self.numerical_functions if nf.input_no == 2
        })
        self.nf_one2one_layers = nn.ModuleDict({
            nf.name: nn.Sequential(
                nn.Linear(self.entity_embedding_size, 4 * self.entity_embedding_size),
                nn.ReLU(),
                nn.Linear(4 * self.entity_embedding_size, self.entity_embedding_size),
            )
            for nf in self.numerical_functions if nf.input_no == 1
        })

        # Logic function layers: E^n -> LG
        self.lf_two2one_layers = nn.ModuleDict({
            lf.name: nn.Sequential(
                nn.Linear(2 * self.entity_embedding_size, 4 * self.entity_embedding_size),
                nn.ReLU(),
                nn.Linear(4 * self.entity_embedding_size, self.entity_embedding_size),
                nn.Tanh()
            )
            for lf in self.logic_functions if lf.input_no == 2
        })
        self.lf_one2one_layers = nn.ModuleDict({
            lf.name: nn.Sequential(
                nn.Linear(self.entity_embedding_size, 4 * self.entity_embedding_size),
                nn.ReLU(),
                nn.Linear(4 * self.entity_embedding_size, self.entity_embedding_size),
                nn.Tanh()
            )
            for lf in self.logic_functions if lf.input_no == 1
        })

        # Theorem layers: THEOREM \times E^n -> A_TENSORIZED
        self.theorem_layers = nn.ModuleDict({
            theorem.name: nn.Sequential(
                nn.Linear(
                    theorem_embedding_size + theorem.input_no * self.entity_embedding_size,
                    self.entity_embedding_size
                ),
                nn.ReLU(),
                nn.Linear(self.entity_embedding_size, self.entity_embedding_size * 2),
                nn.ReLU(),
                nn.Linear(self.entity_embedding_size * 2, self.entity_embedding_size * 4),
                nn.ReLU(),
                nn.Linear(self.entity_embedding_size * 4, self.entity_embedding_size),
                nn.Tanh()
            )
            for theorem in self.theorems
        })

        # Q function network: S_TENSORIZED \times A_TENSORIZED -> R
        input_size = (
            # no. of state tensors
            prover.ent_maxsize + prover.gt_maxsize + prover.lemma_maxsize + prover.obj_maxsize,
            # no. of action tensors
            1 + prover.lemma_operand_maxsize,
            prover.lemma_embedding_size
        )
        concatenated_size = (input_size[0] + input_size[1]) * input_size[2]
        self.q_function = nn.Sequential(
            nn.Linear(concatenated_size, int(concatenated_size / 2)),
            nn.ReLU(),
            nn.Linear(int(concatenated_size / 2), int(concatenated_size / 4)),
            nn.ReLU(),
            nn.Linear(int(concatenated_size / 4), 1)
        )

    # Low level encodings
    def encode_entity(self, entity):
        if not entity.recent_numerical_function:
            return self.input_entity_embedding(self.input_entity_string2index[entity.name])
        else:
            if entity.recent_numerical_function.name in self.nf_two2one_layers:
                operand_tensors = [self.encode_entity(op_ent) for op_ent in entity.rnc_operands]
                concatenated = torch.cat(operand_tensors, dim=1)
                entity_tensor = self.nf_two2one_layers[entity.recent_numerical_function.name](concatenated)
                return entity_tensor
            elif entity.recent_numerical_function.name in self.nf_one2one_layers:
                operand_tensor = self.encode_entity(entity.rnc_operands[0])
                entity_tensor = self.nf_one2one_layers[entity.recent_numerical_function.name](operand_tensor)
                return entity_tensor

    def encode_logic_statement(self, logic_statement):
        if logic_statement.logic_function.input_no == 2:
            operand_tensors = [self.encode_entity(op_ent) for op_ent in logic_statement.entities]
            concatenated = torch.cat(operand_tensors, dim=1)
            lg_statement_tensor = self.lf_two2one_layers[logic_statement.logic_function.name](concatenated)
            return lg_statement_tensor
        elif logic_statement.logic_function.input_no == 1:
            operand_tensor = self.encode_entity(logic_statement.entities[0])
            lg_statement_tensor = self.lf_one2one_layers[logic_statement.logic_function.name](operand_tensor)
            return lg_statement_tensor
        else:
            raise NotImplementedError

    def encode_theorem(self, theorem):
        return self.theorem_embedding(self.theorem_string2index[theorem.name])

    # High level encodings
    def encode_observation(self, observation):
        """

        :param observation: an observation
        :return: the encoded tensor of the observation given
        """
        gt_tensors = [self.encode_logic_statement(gt) for gt in observation["ground_truth"]]
        gt_tensors += [torch.zeros_like(gt_tensors[0])] * (self.ground_truth_maxsize - len(gt_tensors))
        th_tensors = [self.encode_theorem(th) for th in observation["lemmas"]]
        th_tensors += [torch.zeros_like(th_tensors[0])] * (self.theorem_maxsize - len(th_tensors))
        obj_tensors = [self.encode_logic_statement(obj) for obj in observation["objectives"]]
        obj_tensors += [torch.zeros_like(obj_tensors[0])] * (self.objective_maxsize - len(obj_tensors))
        ent_tensors = [self.encode_entity(ent) for ent in observation["entities"]]
        ent_tensors += [torch.zeros_like(ent_tensors[0])] * (self.entity_maxsize - len(ent_tensors))
        observation_tensor = torch.cat(gt_tensors + th_tensors + obj_tensors + ent_tensors, dim=0)
        return torch.unsqueeze(observation_tensor, 0)

    def encode_action(self, action):
        """

        :param action: an action
        :return: the encoded tensor of the action given
        """
        if action["action_type"] == "theorem":
            theorem, operands = action["action"]
            theorem_tensor = [self.encode_theorem(theorem=theorem)]
            entity_tensors = [self.encode_entity(op) for op in operands]
            entity_tensors += [torch.zeros_like(entity_tensors[0])] * (
                        self.theorem_operand_maxsize - len(entity_tensors))
            action_tensor = torch.cat(theorem_tensor + entity_tensors, dim=0)
            return torch.unsqueeze(action_tensor, 0)
        else:
            raise NotImplementedError

    # Forward: Q: S \times A -> R
    def forward(self, obs, act):
        obs_tensor = self.encode_observation(observation=obs)
        act_tensor = self.encode_action(action=act)
        return self.tensor_forward(obs_tensor=obs_tensor, act_tensor=act_tensor)

    def tensor_forward(self, obs_tensor, act_tensor):
        obs_tensor = torch.squeeze(obs_tensor, 1)
        act_tensor = torch.squeeze(act_tensor, 1)
        x = torch.cat((obs_tensor, act_tensor), dim=1)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.q_function(x)
        return x


class TransitionDataset(data_handler.Dataset):
    def __init__(self, replay_buffer, prover):
        self.prover = prover
        self.replay_buffer = replay_buffer

    def __getitem__(self, index):
        transition = self.replay_buffer.buffer[index]
        return {"pt": transition[0],
                "action": transition[1],
                "reward": transition[2],
                "pt+1": transition[3]}

    def __len__(self):
        return len(self.replay_buffer.buffer)


def sample_one_episode(prover, q_net, initial_states, depth=10, randomized=True, modified_her=False):
    prover_copy = deepcopy(prover)
    transitions_in_episode = list()
    all_actions = exhaust_actions(prover_copy)
    objectives = list()

    for _ in range(depth):
        if randomized:
            action = random.choice(all_actions)
        else:
            _, action = end_to_end_max_q_and_action(prover_copy, q_net)
            # Sometimes all actions return negative Q values and none actions, in which case randomize
            if not action:
                action = random.choice(all_actions)
        original_prover, r, done, new_prover = prover_copy.step(action)
        transitions_in_episode.append((original_prover, action, r, new_prover))
        if r > 0:
            new_prover_string = "\n".join([ls.name for ls in new_prover.raw_observe()['ground_truth']])
            if new_prover_string not in initial_states:
                initial_states[new_prover_string] = deepcopy(new_prover)

    if modified_her:
        for transition in transitions_in_episode:
            if transition[2] > 0:  # Theorem proceeded
                objectives.append(what_is_proved(observation=prover.raw_observe(),
                                                 obj_observation=transition[3].raw_observe()))
        for transition in transitions_in_episode:
            if transition[2] > 0:
                for obj in objectives:
                    new_transition = [transition_item for transition_item in transition]
                    new_transition[0].proof.objectives = obj
                    new_transition[3].proof.objectives = obj
                    if set([ls.name for ls in obj]) < \
                            set([ls.name for ls in new_transition[3].raw_observe()['ground_truth']]):
                        new_transition[2] = REWARD_PROOF_COMPLETE
                    else:
                        new_transition[2] = REWARD_THEOREM_PROCEEDED
                    transitions_in_episode.append(tuple(new_transition))
            else:
                continue
    return transitions_in_episode


def sample_episodes_and_cache(q_net, replay_buffer, initial_states, depth=10, no_of_episodes=EPISODES,
                              randomized=True, modified_her=False):
    transitions = list()
    for i in range(no_of_episodes):
        print("Episode {}:".format(i))
        prover = random.choice(list(initial_states.values()))
        for transition in sample_one_episode(prover=prover, q_net=q_net, depth=depth, initial_states=initial_states,
                                             randomized=randomized, modified_her=modified_her):
            transitions.append(transition)
    replay_buffer.cache(transitions)


def train(dataset, optimizer, criterion, q_net, losses):
    running_loss = 0.
    # One epoch through the data loader
    for transition in dataset:
        optimizer.zero_grad()
        max_q, _ = end_to_end_max_q_and_action(transition["pt+1"], q_net)
        target = torch.FloatTensor([transition["reward"] + max_q])
        prediction = q_net.forward(obs=transition["pt"].raw_observe(), act=transition["action"])
        loss = criterion(prediction.squeeze(), target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    losses.append(running_loss)
    print(running_loss)


if __name__ == "__main__":
    set_random_seed(1)

    # Numerical functions
    add = NumericalFunction("add", 2)
    sub = NumericalFunction("sub", 2)
    mul = NumericalFunction("mul", 2)
    sqr = NumericalFunction("sqr", 1)
    inv = NumericalFunction("inv", 1)
    numeric_collection = [add, sub, mul, sqr, inv]

    # Logic functions
    BOE = LogicFunction("BiggerOrEqual", input_no=2)
    Equal = LogicFunction("Equal", input_no=2)
    NonNeg = LogicFunction("Not negative", input_no=1)
    Real = LogicFunction("Real", input_no=1)
    logic_collection = [BOE, Equal, NonNeg, Real]

    # Create non-trivial prover and initiate prover initial states
    nts_prover = simple_prover()
    nts_prover_string = "\n".join([ls.name for ls in nts_prover.raw_observe()['ground_truth']])
    # Initial provers as a dictionary {prover ground truth string: prover}
    initial_provers = {nts_prover_string: deepcopy(nts_prover)}

    # Initialize Q neural network
    e2e_q = End2EndQNetwork(nts_prover, numerical_functions=numeric_collection, logic_functions=logic_collection)
    optimizer = optim.SGD(e2e_q.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    if os.path.isfile("../pt_models/end2end_q_recursive/checkpoint.pt"):
        state = torch.load("../pt_models/end2end_q_recursive/checkpoint.pt")
        iter = state['iter']
        epoch = state['epoch']
        e2e_q = state['model']
        optimizer = state['optimizer']
        training_dataset = state['dataset']
        replay_buffer = state['replay buffer']
        losses = state['losses']
        print("Epoch ", epoch)
    else:
        replay_buffer = ReplayBuffer("End-to-end Q Learning Buffer", max_size=BUFFER_MAX_SIZE)
        sample_episodes_and_cache(e2e_q, replay_buffer, initial_provers, depth=EPISODE_DEPTH,
                                  randomized=True, modified_her=True)
        training_dataset = TransitionDataset(replay_buffer=replay_buffer, prover=nts_prover)
        losses = list()
        iter = 0

    while iter < ITERATION:
        epoch = 0
        while epoch < EPOCHS:
            print("Iteration", iter)
            print("Epoch ", epoch)
            # One training step
            train(dataset=training_dataset, optimizer=optimizer, criterion=criterion, q_net=e2e_q, losses=losses)
            epoch += 1
            state = {
                'iter': iter, 'epoch': epoch, 'model': e2e_q, 'optimizer': optimizer, 'dataset': training_dataset,
                'replay buffer': replay_buffer, 'losses': losses
            }
            torch.save(state, "../pt_models/end2end_q_recursive/checkpoint.pt")
            json.dump(losses, open("../pt_models/end2end_q_recursive/loss.json", "w"))

        sample_episodes_and_cache(q_net=e2e_q, replay_buffer=replay_buffer, initial_states=initial_provers,
                                  depth=EPISODE_DEPTH, randomized=True)
        training_dataset = TransitionDataset(replay_buffer=replay_buffer, prover=nts_prover)
        iter += 1

    torch.save(e2e_q, "../pt_models/end2end_q_recursive/end2end_q.pt")
