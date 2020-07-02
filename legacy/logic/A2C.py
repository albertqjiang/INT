import os
import sys

sys.path.insert(0, os.path.abspath('../..'))
from pprint import pprint
import random
import time
from copy import deepcopy
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_handler

from legacy.helper_functions.RecursiveAttentiveEncoder import RecursiveAttentiveEncoder
from logic.utils import non_trivial_prover, set_random_seed, ReplayBuffer, List2Dataset, \
    element_index_in_list_according_to_name
from legacy.logic.logicRL import REWARD_DUPLICATED_RESULTS

set_random_seed(1)


class Actor(nn.Module):
    def __init__(self, recursive_encoder):
        super(Actor, self).__init__()
        self.encoder = recursive_encoder
        self.theorem_maxsize = self.encoder.theorem_maxsize
        self.theorem_representation_size = self.encoder.theorem_embedding_size
        self.operand_maxsize = self.encoder.theorem_operand_maxsize
        self.operand_representation_size = self.encoder.entity_embedding_size

        self.theorem_chooser = nn.Sequential(
            nn.Linear(self.encoder.observation_tensor_size, self.encoder.observation_tensor_size),
            nn.ReLU(),
            nn.Linear(self.encoder.observation_tensor_size, int(self.encoder.observation_tensor_size / 2)),
            nn.ReLU(),
            nn.Linear(int(self.encoder.observation_tensor_size / 2), int(self.encoder.observation_tensor_size / 4)),
            nn.ReLU(),
            nn.Linear(int(self.encoder.observation_tensor_size / 4), int(self.encoder.observation_tensor_size / 16)),
            nn.ReLU(),
            nn.Linear(int(self.encoder.observation_tensor_size / 16), self.theorem_maxsize),
        )

        self.bilinear_state_and_theorem = nn.Bilinear(
            int(self.encoder.observation_tensor_size / self.operand_representation_size), 1, self.operand_maxsize * 4
        )
        self.operands_chooser_first_part = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.operand_maxsize * 4, self.operand_maxsize * 2),
            nn.ReLU(),
            nn.Linear(self.operand_maxsize * 2, self.operand_maxsize),
            nn.ReLU(),
        )
        self.operands_chooser_second_part = nn.Sequential(
            nn.Linear(self.theorem_representation_size, int(self.theorem_representation_size / 2)),
            nn.ReLU(),
            nn.Linear(int(self.theorem_representation_size / 2), int(self.theorem_representation_size / 4)),
            nn.ReLU(),
            nn.Linear(int(self.theorem_representation_size / 4), int(self.theorem_representation_size / 8)),
            nn.ReLU(),
            nn.Linear(int(self.theorem_representation_size / 8), self.encoder.entity_maxsize)
        )

    def forward(self, state):
        # Choose theorem to apply
        theorem_choice_tensor = self.choose_theorem(state)
        theorem_choice = random.choices(self.encoder.theorems, weights=theorem_choice_tensor)[0]

        # Choose operands for the theorem
        operands_choice_tensor = self.choose_operands(state, theorem_choice)
        operand_choices = list()
        for i in range(operands_choice_tensor.size(0)):
            operand_choices.append(random.choices(
                self.encoder.prover.proof.entities, weights=operands_choice_tensor[i])[0])
        return theorem_choice_tensor, theorem_choice, operands_choice_tensor, operand_choices

    def choose_theorem(self, state):
        state_tensor = self.encoder.encode_observation(state).squeeze()
        # Calculate theorem choice weights
        theorem_choice_tensor = self.theorem_chooser(state_tensor.view(-1))
        theorem_choice_tensor = theorem_choice_tensor.narrow(dim=0, start=0, length=len(self.encoder.theorems))
        theorem_choice_tensor = F.softmax(theorem_choice_tensor, dim=0)
        return theorem_choice_tensor

    def choose_operands(self, state, theorem):
        state_tensor = self.encoder.encode_observation(state).squeeze()
        theorem_embedded = self.encoder.encode_theorem(theorem)

        # Calculate operand choice weights
        operands_choice_tensor = self.bilinear_state_and_theorem(
            state_tensor.transpose(0, 1), theorem_embedded.transpose(0, 1)
        )
        operands_choice_tensor = operands_choice_tensor
        operands_choice_tensor = self.operands_chooser_first_part(operands_choice_tensor).transpose(0, 1)
        operands_choice_tensor = self.operands_chooser_second_part(operands_choice_tensor).squeeze()
        operands_choice_tensor = \
            operands_choice_tensor.narrow(dim=1, start=0, length=len(self.encoder.prover.proof.entities))
        operands_choice_tensor = operands_choice_tensor.narrow(dim=0, start=0, length=theorem.input_no)
        operands_choice_tensor = F.softmax(operands_choice_tensor, dim=1)
        return operands_choice_tensor

    def get_action_prob(self, state, action):
        """
        The pi function
        :param state:
        :param action:
        :return: the probability of choosing the action at the state
        """
        prob = 1.
        theorem_weights = self.choose_theorem(state)
        prob *= theorem_weights[element_index_in_list_according_to_name(self.encoder.theorems, action["action"][0])]
        operand_weights = self.choose_operands(state, action["action"][0])
        for operand_index, operand in enumerate(action["action"][1]):
            prob *= operand_weights[operand_index][element_index_in_list_according_to_name(
                self.encoder.prover.proof.entities, operand)]
        return prob


class Critic(nn.Module):
    def __init__(self, recursive_encoder):
        super(Critic, self).__init__()
        self.encoder = recursive_encoder
        self.state_tensor_size = self.encoder.observation_tensor_size
        self.value_network = nn.Sequential(
            nn.Linear(self.state_tensor_size, int(self.state_tensor_size / 2)),
            nn.ReLU(),
            nn.Linear(int(self.state_tensor_size / 2), int(self.state_tensor_size / 4)),
            nn.ReLU(),
            nn.Linear(int(self.state_tensor_size / 4), int(self.state_tensor_size / 8)),
            nn.ReLU(),
            nn.Linear(int(self.state_tensor_size / 8), 1)
        )

    def forward(self, state):
        state_tensor = self.encoder.encode_observation(state).squeeze().view(-1)
        value_tensor = self.value_network(state_tensor)
        return value_tensor


class ActorCriticAgent:
    def __init__(self, actor, critic, verbose=False):
        self.actor = actor
        self.critic = critic
        if self.actor.encoder.prover is not self.critic.encoder.prover:
            raise RuntimeError
        self.prover = deepcopy(self.actor.encoder.prover)
        self.ground_truth = [ls.name for ls in self.prover.raw_observe()["ground_truth"]]
        self.verbose = verbose

    def step(self):
        _, theorem_choice, _, operand_choices = self.actor.forward(self.prover.raw_observe())
        print("Theorem chosen: {}".format(theorem_choice.name))
        print("Operands chosen: {}".format([op.name for op in operand_choices]))
        action = {"action_type": "theorem", "action": [theorem_choice, operand_choices]}
        _, reward_num, done, next_p = self.prover.step(action)
        print("The reward is {}".format(reward_num))
        print("The problem is proved: {}".format(done))
        new_obs_gt = [ls.name for ls in next_p.raw_observe()["ground_truth"]]
        pprint(set(new_obs_gt) - set(self.ground_truth))
        print()


def sample_one_episode(replay_buffer, initial_prover_dict, episode_depth, actor_network=None, her=False):
    """
    An episode is consisted of transitions presented as (s_i, a, r, s_i+1).
    :param replay_buffer: replay buffer
    :param initial_prover_dict: initial prover s_0s to start from
    :param episode_depth: the depth of the episode to sample
    :param actor_network: the actor to use to determine the policy
    :param her: whether hindsight experience sight is used
    :return: replay buffer stuffed with one new episode and initial states augmented with new states discovered
    """
    episode = list()
    prover_for_episode = deepcopy(random.choice(list(initial_prover_dict.values())))
    for i in range(1, 1 + episode_depth):
        print("Sampling step {} in episode.".format(i))
        if actor_network:
            _, theorem, _, operands = actor_network(prover_for_episode.raw_observe())
        else:
            theorem = random.choice(prover_for_episode.proof.lemmas)
            operands = random.choices(prover_for_episode.proof.entities, k=theorem.input_no)
        action = {"action_type": "theorem", "action": [theorem, operands]}
        p, reward, done, prover_for_episode = prover_for_episode.step(action)

        # If the state changes, register the new state as an initial state
        if reward > 0:
            new_prover_string = str(prover_for_episode)
            if new_prover_string not in initial_prover_dict:
                initial_prover_dict[new_prover_string] = deepcopy(prover_for_episode)
            else:
                pass

        episode.append((p.raw_observe(), action, reward, prover_for_episode.raw_observe()))

    # If Hindsight Experience Replay is used
    if her:
        additional_transitions = list()
        for transition in episode:
            if transition[2] > REWARD_DUPLICATED_RESULTS:
                # The state of the prover changed
                transition_copy = deepcopy(transition)
                alternative_objectives = \
                    [ls for ls in transition_copy[3]["ground_truth"] if ls.name not in
                     [logic_statement.name for logic_statement in transition_copy[0]["ground_truth"]]]
                transition_copy[0]["objectives"] = alternative_objectives
                transition_copy[3]["objectives"] = alternative_objectives
                additional_transitions.append(transition_copy)
        episode.extend(additional_transitions)
        print(len(additional_transitions))
    replay_buffer.cache(episode)

    return episode


def sample_episodes(replay_buffer, initial_prover_dict, episode_depth, episodes, actor_network=None, her=False):
    for e in range(1, 1 + episodes):
        print("Sampling episode {}.".format(e))
        sample_one_episode(replay_buffer=replay_buffer, initial_prover_dict=initial_prover_dict,
                           episode_depth=episode_depth, actor_network=actor_network, her=her)


def optimize_critic(critic, critic_optimizer, critic_criterion, list_of_batches, discount_factor, critic_losses,
                    epoch=1):
    value_tensors = list()
    target_tensors = list()
    for e in range(epoch):
        running_loss = 0.
        running_no_transitions = 0
        for i, batch in enumerate(list_of_batches):
            print("Training the {}th batch for critic.".format(i + 1))
            for transition in batch:
                current_state, action, reward, next_state = transition
                value_tensors.append(critic(current_state))
                target_tensors.append(discount_factor * critic(next_state) + reward)

            # Fit the neural network to the value function
            values = torch.cat(value_tensors)
            targets = torch.cat(target_tensors)
            loss = critic_criterion(values, targets)

            critic_optimizer.zero_grad()
            print("Value function fitting loss(the smaller the better): ", loss.item() / len(batch))
            loss.backward()
            critic_optimizer.step()

            # Reset and update stuff
            value_tensors = list()
            target_tensors = list()
            running_loss += loss.item()
            running_no_transitions += len(batch)

        critic_losses.append(running_loss / running_no_transitions)


def optimize_actor(actor, critic, actor_optimizer, list_of_batches, discount_factor, actor_objectives):
    running_objective = 0.
    running_no_transitions = 0
    for i, batch in enumerate(list_of_batches):
        print("Training the {}th batch for actor.".format(i + 1))
        objective_j_elements = list()
        for transition in batch:
            current_state, action, reward, next_state = transition
            advantage = reward + discount_factor * critic.forward(next_state) - critic.forward(current_state)
            action_probability = actor.get_action_prob(current_state, action)
            stabilizing_epsilon = 1e-45
            objective_j_elements.append(advantage * torch.log(action_probability + stabilizing_epsilon))
        objective_j = -1 * torch.sum(torch.cat(objective_j_elements, dim=0))  # Gradient ascent
        print("Objective(the higher the better)", -1 * objective_j.item() / len(batch))
        actor_optimizer.zero_grad()
        objective_j.backward()
        actor_optimizer.step()

        running_objective += -1 * objective_j.item()
        running_no_transitions += len(batch)
        print(running_objective, running_no_transitions)
    actor_objectives.append(running_objective / running_no_transitions)


def prepare_batches(replay_buffer, batch_size, drop_last=False):
    """

    :param replay_buffer: replay buffer to use
    :param batch_size: batch capacity
    :param drop_last: whether to drop the last batch
    :return: a list of batches of transitions from the replay buffer
    """
    transition_dataset = List2Dataset(replay_buffer.buffer)
    r_sampler = data_handler.RandomSampler(transition_dataset)

    list_of_batches = list()
    current_batch = list()
    batch_index_counter = 1
    for index in r_sampler:
        transition = transition_dataset[index]
        current_batch.append(transition)
        batch_index_counter += 1

        if batch_index_counter % batch_size == 1:
            list_of_batches.append(current_batch)
            current_batch = list()
            batch_index_counter = 1
    if current_batch and not drop_last:
        list_of_batches.append(current_batch)
    return list_of_batches


def train_iterations():
    rb = ReplayBuffer(name="Actor Critic reinforcement learner", max_size=BUFFER_MAX_SIZE)
    nts_prover = non_trivial_prover()
    initial_provers = {str(nts_prover): deepcopy(nts_prover)}

    # Declaring networks and things related to training
    rEncoder = RecursiveAttentiveEncoder(nts_prover)
    actor = Actor(rEncoder)
    actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LEARNING_RATE, amsgrad=True)
    critic = Critic(rEncoder)
    critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LEARNING_RATE, amsgrad=True)
    critic_criterion = nn.MSELoss()

    # Record losses and objectives
    critic_losses = list()
    actor_objectives = list()

    # Start training
    if os.path.isfile("../pt_models/actor_critic/checkpoint.pt"):
        state = torch.load("../pt_models/actor_critic/checkpoint.pt")
        rb = state["replay_buffer"]
        nts_prover = state["prover"]
        initial_provers = state["initial_provers"]
        rEncoder = state["recurrent_encoder"]
        actor = state["actor"]
        actor_optimizer = state["actor_optimizer"]
        critic = state["critic"]
        critic_optimizer = state["critic_optimizer"]
        it = state["iteration"]
        critic_losses = state["critic_losses"]
        actor_objectives = state["actor_objectives"]

    else:
        sample_episodes(replay_buffer=rb, initial_prover_dict=initial_provers, episode_depth=EPISODE_DEPTH,
                        episodes=EPISODES, actor_network=None, her=HER)
        lob = prepare_batches(replay_buffer=rb, batch_size=BATCH_SIZE)
        optimize_critic(critic=critic, critic_optimizer=critic_optimizer, critic_criterion=critic_criterion,
                        list_of_batches=lob, discount_factor=DISCOUNT_FACTOR, epoch=EPOCHS, critic_losses=critic_losses)
        optimize_actor(actor=actor, critic=critic, actor_optimizer=actor_optimizer,
                       list_of_batches=lob, discount_factor=DISCOUNT_FACTOR, actor_objectives=actor_objectives)
        it = 0

    start_time = time.time()
    while it <= ITERATION:
        print("Training iteration {}:".format(it))
        sample_episodes(replay_buffer=rb, initial_prover_dict=initial_provers, episode_depth=EPISODE_DEPTH,
                        episodes=EPISODES, actor_network=actor, her=HER)
        lob = prepare_batches(replay_buffer=rb, batch_size=BATCH_SIZE)
        optimize_critic(critic=critic, critic_optimizer=critic_optimizer, critic_criterion=critic_criterion,
                        list_of_batches=lob, discount_factor=DISCOUNT_FACTOR, epoch=EPOCHS, critic_losses=critic_losses)
        optimize_actor(actor=actor, critic=critic, actor_optimizer=actor_optimizer,
                       list_of_batches=lob, discount_factor=DISCOUNT_FACTOR, actor_objectives=actor_objectives)
        it += 1
        state = {
            "replay_buffer": rb, "prover": nts_prover, "initial_provers": initial_provers,
            "recurrent_encoder": rEncoder, "actor": actor, "actor_optimizer": actor_optimizer,
            "critic": critic, "critic_optimizer": critic_optimizer, "iteration": it,
            "critic_losses": critic_losses, "actor_objectives": actor_objectives
        }
        torch.save(state, "../pt_models/actor_critic/checkpoint.pt")
        json.dump(critic_losses, open("../pt_models/actor_critic/losses.json", "w"))
        try:
            json.dump(actor_objectives, open("../pt_models/actor_critic/objectives.json", "w"))
        except TypeError:
            print(actor_objectives)

        # Agent behavior after each iteration
        ac_agent = ActorCriticAgent(actor=actor, critic=critic)
        for i in range(20):
            ac_agent.step()
    print(time.time() - start_time)
    torch.save(state, "../pt_models/actor_critic/model.pt")


if __name__ == "__main__":
    HER = True
    BUFFER_MAX_SIZE = 2000
    EPISODE_DEPTH = 50
    EPISODES = 10
    DISCOUNT_FACTOR = 0.99
    BATCH_SIZE = 32
    ACTOR_LEARNING_RATE = 1e-5
    CRITIC_LEARNING_RATE = 1e-4
    ITERATION = 10
    EPOCHS = 10

    # TEST_BUFFER_MAX_SIZE = 100
    # TEST_EPISODE_DEPTH = 10
    # TEST_EPISODES = 1
    # TEST_ITERATION = 2
    # BUFFER_MAX_SIZE = TEST_BUFFER_MAX_SIZE
    # EPISODE_DEPTH = TEST_EPISODE_DEPTH
    # EPISODES = TEST_EPISODES
    # ITERATION = TEST_ITERATION

    train_iterations()
