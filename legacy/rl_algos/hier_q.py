# Implementation for theorem proving of the paper
# Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation
# @https://arxiv.org/abs/1604.06057

__author__ = "Albert Jiang"

import json
import random
from legacy import logic_math
import os
import sys

sys.path.insert(0, os.path.abspath('../..'))
from helper_functions import ReplayBuffer, load_config, prepare_batches
from legacy.helper_functions.RecursiveAttentiveEncoder import RecursiveAttentiveEncoder
from logic.utils import non_trivial_prover, set_random_seed
from legacy.pseq.errors import InputError
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class QController(nn.Module):
    def __init__(self, recursive_encoder, criterion, maximum_operand_size=5):
        super(QController, self).__init__()

        self.encoder = recursive_encoder
        self.device = self.encoder.device
        self.maximum_operand_size = maximum_operand_size
        self.input_size = self.encoder.observation_tensor_size + \
                          self.encoder.theorem_embedding_size + (
                                      1 + self.maximum_operand_size) * self.encoder.entity_embedding_size
        self.q_layers = nn.Sequential(
            nn.Linear(self.input_size, 2 * self.input_size),
            nn.ReLU(),
            nn.Linear(2 * self.input_size, int(1.4 * self.input_size)),
            nn.ReLU(),
            nn.Linear(int(1.4 * self.input_size), self.input_size),
            nn.ReLU(),
            nn.Linear(self.input_size, int(0.5 * self.input_size)),
            nn.ReLU(),
            nn.Linear(int(0.5 * self.input_size), int(1. / 16 * self.input_size)),
            nn.ReLU(),
            nn.Linear(int(1. / 16 * self.input_size), 1),
        )
        self.criterion = criterion

        self.to(self.device)

    def forward(self, state, entities_taken, theorem, entity):
        # Combine tensors to be a giant one
        state_tensor = self.encoder.encode_observation(state).squeeze().to(self.device)
        entities_taken_tensors = [self.encoder.encode_entity(entity_taken).to(self.device)
                                  for entity_taken in entities_taken]
        entities_taken_tensors += [torch.zeros(1, self.encoder.entity_embedding_size, device=self.device)] * \
                                  (self.maximum_operand_size - len(entities_taken))
        theorem_tensor = self.encoder.encode_theorem(theorem).to(self.device)
        entity_tensor = self.encoder.encode_entity(entity).to(self.device)
        all_tensors = [state_tensor, theorem_tensor] + entities_taken_tensors + [entity_tensor]
        combined_tensor = torch.cat(all_tensors, dim=0)
        combined_tensor = combined_tensor.view(1, -1)

        assert self.input_size == combined_tensor.size(1)

        return self.q_layers(combined_tensor)


class QMetaController(nn.Module):
    def __init__(self, recursive_encoder, criterion):
        super(QMetaController, self).__init__()

        self.encoder = recursive_encoder
        self.device = self.encoder.device
        self.input_size = self.encoder.observation_tensor_size + self.encoder.theorem_embedding_size
        self.q_layers = nn.Sequential(
            nn.Linear(self.input_size, 2 * self.input_size),
            nn.ReLU(),
            nn.Linear(2 * self.input_size, int(1.4 * self.input_size)),
            nn.ReLU(),
            nn.Linear(int(1.4 * self.input_size), self.input_size),
            nn.ReLU(),
            nn.Linear(self.input_size, int(0.5 * self.input_size)),
            nn.ReLU(),
            nn.Linear(int(0.5 * self.input_size), int(1. / 16 * self.input_size)),
            nn.ReLU(),
            nn.Linear(int(1. / 16 * self.input_size), 1),
        )
        self.criterion = criterion

        self.to(self.device)

    def forward(self, state, theorem):
        state_tensor = self.encoder.encode_observation(state).squeeze()
        theorem_tensor = self.encoder.encode_theorem(theorem)
        combined_tensor = torch.cat([state_tensor, theorem_tensor], dim=0)
        combined_tensor = combined_tensor.view(1, -1)

        assert self.input_size == combined_tensor.size(1)

        return self.q_layers(combined_tensor)


def controller_max_q(state, entities_taken, theorem, action_space, q_func):
    max_q = -9999.
    for action in action_space:
        q = q_func.forward(state, entities_taken, theorem, action)
        if q > max_q:
            max_q = q
    return max_q


def meta_controller_max_q(state, action_space, q_func):
    max_q = -9999.
    for action in action_space:
        q = q_func.forward(state, action)
        if q > max_q:
            max_q = q
    return max_q


def controller_q_epsilon_greedy(state, entities_taken, theorem, action_space, epsilon, q_func, verbose):
    """
    Epsilon greedy algorithm
    :param state:
    :param action_space:
    :param entities_taken:
    :param theorem:
    :param epsilon:
    :param q_func:
    :return: argmax_{a \in A} Q(s, a)
    """
    if random.random() < epsilon:
        return random.sample(population=action_space, k=1)[0]
    else:
        q_values = list()
        for action in action_space:
            q = q_func.forward(state, entities_taken, theorem, action)
            q_values.append(q)
        q_tensor = F.softmax(torch.FloatTensor(q_values).squeeze(), dim=-1)
        action = random.choices(action_space, weights=q_tensor, k=1)[0]
        if verbose:
            print(action.name)
        return action


def meta_controller_q_epsilon_greedy(state, action_space, epsilon, q_func, verbose):
    """
    Epsilon greedy algorithm
    :param state:
    :param action_space:
    :param epsilon:
    :param q_func:
    :return: argmax_{a \in A} Q(s, a)
    """
    if random.random() < epsilon:
        return random.sample(population=action_space, k=1)[0]
    else:
        meta_q_values = list()
        for action in action_space:
            q = q_func.forward(state, action)
            meta_q_values.append(q)
        q_tensor = F.softmax(torch.FloatTensor(meta_q_values).squeeze(), dim=-1)
        action = random.choices(action_space, weights=q_tensor, k=1)[0]
        if verbose:
            print(action.name)
        return action


def controller_update(prover, q_controller, q_buffer, batch_size, controller_optimizer,
                      gamma=0.99, epochs=1, verbose=False):
    losses = list()
    for epoch in range(epochs):
        running_loss = 0.
        running_counter = 0
        lob = prepare_batches(replay_buffer=q_buffer, batch_size=batch_size, drop_last=False)
        for batch in lob:
            output_tensors = list()
            target_tensors = list()
            for transition in batch:

                output_tensors.append(q_controller.forward(
                    state=transition["state"], entities_taken=transition["entities_taken"],
                    theorem=transition["theorem"], entity=transition["entity"]
                ))

                # At terminal state, Q function has the value of the reward
                if transition["theorem"].input_no == 1 + len(transition["entities_taken"]):
                    target_tensors.append(torch.FloatTensor([[transition["reward"]]]).to(q_controller.device))
                else:
                    target_tensors.append(controller_max_q(
                        state=transition["next_state"], action_space=prover.proof.entities, q_func=q_controller,
                        entities_taken=transition["entities_taken"] + [transition["entity"]],
                        theorem=transition["theorem"]
                    ) * gamma + transition["reward"])
            output_tensor = torch.cat(output_tensors, dim=0)
            target_tensor = torch.cat(target_tensors, dim=0).detach()
            loss = q_controller.criterion(output_tensor, target_tensor)
            controller_optimizer.zero_grad()
            loss.backward()
            controller_optimizer.step()
            running_loss += loss.item()
            running_counter += len(batch)

        losses.append(running_loss / running_counter)
        if verbose:
            print(running_loss / running_counter)
    return losses


def meta_controller_update(prover, meta_q_controller, meta_q_buffer, batch_size, meta_controller_optimizer,
                           gamma=0.99, epochs=1, verbose=False):
    losses = list()
    for epoch in range(epochs):
        running_loss = 0.
        running_counter = 0
        lob = prepare_batches(replay_buffer=meta_q_buffer, batch_size=batch_size, drop_last=False)
        for batch in lob:
            output_tensors = list()
            target_tensors = list()
            for transition in batch:
                output_tensors.append(meta_q_controller.forward(
                    state=transition["state"], theorem=transition["theorem"]
                ))

                # At terminal state, Q function has the value of the reward
                if transition["reward"] > 50:
                    target_tensors.append(torch.FloatTensor([[transition["reward"]]]).to(meta_q_controller.device))
                else:
                    target_tensors.append(meta_controller_max_q(
                        state=transition["next_state"], action_space=prover.proof.lemmas, q_func=meta_q_controller
                    ) * gamma + transition["reward"])

            output_tensor = torch.cat(output_tensors, dim=0)
            target_tensor = torch.cat(target_tensors, dim=0).detach()
            loss = meta_q_controller.criterion(output_tensor, target_tensor)
            meta_controller_optimizer.zero_grad()
            loss.backward()
            meta_controller_optimizer.step()
            running_loss += loss.item()
            running_counter += len(batch)

        losses.append(running_loss / running_counter)
        if verbose:
            print(running_loss / running_counter)
    return losses


def meta_q_epsilon_anneal(current_episode, total_episodes):
    return 1. - 0.9 * current_episode / (total_episodes - 1)


def q_epsilon_anneal(success_rate):
    return 1. - 0.9 * success_rate


def run_episodes(prover, q, meta_q, buffer1, buffer2, inner_reward, outer_reward, batch_size,
                 controller_optimizer, meta_controller_optimizer, episodes, saving_dir, saving_affix,
                 episode=0, verbose=False):
    controller_losses = list()
    meta_controller_losses = list()

    success_rate = 0.
    while episode < episodes:
        # Anneal epsilons
        epsilon1 = q_epsilon_anneal(success_rate)
        epsilon2 = meta_q_epsilon_anneal(episode, episodes)
        print("Episode number: {}".format(episode + 1))
        print("Epsilon values are: epsilon1: {}; epsilon2: {}".format(epsilon1, epsilon2))

        counter = 0.
        success_counter = 0.

        prover_copy = deepcopy(prover)
        theorem_to_use = meta_controller_q_epsilon_greedy(state=prover_copy.raw_observe(),
                                                          action_space=prover_copy.raw_observe()["lemmas"],
                                                          epsilon=epsilon2, q_func=meta_q, verbose=verbose)
        while not prover_copy.proof.proved:
            s0 = deepcopy(prover_copy.raw_observe())
            entities_taken = list()

            for intrinsic_step in range(theorem_to_use.input_no):

                # Inner MDP and cache samples
                entity_to_use = \
                    controller_q_epsilon_greedy(state=prover_copy.raw_observe(), entities_taken=entities_taken,
                                                theorem=theorem_to_use, action_space=prover_copy.proof.entities,
                                                epsilon=epsilon1, q_func=q, verbose=verbose)
                inner_transition = {"state": s0, "entities_taken": deepcopy(entities_taken), "theorem": theorem_to_use,
                                    "entity": entity_to_use}
                entities_taken.append(entity_to_use)
                if len(entities_taken) == theorem_to_use.input_no:
                    try:
                        reward_string = prover_copy.proof.apply_theorem(theorem_to_use, entities_taken)
                    except InputError:
                        reward_string = "REWARD_INPUT_INVALID"
                else:
                    reward_string = "REWARD_NULL"

                # Scale negative rewards by 1/exp(#operands of the theorem)
                if inner_reward[reward_string] < 0:
                    inner_scaling = 1. / logic_math.exp(theorem_to_use.input_no)
                else:
                    inner_scaling = 1.
                inner_transition["reward"] = inner_reward[reward_string] * inner_scaling

                # if verbose:
                #     print("Inner reward: ", inner_transition["reward"])
                inner_transition["next_state"] = prover_copy.raw_observe()
                buffer1.cache([inner_transition])

                # Update the q and meta q controllers
                # The more successes the solver gets, the more often it fits the curves
                if not (reward_string == "REWARD_NULL" or reward_string == "REWARD_INPUT_INVALID"):
                    controller_losses.extend(
                        controller_update(prover=prover_copy, q_controller=q, q_buffer=buffer1, batch_size=batch_size,
                                          controller_optimizer=controller_optimizer, verbose=verbose)
                    )
                    meta_controller_losses.extend(
                        meta_controller_update(prover=prover_copy, meta_q_controller=meta_q, meta_q_buffer=buffer2,
                                               batch_size=batch_size, verbose=verbose,
                                               meta_controller_optimizer=meta_controller_optimizer)
                    )
                else:
                    random_number = random.random()
                    if random_number > epsilon2:
                        controller_losses.extend(
                            controller_update(prover=prover_copy, q_controller=q, q_buffer=buffer1,
                                              batch_size=batch_size,
                                              controller_optimizer=controller_optimizer, verbose=verbose)
                        )
                        meta_controller_losses.extend(
                            meta_controller_update(prover=prover_copy, meta_q_controller=meta_q, meta_q_buffer=buffer2,
                                                   batch_size=batch_size, verbose=verbose,
                                                   meta_controller_optimizer=meta_controller_optimizer)
                        )

            # Outer MDP and cache samples
            outer_transition = {"state": s0, "theorem": theorem_to_use, "next_state": prover_copy.raw_observe(),
                                "reward": outer_reward[reward_string]}
            if verbose:
                print("Theorem used: ", theorem_to_use.name)
                print("Entities taken: ", [entity.name for entity in entities_taken])
                print("Outer reward: ", outer_reward[reward_string])

            # Scaling negative outer reward by 1 / exp(#operands of the theorem)
            if outer_reward[reward_string] < 0:
                outer_scaling = 1. / logic_math.exp(theorem_to_use.input_no)
            else:
                outer_scaling = 1.
            outer_transition["reward"] *= outer_scaling
            buffer2.cache([outer_transition])

            if not prover_copy.proved:
                theorem_to_use = \
                    meta_controller_q_epsilon_greedy(state=prover_copy.raw_observe(), epsilon=epsilon2,
                                                     action_space=prover_copy.proof.lemmas, q_func=meta_q,
                                                     verbose=verbose)

            counter += 1.
            if reward_string == "REWARD_THEOREM_PROCEEDED" or reward_string == "REWARD_PROOF_COMPLETE":
                success_counter += 1.

        # Update success rate
        success_rate = success_counter / counter
        print("Success rate is {}".format(success_rate))
        print("Theorem is proved.")

        episode += 1
        state = {"episode": episode, "q": q, "meta_q": meta_q, "buffer1": buffer1, "buffer2": buffer2,
                 "controller_optimizer": controller_optimizer, "meta_controller_optimizer": meta_controller_optimizer}
        torch.save(state, saving_dir + "checkpoint{}.pt".format(saving_affix))
        json.dump(controller_losses, open(saving_dir + "controller_losses{}.json".format(saving_affix), "w"))
        json.dump(meta_controller_losses, open(saving_dir + "meta_controller_losses{}.json".format(saving_affix), "w"))
    torch.save(state, saving_dir + "model{}.pt".format(saving_affix))


if __name__ == "__main__":
    # Set all random seeds
    set_random_seed(1)

    # Load configuration parameters
    config = load_config()["hierq"]
    CONTROLLER_BUFFER_SIZE = config["CONTROLLER_BUFFER_SIZE"]
    META_CONTROLLER_BUFFER_SIZE = config["META_CONTROLLER_BUFFER_SIZE"]
    INNER_REWARD = config["INNER_REWARD"]
    OUTER_REWARD = config["OUTER_REWARD"]
    EPISODES = config["EPISODES"]
    BATCH_SIZE = config["BATCH_SIZE"]
    CONTROLLER_LEARNING_RATE = config["CONTROLLER_LEARNING_RATE"]
    META_CONTROLLER_LEARNING_RATE = config["META_CONTROLLER_LEARNING_RATE"]
    WEIGHT_DECAY = config["WEIGHT_DECAY"]
    SAVING_DIR = config["SAVING_DIR"]
    SAVING_AFFIX = config["SAVING_AFFIX"]

    # Initialize models
    replay_buffer_controller = ReplayBuffer(name="D1", capacity=CONTROLLER_BUFFER_SIZE)
    replay_buffer_m_controller = ReplayBuffer(name="D2", capacity=META_CONTROLLER_BUFFER_SIZE)
    nt_prover = non_trivial_prover()
    encoder = RecursiveAttentiveEncoder(nt_prover)
    criterion = nn.MSELoss()
    q_controller = QController(encoder, criterion=criterion)
    q_meta_controller = QMetaController(encoder, criterion=criterion)
    q_optimizer = optim.Adam(q_controller.parameters(), lr=CONTROLLER_LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY, amsgrad=True)
    q_meta_optimizer = optim.Adam(q_meta_controller.parameters(), lr=META_CONTROLLER_LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY, amsgrad=True)
    epsilon1 = 1.
    epsilon2 = 1.

    # Training
    run_episodes(prover=nt_prover, q=q_controller, meta_q=q_meta_controller,
                 buffer1=replay_buffer_controller, buffer2=replay_buffer_m_controller,
                 inner_reward=INNER_REWARD, outer_reward=OUTER_REWARD,
                 batch_size=BATCH_SIZE,
                 controller_optimizer=q_optimizer, meta_controller_optimizer=q_meta_optimizer, episodes=EPISODES,
                 saving_dir=SAVING_DIR, saving_affix=SAVING_AFFIX, verbose=True)
