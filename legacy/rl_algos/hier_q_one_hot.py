__author__ = "Albert Jiang"

import json
import random
from legacy import logic_math
from copy import deepcopy
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import sys

sys.path.insert(0, os.path.abspath('../..'))
from legacy.helper_functions.utils import ReplayBuffer, load_config, prepare_batches
from legacy.logic.OneHotEncoder import OneHotEncoder
from logic.utils import non_trivial_prover, set_random_seed
from legacy.pseq.errors import InputError


class AgnosticOneHotQController(nn.Module):
    def __init__(self, recursive_encoder, criterion,
                 maximum_operand_size=5, verbose=False):
        super(AgnosticOneHotQController, self).__init__()

        # Device to operate on
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = recursive_encoder
        self.prover = self.encoder.prover
        self.proof = self.prover.proof
        self.criterion = criterion
        self.maximum_operand_size = maximum_operand_size
        self.verbose = verbose
        self.input_size = len(self.encoder.theorems) + (1 + self.maximum_operand_size) * len(self.encoder.entities)
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

        self.to(self.device)

    def forward(self, theorem, entities_taken, entity):
        # Combine tensors to be a giant one
        theorem_tensor = self.encoder.encode_theorem(theorem).unsqueeze(0).to(self.device)
        entities_taken_tensors = [self.encoder.encode_entity(entity_taken).unsqueeze(0).to(self.device)
                                  for entity_taken in entities_taken]
        entities_taken_tensors += [torch.zeros(1, len(self.encoder.entities), device=self.device)] * \
                                  (self.maximum_operand_size - len(entities_taken))
        entity_tensor = self.encoder.encode_entity(entity).unsqueeze(0).to(self.device)
        combined_tensor = torch.cat([theorem_tensor] + entities_taken_tensors + [entity_tensor], dim=1).view(1, -1)

        # Check tensor sizes are right
        assert self.input_size == combined_tensor.size(1)
        assert len(entities_taken) <= theorem.input_no
        return self.q_layers(combined_tensor)

    def _max_action(self, theorem, entities_taken):
        max_q, max_action = -9999., None
        for entity in self.proof.entities:
            entity_q_value = self.forward(theorem=theorem, entities_taken=entities_taken, entity=entity).item()
            if entity_q_value > max_q:
                max_q = entity_q_value
                max_action = entity
        return max_action

    def choose_next_entity(self, theorem, entities_taken):
        entity_q_values = list()
        for entity in self.proof.entities:
            entity_q_values.append(self.forward(theorem=theorem, entities_taken=entities_taken, entity=entity).item())
        probability_weights = F.softmax(torch.Tensor(entity_q_values), dim=0).numpy().tolist()

        assert len(self.proof.entities) == len(probability_weights)
        return random.choices(population=self.proof.entities, weights=probability_weights)[0]

    def epsilon_choose_next_entity(self, theorem, entities_taken, epsilon):
        print("Theorem to use: {}".format(theorem.name))
        print("Entities taken: {}".format([entity.name for entity in entities_taken]))
        if random.random() < epsilon:
            action = random.choice(self.proof.entities)
            print("Randomly chosen: {}\n".format(action.name))
            return action
        else:
            action = self.choose_next_entity(theorem=theorem, entities_taken=entities_taken)
            print("Max q chosen: {}\n".format(action.name))
            return action

    def update(self, null_buffer, complete_buffer, batch_size, optimizer):
        running_loss = 0.
        running_counter = 0
        buffer = ReplayBuffer("Temp buffer", capacity=2 * null_buffer.size)
        buffer.cache(complete_buffer.pool + null_buffer.pool[:len(complete_buffer.pool)])
        lob = prepare_batches(replay_buffer=buffer, batch_size=batch_size, drop_last=False)
        for batch in lob:
            output_tensors = list()
            target_tensors = list()
            for transition in batch:
                output_tensors.append(self.forward(theorem=transition["current state"]["theorem"],
                                                   entities_taken=transition["current state"]["entities_taken"],
                                                   entity=transition["action"]))

                # At terminal state, Q function has the value of the reward
                if transition["reward"] != 0:
                    target_tensors.append(torch.FloatTensor([[transition["reward"]]]).to(self.device))
                else:
                    max_q_action = self._max_action(theorem=transition["next state"]["theorem"],
                                                    entities_taken=transition["next state"]["entities_taken"])
                    target_tensors.append(self.forward(theorem=transition["next state"]["theorem"],
                                                       entities_taken=transition["next state"]["entities_taken"],
                                                       entity=max_q_action) + transition["reward"])

            # Organize tensors and update
            output_tensor = torch.cat(output_tensors, dim=0)
            target_tensor = torch.cat(target_tensors, dim=0).detach()
            loss = self.criterion(output_tensor, target_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_counter += len(batch)

        if running_counter == 0:
            assert running_loss == 0
            average_loss = 0
        else:
            average_loss = running_loss / running_counter
        if self.verbose:
            print(average_loss)
        return average_loss


def choose_epsilon(episode, episodes):
    return 0.1 - 0.095 * (episode - 1) / (episodes - 1)
    # return 1. - 0.9 * (episode-1)/(episodes-1)


def sample_q_episodes(q_controller, q_null_buffer, q_complete_buffer, config, optimizer):
    episodes = config["EPISODES"]
    depth = config["DEPTH"]
    reward_dict = config["INNER_REWARD"]
    saving_dir = config["SAVING_DIR"]
    saving_affix = config["SAVING_AFFIX"]
    prover = q_controller.prover

    episode = 0
    controller_losses = list()
    episode_rewards = list()
    while episode < episodes:
        theorem = random.choices(population=prover.proof.lemmas, weights=[logic_math.exp(theorem.input_no)
                                                                          for theorem in prover.proof.lemmas], k=1)[0]
        episode += 1
        prover_copy = deepcopy(prover)
        epsilon = choose_epsilon(episode, episodes)

        # Initialize some empty objects
        entities_taken = list()
        aggregate_reward = 0.
        finished_proof = 0
        current_depth = 0

        while current_depth < depth and not prover.proved:
            current_depth += 1

            entity = q_controller.epsilon_choose_next_entity(theorem=theorem,
                                                             entities_taken=entities_taken, epsilon=epsilon)
            inner_transition = {"current state": {"theorem": theorem, "entities_taken": entities_taken},
                                "action": entity,
                                "next state": {"theorem": theorem, "entities_taken": entities_taken + [entity]}}
            entities_taken += [entity]
            if len(entities_taken) == theorem.input_no:
                try:
                    reward_string = prover_copy.proof.apply_theorem(theorem=theorem, operands=entities_taken)
                except InputError:
                    reward_string = "REWARD_INPUT_INVALID"
                entities_taken = list()
            else:
                reward_string = "REWARD_NULL"

            if reward_dict[reward_string] < 0:
                inner_transition["reward"] = reward_dict[reward_string] / logic_math.exp(theorem.input_no)
                q_complete_buffer.cache([inner_transition])
            elif reward_dict[reward_string] > 0:
                inner_transition["reward"] = reward_dict[reward_string] * logic_math.exp(theorem.input_no)
                q_complete_buffer.cache([inner_transition])
            else:
                inner_transition["reward"] = reward_dict[reward_string]
                q_null_buffer.cache([inner_transition])

            aggregate_reward += inner_transition["reward"]
            finished_proof += 1

        episode_rewards.append(aggregate_reward / finished_proof)
        controller_losses.append(q_controller.update(q_null_buffer, q_complete_buffer,
                                                     batch_size=config["BATCH_SIZE"], optimizer=optimizer))

        state = {"q_controller": q_controller, "q_null_buffer": q_null_buffer, "config": config, "optimizer": optimizer,
                 "episode_rewards": episode_rewards, "controller_losses": controller_losses, "episode": episode}
        torch.save(state, saving_dir + "checkpoint_{}.pt".format(saving_affix))
        json.dump(episode_rewards, open(saving_dir + "episode_rewards_{}.json".format(saving_affix), "w"))
        json.dump(controller_losses, open(saving_dir + "controller_losses_{}.json".format(saving_affix), "w"))
    torch.save(state, saving_dir + "final_state{}.pt".format(saving_affix))


if __name__ == "__main__":
    # Set all random seeds
    set_random_seed(1)

    # Load configuration parameters
    config = load_config()["hierqonehot"]
    CONTROLLER_BUFFER_SIZE = config["CONTROLLER_BUFFER_SIZE"]
    CONTROLLER_LEARNING_RATE = config["CONTROLLER_LEARNING_RATE"]
    WEIGHT_DECAY = config["WEIGHT_DECAY"]

    # Define model
    nt_prover = non_trivial_prover()
    proof = nt_prover.proof
    encoder = OneHotEncoder(nt_prover)
    criterion = nn.MSELoss()
    q_controller = AgnosticOneHotQController(recursive_encoder=encoder, criterion=criterion, verbose=True)
    q_buffer1 = ReplayBuffer(name="Q null buffer", capacity=CONTROLLER_BUFFER_SIZE)
    q_buffer2 = ReplayBuffer(name="Q complete buffer", capacity=CONTROLLER_BUFFER_SIZE)
    optimizer = optim.Adam(q_controller.parameters(), lr=CONTROLLER_LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                           amsgrad=True)
    pprint(sample_q_episodes(q_controller=q_controller, q_null_buffer=q_buffer1, q_complete_buffer=q_buffer2,
                             config=config, optimizer=optimizer))
