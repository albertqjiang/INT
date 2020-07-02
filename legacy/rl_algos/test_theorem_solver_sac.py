import torch
import random
from copy import deepcopy
import os
import sys

sys.path.insert(0, os.path.abspath('../..'))
from legacy.helper_functions.utils import set_random_seed, load_config

set_random_seed(1)
config = load_config()["soft_actor_critic"]
state_dir = config["sac_model"]
max_steps = config["max_steps"]

# Load encoder and disable its training
encoder = torch.load(config["encoder_path"])
for encoder_para in encoder.parameters():
    encoder_para.requires_grad = False

state = torch.load(state_dir)
policy_net = state["policy_net"]
for policy_net_para in policy_net.parameters():
    policy_net_para.requires_grad = False

steps = list()

for i in range(100):
    prover = deepcopy(encoder.prover)
    step = 0
    while step < max_steps:
        theorem = random.choices(prover.proof.lemmas, k=1)[0]
        entities_taken = list()
        for entity_idx in range(1, 1 + theorem.input_no):
            state_tensor = torch.cat([encoder.encode_theorem(theorem), encoder.encode_entities(entities_taken)], dim=-1)
            action = policy_net.get_action(state=state_tensor, action_space=prover.proof.entities)
            entities_taken += [action]
        reward_string = prover.proof.apply_theorem(theorem, entities_taken)
        print(reward_string)
        step += 1
        if prover.proof.proved:
            steps.append(step)
            break
print(sum(steps) / len(steps))
