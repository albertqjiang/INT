import os
import sys

sys.path.insert(0, os.path.abspath('../..'))
from logic.utils import non_trivial_prover
from copy import deepcopy
import random

import torch


def prove_theorem(q_controller, no_proofs):
    for _ in range(no_proofs):
        prover = deepcopy(q_controller.prover)
        step_count = 0
        while not prover.proved:
            step_count += 1
            theorem = random.sample(prover.proof.lemmas, k=1)[0]
            entities_taken = list()
            for operand_ind in range(theorem.input_no):
                entity = q_controller.choose_next_entity(theorem=theorem, entities_taken=entities_taken)
                entities_taken.append(entity)
            reward_string = prover.proof.apply_theorem(theorem=theorem, operands=entities_taken)
            print(reward_string)

        print("Steps to prove the whole theorem: {}".format(step_count))


if __name__ == "__main__":
    nt_prover = non_trivial_prover()
    q_controller = torch.load("../pt_models/hierq/final_state_onehot.pt")["q_controller"]
    prove_theorem(q_controller=q_controller, no_proofs=1)
