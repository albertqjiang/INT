from legacy.nn.autoencoder import Lang, readLangs
from legacy.pseq.errors import InputError

from copy import deepcopy

import torch.nn as nn

MAX_LENGTH = 100
REWARD_INPUT_INVALID = 0.
REWARD_ASSUMPTION_INVALID = 0.
REWARD_DUPLICATED_RESULTS = 1
REWARD_THEOREM_PROCEEDED = 100.
REWARD_PROOF_COMPLETE = 10000.


def filterPair(p, max_length=MAX_LENGTH):
    return len(p[0].split(' ')) < max_length and \
           len(p[1].split(' ')) < max_length


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang_name, data_path):
    lang = Lang(lang_name)
    pairs = readLangs(file_path=data_path)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        lang.addSentence(pair[0])
        lang.addSentence(pair[1])
    print("Counted words:")
    print(lang.n_words)
    return lang, pairs


class LogicBasedProver:
    def __init__(self, proof, ent_maxsize, gt_maxsize, lemma_maxsize,
                 lemma_embedding_size, lemma_operand_size, objective_maxsize,
                 verbosity=False):
        self.original_proof = deepcopy(proof)
        self.proof = proof
        self.ent_maxsize = ent_maxsize
        self.gt_maxsize = gt_maxsize
        self.lemma_maxsize = lemma_maxsize
        self.lemma_embedding_size = lemma_embedding_size
        self.lemma_name2index = dict()
        for ind, lemma in enumerate(self.proof.lemmas):
            self.lemma_name2index[lemma.name] = ind
        self.lemma_embedding = nn.Embedding(self.lemma_maxsize, self.lemma_embedding_size)
        self.lemma_operand_maxsize = lemma_operand_size

        self.obj_maxsize = objective_maxsize
        # self.lang, self.pairs = prepareData(lang_name, data_path)

        self.verbosity = verbosity

    def raw_observe(self):
        raw_observation = dict()
        raw_observation["ground_truth"] = self.proof.ground_truth
        raw_observation["lemmas"] = self.proof.lemmas
        raw_observation["entities"] = self.proof.entities
        raw_observation["objectives"] = self.proof.objectives
        return raw_observation

    def raw_objectives(self):
        return self.proof.objectives

    def step(self, action):
        """

        :param action: an action that is of the form as follows:
                {"action_type": "theorem" or "operation", "action"}
        :return: observation, reward, done, next_observation
        """
        prover_copy = deepcopy(self)
        if action["action_type"] == "theorem":
            try:
                reward = self.proof.apply_theorem(theorem=action["action"][0], operands=action["action"][1])
            except InputError as ie:
                if self.verbosity:
                    print("InputError: " + ie.expression)
                reward = "REWARD_INPUT_INVALID"
            # Setting reward
            # None for theorem assumptions not satisfied, True for objectives proved, and False for theorem concludes
            # but objectives not all proved
            reward_float = {
                "REWARD_INPUT_INVALID": REWARD_INPUT_INVALID, "REWARD_ASSUMPTION_INVALID": REWARD_ASSUMPTION_INVALID,
                "REWARD_PROOF_COMPLETE": REWARD_PROOF_COMPLETE, "REWARD_THEOREM_PROCEEDED": REWARD_THEOREM_PROCEEDED,
                "REWARD_DUPLICATED_RESULTS": REWARD_DUPLICATED_RESULTS
            }[reward]
            done = self.proof.proved
            next_prover_copy = deepcopy(self)
            if done:
                self.reset()
            return prover_copy, reward_float, done, next_prover_copy
        elif action["action_type"] == "operation":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def reset(self):
        self.proof = deepcopy(self.original_proof)

    def get_status(self):
        return self.proof.proved

    def __str__(self):
        return "\n".join([ls.name for ls in self.raw_observe()['ground_truth']])
