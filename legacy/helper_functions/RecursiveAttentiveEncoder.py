import torch
import torch.nn as nn
import torch.nn.functional as F

from logic.utils import standard_numerical_functions, standard_logic_functions
from legacy.rl_algos.hierarchical import objectives_to_coupled_proofs_and_provers

import pickle


class RecursiveAttentiveEncoder(nn.Module):
    def __init__(self, proof, configuration, numerical_functions=None, logic_functions=None,
                 entity_embedding_size=128, logic_statement_embedding_size=128, theorem_embedding_size=128, cuda=True):
        super(RecursiveAttentiveEncoder, self).__init__()

        if cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Define default numerical and logic functions
        if not numerical_functions:
            numerical_functions = [nf for nf in standard_numerical_functions.values()]
        if not logic_functions:
            # Logic functions
            logic_functions = [lf for lf in standard_logic_functions.values()]

        # Getting parameters from prover and other inputs
        self.theorems = proof.lemmas
        self.numerical_functions = numerical_functions
        self.logic_functions = logic_functions
        self.input_entities = proof.entities

        if "entity_embedding_size" in configuration.keys() \
                and "logic_statement_embedding_size" in configuration.keys() \
                and "theorem_embedding_size" in configuration.keys():
            entity_embedding_size = configuration["entity_embedding_size"]
            logic_statement_embedding_size = configuration["logic_statement_embedding_size"]
            theorem_embedding_size = configuration["theorem_embedding_size"]
        self.entity_embedding_size = entity_embedding_size
        self.logic_statement_embedding_size = logic_statement_embedding_size
        self.theorem_embedding_size = theorem_embedding_size

        # Maximum sizes of properties to fix dimensions
        self.ground_truth_maxsize = configuration["gt_maxsize"]
        self.entity_maxsize = configuration["ent_maxsize"]
        self.theorem_maxsize = configuration["lemma_maxsize"]
        self.theorem_operand_maxsize = configuration["lemma_operand_maxsize"]
        self.objective_maxsize = configuration["obj_maxsize"]
        self.observation_tensor_size = \
            self.entity_embedding_size * self.entity_maxsize + \
            self.logic_statement_embedding_size * (self.ground_truth_maxsize + self.objective_maxsize) + \
            self.theorem_embedding_size * self.theorem_maxsize
        self.higher_attentive_size = \
            2 * self.logic_statement_embedding_size + self.entity_embedding_size + self.theorem_embedding_size
        self.lower_attentive_size = \
            2 * self.logic_statement_embedding_size + 2 * self.entity_embedding_size + self.theorem_embedding_size

        # Define embeddings
        self.input_entity_embedding = nn.Embedding(num_embeddings=len(self.input_entities),
                                                   embedding_dim=self.entity_embedding_size)
        self.theorem_embedding = nn.Embedding(num_embeddings=len(self.theorems),
                                              embedding_dim=self.theorem_embedding_size)

        # Memorize tensors corresponding to entity strings
        self.input_entity_string2index = {ie.to_string(): torch.LongTensor([ind]).to(self.device)
                                          for ind, ie in enumerate(self.input_entities)}
        self.theorem_string2index = {theorem.name: torch.LongTensor([ind]).to(self.device)
                                     for ind, theorem in enumerate(self.theorems)}

        # Numerical function layers: E^n -> E
        self.nf_two2one_layers = nn.ModuleDict({
            nf.name: nn.Sequential(
                nn.Linear(2 * self.entity_embedding_size, 2 * self.entity_embedding_size),
                nn.ReLU(inplace=True),
                nn.Linear(2 * self.entity_embedding_size, self.entity_embedding_size),
            )
            for nf in self.numerical_functions if nf.input_no == 2
        })
        self.nf_one2one_layers = nn.ModuleDict({
            nf.name: nn.Sequential(
                nn.Linear(self.entity_embedding_size, 2 * self.entity_embedding_size),
                nn.ReLU(inplace=True),
                nn.Linear(2 * self.entity_embedding_size, self.entity_embedding_size),
            )
            for nf in self.numerical_functions if nf.input_no == 1
        })

        # Logic function layers: E^n -> LG
        self.lf_two2one_layers = nn.ModuleDict({
            lf.name: nn.Sequential(
                nn.Linear(2 * self.entity_embedding_size, 4 * self.entity_embedding_size),
                nn.ReLU(inplace=True),
                nn.Linear(4 * self.entity_embedding_size, self.entity_embedding_size),
            )
            for lf in self.logic_functions if lf.input_no == 2
        })
        self.lf_one2one_layers = nn.ModuleDict({
            lf.name: nn.Sequential(
                nn.Linear(self.entity_embedding_size, 4 * self.entity_embedding_size),
                nn.ReLU(inplace=True),
                nn.Linear(4 * self.entity_embedding_size, self.entity_embedding_size),
            )
            for lf in self.logic_functions if lf.input_no == 1
        })

        # Attentions
        # Higher state: objective attend -> entities, theorems and ground truths
        self.higher_entity_attn = nn.Linear(
            self.logic_statement_embedding_size, self.entity_embedding_size
        )
        self.higher_ground_truth_attn = nn.Linear(
            self.logic_statement_embedding_size, self.entity_embedding_size
        )
        self.higher_theorem_attn = nn.Linear(
            self.logic_statement_embedding_size, self.theorem_embedding_size
        )
        # Lower state: objective and theorem attend -> entities, ground truths and sequence
        self.lower_entity_attn = nn.Linear(
            self.logic_statement_embedding_size + self.theorem_embedding_size, self.entity_embedding_size
        )
        self.lower_ground_truth_attn = nn.Linear(
            self.logic_statement_embedding_size + self.theorem_embedding_size, self.entity_embedding_size
        )
        self.lower_sequence_attn = nn.Linear(
            self.logic_statement_embedding_size + self.theorem_embedding_size, self.entity_embedding_size
        )

        # Device to operate on
        self.to(self.device)

    # Low level encodings
    def encode_entity(self, entity):
        if not entity.recent_numerical_function:
            return self.input_entity_embedding(
                self.input_entity_string2index[entity.name].to(self.device)).to(self.device)
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
            else:
                pass

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
        return self.theorem_embedding(self.theorem_string2index[theorem.name]).to(self.device)

    # High level encodings
    def encode_observation(self, observation):
        """

        :param observation: an observation
        :return: the encoded tensor of the observation given
        """
        if (len(observation["ground_truth"]) > self.ground_truth_maxsize) or \
                (len(observation["lemmas"]) > self.theorem_maxsize) or \
                (len(observation["entities"]) > self.entity_maxsize) or \
                (len(observation["objectives"]) > self.objective_maxsize):
            print(len(observation["ground_truth"]), self.ground_truth_maxsize)
            print(len(observation["lemmas"]), self.theorem_maxsize)
            print(len(observation["entities"]), self.entity_maxsize)
            print(len(observation["objectives"]), self.objective_maxsize)
        gt_tensors = self.encode_gt(observation)
        th_tensors = self.encode_theorems(observation)
        ent_tensors = self.encode_entities(observation)
        obj_tensors = self.encode_objectives(observation)
        observation_tensor = torch.cat(gt_tensors + th_tensors + ent_tensors + obj_tensors, dim=0)
        observation_tensor = observation_tensor.view(-1, self.observation_tensor_size)
        return observation_tensor

    def encode_observation_gt(self, observation):
        gt_tensors = [self.encode_logic_statement(gt) for gt in observation["ground_truth"]]
        gt_tensors += [torch.zeros(1, self.entity_embedding_size).to(self.device)] \
                      * (self.ground_truth_maxsize - len(gt_tensors))
        return gt_tensors

    def encode_observation_theorems(self, observation):
        th_tensors = [self.encode_theorem(th) for th in observation["lemmas"]]
        th_tensors += [torch.zeros(1, self.theorem_embedding_size).to(self.device)] \
                      * (self.theorem_maxsize - len(th_tensors))
        return th_tensors

    def encode_observation_entities(self, observation):
        ent_tensors = [self.encode_entity(ent) for ent in observation["entities"]]
        ent_tensors += [torch.zeros(1, self.entity_embedding_size).to(self.device)] \
                       * (self.entity_maxsize - len(ent_tensors))
        return ent_tensors

    def encode_observation_objectives(self, observation):
        obj_tensors = [self.encode_logic_statement(objective) for objective in observation["objectives"]]
        obj_tensors += [torch.zeros(1, self.entity_embedding_size).to(self.device)] \
                       * (self.objective_maxsize - len(obj_tensors))
        return obj_tensors

    def encode_sequence_of_operands(self, sequence):
        entity_tensors = list(self.encode_entity(entity) for entity in sequence)
        entity_tensors += [torch.zeros(1, self.entity_embedding_size).to(self.device)] * \
                          (self.theorem_operand_maxsize - len(sequence))
        return torch.cat(entity_tensors, dim=0)

    @staticmethod
    def concatenate_encodings(encodings):
        encoding_tensor = torch.cat(encodings, dim=0)
        return encoding_tensor

    def higher_attentive_state(self, observation):
        # Attend to objective, represent observation
        gt_tensors = self.encode_observation_gt(observation)
        th_tensors = self.encode_observation_theorems(observation)
        ent_tensors = self.encode_observation_entities(observation)
        obj_tensors = self.encode_observation_objectives(observation)
        ground_truths, theorems, entities, objectives = self.concatenate_encodings(gt_tensors), \
                                                        self.concatenate_encodings(
                                                            th_tensors), self.concatenate_encodings(ent_tensors), \
                                                        self.concatenate_encodings(obj_tensors)
        ground_truth_evaluator = self.higher_ground_truth_attn(objectives)
        ground_truth_weights = F.softmax(torch.matmul(ground_truths, torch.t(ground_truth_evaluator)), dim=0)
        weighted_ground_truth = torch.matmul(torch.t(ground_truth_weights), ground_truths)
        theorem_evaluator = self.higher_theorem_attn(objectives)
        theorem_weights = F.softmax(torch.matmul(theorems, torch.t(theorem_evaluator)), dim=0)
        weighted_theorem = torch.matmul(torch.t(theorem_weights), theorems)
        entity_evaluator = self.higher_entity_attn(objectives)
        entity_weights = F.softmax(torch.matmul(entities, torch.t(entity_evaluator)), dim=0)
        weighted_entity = torch.matmul(torch.t(entity_weights), entities)
        attentive_state = torch.cat([weighted_ground_truth, weighted_theorem, weighted_entity, objectives], dim=-1)
        return attentive_state

    def lower_attentive_state(self, observation, lemma_chosen, sequence_of_entities):
        # Attend to objective and lemma chosen, represent observation
        ground_truth_tensor = self.concatenate_encodings(self.encode_observation_gt(observation))
        entity_tensor = self.concatenate_encodings(self.encode_observation_entities(observation))
        sequence_tensor = self.encode_sequence_of_operands(sequence_of_entities)
        objective_tensor = self.concatenate_encodings(self.encode_observation_objectives(observation))
        lemma_tensor = self.encode_theorem(lemma_chosen)
        attend_to = torch.cat([objective_tensor, lemma_tensor], dim=-1)

        ground_truth_evaluator = self.lower_ground_truth_attn(attend_to)
        ground_truth_weights = F.softmax(torch.matmul(ground_truth_tensor, torch.t(ground_truth_evaluator)), dim=0)
        weighted_ground_truth = torch.matmul(torch.t(ground_truth_weights), ground_truth_tensor)
        entity_evaluator = self.lower_entity_attn(attend_to)
        entity_weights = F.softmax(torch.matmul(entity_tensor, torch.t(entity_evaluator)), dim=0)
        weighted_entity = torch.matmul(torch.t(entity_weights), entity_tensor)
        sequence_evaluator = self.lower_sequence_attn(attend_to)
        sequence_weights = F.softmax(torch.matmul(sequence_tensor, torch.t(sequence_evaluator)), dim=0)
        weighted_sequence = torch.matmul(torch.t(sequence_weights), sequence_tensor)
        attentive_state = torch.cat([weighted_ground_truth, weighted_entity, weighted_sequence, attend_to], dim=-1)
        return attentive_state

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

    def encode_entities(self, entities):
        entity_tensors = list(self.encode_entity(entity) for entity in entities)
        entity_tensors += [torch.zeros(1, self.entity_embedding_size).to(self.device)] * \
                          (self.theorem_operand_maxsize - len(entities))
        return torch.cat(entity_tensors, dim=-1)

    def encode_theorem_space(self, theorem_space):
        return torch.cat([self.encode_theorem(lemma) for lemma in theorem_space], dim=0)

    def encode_entity_space(self, entity_space):
        return torch.cat([self.encode_entity(entity) for entity in entity_space], dim=0)

    def forward(self, x):
        raise NotImplementedError


if __name__ == "__main__":
    encoder = torch.load("../pt_models/logic_recur_nn/logic_recursive_nn.pt")
    simple_proof = pickle.load(open("../data/simple_proof.p", "rb"))
    objectives = pickle.load(open("../data/simple_proof_degree_1_10_objectives.p", "rb"))
    proofs_and_provers = objectives_to_coupled_proofs_and_provers(objectives, encoder, simple_proof)
    proof, prover = proofs_and_provers[0]
    sequence_of_entities = list(proof.initial_entities)
    lemma_chosen = proof.lemmas[0]
    print(encoder.higher_attentive_state(prover.raw_observe()).size())
    print(encoder.lower_attentive_state(prover.raw_observe(), lemma_chosen, sequence_of_entities).size())
