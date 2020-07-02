import torch
import torch.nn as nn
import torch.nn.functional as F

import collections
import os
import sys

sys.path.insert(0, os.path.abspath('../../..'))
from logic.utils import standard_numerical_functions, standard_logic_functions
from legacy.logic_math import complete_theorems as standard_theorems
from legacy.helper_functions.utils import load_config


class ExecutionQueue:
    def __init__(self, name, batch_encoder):
        self.name = name
        self.execution_queue = dict()
        self.unique_items = dict()
        self.batch_encoder = batch_encoder

    def enqueue(self, batch_action):
        for degree, same_degree_actions in batch_action.items():
            for action in same_degree_actions:
                specific_type = action['method'] + action['type']
                if degree not in self.execution_queue.keys():
                    self.execution_queue[degree] = dict()
                if specific_type not in self.execution_queue[degree].keys():
                    self.execution_queue[degree][specific_type] = list()
                self.execution_queue[degree][specific_type].append(action)

    def dequeue(self):
        raise NotImplementedError

    def execute_element(self, element):
        raise NotImplementedError

    def execute_queue(self):
        ordered_dict = collections.OrderedDict(sorted(self.execution_queue.items()))
        for degree, same_degree_actions in ordered_dict.items():
            for specific_type, batch_actions in same_degree_actions.items():
                if specific_type == 'embed_entity':
                    entity_strings = [datapoint['operands'] for datapoint in batch_actions]
                    uids = [datapoint['uid'] for datapoint in batch_actions]
                    embeddings = self.batch_encoder.embed_entity(entity_strings)
                    for uid, embedding in zip(uids, embeddings):
                        self.unique_items[uid] = embedding
                elif specific_type == "embed_theorem":
                    theorem_strings = [datapoint['operands'] for datapoint in batch_actions]
                    uids = [datapoint['uid'] for datapoint in batch_actions]
                    embeddings = self.batch_encoder.embed_theorem(theorem_strings)
                    for uid, embedding in zip(uids, embeddings):
                        self.unique_items[uid] = embedding
                elif specific_type.startswith('operation'):
                    concatenations = list()
                    uids = list()
                    for datapoint in batch_actions:
                        concatenated = torch.cat(
                            list(map(self.unique_items.get, datapoint['operands'])),
                            dim=0,
                        )
                        concatenated = concatenated.unsqueeze(0)
                        concatenations.append(concatenated)
                        uids.append(datapoint['uid'])
                    concatenations = torch.cat(concatenations, dim=0)

                    operation_stripped = specific_type.replace('operation', '')
                    if operation_stripped in self.batch_encoder.nf_one2one_layers.keys():
                        outputs = self.batch_encoder.nf_one2one_layers[operation_stripped].forward(concatenations)
                    elif operation_stripped in self.batch_encoder.nf_two2one_layers.keys():
                        outputs = self.batch_encoder.nf_two2one_layers[operation_stripped].forward(concatenations)
                    elif operation_stripped in self.batch_encoder.lf_one2one_layers.keys():
                        outputs = self.batch_encoder.lf_one2one_layers[operation_stripped].forward(concatenations)
                    elif operation_stripped in self.batch_encoder.lf_two2one_layers.keys():
                        outputs = self.batch_encoder.lf_two2one_layers[operation_stripped].forward(concatenations)
                    else:
                        print(specific_type)
                        raise NotImplementedError

                    for uid, output in zip(uids, outputs):
                        self.unique_items[uid] = output
                else:
                    raise NotImplementedError

        return self.unique_items


class BatchEncoder(nn.Module):
    """
    This version of the batch encoder doesn't remove redundant expressions.
    The batch encoder adopts folding to accelerate the representation and optimization of
    the encoder graph neural network by batching similar operations together.
    Similar methods like tf.fold and torchfold were developed before but couldn't be
    adopted here for various reasons.
    """

    def __init__(self, configuration, numerical_functions=None, logic_functions=None, theorems=None,
                 entity_embedding_size=128, logic_statement_embedding_size=128, theorem_embedding_size=128, cuda=True):
        super(BatchEncoder, self).__init__()
        # Stuff useful for batching
        self.uid = 0
        self.encoding_by_degree = dict()

        # Stuff useful for encoding
        if cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # Loading some stuff from configuration
        self.entity_maxsize = configuration["entity_maxsize"]
        self.entity_embedding_capacity = configuration["entity_embedding_capacity"]
        self.ground_truth_maxsize = configuration["ground_truth_maxsize"]
        self.theorem_maxsize = configuration["theorem_maxsize"]
        self.objective_maxsize = configuration["objective_maxsize"]
        self.theorem_operand_maxsize = configuration["theorem_operand_maxsize"]

        self.entity_embedding_size = entity_embedding_size
        self.logic_statement_embedding_size = logic_statement_embedding_size
        self.theorem_embedding_size = theorem_embedding_size
        self.higher_attentive_size = \
            2 * self.logic_statement_embedding_size + self.entity_embedding_size + self.theorem_embedding_size
        self.lower_attentive_size = \
            self.higher_attentive_size + self.entity_embedding_size + self.theorem_embedding_size

        # Getting numerical functions, logic functions and theorems available initiated
        if not numerical_functions:
            self.numerical_functions = standard_numerical_functions
        else:
            self.numerical_functions = numerical_functions
        if not logic_functions:
            self.logic_functions = standard_logic_functions
        else:
            self.logic_functions = logic_functions
        if not theorems:
            self.theorems = standard_theorems
        else:
            self.theorems = theorems

        # Numerical function layers: E^n -> E
        self.nf_one2one_layers = nn.ModuleDict({
            nf.name: nn.Sequential(
                nn.Linear(self.entity_embedding_size, 2 * self.entity_embedding_size),
                nn.ReLU(inplace=True),
                nn.Linear(2 * self.entity_embedding_size, self.entity_embedding_size),
            )
            for nf in list(self.numerical_functions.values()) if nf.input_no == 1
        })
        self.nf_two2one_layers = nn.ModuleDict({
            nf.name: nn.Sequential(
                nn.Linear(2 * self.entity_embedding_size, 2 * self.entity_embedding_size),
                nn.ReLU(inplace=True),
                nn.Linear(2 * self.entity_embedding_size, self.entity_embedding_size),
            )
            for nf in list(self.numerical_functions.values()) if nf.input_no == 2
        })

        # Logic function layers: E^n -> LG
        self.lf_two2one_layers = nn.ModuleDict({
            lf.name: nn.Sequential(
                nn.Linear(2 * self.entity_embedding_size, 2 * self.entity_embedding_size),
                nn.ReLU(inplace=True),
                nn.Linear(2 * self.entity_embedding_size, self.entity_embedding_size),
            )
            for lf in self.logic_functions.values() if lf.input_no == 2
        })
        self.lf_one2one_layers = nn.ModuleDict({
            lf.name: nn.Sequential(
                nn.Linear(self.entity_embedding_size, 2 * self.entity_embedding_size),
                nn.ReLU(inplace=True),
                nn.Linear(2 * self.entity_embedding_size, self.entity_embedding_size),
            )
            for lf in self.logic_functions.values() if lf.input_no == 1
        })
        if "entity_strings" in configuration.keys():
            self.entity_string2index = {
                entity_string: torch.LongTensor([ind]).to(self.device)
                for ind, entity_string in enumerate(configuration["entity_strings"])
            }
        else:
            self.entity_string2index = {
                entity_string: torch.LongTensor([ind]).to(self.device)
                for ind, entity_string in enumerate(["input1", "input2", "input3", "0", "1", "2"])
            }
        if "theorem_strings" in configuration.keys():
            self.theorem_string2index = {
                theorem_string: torch.LongTensor([ind]).to(self.device)
                for ind, theorem_string in enumerate(configuration["theorem_strings"])
            }
        else:
            theorem_names = [theorem.name for theorem in list(standard_theorems.values())]
            self.theorem_string2index = {
                theorem_string: torch.LongTensor([ind]).to(self.device)
                for ind, theorem_string in enumerate(theorem_names)
            }

        # Attention weights
        self.ground_truth_aam = nn.Linear(self.logic_statement_embedding_size * 2, 1)
        self.entity_aam = nn.Linear(self.entity_embedding_size + self.logic_statement_embedding_size, 1)
        self.theorem_aam = nn.Linear(self.theorem_embedding_size + self.logic_statement_embedding_size, 1)

        self.to(self.device)

    def batch_encode(self, batch, value_type):
        if value_type in ["entity", "logic_statement", "theorem"]:
            # If the batch is empty, return a zero tensor as placeholder
            if not batch:
                if value_type == "entity":
                    return torch.zeros(1, self.entity_embedding_size, device=self.device)
                elif value_type == "logic_statement":
                    return torch.zeros(1, self.logic_statement_embedding_size, device=self.device)
                elif value_type == "theorem":
                    return torch.zeros(1, self.theorem_embedding_size, device=self.device)
                else:
                    raise ValueError
            else:
                eq = ExecutionQueue(name="Batch queue", batch_encoder=self)
                uid_list = list()
                for element in batch:
                    self.generate_action(element, value_type)
                    uid_list.append(self.encoding_by_degree[max(self.encoding_by_degree.keys())][-1]['uid'])
                    eq.enqueue(self.encoding_by_degree)
                    self.encoding_by_degree.clear()
                uni_items = eq.execute_queue()
                return torch.cat([uni_items[key].unsqueeze(0) for key in uid_list], dim=0)
        elif value_type == "observation":
            # Encode observations one by one, the power of batching is only achieved in each individual observation
            # This is not the best way to do it, and should be replaced in the future
            observations = list()
            for observation in batch:
                observations.append(self.encode_single_observation(observation=observation))
            return torch.cat(observations, dim=0)
        else:
            raise NotImplementedError

    def generate_action(self, raw_data, value_type):
        if value_type == "entity":
            self.encode_entity(raw_data)
        elif value_type == "logic_statement":
            self.encode_logic_statement(raw_data)
        elif value_type == "theorem":
            self.encode_theorem(raw_data)
        else:
            raise NotImplementedError

    def encode_single_observation(self, observation):
        """
        Encode a single observation, this is temporary and should be deprecated in the future
        :param observation:
        :return: encoded observation tensor: 1 * observation tensor size, arranged in the order of
        entities, ground_truth, theorems, objectives
        """
        gt_tensors = self.batch_encode(observation["ground_truth"], "logic_statement")
        th_tensors = self.batch_encode(observation["lemmas"], "theorem")
        ent_tensors = self.batch_encode(observation["entities"], "entity")
        obj_tensors = self.batch_encode(observation["objectives"], "logic_statement")
        weighted_gt = self.attend_to(objective_tensor=obj_tensors,
                                     input_tensor=gt_tensors, attention_type="ground_truth")
        weighted_th = self.attend_to(objective_tensor=obj_tensors,
                                     input_tensor=th_tensors, attention_type="theorem")
        weighted_ent = self.attend_to(objective_tensor=obj_tensors,
                                      input_tensor=ent_tensors, attention_type="entity")
        observation_tensor = torch.cat([weighted_ent, weighted_gt, weighted_th, obj_tensors], dim=1)
        return observation_tensor

    def encode_entity(self, entity):
        if not entity.recent_numerical_function:
            action = {
                "method": "embed_entity",
                "type": "",
                "uid": self.uid,
                "operands": entity.to_string()
            }
            self.uid += 1
            if 0 not in self.encoding_by_degree.keys():
                self.encoding_by_degree[0] = list()
            self.encoding_by_degree[0].append(action)
            return {"action": action, "degree": 0}
        else:
            operand_encodings = [self.encode_entity(operand) for operand in entity.rnc_operands]
            action = {
                "method": "operation",
                "type": entity.recent_numerical_function.to_string(),
                "uid": self.uid,
                "operands": [encoding["action"]["uid"] for encoding in operand_encodings]
            }
            self.uid += 1
            degree = max([encoding["degree"] for encoding in operand_encodings]) + 1
            if degree not in self.encoding_by_degree.keys():
                self.encoding_by_degree[degree] = list()
            self.encoding_by_degree[degree].append(action)
            return {"action": action, "degree": degree}

    def encode_logic_statement(self, logic_statement):
        operand_encodings = [self.encode_entity(operand) for operand in logic_statement.entities]
        action = {
            "method": "operation",
            "type": logic_statement.logic_function.to_string(),
            "uid": self.uid,
            "operands": [encoding["action"]["uid"] for encoding in operand_encodings]
        }
        self.uid += 1
        degree = max([encoding["degree"] for encoding in operand_encodings]) + 1
        if degree not in self.encoding_by_degree.keys():
            self.encoding_by_degree[degree] = list()
        self.encoding_by_degree[degree].append(action)
        return {"action": action, "degree": degree}

    def encode_theorem(self, theorem):
        action = {
            "method": "embed_theorem",
            "type": "",
            "uid": self.uid,
            "operands": theorem.name
        }
        self.uid += 1
        if 0 not in self.encoding_by_degree.keys():
            self.encoding_by_degree[0] = list()
        self.encoding_by_degree[0].append(action)
        return {"action": action, "degree": 0}

    # Actual encoding instead of scheduling
    def embed_entity(self, entity_strings):
        index_tensors = list(map(self.entity_string2index.get, entity_strings))
        index_tensors = torch.cat(index_tensors, dim=0).unsqueeze(1)
        out = torch.zeros(len(entity_strings), self.entity_embedding_size, device=self.device)
        return out.scatter_(1, index_tensors, 1)

    def embed_theorem(self, theorem_strings):
        index_tensors = list(map(self.theorem_string2index.get, theorem_strings))
        index_tensors = torch.cat(index_tensors, dim=0).unsqueeze(1)
        out = torch.zeros(len(theorem_strings), self.entity_embedding_size, device=self.device)
        return out.scatter_(1, index_tensors, 1)

    def forward(self, batch, batch_type):
        raise NotImplementedError

    def attend_to(self, objective_tensor, input_tensor, attention_type):
        """
        The objective tends to a certain type of inputs
        :param objective_tensor: 1 * logic_statement_embedding_size
        :param input_tensor: N * input_size
        :param attention_type: specify the type of the input
        :return: weighted_tensor: 1 * input_size
        """
        objective_tensor_repeated = objective_tensor.repeat(input_tensor.size(0), 1)
        concat = torch.cat([objective_tensor_repeated, input_tensor], dim=1)
        if attention_type == "ground_truth":
            attention_weights = self.ground_truth_aam(concat)
        elif attention_type == "entity":
            attention_weights = self.entity_aam(concat)
        elif attention_type == "theorem":
            attention_weights = self.theorem_aam(concat)
        else:
            raise NotImplementedError

        attention_weights = F.softmax(attention_weights, dim=0)
        weighted_input = torch.mm(torch.t(attention_weights), input_tensor)
        return weighted_input

    def higher_attentive_state(self, observation):
        return self.encode_single_observation(observation=observation)

    def lower_attentive_state(self, observation, lemma_chosen, sequence_of_entities):
        objective_tensor = self.batch_encode(observation["objectives"], "logic_statement")
        observation_tensor = self.encode_single_observation(observation=observation)
        lemma_tensor = self.embed_theorem(theorem_strings=[lemma_chosen.name])
        sequence_tensor = self.batch_encode(batch=sequence_of_entities, value_type="entity")
        weighted_sequence_tensor = self.attend_to(objective_tensor=objective_tensor,
                                                  input_tensor=sequence_tensor,
                                                  attention_type="entity")
        return torch.cat([observation_tensor, lemma_tensor, weighted_sequence_tensor], dim=1)

    def encode_theorem_space(self, theorem_space):
        return self.batch_encode(batch=theorem_space, value_type="theorem")

    def encode_entity_space(self, entity_space):
        return self.batch_encode(batch=entity_space, value_type="entity")


if __name__ == "__main__":
    from logic.logic import Entity

    import time
    import random

    criterion = nn.MSELoss()
    config = load_config(config_path='../rl_algos/config.json')["batch_encode"]
    batch_encoder = BatchEncoder(configuration=config)
    single_batch_length = 32
    y_bar = torch.rand(single_batch_length, 128)

    # Testing encoding entities
    input1 = Entity("input1")
    input2 = Entity("input2")
    input3 = standard_numerical_functions["add"].execute_nf([input1, input2])
    input4 = standard_numerical_functions["mul"].execute_nf([input3, input1])
    input5 = standard_numerical_functions["sub"].execute_nf([input4, input1])
    input6 = standard_numerical_functions["opp"].execute_nf([input4])
    single_batch = [input3, input4, input5, input6] * int(single_batch_length / 4)
    random.shuffle(single_batch)

    starting_time = time.time()
    y = batch_encoder.batch_encode(batch=single_batch, value_type="entity")
    loss = criterion(y, y_bar)
    print(loss.item())
    loss.backward()
    time1 = time.time() - starting_time
    print("Time taken: ", time1)

    starting_time = time.time()
    encoded_list = list()
    for element in single_batch:
        y = batch_encoder.batch_encode(batch=[element], value_type="entity")
        encoded_list.append(y)
    batch_tensor = torch.cat(encoded_list, dim=0)
    loss = criterion(batch_tensor, y_bar)
    loss.backward()
    print(loss.item())
    time2 = time.time() - starting_time
    print("Time taken: ", time2)
    print("Speed up: ", time2 / time1)

    # Testing encoding logic statements
    ls1 = standard_logic_functions["Real"].execute_lf([input1])
    ls2 = standard_logic_functions["BiggerOrEqual"].execute_lf([input2, input6])
    ls3 = standard_logic_functions["SmallerOrEqual"].execute_lf([input3, input4])
    ls4 = standard_logic_functions["Equivalent"].execute_lf([input1, input4])
    single_batch = [ls1, ls2, ls3, ls4] * int(single_batch_length / 4)
    random.shuffle(single_batch)

    starting_time = time.time()
    y = batch_encoder.batch_encode(batch=single_batch, value_type="logic_statement")
    loss = criterion(y, y_bar)
    print(loss.item())
    loss.backward()
    time1 = time.time() - starting_time
    print("Time taken: ", time1)

    starting_time = time.time()
    encoded_list = list()
    for element in single_batch:
        y = batch_encoder.batch_encode(batch=[element], value_type="logic_statement")
        encoded_list.append(y)
    batch_tensor = torch.cat(encoded_list, dim=0)
    loss = criterion(batch_tensor, y_bar)
    loss.backward()
    print(loss.item())
    time2 = time.time() - starting_time
    print("Time taken: ", time2)
    print("Speed up: ", time2 / time1)

    # Testing encoding theorems
    single_batch = list(standard_theorems.values())[:4] * int(single_batch_length / 4)
    random.shuffle(single_batch)
    starting_time = time.time()
    y = batch_encoder.batch_encode(batch=single_batch, value_type="theorem")
    print(y)
    time1 = time.time() - starting_time
    print("Time taken: ", time1)

    starting_time = time.time()
    encoded_list = list()
    for element in single_batch:
        y = batch_encoder.batch_encode(batch=[element], value_type="theorem")
        encoded_list.append(y)
    batch_tensor = torch.cat(encoded_list, dim=0)
    print(batch_tensor)
    time2 = time.time() - starting_time
    print("Time taken: ", time2)
    print("Speed up: ", time2 / time1)
