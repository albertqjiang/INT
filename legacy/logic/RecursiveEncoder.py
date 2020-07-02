import torch
import torch.nn as nn

from logic.utils import standard_numerical_functions, standard_logic_functions


class RecursiveEncoder(nn.Module):
    def __init__(self, prover, numerical_functions=None, logic_functions=None,
                 entity_and_theorem_embedding_size=128):
        super(RecursiveEncoder, self).__init__()

        # Define default numerical and logic functions
        if not numerical_functions:
            numerical_functions = [nf for nf in standard_numerical_functions.values()]
        if not logic_functions:
            # Logic functions
            logic_functions = [lf for lf in standard_logic_functions.values()]

        # Device to operate on
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dimensions of the spaces used
        # Entity(E): R^128
        # Theorem(THEOREM): R^8
        # Logic statements(LG): (-1, 1)^128
        # A_TENSORIZED = (-1, 1)^128

        # Getting parameters from prover and other inputs
        self.prover = prover
        self.theorems = prover.proof.lemmas
        self.numerical_functions = numerical_functions
        self.logic_functions = logic_functions
        self.input_entities = prover.proof.entities
        self.entity_embedding_size = entity_and_theorem_embedding_size
        self.theorem_embedding_size = entity_and_theorem_embedding_size

        # Maximum sizes of properties to fix dimensions
        self.ground_truth_maxsize = prover.gt_maxsize
        self.entity_maxsize = prover.ent_maxsize
        self.theorem_maxsize = prover.lemma_maxsize
        self.theorem_operand_maxsize = prover.lemma_operand_maxsize
        self.objective_maxsize = prover.obj_maxsize
        self.observation_tensor_size = \
            entity_and_theorem_embedding_size * \
            (self.ground_truth_maxsize + self.entity_maxsize + self.theorem_maxsize + self.objective_maxsize)

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
                nn.Linear(2 * self.entity_embedding_size, 4 * self.entity_embedding_size),
                nn.ReLU(),
                nn.Linear(4 * self.entity_embedding_size, 2 * self.entity_embedding_size),
                nn.ReLU(),
                nn.Linear(2 * self.entity_embedding_size, self.entity_embedding_size),
            )
            for nf in self.numerical_functions if nf.input_no == 2
        })
        self.nf_one2one_layers = nn.ModuleDict({
            nf.name: nn.Sequential(
                nn.Linear(self.entity_embedding_size, 4 * self.entity_embedding_size),
                nn.ReLU(),
                nn.Linear(4 * self.entity_embedding_size, 2 * self.entity_embedding_size),
                nn.ReLU(),
                nn.Linear(2 * self.entity_embedding_size, self.entity_embedding_size),
            )
            for nf in self.numerical_functions if nf.input_no == 1
        })

        # Logic function layers: E^n -> LG
        self.lf_two2one_layers = nn.ModuleDict({
            lf.name: nn.Sequential(
                nn.Linear(2 * self.entity_embedding_size, 4 * self.entity_embedding_size),
                nn.ReLU(),
                nn.Linear(4 * self.entity_embedding_size, self.entity_embedding_size),
            )
            for lf in self.logic_functions if lf.input_no == 2
        })
        self.lf_one2one_layers = nn.ModuleDict({
            lf.name: nn.Sequential(
                nn.Linear(self.entity_embedding_size, 4 * self.entity_embedding_size),
                nn.ReLU(),
                nn.Linear(4 * self.entity_embedding_size, self.entity_embedding_size),
            )
            for lf in self.logic_functions if lf.input_no == 1
        })

        # Attentions
        self.objective_entity_attn = nn.Linear(
            self.entity_embedding_size, self.entity_embedding_size
        )

        self.to(self.device)

    # Low level encodings
    def encode_entity(self, entity):
        if not entity.recent_numerical_function:
            return self.input_entity_embedding(self.input_entity_string2index[entity.name]).to(self.device)
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

    @staticmethod
    def concatenate_encodings(encodings):
        encoding_tensor = torch.cat(encodings, dim=0)
        encoding_tensor = encoding_tensor.view(1, -1)
        return encoding_tensor

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

    def forward(self, x):
        raise NotImplementedError
