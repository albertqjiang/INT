import torch


class OneHotEncoder:
    def __init__(self, prover):
        # Device to operate on
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.prover = prover
        self.entities = prover.proof.entities
        self.theorems = prover.proof.lemmas
        self.entity_string2index = {
            ent.name: torch.zeros(len(self.entities)).scatter_(0, torch.LongTensor([ind]), torch.FloatTensor([1])).
                to(self.device) for ind, ent in enumerate(self.entities)
        }
        self.theorem_string2index = {
            ent.name: torch.zeros(len(self.theorems)).scatter_(0, torch.LongTensor([ind]), torch.FloatTensor([1])).
                to(self.device) for ind, ent in enumerate(self.theorems)
        }

    def encode_entity(self, entity):
        return self.entity_string2index[entity.name]

    def encode_theorem(self, theorem):
        return self.theorem_string2index[theorem.name]
