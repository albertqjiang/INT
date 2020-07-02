import torch
import torch.nn as nn

from logic.logic import Entity, NumericalFunction, LogicFunction


class RecurrentEncoder(nn.Module):
    def __init__(self, independent_variable_maxsize=2, numerical_functions=None, logic_functions=None,
                 num_embedding=20, embedding_dim=128):
        super(RecurrentEncoder, self).__init__()

        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim

        # Define default numerical and logic functions
        if not numerical_functions:
            # Numerical functions
            add = NumericalFunction("add", 2)
            sub = NumericalFunction("sub", 2)
            mul = NumericalFunction("mul", 2)
            sqr = NumericalFunction("sqr", 1)
            inv = NumericalFunction("inv", 1)
            self.numerical_functions = [add, sub, mul, sqr, inv]
        if not logic_functions:
            # Logic functions
            BOE = LogicFunction("BiggerOrEqual", input_no=2)
            Equal = LogicFunction("Equal", input_no=2)
            NonNeg = LogicFunction("Not negative", input_no=1)
            Real = LogicFunction("Real", input_no=1)
            self.logic_functions = [BOE, Equal, NonNeg, Real]

        # Device to operate on
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # String to index and embedding
        self.input_entity_names = ["input{}".format(i) for i in range(1, 1 + independent_variable_maxsize)]
        self.special_strings = ["(", ")", ",", "<SOS>", "<EOS>"]
        self.string2embed = [nf.name for nf in self.numerical_functions] + [lf.name for lf in self.logic_functions] + \
                            self.input_entity_names + self.special_strings
        self.string2index = {self.string2embed[i]: torch.LongTensor([i]).to(self.device)
                             for i in range(len(self.string2embed))}

        self.embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim)

        self.to(self.device)

    def forward(self, gru_input, gru_hidden):
        gru_input_index = self.string2index[gru_input]
        embedded = self.embedding(gru_input_index).view(1, 1, -1)
        gru_output = embedded
        gru_output, gru_hidden = self.gru(gru_output, gru_hidden)
        return gru_output, gru_hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.embedding_dim, device=self.device)

    def encode_entity(self, entity):
        entity_chars = entity.name.split()
        hidden = self.init_hidden()
        for character in entity_chars:
            output, hidden = self.forward(character, hidden)
            if character == "EOS":
                break
        return output

    def test_encoding(self):
        entity1 = Entity("input1")
        entity2 = Entity("input2")
        entity3 = self.numerical_functions[0].execute_nf([entity1, entity2])
        return self.encode_entity(entity3)


if __name__ == "__main__":
    recurrent_encoder = RecurrentEncoder()
    print(recurrent_encoder.test_encoding().size())
