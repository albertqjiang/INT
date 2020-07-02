import torch
from tensorboardX import SummaryWriter

from legacy.helper_functions.QNetwork import GeneralQNetwork

encoder = torch.load("../pt_models/logic_recur_nn/logic_recursive_nn.pt")
higher_action_dim = encoder.theorem_embedding_size
writer = SummaryWriter()
higher_q_net = GeneralQNetwork(state_dim=encoder.higher_attentive_size, action_dim=higher_action_dim)

dummy_input1 = torch.rand(1, 512)
dummy_input2 = torch.rand(1, 128)
writer.add_graph(higher_q_net, [dummy_input1, dummy_input2], True)

writer.close()
