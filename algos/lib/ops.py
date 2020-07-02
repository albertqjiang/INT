import torch
from torch_scatter import scatter_add, scatter_max
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def one_hot(batch, depth):
    batch_cpu = batch.cpu()
    ones = torch.sparse.torch.eye(depth)
    return ones.index_select(0, batch_cpu).to(device)


def graph_softmax(score, ind):
    max_score = torch.index_select(
        scatter_max(score, ind)[0], 0, ind)
    score_exp = torch.exp(score - max_score)
    softmax_denom = torch.index_select(
        scatter_add(score_exp, ind), 0, ind)
    attention = score_exp / softmax_denom
    return attention


def turn_grad_on_off(model, name, on_off):
    for param_name, param in model.named_parameters():
        if param_name.startswith(name):
            if on_off == "on":
                param.requires_grad = True
            elif on_off == "off":
                param.requires_grad = False
            else:
                raise NotImplementedError


def flatten(observations):
    flatten_obs = []
    for i in range(len(observations)):
        flatten_obs.extend(observations[i])
    observations = flatten_obs
    return observations