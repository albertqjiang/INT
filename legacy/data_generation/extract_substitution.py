import argparse
import pickle
import torch
import os
from torch_geometric.data import Batch
import torch.nn as nn
import torch.optim as optim
import random
from algos import batch_process, tile_obs_acs, compute_mask, compute_trans_ind


class LinearClassifier(nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(1536, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.fc(x))


def accuracy(prediction, label):
    return ((prediction >= 0.5) == label).sum().item() / label.size(0)


lc = LinearClassifier()
loss_func = nn.BCELoss()
optimizer = optim.SGD(lc.parameters(), lr=3e-4)

parser = argparse.ArgumentParser(description='Extract substitution dataset')

parser.add_argument("--model_path", required=False, type=str,
                    default="/Users/qj213/Downloads/nov15")
parser.add_argument("--test_dirs", required=False, type=str, default="k=3_l=7")
parser.add_argument("--combo_path", required=False, type=str,
                    default="/Users/qj213/Downloads/nov15/backward")
args = parser.parse_args()

model = torch.load(os.path.join(args.model_path, "model_checkpoint.pt"), map_location="cpu")
test_dataset = pickle.load(open(os.path.join(args.combo_path, args.test_dirs, "train.pkl"), "rb"))

model.eval()
with torch.no_grad():
    all_x = list()
    all_y = list()
    datapoints = [point for point in test_dataset.io_tuples if point[1].name == "EquivalenceSubstitution"]
    batch_states, batch_actions, batch_name_actions = batch_process(datapoints)

    print(batch_states, batch_actions)

    (obj_gnns, obj_gnns_ind, obj_batch_gnn_ind, obj_node_ent, obj_trans_ind,
     gt_gnns, gt_gnns_ind, gt_batch_gnn_ind, gt_node_ent, gt_trans_ind,
     batch_ind, actions, entity_actions, prev_max_ents) = tile_obs_acs(
        batch_states, batch_actions, sl_train=True)

    lemma_actions = torch.LongTensor(actions[:, 0])
    entity_actions = torch.LongTensor(entity_actions)

    batch_obj_state = Batch.from_data_list(obj_gnns)
    obj_state_tensor, obj_out = model.obj_encoder(
        batch_obj_state, obj_gnns_ind, obj_batch_gnn_ind)
    if len(gt_gnns) > 0:
        batch_gt_state = Batch.from_data_list(gt_gnns)
        if model.attention_type is 1:
            gt_state_tensor, gt_out = model.gt_encoder(
                batch_gt_state, gt_gnns_ind, gt_batch_gnn_ind, obj_out)
        else:
            gt_state_tensor, gt_out = model.gt_encoder(
                batch_gt_state, gt_gnns_ind, gt_batch_gnn_ind)
        state_tensor = torch.cat((obj_state_tensor, gt_state_tensor), 0)
    else:
        gt_out = torch.zeros_like(obj_out)
        state_tensor = obj_state_tensor

    ent_mask = compute_mask(obj_node_ent, gt_node_ent)
    rev_trans_ind = compute_trans_ind(obj_trans_ind, gt_trans_ind)
    ent_rep = torch.index_select(state_tensor[ent_mask, :].clone(), 0, rev_trans_ind)
    total_ents = ent_rep.size(0)

    # Randomly roll an entity for each second operand in the substitutions
    random_entity_addition = torch.randint(1, total_ents, (entity_actions.size(0), 1)).squeeze()
    random_entity_actions = entity_actions.clone().detach()
    random_entity_actions[:, 1] = torch.fmod(entity_actions[:, 1] + random_entity_addition, total_ents)
    # print(entity_actions)
    # print(random_entity_actions)

    mask = (entity_actions[:, 0] != -1)
    context = torch.index_select(ent_rep, 0, entity_actions[:, 0][mask])
    print(context.size(), context)

    mask = (entity_actions[:, 1] != -1)
    cur_ent_rep = torch.index_select(ent_rep, 0, entity_actions[:, 1][mask])
    print(cur_ent_rep.size(), cur_ent_rep)
    cur_random_ent_rep = torch.index_select(ent_rep, 0, random_entity_actions[:, 1][mask])
    print(cur_random_ent_rep.size(), cur_random_ent_rep)

    for i in range(context.size(0)):
        coin = random.choice([0, 1])
        if coin == 0:
            pair = (
                torch.cat([context[i], cur_ent_rep[i], cur_random_ent_rep[i]], dim=-1).unsqueeze(0), coin
            )
        elif coin == 1:
            pair = (
                torch.cat([context[i], cur_random_ent_rep[i], cur_ent_rep[i]], dim=-1).unsqueeze(0), coin
            )
        else:
            raise AssertionError

        all_x.append(pair[0])
        all_y.append(pair[1])

    X = torch.cat(all_x, dim=0)
    y = torch.FloatTensor(all_y).unsqueeze(1)

    print(X.size(), y.size())

X_train = X[:int(0.9 * X.size(0))]
X_test = X[int(0.9 * X.size(0)):]
y_train = y[:int(0.9 * y.size(0))]
y_test = y[int(0.9 * y.size(0)):]
for i in range(1000):
    predict = lc(X_train)
    loss = loss_func(predict, y_train)
    print("Epoch: ", i)
    print("Loss: ", loss.item())
    print("Train accuracy: ", accuracy(predict, y_train))
    print("Test accuracy: ", accuracy(lc(X_test), y_test))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
