import os
import sys

sys.path.insert(0, os.path.abspath('../../Inequality'))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from algos.model.gnns import TransGATEncoder, GraphEncoder, GraphTransformingEncoder, GraphIsomorphismEncoder,\
    FCResBlock, RawGATEncoder
from algos.lib.ops import one_hot, graph_softmax
from algos.lib.obs import tile_obs_acs, thm2index, compute_mask, compute_trans_ind, theorem_no_input, index2thm, \
    thm_index2no_input, convert_obs_to_dict
from torch_geometric.data import Batch
from torch.distributions import Categorical, Uniform
from torch_scatter import scatter_add, scatter_max
import collections

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = torch.cuda.is_available()
torch.manual_seed(0)
if cuda:
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(0)
np.random.seed(0)


# device = torch.device("cpu")
# cuda = False


class GroundTruthEncoder(torch.nn.Module):
    def __init__(self, num_in, num_out, conv_dim=64, gnn_type="GAT", hidden_layers=1, norm=None,
                 **options):
        super(GroundTruthEncoder, self).__init__()

        self.gnn_type = gnn_type

        if gnn_type == "GCN":
            self.graph_encoder = GraphEncoder(
                num_in, num_out, conv_dim=conv_dim, hidden_layers=hidden_layers, norm=norm)
        elif gnn_type == "GAT":
            self.graph_encoder = GraphTransformingEncoder(
                num_in, num_out, conv_dim=conv_dim, hidden_layers=hidden_layers, norm=norm)
        elif gnn_type == "GIN":
            self.graph_encoder = GraphIsomorphismEncoder(
                num_in, num_out, nn_dim=conv_dim, hidden_layers=hidden_layers, norm=norm)
        elif gnn_type == "GICN":
            self.graph_encoder1 = GraphIsomorphismEncoder(
                num_in, num_out, nn_dim=conv_dim, hidden_layers=hidden_layers, norm=norm)
            self.graph_encoder2 = GraphEncoder(
                num_in, num_out, conv_dim=conv_dim, hidden_layers=hidden_layers, norm=norm)
        elif gnn_type == "TransGAT":
            self.graph_encoder = TransGATEncoder(input_dim=num_in, hidden_dim=num_out, inception=hidden_layers,
                                                 heads=options["attention_heads"],
                                                 gat_dropout_rate=options["gat_dropout_rate"],
                                                 dropout_rate=options["dropout_rate"])
        elif gnn_type == "RawGAT":
            self.graph_encoder = RawGATEncoder(num_in=num_in, hidden_layers=hidden_layers,
                                               num_out=num_out, heads=options["attention_heads"])
        else:
            raise NotImplementedError
        self.to(device)

    def forward(self, data, gnn_ind, batch_gnn_ind=None, attention_keys=None):

        if self.gnn_type == "GICN":
            state_tensor1 = self.graph_encoder1(data)
            state_tensor2 = self.graph_encoder2(data)
            state_tensor = state_tensor1 + state_tensor2
        else:
            state_tensor = self.graph_encoder(data)

        out = scatter_add(state_tensor, torch.LongTensor(gnn_ind).to(device), 0)

        if attention_keys is not None:
            batch_gnn_ind = torch.LongTensor(batch_gnn_ind).to(device)
            key = torch.index_select(attention_keys, 0, batch_gnn_ind)
            score = torch.sum(key * out, 1)
            attention = graph_softmax(score, batch_gnn_ind)
            out = out * attention.view(-1, 1)
        if isinstance(batch_gnn_ind, list):
            batch_gnn_ind = torch.LongTensor(batch_gnn_ind).to(device)
        out = scatter_add(out, batch_gnn_ind, 0)

        return state_tensor, out


class ThmNet(torch.nn.Module):

    def __init__(self, **options):
        super().__init__()
        num_nodes = options["num_nodes"]
        num_lemmas = options["num_lemmas"]
        hidden_dim = options["hidden_dim"]
        gnn_type = options["gnn_type"]
        combined_gt_obj = options["combined_gt_obj"]
        attention_type = options["attention_type"]
        hidden_layers = options["hidden_layers"]
        self.entity_cost = options["entity_cost"]
        self.lemma_cost = options["lemma_cost"]
        norm = options["norm"]

        self.attention_type = attention_type
        assert attention_type is not None
        # if attention_type == 2:
        #     self.gt_encoder = GroundTruthEncoder(num_nodes + 4, hidden_dim, hidden_dim,
        #                                          gnn_type=gnn_type, hidden_layers=hidden_layers, norm=norm,
        #                                          prop_attention=True)
        #     combined_gt_obj = False
        # else:
        self.gt_encoder = GroundTruthEncoder(num_nodes + 4, hidden_dim, hidden_dim,
                                             # gnn_type=gnn_type, hidden_layers=hidden_layers, norm=norm,
                                             **options)
        self.combined_gt_obj = combined_gt_obj
        if not combined_gt_obj:
            self.obj_encoder = GroundTruthEncoder(num_nodes + 4, hidden_dim, hidden_dim,
                                                  # gnn_type=gnn_type, hidden_layers=hidden_layers, norm=norm,
                                                  **options)
        else:
            self.obj_encoder = self.gt_encoder
        self.lemma_encoder = nn.Linear(num_lemmas, 2*hidden_dim)
        self.key_transform = nn.Linear(2*hidden_dim, hidden_dim, bias=False)
        self.ent_transform = nn.Linear(hidden_dim, 2*hidden_dim, bias=False)
        self.gt_ent_transform = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.obj_ent_transform = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # self.lemma_q = nn.Sequential(
        #     nn.Linear(2*hidden_dim, hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, num_lemmas))
        self.lemma_q = FCResBlock(2*hidden_dim)
        self.lemma_out = nn.Linear(2*hidden_dim, num_lemmas)
        self.vf_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.entity_q = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))
        self.softmax = nn.Softmax(dim=1)
        self.to(device)

    def vf(self, observations):
        actions, sl_train = None, False

        (obj_gnns, obj_gnns_ind, obj_batch_gnn_ind, obj_node_ent, obj_trans_ind,
         gt_gnns, gt_gnns_ind, gt_batch_gnn_ind, gt_node_ent, gt_trans_ind,
         batch_ind, actions, entity_actions, prev_max_ents) = tile_obs_acs(
            observations, actions, sl_train)

        # Process inputs with GNNs
        batch_obj_state = Batch.from_data_list(obj_gnns)
        obj_state_tensor, obj_out = self.obj_encoder(
            batch_obj_state, obj_gnns_ind, obj_batch_gnn_ind)
        if len(gt_gnns) > 0:
            batch_gt_state = Batch.from_data_list(gt_gnns)
            if self.attention_type == 1:
                gt_state_tensor, gt_out = self.gt_encoder(
                    batch_gt_state, gt_gnns_ind, gt_batch_gnn_ind, obj_out)
            else:
                gt_state_tensor, gt_out = self.gt_encoder(
                    batch_gt_state, gt_gnns_ind, gt_batch_gnn_ind)
        else:
            gt_out = torch.zeros_like(obj_out)

        if gt_out.shape[0] < obj_out.shape[0]:
            zero_out_tensor = torch.zeros(obj_out.shape[0]-gt_out.shape[0], gt_out.shape[1]).to(device)
            gt_out = torch.cat((gt_out, zero_out_tensor), 0)
        # output by GNNs
        out = torch.cat((obj_out, gt_out), 1)
        # value function
        vf = self.vf_net(out)
        return vf

    def forward(self, observations, actions=None, sl_train=True):
        if actions is None:
            sl_train = False
        (obj_gnns, obj_gnns_ind, obj_batch_gnn_ind, obj_node_ent, obj_trans_ind,
         gt_gnns, gt_gnns_ind, gt_batch_gnn_ind, gt_node_ent, gt_trans_ind,
         batch_ind, actions, entity_actions, prev_max_ents) = tile_obs_acs(
            observations, actions, sl_train)

        # Process inputs with GNNs
        batch_obj_state = Batch.from_data_list(obj_gnns)
        obj_state_tensor, obj_out = self.obj_encoder(
            batch_obj_state, obj_gnns_ind, obj_batch_gnn_ind)
        if len(gt_gnns) > 0:
            batch_gt_state = Batch.from_data_list(gt_gnns)
            if self.attention_type == 1:
                gt_state_tensor, gt_out = self.gt_encoder(
                    batch_gt_state, gt_gnns_ind, gt_batch_gnn_ind, obj_out)
            else:
                gt_state_tensor, gt_out = self.gt_encoder(
                    batch_gt_state, gt_gnns_ind, gt_batch_gnn_ind)
            state_tensor = torch.cat((obj_state_tensor, gt_state_tensor), 0)
        else:
            gt_out = torch.zeros_like(obj_out)
            state_tensor = obj_state_tensor
        if gt_out.shape[0] < obj_out.shape[0]:
            zero_out_tensor = torch.zeros(obj_out.shape[0]-gt_out.shape[0], gt_out.shape[1]).to(device)
            gt_out = torch.cat((gt_out, zero_out_tensor), 0)
        # output by GNNs
        out = torch.cat((obj_out, gt_out), 1)
        # value function
        vf = self.vf_net(out)
        # Lemma outputs
        lemma_outputs = self.lemma_out(F.relu(self.lemma_q(out)))
        lemma_m = Categorical(probs=self.softmax(lemma_outputs))
        if actions is not None:
            if cuda:
                lemma_actions = torch.cuda.LongTensor(actions[:, 0])
                entity_actions = torch.cuda.LongTensor(entity_actions)
            else:
                lemma_actions = torch.LongTensor(actions[:, 0])
                entity_actions = torch.LongTensor(entity_actions)
        else:
            lemma_m = Categorical(probs=self.softmax(lemma_outputs))
            lemma_actions = lemma_m.sample()
            lemma_args = [theorem_no_input[index2thm[int(lemma)]] for lemma in lemma_actions]
            max_num_ent = 4
            masks = -1 * torch.ones(len(lemma_args), max_num_ent)
            for i in range(len(lemma_args)):
                for j in range(lemma_args[i]):
                    masks[i][j] = 1
        # compute logprobs and entropy for training
        if actions is not None:
            lemma_logprobs = lemma_m.log_prob(lemma_actions)
            lemma_entropy = torch.mean(lemma_m.entropy())
        batch_lemmas = one_hot(lemma_actions, len(thm2index))
        # condition on lemma actions
        out = F.relu(out.to(device) + F.relu(self.lemma_encoder(batch_lemmas).to(device)))

        # Prepare for entity actions
        ent_mask = compute_mask(obj_node_ent, gt_node_ent)
        rev_trans_ind = compute_trans_ind(obj_trans_ind, gt_trans_ind)
        ent_rep = torch.index_select(state_tensor[ent_mask, :].clone(), 0, rev_trans_ind)
        if actions is None:
            entity_actions = []
            h_uniform = Uniform(torch.zeros(ent_rep.shape[0]), torch.ones(ent_rep.shape[0]))
            for ii in range(max_num_ent):
                if cuda:
                    mask = (masks[:, ii] != -1).cuda()
                else:
                    mask = (masks[:, ii] != -1)
                if sum(mask) > 0:
                    key = torch.index_select(out, 0, batch_ind)
                    h = torch.mean(self.key_transform(key) * ent_rep, 1)
                    gumble_h = h - torch.log(-torch.log(h_uniform.sample().to(device)))
                    # calculate loss term.
                    entity_action = scatter_max(gumble_h, batch_ind)[1]

                    # use a zero tensor to fill in empty ent representation that
                    # refers to an empty action slot
                    zero_ind = torch.LongTensor(np.arange(entity_action.shape[0])).to(device)
                    cur_ent_rep = torch.index_select(
                        ent_rep, 0, entity_action[mask].to(device))
                    zero_tensor = torch.zeros(entity_action.shape[0], cur_ent_rep.shape[1]).to(device)
                    cur_ent_rep = \
                        scatter_add(torch.cat((cur_ent_rep, zero_tensor), 0),
                                    torch.cat((torch.LongTensor(np.argwhere(mask.cpu().numpy() == 1)).squeeze(1).to(
                                        device), zero_ind), 0), 0)
                    # get new key vector
                    out = F.relu(out + F.relu(self.ent_transform(cur_ent_rep)))
                    # append entity actions; plus 1 because 0 is saved for no_op
                    # entity_action = entity_action[mask] + 1 - torch.LongTensor(prev_max_ents)
                    entity_action = (entity_action - torch.LongTensor(prev_max_ents).to(device))
                    entity_action = ((entity_action.float() + 1) * mask.float()).long()
                    entity_action = torch.cat(
                        [entity_action,
                         torch.zeros(masks.shape[0] - entity_action.shape[0]).long().to(device)])
                    entity_actions.append(entity_action)
            entity_actions = torch.stack(entity_actions).t()
            entity_actions = torch.cat([entity_actions,
                                        torch.zeros([masks.shape[0],
                                                     masks.shape[1] - entity_actions.shape[1]]).long().to(device)], 1)
            actions = torch.cat([lemma_actions.view(-1, 1), entity_actions], 1)
            actions = actions.cpu().numpy()
            # print(actions)
            return actions, vf
        else:
            max_num_ent = 4
            ent_logprob = 0.
            ent_entropy = 0.
            if sl_train:
                num_ent_correct = 0
                num_name_correct = 0
                num_ent = 0
                all_lemma_wrong_ent_indices = []
                right_lemma_wrong_ent_indices = []
            for ii in range(max_num_ent):
                if cuda:
                    mask = (entity_actions[:, ii] != -1).cuda()
                else:
                    mask = (entity_actions[:, ii] != -1)
                if sum(mask) > 0:
                    key = torch.index_select(out, 0, batch_ind)
                    h = torch.mean(self.key_transform(key) * ent_rep, 1)
                    # calculate loss term.
                    max_h = torch.index_select(
                        scatter_max(h, batch_ind)[0], 0, batch_ind)
                    logsoftmax = h - max_h - torch.index_select(
                        torch.log(scatter_add(torch.exp(h-max_h), batch_ind)), 0, batch_ind)
                    # mask out unused loss.
                    ent_logprob = ent_logprob + torch.mean(torch.index_select(
                        logsoftmax, 0, entity_actions[:, ii][mask].to(device)))
                    zero_tensor = torch.zeros(entity_actions.shape[0]).to(device)
                    zero_ind = torch.LongTensor(np.arange(entity_actions.shape[0])).to(device)
                    cur_logsoftmax = torch.index_select(
                        logsoftmax, 0, entity_actions[:, ii][mask].to(device))
                    cur_logsoftmax = \
                        scatter_add(torch.cat((cur_logsoftmax, zero_tensor),0),
                                    torch.cat(
                                        (torch.LongTensor(np.argwhere(mask.cpu().numpy() == 1)).squeeze(1).to(device),
                                         zero_ind), 0))
                    ent_logprob = ent_logprob + cur_logsoftmax

                    # calculate entropy.
                    prob = h / torch.index_select(
                        scatter_add(torch.exp(h), batch_ind), 0, batch_ind)
                    # mask out unused probs.
                    # TODO: Check whether scale makes sense? sum entropy of different entity action? but mean over data?
                    ent_entropy = ent_entropy + torch.mean(scatter_add(logsoftmax * prob, batch_ind)[mask])
                    cur_ent_rep = torch.index_select(
                        ent_rep, 0, entity_actions[:, ii][mask].to(device))
                    zero_tensor = torch.zeros(entity_actions.shape[0], cur_ent_rep.shape[1]).to(device)
                    cur_ent_rep = \
                        scatter_add(torch.cat((cur_ent_rep, zero_tensor), 0),
                                    torch.cat(
                                        (torch.LongTensor(np.argwhere(mask.cpu().numpy() == 1)).squeeze(1).to(device),
                                         zero_ind), 0), 0)

                    if sl_train:
                        h_max = scatter_max(h, batch_ind)[1].to(device)
                        num_ent += mask.float().sum()
                        num_ent_correct += (entity_actions[:, ii].to(device) == h_max).float()[mask].sum()
                        right_lemma_mask = lemma_outputs.argmax(1) != lemma_actions
                        wrong_ent_lemma_mask = entity_actions[:, ii].to(device) != h_max
                        all_lemma_wrong_ent_indices.extend(list(lemma_actions[wrong_ent_lemma_mask & mask].cpu().numpy()))
                        right_lemma_wrong_ent_indices.extend(list(lemma_actions.cpu()[wrong_ent_lemma_mask.cpu() & mask.cpu() & right_lemma_mask.cpu()].numpy()))

                    out = F.relu(out + F.relu(self.ent_transform(cur_ent_rep)))

                logprobs = self.lemma_cost * lemma_logprobs + self.entity_cost * ent_logprob
                entropy = lemma_entropy #+ ent_entropy
                # action = torch.LongTensor(actions)

            if sl_train:

                lemma_acc = torch.mean((lemma_outputs.argmax(1)==lemma_actions).float())
                different_lemma_indices = []
                if lemma_acc < 1:
                    different_obs_indices = (lemma_outputs.argmax(1).cpu() != lemma_actions.cpu())
                    obs_indices = list(range(len(different_obs_indices.numpy().tolist())))
                    select_indices = different_obs_indices.numpy().tolist()
                    obs_indices = [obs_indices[ind] for ind in range(len(select_indices)) if select_indices[ind] == 1]
                    label_lemma = lemma_actions.cpu()[different_obs_indices]
                    chosen_lemma = lemma_outputs.argmax(1).cpu()[different_obs_indices]

                    for ind in range(len(obs_indices)):
                        different_lemma_indices.append(
                            dict(obs_index=obs_indices[ind],
                                 label_lemma=label_lemma.cpu().numpy()[ind],
                                 chosen_lemma=chosen_lemma.cpu().numpy()[ind])
                        )
                ent_acc = num_ent_correct / num_ent
                name_acc = num_name_correct / num_ent
                # assert name_acc >= ent_acc
                different_ent_lemma_indices = dict(all_lemma_wrong_ent_indices=collections.Counter(all_lemma_wrong_ent_indices),
                                                   right_lemma_wrong_ent_indices=collections.Counter(right_lemma_wrong_ent_indices))
                acc = (lemma_acc, ent_acc, name_acc, different_lemma_indices, different_ent_lemma_indices)
                return logprobs, vf.squeeze(1), entropy, acc
            return logprobs, vf.squeeze(1), entropy