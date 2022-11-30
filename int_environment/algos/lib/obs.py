import torch
from torch_geometric.data import Data
from algos.lib.ops import one_hot
from proof_system.all_axioms import all_axioms_to_prove
from proof_system.numerical_functions import necessary_numerical_functions
from proof_system.logic_functions import necessary_logic_functions
import numpy as np

# import dgl


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = torch.cuda.is_available()

logic_function_names = [lf.name for lf in list(necessary_logic_functions.values())]
numerical_function_names = [nf.name for nf in list(necessary_numerical_functions.values())]
input_names = [chr(ord('a') + i) for i in range(25)] + [str(i) for i in range(10)]
all_node_names = logic_function_names + numerical_function_names + input_names
nodename2index = {
    node: torch.LongTensor([ind]).to(device)
    for ind, node in enumerate(all_node_names)
}
theorem_names = [theorem.name for theorem in list(all_axioms_to_prove.values())]
thm2index = {
    # node: torch.LongTensor([ind]).to(device)
    node: ind
    for ind, node in enumerate(theorem_names)
}
index2thm = {
    ind: node for ind, node in enumerate(theorem_names)
}
theorem_no_input = {theorem.name: theorem.input_no for theorem in list(all_axioms_to_prove.values())}
thm_index2no_input = {
    i: theorem_no_input[index2thm[i]]
    for i in range(len(theorem_names))
}


def gt_2_list(gt, node_ent, node_name, ent_dic, name_dic):

    node_op = []
    ivs = []

    for node in list(gt.ent_dic.values()):
        assert node not in ent_dic.keys()
        ent_dic[node] = (len(ent_dic), gt)
        if node.name not in name_dic.keys():
            name_dic[node.name] = len(name_dic)

        if hasattr(node, "logic_function"):
            node_op.append(nodename2index[node.logic_function.name])
            ivs.append(0)
            node_ent.append(-1)
            node_name.append(-1)
        else:
            node_ent.append(ent_dic[node][0])
            node_name.append(name_dic[node.name])
            if node.recent_numerical_function is not None:
                node_op.append(nodename2index[node.recent_numerical_function.name])
                ivs.append(0)
            else:
                node_op.append(nodename2index[node.name])
                ivs.append(1)

    return node_op, ivs


def gt_2_graph(gt, node_ent, node_name, ent_dic, name_dic, unit_direction=None):
    root = nodename2index[gt.logic_function.to_string()]
    node_op = [root]
    graph_data = []
    node_ent.append(-1)
    node_name.append(-1)
    node_count = []
    local_poses = []
    node_count = []
    local_poses = []
    ivs = [0]

    def data_to_graph(entity, source, pos):
        local_poses.append(pos)
        if entity.name in [chr(ord('a') + i) for i in range(8)]:# or entity.is_iv:
            ivs.append(1)
        else:
            ivs.append(0)
        if entity.operands is None:
            # node feature input
            node_op.append(nodename2index[entity.name])

            # entity names
            if entity.name not in name_dic.keys():
                name_dic[entity.name] = len(name_dic)
            node_name.append(name_dic[entity.name])

            # if entity not in ent_dic.keys():
            assert entity not in ent_dic.keys()
            ent_dic[entity] = (len(ent_dic), gt)
            node_ent.append(ent_dic[entity][0])

            # calculate new source index
            node_count.append(1)
            if unit_direction is not None:
                if unit_direction == "topdown":
                    return [[source, len(node_count)]]
                elif unit_direction == "bottomup":
                    return [[len(node_count), source]]
            else:
                return [[len(node_count), source], [source, len(node_count)]]
        else:
            # node feature input
            node_op.append(nodename2index[entity.recent_numerical_function.name])

            # entity names
            if entity.name not in name_dic.keys():
                name_dic[entity.name] = len(name_dic)
            node_name.append(name_dic[entity.name])

            # if entity not in ent_dic.keys():
            assert entity not in ent_dic.keys()
            ent_dic[entity] = (len(ent_dic), gt)
            node_ent.append(ent_dic[entity][0])

            # calculate new source index
            node_count.append(1)
            new_source = len(node_count)
            data = [[source, new_source]]
            if unit_direction is not None:
                if unit_direction == "topdown":
                    data = [[source, len(node_count)]]
                elif unit_direction == "bottomup":
                    data = [[len(node_count), source]]
            else:
                data = [[len(node_count), source], [source, len(node_count)]]
            for pos, e in enumerate(entity.operands):
                data += data_to_graph(e, new_source, pos)
            return data

    for pos, ent in enumerate(gt.operands):
        graph_data += data_to_graph(ent, 0, pos)

    return graph_data, node_op, local_poses, ivs


def obs_to_graphs(obs, bag=False):
    gt_graph = []
    gt_gnn_ind = []
    obj_graph = []
    obj_gnn_ind = []
    ent_dic = dict()
    name_dic = dict()
    node_ent = []
    node_name = []
    g_ind = 0
    for gt in obs['objectives']:
        graph_data, node_op, local_poses, ivs = gt_2_graph(
            gt, node_ent, node_name, ent_dic, name_dic
        )
        if bag:
            graph_data = list(zip(range(len(node_op)),
                                  range(len(node_op))))

        for _ in range(len(node_op)):
            obj_gnn_ind.append(g_ind)
        g_ind += 1
        if cuda:
            node_op_onehot = one_hot(torch.LongTensor(node_op), len(nodename2index))
            local_poses_onehot = torch.cat((torch.cuda.FloatTensor(1, 2).fill_(0),
                                            one_hot(torch.LongTensor(local_poses), 2)), 0)
            ivs_onehot = one_hot(torch.LongTensor(ivs), 2).to(device)
            node_input = torch.cat((node_op_onehot, local_poses_onehot, ivs_onehot), 1)
            graph_data = torch.cuda.LongTensor(graph_data).transpose(0, 1)
        else:
            node_op_onehot = one_hot(torch.LongTensor(node_op), len(nodename2index))
            local_poses_onehot = torch.cat((torch.zeros(1, 2), one_hot(torch.LongTensor(local_poses), 2)),
                                           0)
            ivs_onehot = one_hot(torch.LongTensor(ivs), 2)
            node_input = torch.cat((node_op_onehot, local_poses_onehot, ivs_onehot), 1)
            graph_data = torch.LongTensor(graph_data).transpose(0, 1)
        obj_graph.append(Data(x=node_input, edge_index=graph_data))
    # obj = dict(obj_graph=obj_graph, obj_gnn_ind=obj_gnn_ind)
    obj = [obj_graph, obj_gnn_ind]

    g_ind = 0
    for gt in obs["ground_truth"]:
        graph_data, node_op, local_poses, ivs = gt_2_graph(
            gt, node_ent, node_name, ent_dic, name_dic
        )
        if bag:
            graph_data = list(zip(range(len(node_op)),
                                  range(len(node_op))))
        for _ in range(len(node_op)):
            gt_gnn_ind.append(g_ind)
        g_ind += 1
        if cuda:
            node_op_onehot = one_hot(torch.LongTensor(node_op), len(nodename2index))
            local_poses_onehot = torch.cat((torch.cuda.FloatTensor(1, 2).fill_(0),
                                            one_hot(torch.LongTensor(local_poses), 2)), 0)
            ivs_onehot = one_hot(torch.LongTensor(ivs), 2).to(device)
            node_input = torch.cat((node_op_onehot, local_poses_onehot, ivs_onehot), 1)
            graph_data = torch.cuda.LongTensor(graph_data).transpose(0, 1)
        else:
            node_op_onehot = one_hot(torch.LongTensor(node_op), len(nodename2index))
            local_poses_onehot = \
                torch.cat((torch.zeros(1, 2), one_hot(torch.LongTensor(local_poses), 2)), 0)
            ivs_onehot = one_hot(torch.LongTensor(ivs), 2)
            node_input = torch.cat((node_op_onehot, local_poses_onehot, ivs_onehot), 1)
            graph_data = torch.LongTensor(graph_data).transpose(0, 1)
        gt_graph.append(Data(x=node_input, edge_index=graph_data))
    # gt = dict(gt_graph=gt_graph, gt_gnn_ind=gt_gnn_ind)
    gt = [gt_graph, gt_gnn_ind]
    # observation = dict(obj=obj,
    #                    gt=gt,
    #                    node_ent=node_ent,
    #                    node_name=node_name,
    #                    ent_dic=ent_dic,
    #                    name_dic=name_dic)
    observation = [obj, gt, node_ent, node_name, ent_dic, name_dic]
    return observation


def obs_to_graphs_dgl(obs, debug=False):
    gt_graph = []
    gt_gnn_ind = []
    obj_graph = []
    obj_gnn_ind = []
    ent_dic = dict()
    name_dic = dict()
    node_ent = []
    node_name = []
    g_ind = 0
    for gt in obs['objectives']:
        graph_data, node_op, local_poses, ivs = gt_2_graph(
            gt, node_ent, node_name, ent_dic, name_dic, "topdown"
        )

        for _ in range(len(node_op)):
            obj_gnn_ind.append(g_ind)
        g_ind += 1
        if cuda:
            node_op_onehot = one_hot(torch.LongTensor(node_op), len(nodename2index))
            local_poses_onehot = torch.cat((torch.cuda.FloatTensor(1, 2).fill_(0),
                                            one_hot(torch.LongTensor(local_poses), 2)), 0)
            ivs_onehot = one_hot(torch.LongTensor(ivs), 2).to(device)
            node_input = torch.cat((node_op_onehot, local_poses_onehot, ivs_onehot), 1)
        else:
            node_op_onehot = one_hot(torch.LongTensor(node_op), len(nodename2index))
            local_poses_onehot = torch.cat((torch.zeros(1, 2), one_hot(torch.LongTensor(local_poses), 2)),
                                           0)
            ivs_onehot = one_hot(torch.LongTensor(ivs), 2)
            node_input = torch.cat((node_op_onehot, local_poses_onehot, ivs_onehot), 1)
        g = dgl.DGLGraph()
        g_rev = dgl.DGLGraph()
        src, dst = tuple(zip(*graph_data))
        g.add_nodes(max(max(src), max(dst))+1)#, data=torch.cat((node_op_onehot, local_poses_onehot, ivs_onehot), 1))
        g_rev.add_nodes(max(max(src), max(dst))+1)#, data=torch.cat((node_op_onehot, local_poses_onehot, ivs_onehot), 1))
        g.add_edges(src, dst)
        g_rev.add_edges(dst, src)
        g.ndata['feat'] = node_input
        g_rev.ndata['feat'] = node_input
        obj_graph.append([g, g_rev])
    obj = [obj_graph, obj_gnn_ind]

    g_ind = 0
    for gt in obs["ground_truth"]:
        graph_data, node_op, local_poses, ivs = gt_2_graph(
            gt, node_ent, node_name, ent_dic, name_dic, "topdown"
        )
        for _ in range(len(node_op)):
            gt_gnn_ind.append(g_ind)
        g_ind += 1
        if cuda:
            node_op_onehot = one_hot(torch.LongTensor(node_op), len(nodename2index))
            local_poses_onehot = torch.cat((torch.cuda.FloatTensor(1, 2).fill_(0),
                                            one_hot(torch.LongTensor(local_poses), 2)), 0)
            ivs_onehot = one_hot(torch.LongTensor(ivs), 2).to(device)
            node_input = torch.cat((node_op_onehot, local_poses_onehot, ivs_onehot), 1)
        else:
            node_op_onehot = one_hot(torch.LongTensor(node_op), len(nodename2index))
            local_poses_onehot = \
                torch.cat((torch.zeros(1, 2), one_hot(torch.LongTensor(local_poses), 2)), 0)
            ivs_onehot = one_hot(torch.LongTensor(ivs), 2)
            node_input = torch.cat((node_op_onehot, local_poses_onehot, ivs_onehot), 1)
        g = dgl.DGLGraph()
        g_rev = dgl.DGLGraph()
        src, dst = tuple(zip(*graph_data))
        g.add_nodes(max(max(src), max(dst)) + 1)
        g_rev.add_nodes(max(max(src), max(dst)) + 1)
        g.add_edges(src, dst)
        g_rev.add_edges(dst, src)
        g.ndata['feat'] = node_input
        g_rev.ndata['feat'] = node_input
        gt_graph.append([g, g_rev])
    gt = [gt_graph, gt_gnn_ind]

    observation = [obj, gt, node_ent, node_name, ent_dic, name_dic]
    return observation


def convert_t2t_obs_to_dict(obs):
    g, node_ent, node_name = obs
    observation = dict(g=g,
                       node_ent=node_ent,
                       node_name=node_name)
    return observation


def convert_obs_to_dict(obs):
    obj, gt, node_ent, node_name = obs
    obj_graph, obj_gnn_ind = obj
    gt_graph, gt_gnn_ind = gt
    obj = dict(obj_graph=obj_graph, obj_gnn_ind=obj_gnn_ind)
    gt = dict(gt_graph=gt_graph, gt_gnn_ind=gt_gnn_ind)
    observation = dict(obj=obj,
                       gt=gt,
                       node_ent=node_ent,
                       node_name=node_name)
    return observation

def convert_batch_obs_to_dict(obs):
    batch_obs = list()
    for ob in obs:
        observation = convert_obs_to_dict(list(ob))
        batch_obs.append(observation)
    return batch_obs


def construct_connected_graph(n):
    graph_data = []
    for i in range(n):
        for j in range(i+1, n):
            graph_data.append([i,j])
    return graph_data


def obs_to_graphs_transformer(obs):
    ent_dic = dict()
    name_dic = dict()
    node_ent = []
    node_name = []
    all_node_op = []
    all_local_poses = []
    all_ivs = []
    gt_or_obj = []

    for gt in obs['objectives']:
        _, node_op, local_poses, ivs = gt_2_graph(
            gt, node_ent, node_name, ent_dic, name_dic
        )
        all_node_op.extend(node_op)
        local_poses_onehot = \
            torch.cat((torch.zeros(1, 2).to(device), one_hot(torch.LongTensor(local_poses), 2)), 0).to(device)
        all_local_poses.extend(local_poses_onehot)
        all_ivs.extend(ivs)
        for _ in range(len(node_op)):
            gt_or_obj.append(0)

    for gt in obs["ground_truth"]:
        _, node_op, local_poses, ivs = gt_2_graph(
            gt, node_ent, node_name, ent_dic, name_dic
        )
        all_node_op.extend(node_op)
        local_poses_onehot = \
            torch.cat((torch.zeros(1, 2).to(device), one_hot(torch.LongTensor(local_poses), 2)), 0).to(device)
        all_local_poses.extend(local_poses_onehot)
        all_ivs.extend(ivs)
        for _ in range(len(node_op)):
            gt_or_obj.append(1)

    if cuda:
        node_op_onehot = one_hot(torch.LongTensor(all_node_op), len(nodename2index))
        local_poses_onehot = torch.stack(all_local_poses)
        ivs_onehot = one_hot(torch.LongTensor(all_ivs), 2).to(device)
        node_input = torch.cat((node_op_onehot, local_poses_onehot, ivs_onehot), 1)
    else:
        node_op_onehot = one_hot(torch.LongTensor(all_node_op), len(nodename2index))
        local_poses_onehot = torch.stack(all_local_poses)
        ivs_onehot = one_hot(torch.LongTensor(all_ivs), 2)
        node_input = torch.cat((node_op_onehot, local_poses_onehot, ivs_onehot), 1)
    graph_data = construct_connected_graph(len(all_node_op))
    self_loops = tuple(range(len(all_node_op)))
    g = dgl.DGLGraph()
    src, dst = tuple(zip(*graph_data))
    num_nodes = max(max(src), max(dst)) + 1
    g.add_nodes(num_nodes)
    g.add_edges(src, dst)
    g.add_edges(dst, src)
    g.add_edges(self_loops, self_loops)
    g.ndata['feat'] = node_input
    g.ndata['raw_pos'] = torch.LongTensor(range(num_nodes))

    observation = [g, node_ent, node_name, ent_dic, name_dic]
    return observation


def convert_np_obs_to_dict(obs):
    batch_obs = list()
    for ob in obs:
        obj, gt, node_ent, node_name = list(ob)
        obj_graph, obj_gnn_ind = obj
        gt_graph, gt_gnn_ind = gt
        obj = dict(obj_graph=obj_graph, obj_gnn_ind=obj_gnn_ind)
        gt = dict(gt_graph=gt_graph, gt_gnn_ind=gt_gnn_ind)
        observation = dict(obj=obj,
                           gt=gt,
                           node_ent=node_ent,
                           node_name=node_name)
        batch_obs.append(observation)
    return batch_obs


def batch_process(batch, mode="geometric", bag=False):
    batch_actions = list()
    batch_name_actions = list()
    batch_states = list()

    for datapoint in batch:
        # Only use the last ground truth both as an info provider and where entities are chosen from
        obs = datapoint[0]
        if mode == "dgl":
            obj, gt, node_ent, node_name, ent_dic, name_dic = obs_to_graphs_dgl(obs)
            obs = convert_obs_to_dict([obj, gt, node_ent, node_name])
        elif mode == "geometric":
            obj, gt, node_ent, node_name, ent_dic, name_dic = obs_to_graphs(obs, bag=bag)
            obs = convert_obs_to_dict([obj, gt, node_ent, node_name])
        elif mode == "transformer":
            g, node_ent, node_name, ent_dic, name_dic = obs_to_graphs_transformer(obs)
            obs = convert_t2t_obs_to_dict([g, node_ent, node_name])
        batch_states.append(obs)
        lemma_action = thm2index[datapoint[1].name]
        actions = [lemma_action]
        entity_actions = [(ent_dic[et][0]) for et in datapoint[2]]
        # why plus 1? We will subtract 1 in model.py. This is mainly for consistency in RL and SL. RL's action
        # zero is not usable.
        actions += [entity_action + 1 for entity_action in entity_actions]
        while len(actions) < 5:
            actions.append(0)
        batch_actions.append(np.array(actions)[np.newaxis, :])

        all_name_actions = []
        name_actions = [(name_dic[et.name]) for et in datapoint[2]]
        all_name_actions += [name_action for name_action in name_actions]
        while len(all_name_actions) < 5:
            all_name_actions.append(-1)
        batch_name_actions.append(np.array(all_name_actions)[np.newaxis, :])

    batch_actions = np.concatenate(batch_actions)
    batch_name_actions = np.concatenate(batch_name_actions)
    # print("Entity indices:\n", batch_actions)
    # print("Name indices:\n", batch_name_actions)
    return batch_states, batch_actions, batch_name_actions


def tile_obs_acs(observations, actions, sl_train):
    """
    Each obs in observations is a problem, which consists of objectives, ground truths.
    This function first group all nodes in objectives in all problems of observations
    as a batch, and all nodes in ground truths as a batch, and labels each node an index
    that indicates which graph it comes from, and each graph an index that indicates
    which problem the graph is from.

    :param observations:
    :param actions:
    :param sl_train:
    :return:
    obj_gnns: all obj gnns
    obj_gnns_ind: the obj graph indices, which obj graph the node comes from?
    obj_batch_gnn_ind: the problem indices, which problem the obj graph comes from?
    obj_node_ent:
    obj_trans_id:
    gt_gnns: all gt gnns
    gt_gnns_ind: the gt graph indices, which gt graph the node comes from?
    gt_batch_gnn_ind: the problem indices, which problem the gt graph comes from?
 indices
    """
    # actions:
    if actions is not None:
        if not sl_train:
            actions = np.vstack([ac for ac in actions])
        entity_actions = actions[:, 1:] - 1  # recover entity indices
    else:
        entity_actions = None
    all_gnns = []
    obj_gnns = []
    obj_gnns_ind = []
    obj_batch_gnn_ind = []
    obj_prev_num_graphs = 0
    obj_node_ent = []
    obj_node_name = []
    gt_gnns = []
    gt_gnns_ind = []
    gt_batch_gnn_ind = []
    gt_prev_num_graphs = 0
    gt_node_ent = []
    gt_node_name = []
    batch_ind = []
    prev_max_ent = 0
    prev_max_ents = []
    obj_trans_ind = []
    gt_trans_ind = []
    # if isinstance(observations, np.ndarray) or isinstance(observations, list):
    if not sl_train:
        observations = convert_batch_obs_to_dict(observations)
    for ii, obs in enumerate(observations):
        obj_gnn = obs["obj"]["obj_graph"]
        obj_gnn_ind = obs["obj"]["obj_gnn_ind"]
        # node index -> graph index
        obj_gnns_ind += [obj_prev_num_graphs+ind for ind in obj_gnn_ind]
        obj_prev_num_graphs = max(obj_gnns_ind) + 1
        obj_batch_gnn_ind += [ii] * (max(obj_gnn_ind) + 1)
        # graph index -> problem index
        obj_gnns += obj_gnn
        all_gnns += obj_gnn
        if len(obs["gt"]["gt_graph"]) > 0:
            gt_gnn = obs["gt"]["gt_graph"]
            gt_gnn_ind = obs["gt"]["gt_gnn_ind"]
            gt_gnns_ind += [gt_prev_num_graphs+ind for ind in gt_gnn_ind]
            gt_prev_num_graphs = max(gt_gnns_ind) + 1
            gt_batch_gnn_ind += [ii] * (max(gt_gnn_ind) + 1)
            all_gnns += gt_gnn
            gt_gnns += gt_gnn
        else:
            gt_gnn_ind = []

        assert len(obs["node_ent"]) == len(obj_gnn_ind) + len(gt_gnn_ind)
        obj_node_ent += obs["node_ent"][:len(obj_gnn_ind)]
        obj_node_name += obs["node_name"][:len(obj_gnn_ind)]
        gt_node_ent += obs["node_ent"][len(obj_gnn_ind):]
        gt_node_name += obs["node_name"][len(obj_gnn_ind):]

        num_ent = sum(np.array(obs["node_ent"])!=-1)
        num_obj_ent = len(np.array(obs["node_ent"][:len(obj_gnn_ind)])[np.array(obs["node_ent"][:len(obj_gnn_ind)])!=-1])
        obj_trans_ind += list(np.arange(num_ent)[:num_obj_ent] + prev_max_ent)
        gt_trans_ind += list(np.arange(num_ent)[num_obj_ent:] + prev_max_ent)
        prev_max_ents.append(prev_max_ent)
        # shift index for entity actions
        if actions is not None:
            entity_actions[ii] = entity_actions[ii] + prev_max_ent * (entity_actions[ii] != -1).astype(float)
        prev_max_ent += num_ent
        # entity index -> problem index
        batch_ind += [ii] * sum(np.array(obs["node_ent"])!=-1)
        # print(sum(np.array(obs[2])!=-1))
    if cuda:
        batch_ind = torch.cuda.LongTensor(batch_ind)
    else:
        batch_ind = torch.LongTensor(batch_ind)
    return (obj_gnns, obj_gnns_ind, obj_batch_gnn_ind, obj_node_ent, obj_trans_ind,
            gt_gnns, gt_gnns_ind, gt_batch_gnn_ind, gt_node_ent, gt_trans_ind,
            batch_ind, actions, entity_actions, prev_max_ents)


def compute_mask(obj_node_ent, gt_node_ent):
    if cuda:
        obj_node_ent = torch.cuda.LongTensor(obj_node_ent)
        gt_node_ent = torch.cuda.LongTensor(gt_node_ent)
        node_ent = torch.cuda.LongTensor(torch.cat((obj_node_ent, gt_node_ent), 0))
        ent_mask = (node_ent != -1).cuda()
    else:
        obj_node_ent = torch.LongTensor(obj_node_ent)
        gt_node_ent = torch.LongTensor(gt_node_ent)
        node_ent = torch.LongTensor(torch.cat((obj_node_ent, gt_node_ent), 0))
        ent_mask = (node_ent != -1)
    return ent_mask


def compute_name(obj_node_name, gt_node_name):
    if cuda:
        obj_node_name = torch.cuda.LongTensor(obj_node_name)
        gt_node_name = torch.cuda.LongTensor(gt_node_name)
        node_name = torch.cuda.LongTensor(torch.cat((obj_node_name, gt_node_name), 0))[ent_mask]
    else:
        obj_node_name = torch.LongTensor(obj_node_name)
        gt_node_name = torch.LongTensor(gt_node_name)
        node_name = torch.LongTensor(torch.cat((obj_node_name, gt_node_name), 0))[ent_mask]
    return node_name


def compute_trans_ind(obj_trans_ind, gt_trans_ind):
    if cuda:
        obj_trans_ind = torch.cuda.LongTensor(obj_trans_ind)
        gt_trans_ind = torch.cuda.LongTensor(gt_trans_ind)
        trans_ind = torch.cuda.LongTensor(torch.cat((obj_trans_ind, gt_trans_ind), 0))
        rev_trans_ind = torch.cuda.LongTensor(
            np.hstack([np.where(trans_ind.cpu().numpy() == i)[0] for i in range(len(trans_ind))]))
    else:
        obj_trans_ind = torch.LongTensor(obj_trans_ind)
        gt_trans_ind = torch.LongTensor(gt_trans_ind)
        trans_ind = torch.LongTensor(torch.cat((obj_trans_ind, gt_trans_ind), 0))
        rev_trans_ind = torch.LongTensor(
            np.hstack([np.where(trans_ind.cpu().numpy() == i)[0] for i in range(len(trans_ind))]))
    return rev_trans_ind
