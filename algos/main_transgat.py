import json
import os
import pickle
import random
import numpy as np
from datetime import datetime
from time import time

import torch
import torch.optim as optim
import torch.utils.data as data_handler
from transformers import get_cosine_schedule_with_warmup

from algos.eval import eval_agent
from algos.lib.obs import nodename2index, thm2index, batch_process
from data_generation.generate_problems import generate_multiple_problems
from data_generation.utils import Dataset
from TransGrapher.models.TransGAT import GATThmNet
from algos.model.thm_model import ThmNet


use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')

seed = 21300
torch.manual_seed(21300)
if use_gpu:
    torch.cuda.manual_seed_all(0)
    os.environ['PYTHONHASHSEED'] = str(0)
random.seed(seed)
np.random.seed(seed)


def load_data(data_path):
    train_problems, valid_problems, test_problems = \
        pickle.load(open(os.path.join(data_path, "train_problems.pkl"), "rb")), \
        pickle.load(open(os.path.join(data_path, "valid_problems.pkl"), "rb")), \
        pickle.load(open(os.path.join(data_path, "test_problems.pkl"), "rb"))
    train_trajs = list()
    for train_problem in train_problems:
        train_trajs.extend(train_problem)
    valid_trajs = list()
    for valid_problem in valid_problems:
        valid_trajs.extend(valid_problem)
    test_trajs = list()
    for test_problem in test_problems:
        test_trajs.extend(test_problem)
    train_dataset, valid_dataset, test_dataset = \
        Dataset(train_trajs), Dataset(valid_trajs), Dataset(test_trajs)
    return train_dataset, valid_dataset, test_dataset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='INTGAT')
    parser.add_argument('-dp', "--data-path", type=str,
                        help="the path to collect data")
    parser.add_argument("-le", "--lemma-cost", required=False, type=float, default=1.0,
                        help="lemma cost")
    parser.add_argument("-ent", "--entity-cost", required=False, type=float, default=1.0,
                        help="entity cost")
    parser.add_argument("-lr", "--learning-rate", required=False, type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("-wd", "--weight-decay", required=False, type=float, default=0.,
                        help="learning rate")
    parser.add_argument('-wp', '--warmup-percentage', type=float, default=0.1,
                        help="the proportion of warmup updates")
    parser.add_argument('-c', '--cycles', type=float, default=2.5, help="how many cycles")
    parser.add_argument('-ns', '--num-steps', type=int, default=3000, help="how many steps")
    parser.add_argument("-o", "--obs_mode", required=False, type=str, default="geometric",
                        help="which mode of observation to use")
    parser.add_argument('--inception', type=int, default=3, help="how many intermediate layers")
    parser.add_argument("-d", "--dump", required=True, type=str,
                        help="which mode of observation to use")
    parser.add_argument('-hd', '--hidden-dim', type=int, default=128, help="how many hidden dimensions")
    parser.add_argument("-bs", "--batch-size", required=False, type=int, default=32,
                        help="what batch size to use")
    parser.add_argument("-gt", "--gnn-type", required=True, type=str,
                        help="which type of gnn to use")
    args = parser.parse_args()

    if args.gnn_type == "transgat":
        options = dict(
            num_nodes=len(nodename2index),
            num_lemmas=len(thm2index),
            hidden_dim=args.hidden_dim,
            inception=args.inception,
            entity_cost=args.entity_cost,
            lemma_cost=args.lemma_cost,
        )
        model = GATThmNet(**options)
    elif args.gnn_type == "GIN":
        options = dict(
            num_nodes=len(nodename2index),
            num_lemmas=len(thm2index),
            state_dim=args.hidden_dim,
            gnn_type=args.gnn_type,
            combined_gt_obj=True,
            attention_type=1,
            hidden_layers=args.inception,
            norm=None,
            entity_cost=args.entity_cost,
            lemma_cost=args.lemma_cost,
            cuda=use_gpu
        )
        model = ThmNet(**options)
    else:
        raise NotImplementedError

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_percentage * args.num_steps),
        num_training_steps=args.num_steps,
        num_cycles=args.cycles
    )

    if not os.path.isdir(args.dump):
        os.makedirs(args.dump)

    updates = 0
    train_lemma_accs, train_ent_accs, train_name_accs = list(), list(), list()
    valid_lemma_accs, valid_ent_accs, valid_name_accs = list(), list(), list()
    train_d, valid_d, test_d = load_data(args.data_path)
    for epoch in range(10000):
        minibatch = 0
        total_loss = 0
        total_lemma_acc = 0
        total_ent_acc = 0
        total_name_acc = 0
        sampler = data_handler.RandomSampler(train_d)
        batcher = data_handler.BatchSampler(sampler, batch_size=args.batch_size, drop_last=False)
        for indices in batcher:
            batch = train_d.get_multiple(indices=indices)
            batch_states, batch_actions, batch_name_actions = batch_process(batch, mode=args.obs_mode)
            log_probs, _, _, (lemma_acc, ent_acc, name_acc, _, _) = \
                model.forward(batch_states, batch_actions)
            loss = -log_probs.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            updates += 1
            minibatch += 1
            total_loss += loss.cpu().item()
            total_lemma_acc += lemma_acc.cpu().item()
            total_ent_acc += ent_acc.cpu().item()
            total_name_acc += name_acc.cpu().item()

        lemma_acc = total_lemma_acc / minibatch
        ent_acc = total_ent_acc / minibatch
        name_acc = total_name_acc / minibatch
        train_lemma_accs.append(lemma_acc)
        train_ent_accs.append(ent_acc)
        train_name_accs.append(name_acc)

        valid_batch = valid_d.io_tuples
        valid_batch_states, valid_batch_actions, valid_batch_name_actions = \
            batch_process(valid_batch, mode=args.obs_mode)
        _, _, _, (valid_lemma_acc, valid_ent_acc, valid_name_acc, _, _) = \
            model.forward(valid_batch_states, valid_batch_actions)
        valid_lemma_accs.append(valid_lemma_acc.cpu().item())
        valid_ent_accs.append(valid_ent_acc.cpu().item())
        valid_name_accs.append(valid_name_acc.cpu().item())

        torch.save(model, os.path.join(args.dump, "model.pt"))
        json.dump(train_lemma_accs, open(os.path.join(args.dump, "train_lemma_accs.json"), "w"))
        json.dump(train_ent_accs, open(os.path.join(args.dump, "train_ent_accs.json"), "w"))
        json.dump(train_name_accs, open(os.path.join(args.dump, "train_name_accs.json"), "w"))
        json.dump(valid_lemma_accs, open(os.path.join(args.dump, "valid_lemma_accs.json"), "w"))
        json.dump(valid_ent_accs, open(os.path.join(args.dump, "valid_ent_accs.json"), "w"))
        json.dump(valid_name_accs, open(os.path.join(args.dump, "valid_name_accs.json"), "w"))

        if updates > args.num_steps:
            break
