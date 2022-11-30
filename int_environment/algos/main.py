"""
Usage:
python supervised_curriculum.py
"""

# TODO: get curriculum in here
# TODO: get rid of all random choices and use sample instead

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

from algos.eval import eval_agent
from algos.lib.obs import nodename2index, thm2index, batch_process
from algos.lib.arguments import get_args
from data_generation.generate_problems import generate_multiple_problems
from data_generation.utils import Dataset

timestamp = str(datetime.fromtimestamp(time())).replace(" ", "_").replace(":", "_").replace("-", "_").replace(".", "_")
args = get_args()

args.use_gpu = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.use_gpu else "cpu")

if args.num_order_or_combo < 0:
    args.num_order_or_combo = None

dgl = (args.obs_mode == "dgl")
bow = args.bag_of_words
print(args.transform_gt)
if dgl and (not bow):
    from algos.model.thm_model_dgl import ThmNet
elif bow and (not dgl):
    from algos.model.thm_model import ThmNet
elif (not bow) and (not dgl):
    from algos.model.thm_model import ThmNet
else:
    raise AssertionError

torch.manual_seed(args.seed)
if args.use_gpu:
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(0)
random.seed(args.seed)
np.random.seed(args.seed)

os.makedirs(os.path.join(args.dump, str(timestamp)))


def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['extra']


def resume_checkpoint(model, optimizer, resume_dir):
    name = 'last_epoch.pth'
    fname = os.path.join(resume_dir, name)
    if os.path.exists(fname):
        extra = load_checkpoint(model, optimizer, fname)
        updates = extra['updates']
        start_epoch = extra['epoch']
        best_val_succ = extra['best_val_succ']
        return start_epoch, updates, best_val_succ
    return 0, 0, 0.0


def save_checkpoint(model, optimizer, ckpt_dir, epoch, extra,
                    is_last=False, is_best=False):
    name = 'epoch_{}.pth'.format(epoch)
    if is_last:
        name = 'last_epoch.pth'
    elif is_best:
        name = 'best_epoch.pth'
    fname = os.path.join(ckpt_dir, name)
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'extra': extra
    }
    torch.save(state, fname)


def load_data(data_dir, mode="train"):
    file_name = os.path.join(data_dir, '{}.pkl'.format(mode))
    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def load_all_data(train_dirs, test_dirs):
    train_dataset = Dataset([])
    val_dataset = Dataset([])
    train_first_dataset = Dataset([])
    val_first_dataset = Dataset([])
    for train_dir in train_dirs:
        train_ds = load_data(train_dir, mode="train")
        train_dataset.merge(train_ds)
        val_ds = load_data(train_dir, mode="val")
        val_dataset.merge(val_ds)
        train_first_ds = load_data(train_dir, mode="train_first")
        train_first_dataset.merge(train_first_ds)
        val_first_ds = load_data(train_dir, mode="val_first")
        val_first_dataset.merge(val_first_ds)

    test_dataset = Dataset([])
    test_first_dataset = Dataset([])
    for test_dir in test_dirs:
        test_ds = load_data(test_dir, mode="test")
        test_dataset.merge(test_ds)
        test_first_ds = load_data(test_dir, mode="test_first")
        test_first_dataset.merge(test_first_ds)
    return {
        "train": train_dataset,
        "train_first": train_first_dataset,
        "val": val_dataset,
        "val_first": val_first_dataset,
        "test": test_dataset,
        "test_first": test_first_dataset
    }


def load_model():
    options = dict(
        num_nodes=len(nodename2index),
        num_lemmas=len(thm2index),
        hidden_dim=args.hidden_dim,
        gnn_type=args.gnn_type,
        combined_gt_obj=args.combined_gt_obj,
        attention_type=args.atten_type,
        hidden_layers=args.hidden,
        norm=args.norm,
        entity_cost=args.entity_cost,
        lemma_cost=args.lemma_cost,
        cuda=args.use_gpu,
        attention_heads=args.attention_heads,
        gat_dropout_rate=args.gat_dropout_rate,
        dropout_rate=args.dropout_rate,
    )
    return ThmNet(**options)


def load_optimizer(model):
    return optim.Adam(model.parameters(), lr=args.lr)


# Train epoch
def train_epoch(model, dataset, optimizer, updates):
    time0 = time()
    total_loss = 0
    total_lemma_acc = 0
    total_ent_acc = 0
    total_name_acc = 0
    minibatch = 0

    sampler = data_handler.RandomSampler(dataset)
    batcher = data_handler.BatchSampler(sampler, batch_size=args.batch_size, drop_last=False)

    # Main training steps
    for indices in batcher:
        batch = dataset.get_multiple(indices=indices)
        batch_states, batch_actions, batch_name_actions = batch_process(batch, mode=args.obs_mode)
        log_probs, _, _, (
            lemma_acc, ent_acc, name_acc, diff_lemma_indices, diff_ent_lemma_indices) = model.forward(
            batch_states, batch_actions)
        loss = -log_probs.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        updates += 1

        minibatch += 1
        total_loss += loss.cpu().item()
        total_lemma_acc += lemma_acc.cpu().item()
        total_ent_acc += ent_acc.cpu().item()
        total_name_acc += name_acc.cpu().item()

    loss = total_loss / minibatch
    lemma_acc = total_lemma_acc / minibatch
    ent_acc = total_ent_acc / minibatch
    name_acc = total_name_acc / minibatch

    print("Epoch time: {}s\n".format(time() - time0))
    return loss, lemma_acc, ent_acc, name_acc, updates


# Validate every epoch
def validate(model, e_dataset):
    model.eval()
    validation_batch = e_dataset.io_tuples[:args.evaluation_size]
    batch_states, batch_actions, batch_name_actions = batch_process(validation_batch, mode=args.obs_mode)
    with torch.no_grad():
        log_probs, _, _, \
        (lemma_acc, ent_acc, name_acc, different_lemma_indices, different_ent_lemma_indices) = \
            model.forward(
                batch_states, batch_actions, sl_train=True
            )
        loss = -log_probs.mean()
    model.train()
    return loss.cpu().item(), lemma_acc.cpu().item(), ent_acc.cpu().item(), name_acc.cpu().item()


# Test every epoch
def test_rollout(model, test_dataset, whole_dataset=False):
    model.eval()
    if whole_dataset:
        test_starting_points = test_dataset.trajectories
    else:
        indices = range(len(test_dataset))
        test_starting_points = [test_dataset.trajectories[index]
                                for index in random.sample(indices, k=args.num_test_probs)]
    env_config = {
        "mode": "eval",
        "eval_dataset": test_starting_points,
        "online": False,
        "batch_eval": False,
        "verbo": True,
        "obs_mode": args.obs_mode,
        "bag_of_words": args.bag_of_words,
        "time_limit": args.time_limit,
        "degree": args.degree
    }
    success_rate, wrong_cases, success_cases, avg_num_steps = \
        eval_agent(model, env_config=env_config)

    model.train()
    return success_rate, wrong_cases, success_cases, avg_num_steps


def train_eval_test(model, optimizer, kl_dict=None, all_data=None, resume_dir=None, axiom_combo_dict=None):
    if resume_dir:
        start_epoch, updates, best_val_succ = resume_checkpoint(model, optimizer, resume_dir)
    else:
        start_epoch, updates, best_val_succ = 0, 0, 0.0

    if start_epoch > 0:
        regenerate_dataset = True
    else:
        regenerate_dataset = False

    if not args.online:
        (train_dataset, val_dataset, eval_dataset, train_first_dataset,
         val_first_dataset, eval_first_dataset) = (all_data["train"], all_data["val"], all_data["test"],
                                                   all_data["train_first"], all_data["val_first"],
                                                   all_data["test_first"])
    else:
        val_dataset = Dataset([])
        eval_dataset = Dataset([])
        eval_first_dataset = Dataset([])
        for kl in args.test_sets:
            k = kl.split("_")[0][-1]
            l = int(kl[-1])

            data_path = os.path.join(args.combo_path,
                                     "test_first_dataset_prob{}_k{}l{}_oog{}_nooc{}_degree{}.pkl".format(
                                         args.num_probs, k, l,
                                         args.online_order_generation, args.num_order_or_combo,
                                         args.degree)
                                     )

            if os.path.isfile(data_path):
                with pickle.load(open(data_path, "rb")) as existent_dataset:
                    eval_first_dataset.merge(existent_dataset)
            else:
                if args.online_order_generation:
                    keyword_arguments = {"combos": kl_dict}
                else:
                    keyword_arguments = {"orders": kl_dict}
                one_piece_of_data, _ = generate_multiple_problems(k, l, num_probs=args.num_probs,
                                                                  train_test="test", backwards=True,
                                                                  transform_gt=args.transform_gt, degree=args.degree,
                                                                  num_order_or_combo=args.num_order_or_combo,
                                                                  **keyword_arguments)

                eval_dataset.merge(one_piece_of_data["all"])
                eval_first_dataset.merge(one_piece_of_data["all_first"])

        eval_objectives = set([problem[0]["objectives"][0].name for problem in eval_first_dataset])

        print("Eval dataset length ", len(eval_dataset))
        print("Eval first step dataset length ", len(eval_first_dataset))

    train_losses, train_lemma_accs, train_ent_accs, train_name_accs = [], [], [], []
    val_losses, val_lemma_accs, val_ent_accs, val_name_accs = [], [], [], []
    test_losses, test_lemma_accs, test_ent_accs, test_name_accs = [], [], [], []

    new_dataset_success_rates = []
    train_first_success_rates = []
    test_first_success_rates = []
    val_first_success_rates = []

    new_dataset_avg_proof_lengths = []
    train_first_avg_proof_lengths = []
    test_first_avg_proof_lengths = []

    args_dict = vars(args)
    # args_dict["train_combos_and_lengths"] = train_combos_and_lengths
    # args_dict["test_combos_and_lengths"] = test_combos_and_lengths
    json.dump(args_dict, open(os.path.join(args.dump, str(timestamp), "env_config.json"), "w"))

    timing = dict()
    record = dict()
    for epoch in range(start_epoch, args.epoch):
        print(epoch)
        if args.online and (regenerate_dataset or epoch % args.epochs_per_online_dataset == 0):
            regenerate_dataset = False
            train_dataset = Dataset([])
            train_first_dataset = Dataset([])
            for kl in args.train_sets:
                k = kl.split("_")[0][-1]
                l = int(kl[-1])

                if args.online_order_generation:
                    keyword_arguments = {"combos": kl_dict}
                else:
                    keyword_arguments = {"orders": kl_dict}
                one_piece_of_data, _ = generate_multiple_problems(k, l, num_probs=args.num_probs,
                                                                  train_test="train", backwards=True,
                                                                  transform_gt=args.transform_gt, degree=args.degree,
                                                                  num_order_or_combo=args.num_order_or_combo,
                                                                  avoid_objective_names=eval_objectives,
                                                                  **keyword_arguments)
                train_dataset.merge(one_piece_of_data["all"])
                train_first_dataset.merge(one_piece_of_data["all_first"])

            new_dataset_success_rate, new_dataset_wrong_case, new_dataset_right_case, new_dataset_avg_proof_length = \
                test_rollout(model, train_first_dataset)
            new_dataset_success_rates.append(new_dataset_success_rate)
            new_dataset_avg_proof_lengths.append(new_dataset_avg_proof_length)
            record["new_dataset_success_rates"] = new_dataset_success_rates
            record["new_dataset_avg_proof_lengths"] = new_dataset_avg_proof_lengths

        # Train
        time0 = time()
        if updates >= args.updates:
            break
        train_loss, train_lemma_acc, train_ent_acc, train_name_acc, updates = \
            train_epoch(model, train_dataset, optimizer, updates)
        train_losses.append(train_loss)
        train_lemma_accs.append(train_lemma_acc)
        train_ent_accs.append(train_ent_acc)
        train_name_accs.append(train_name_acc)
        timing["epoch_train"] = time() - time0

        # Eval
        if len(val_dataset) > 0:
            time0 = time()
            val_loss, val_lemma_acc, val_ent_acc, val_name_acc = validate(model, val_dataset)
            val_losses.append(val_loss)
            val_lemma_accs.append(val_lemma_acc)
            val_ent_accs.append(val_ent_acc)
            val_name_accs.append(val_name_acc)
            timing["epoch_val"] = time() - time0

        # Test
        time0 = time()
        test_loss, test_lemma_acc, test_ent_acc, test_name_acc = validate(model, eval_dataset)
        test_losses.append(test_loss)
        test_lemma_accs.append(test_lemma_acc)
        test_ent_accs.append(test_ent_acc)
        test_name_accs.append(test_name_acc)
        timing["epoch_test"] = time() - time0

        time0 = time()
        record["progress"] = "{}%".format((epoch) / args.epoch * 100)
        record["train_losses"] = train_losses
        record["train_lemma_accs"] = train_lemma_accs
        record["train_ent_accs"] = train_ent_accs
        record["train_name_accs"] = train_name_accs
        record["val_losses"] = val_losses
        record["val_lemma_accs"] = val_lemma_accs
        record["val_ent_accs"] = val_ent_accs
        record["val_name_accs"] = val_name_accs
        record["test_lemma_accs"] = test_lemma_accs
        record["test_ent_accs"] = test_ent_accs
        record["test_name_accs"] = test_name_accs
        print("Train Lemma acc: {}".format(train_lemma_acc))
        print("Train ent acc: {}".format(train_ent_acc))
        if len(val_dataset) > 0:
            print("Val Lemma acc: {}".format(val_lemma_acc))
            print("Val ent acc: {}".format(val_ent_acc))
        print("Test Lemma acc: {}".format(test_lemma_acc))
        print("Test ent acc: {}".format(test_ent_acc))
        # print("name acc: {}".format(name_acc))

        json.dump(record, open(os.path.join(args.dump, str(timestamp), "record.json"), "w"))

        timing["epoch_saving"] = time() - time0

        if resume_dir:
            save_checkpoint(model, optimizer, ckpt_dir=resume_dir, epoch=epoch,
                            extra=dict(epoch=epoch, updates=updates, best_val_succ=best_val_succ), is_last=True)

        if epoch % args.epoch_per_case_record == 0:
            print("rollout")
            # First-step rollouts
            time0 = time()
            train_first_success_rate, train_first_wrong_case, train_first_right_case, train_first_avg_proof_length = \
                test_rollout(model, train_first_dataset)
            train_first_success_rates.append(train_first_success_rate)
            train_first_avg_proof_lengths.append(train_first_avg_proof_length)
            print("train success rate {}".format(train_first_success_rate))

            if len(val_dataset) > 0:
                val_first_success_rate, val_first_wrong_case, val_first_right_case = \
                    test_rollout(model, val_first_dataset)
                val_first_success_rates.append(val_first_success_rate)
                print("val success rate {}".format(val_first_success_rate))
            else:
                val_first_wrong_case = ""
                val_first_right_case = ""
            test_first_success_rate, test_first_wrong_case, test_first_right_case, test_first_avg_proof_length = \
                test_rollout(model, eval_first_dataset, whole_dataset=True)
            test_first_success_rates.append(test_first_success_rate)
            test_first_avg_proof_lengths.append(test_first_avg_proof_length)
            print("test success rate {}".format(test_first_success_rate))
            record["train_first_success_rates"] = train_first_success_rates
            record["val_first_success_rates"] = val_first_success_rates
            record["test_first_success_rates"] = test_first_success_rates
            record["train_first_avg_proof_lengths"] = train_first_avg_proof_lengths
            # record["val_first_avg_proof_lengths"] = val_first_avg_proof_lengths
            record["test_first_avg_proof_lengths"] = test_first_avg_proof_lengths
            timing["{}epoch_rollout".format(args.epoch_per_case_record)] = time() - time0

            time0 = time()
            cases_record = {
                "train_first_wrong_case": train_first_wrong_case,
                "train_first_right_case": train_first_right_case,
                "val_first_wrong_case": val_first_wrong_case,
                "val_first_right_case": val_first_right_case,
                "test_first_wrong_case": test_first_wrong_case,
                "test_first_right_case": test_first_right_case
            }

            json.dump(cases_record,
                      open(
                          os.path.join(
                              args.dump,
                              str(timestamp),
                              "cases_record{0:.0%}.json".format(int(updates / args.updates))),
                          "w")
                      )
            if test_first_success_rate > best_val_succ:
                save_checkpoint(model, optimizer, ckpt_dir=os.path.join(args.dump, str(timestamp)), epoch=epoch,
                                extra=dict(epoch=epoch, updates=updates, best_val_acc=best_val_succ), is_best=True)
                best_val_succ = test_first_success_rate
            timing["{}epoch_cases".format(args.epoch_per_case_record)] = time() - time0
            json.dump(record, open(os.path.join(args.dump, str(timestamp), "record.json"), "w"))
            torch.save(model, os.path.join(args.dump, str(timestamp), "model_checkpoint.pt"))
        json.dump(timing, open(os.path.join(args.dump, str(timestamp), "timing.json"), "w"))


def main():
    model = load_model()
    optimizer = load_optimizer(model)
    if args.online:
        if args.online_order_generation:
            kl_dict = json.load(open(os.path.join(args.combo_path, "combinations.json"), "r"))
        else:
            kl_dict = json.load(open(os.path.join(args.combo_path, "orders.json"), "r"))
        train_eval_test(model, optimizer, kl_dict=kl_dict, resume_dir=args.resume_dir)
    else:
        train_dirs = [os.path.join(args.path_to_data, train_dir) for train_dir in args.train_sets]
        test_dirs = [os.path.join(args.path_to_data, test_dir) for test_dir in args.test_sets]
        all_data = load_all_data(train_dirs, test_dirs)
        train_eval_test(model, optimizer, all_data=all_data, resume_dir=args.resume_dir)


if __name__ == "__main__":
    main()
