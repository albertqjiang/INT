import json
import os
import pickle
import random
import sys

sys.path.insert(0, os.path.abspath('..'))

import torch
import torch.utils.data as data_handler
import torch.optim as optim
import datetime
from algos import ThmNet
from algos import eval_agent
# from algos.model.eval_multi import eval_agent
from algos import nodename2index, thm2index, batch_process
from data_generation import Dataset
from time import time
import glob
import collections
import numpy as np

torch.manual_seed(12)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = torch.cuda.is_available()

random.seed(123)


def train(all_datapoints, batch_size, epochs, imitation_net, optimizer,
          saving_directory, timelimit,
          update=0, training_losses=[], lemma_acces=[], ent_acces=[], name_acces=[],
          val_losses=[], val_lemma_acces=[], val_ent_acces=[], val_name_acces=[],
          test_losses=[], test_lemma_acces=[], test_ent_acces=[], test_name_acces=[],
          train_succ_rates=[], val_succ_rates=[], test_succ_rates=[], lr_scheduler=None,
          update_per_record=None, bag_of_words=False):
    train_difficulties = list(all_datapoints["train"].keys())
    test_difficulties = list(all_datapoints["test"].keys())

    # Prepare recording data
    if len(val_losses) == 0:
        val_losses = dict()
        val_lemma_acces = dict()
        val_ent_acces = dict()
        val_name_acces = dict()
        test_losses = dict()
        test_lemma_acces = dict()
        test_ent_acces = dict()
        test_name_acces = dict()
        train_succ_rates = list()
        val_succ_rates = dict()
        max_val_succ_rates = dict()
        train_succ_rates = dict()
        test_succ_rates = dict()
        val_lemma_cases = dict()
        test_lemma_cases = dict()
        val_ent_lemma_cases = dict()
        test_ent_lemma_cases = dict()
        for diff in train_difficulties:
            val_losses[diff] = list()
            val_lemma_acces[diff] = list()
            val_ent_acces[diff] = list()
            val_name_acces[diff] = list()
            val_succ_rates[diff] = list()
            train_succ_rates[diff] = list()
            max_val_succ_rates[diff] = list()
            val_lemma_cases[diff] = collections.deque(maxlen=100)
            val_ent_lemma_cases[diff] = collections.deque(maxlen=100)

        for diff in test_difficulties:
            test_losses[diff] = list()
            test_lemma_acces[diff] = list()
            test_ent_acces[diff] = list()
            test_name_acces[diff] = list()
            test_succ_rates[diff] = list()
            test_lemma_cases[diff] = collections.deque(maxlen=100)
            test_ent_lemma_cases[diff] = collections.deque(maxlen=100)

    time_record = dict()
    train_data = list()
    for train_list in all_datapoints["train"].values():
        train_data.extend(train_list)
    train_dataset = Dataset(train_data)
    val_datasets = {
        diff: Dataset(val_data) for diff, val_data in all_datapoints["val"].items()
    }
    test_datasets = {
        diff: Dataset(test_data) for diff, test_data in all_datapoints["test"].items()
    }

    for epoch in range(epochs):
        print("epoch%i" % epoch)
        time0 = time()
        sampler = data_handler.RandomSampler(train_dataset)
        batcher = data_handler.BatchSampler(sampler, batch_size=batch_size, drop_last=False)
        time_record["initial_batching"] = time() - time0

        # Main training steps
        for indices in batcher:
            time0 = time()
            batch = train_dataset.get_multiple(indices=indices)
            time_record["sample"] = time() - time0
            time0 = time()
            batch_states, batch_actions, batch_name_actions = batch_process(batch)
            time_record["graph_construction"] = time() - time0
            time0 = time()
            log_probs, _, _, (
            lemma_acc, ent_acc, name_acc, diff_lemma_indices, diff_ent_lemma_indices) = imitation_net.batch_forward(
                batch_states, batch_actions, batch_name_actions, acc=True)
            time_record["batch_graph_forward"] = time() - time0
            time0 = time()
            loss = -log_probs.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            time_record["batch_graph_backward"] = time() - time0

            update += 1
            time_record["update"] = update
            print("training_losses:%f" % loss.cpu().detach().numpy())

            if update % (10 * update_per_record) == 1:
                batch = train_dataset.get_multiple(indices=random.choices(list(range(len(train_dataset))), k=100))
                batch_states, batch_actions, batch_name_actions = batch_process(batch, bag=bag_of_words)
                log_probs, _, _, (lemma_acc, ent_acc, name_acc, diff_lemma_indices, _) = imitation_net.batch_forward(
                    batch_states, batch_actions, batch_name_actions, acc=True)
                loss = -log_probs.mean()
                training_losses.append(loss.cpu().item())
                lemma_acces.append(lemma_acc.cpu().item())
                # print("eval train set size: {}".format(len(batch)))
                print("train lemma accs: {}".format(lemma_acc.cpu().item()))
                ent_acces.append(ent_acc.cpu().item())
                name_acces.append(name_acc.cpu().item())

                torch.save({
                    'model': imitation_net,
                    'optimizer': optimizer,
                    'model_state_dict': imitation_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'update': update,
                }, "{}/model_checkpoint.pt".format(saving_directory))
                json.dump(training_losses, open("{}/training_losses.json".format(saving_directory), "w"))
                json.dump(lemma_acces, open("{}/lemma_accuracies.json".format(saving_directory), "w"))
                json.dump(ent_acces, open("{}/ent_accuracies.json".format(saving_directory), "w"))
                json.dump(name_acces, open("{}/name_accuracies.json".format(saving_directory), "w"))

            if update % (10 * update_per_record) == 1:
                all_wrong_cases = dict()
                all_success_cases = dict()

                if len(train_data) > 0:
                    for diff in all_datapoints["train_first"].keys():
                        train_success_rate, train_wrong_cases, train_success_cases = \
                            eval_agent(ineq_net, random.choices(all_datapoints["train_first"][diff], k=100),
                                       num_processes=5, timelimit=timelimit, bag_of_words=bag_of_words)
                        train_succ_rates[diff].append(train_success_rate)
                        all_wrong_cases[diff] = train_wrong_cases
                        all_success_cases[diff] = train_success_cases

                    json.dump(train_succ_rates, open("{}/train_succ_rate.json".format(saving_directory), "w"))
                    json.dump(all_wrong_cases, open("{}/train_wrong_cases.json".format(saving_directory), "w"))
                    json.dump(all_success_cases, open("{}/train_success_cases.json".format(saving_directory), "w"))

                if len(val_datasets[train_difficulties[0]]) > 0:
                    time0 = time()
                    for diff, val_dataset in val_datasets.items():
                        val_loss, val_lemma_acc, val_ent_acc, val_name_acc, diff_lemma_indices, different_ent_lemma_indices = \
                            validate(imitation_net, val_dataset)
                        val_losses[diff].append(val_loss.cpu().item())
                        val_lemma_acces[diff].append(val_lemma_acc.cpu().item())
                        val_ent_acces[diff].append(val_ent_acc.cpu().item())
                        val_name_acces[diff].append(val_name_acc.cpu().item())
                        val_lemma_cases[diff].append(diff_lemma_indices)
                        val_ent_lemma_cases[diff].append(different_ent_lemma_indices)
                        # print("val set size: {}".format(len(val_dataset)))
                        print("val lemma accs: {}".format(val_lemma_acc.cpu().item()))
                    time_record["validation"] = time() - time0

                    json.dump(val_losses, open("{}/val_losses.json".format(saving_directory), "w"))
                    json.dump(val_lemma_acces, open("{}/val_lemma_accuracies.json".format(saving_directory), "w"))
                    json.dump(val_ent_acces, open("{}/val_ent_accuracies.json".format(saving_directory), "w"))
                    json.dump(val_name_acces, open("{}/val_name_accuracies.json".format(saving_directory), "w"))
                    np.save("{}/val_lemma_cases.npy".format(saving_directory), val_lemma_cases)
                    np.save("{}/val_ent_lemma_cases.npy".format(saving_directory), val_ent_lemma_cases)
                    # json.dump({diff: list(value) for diff, value in val_ent_lemma_cases.items()},
                    #           open("{}/val_ent_lemma_cases.json".format(saving_directory), "w"))

                    time0 = time()
                    for diff, test_dataset in test_datasets.items():
                        test_loss, test_lemma_acc, test_ent_acc, test_name_acc, diff_lemma_indices, different_ent_lemma_indices = \
                            validate(imitation_net, test_dataset)
                        test_losses[diff].append(test_loss.cpu().item())
                        test_lemma_acces[diff].append(test_lemma_acc.cpu().item())
                        test_ent_acces[diff].append(test_ent_acc.cpu().item())
                        test_name_acces[diff].append(test_name_acc.cpu().item())
                        test_lemma_cases[diff].append(diff_lemma_indices)
                        test_ent_lemma_cases[diff].append(different_ent_lemma_indices)
                        # print("test set size: {}".format(len(test_dataset)))
                        print("test lemma accs: {}".format(test_lemma_acc.cpu().item()))
                    time_record["test"] = time() - time0
                    json.dump(test_losses, open("{}/test_losses.json".format(saving_directory), "w"))
                    json.dump(test_lemma_acces, open("{}/test_lemma_accuracies.json".format(saving_directory), "w"))
                    json.dump(test_ent_acces, open("{}/test_ent_accuracies.json".format(saving_directory), "w"))
                    json.dump(test_name_acces, open("{}/test_name_accuracies.json".format(saving_directory), "w"))
                    np.save("{}/test_lemma_cases.npy".format(saving_directory), test_lemma_cases)
                    np.save("{}/test_ent_lemma_cases.npy".format(saving_directory), test_ent_lemma_cases)

                if len(all_datapoints["test"][test_difficulties[0]]) > 0:
                    all_wrong_cases = dict()
                    all_success_cases = dict()
                    for diff in test_difficulties:
                        test_success_rate, test_wrong_cases, test_success_cases = \
                            eval_agent(ineq_net, random.choices(all_datapoints["test_first"][diff], k=100),
                                       num_processes=5, timelimit=timelimit, bag_of_words=bag_of_words)
                        test_succ_rates[diff].append(test_success_rate)
                        all_wrong_cases[diff] = test_wrong_cases
                        all_success_cases[diff] = test_success_cases
                    json.dump(test_succ_rates, open("{}/test_succ_rates.json".format(saving_directory), "w"))
                    json.dump(all_wrong_cases, open("{}/test_wrong_cases.json".format(saving_directory), "w"))
                    json.dump(all_success_cases, open("{}/test_success_cases.json".format(saving_directory), "w"))

            json.dump(time_record, open("{}/time_record.json".format(saving_directory), "w"))

        if lr_scheduler is not None:
            lr_scheduler.step()
    return update


def validate(model, val_dataset):
    indices = range(len(val_dataset))
    whole_batch = val_dataset.get_multiple(indices)
    validation_batch = random.choices(whole_batch, k=100)
    batch_states, batch_actions, batch_name_actions = batch_process(validation_batch)
    with torch.no_grad():
        log_probs, _, _, (
        lemma_acc, ent_acc, name_acc, diff_lemma_indices, different_ent_lemma_indices) = model.batch_forward(
            batch_states, batch_actions, batch_name_actions, acc=True)
        loss = -log_probs.mean()

    return loss, lemma_acc, ent_acc, name_acc, diff_lemma_indices, different_ent_lemma_indices


def initialize(env_config):
    # Unwrapping environment configuration
    lr = env_config["lr"]
    resume = env_config["resume"]
    batch_size = env_config["batch_size"]
    train_ratio = env_config["train_ratio"]
    val_ratio = env_config["val_ratio"]
    test_ratio = env_config["test_ratio"]
    train_val_diffs = env_config["train_val_diffs"]
    test_diffs = env_config["test_diffs"]
    custom_objective_dir = env_config["custom_objective_dir"]
    state_dim = env_config["state_dim"]
    gnn_type = env_config["gnn_type"]
    combined_gt_obj = env_config["combined_gt_obj"]
    attention_type = env_config["attention_type"]
    lr_decay = env_config["lr_decay"]
    epoch = env_config["epoch"]
    hidden_layers = env_config["hidden_layers"]
    norm = env_config["norm"]
    saving_directory = env_config['saving_directory']
    directory_name = custom_objective_dir

    # Getting train, val and test sets from the ratios
    # Train and val steps are split randomly from the mixed steps from the problems
    # Test steps are the first steps of every problem not seen before
    normalization_c = train_ratio + val_ratio + test_ratio
    train_ratio, val_ratio, test_ratio = \
        train_ratio / normalization_c, val_ratio / normalization_c, test_ratio / normalization_c

    all_datapoints = {
        "train": dict(),
        "train_first": dict(),
        "val": dict(),
        "val_first": dict(),
        "test": dict(),
        "test_first": dict(),
    }

    # If problems are not categorized, do that then
    # total_categories = 5
    # if not os.path.exists(directory_name + "1/"):
    #     categorize_problems(directory_name, total_categories=total_categories)

    normalization_c = train_ratio + val_ratio
    train_ratio, val_ratio = train_ratio / normalization_c, val_ratio / normalization_c
    problem_no = 0
    for category in train_val_diffs:
        train_directory_name = directory_name + "{}/train/".format(category)
        train_val_steps = list()
        train_val_first_steps = list()
        fnames = glob.glob(train_directory_name + "*")

        for fname in fnames[:1000]:
            problem_no += 1
            steps = pickle.load(open(fname, "rb"))
            train_val_steps.extend(steps)
            train_val_first_steps.extend([steps[0]])

        random.shuffle(train_val_steps)
        random.shuffle(train_val_first_steps)
        all_datapoints["train"][category] = train_val_steps[:int(train_ratio * len(train_val_steps))]
        all_datapoints["val"][category] = train_val_steps[int(train_ratio * len(train_val_steps)):]
        all_datapoints["train_first"][category] = train_val_first_steps[:int(train_ratio * len(train_val_first_steps))]
        all_datapoints["val_first"][category] = train_val_first_steps[int(train_ratio * len(train_val_first_steps)):]
    print("finish reading train data")

    # Log progress whenever 100 problems are trained
    total_train_steps = int(len(train_val_steps) * train_ratio / (train_ratio + val_ratio))
    steps_per_problem = total_train_steps / problem_no
    ten_problem_steps = 10 * steps_per_problem
    update_per_record = 100

    for category in test_diffs:
        test_directory_name = directory_name + "{}/test/".format(category)
        test_steps = list()
        test_first_steps = list()
        fnames = glob.glob(test_directory_name + "*")
        for fname in fnames[:100]:
            steps = pickle.load(open(fname, "rb"))
            test_steps.extend(steps)
            test_first_steps.extend([steps[0]])
        all_datapoints["test"][category] = test_steps
        all_datapoints["test_first"][category] = test_first_steps
    print("finish reading test data")

    options = dict(
        num_nodes=len(nodename2index),
        num_lemmas=len(thm2index),
        state_dim=state_dim,
        gnn_type=gnn_type,
        combined_gt_obj=combined_gt_obj,
        attention_type=attention_type,
        hidden_layers=hidden_layers,
        norm=norm,
        entity_cost=env_config["entity_cost"],
        lemma_cost=env_config["lemma_cost"],
        pretrain=None,
    )
    ineq_net = ThmNet(**options)
    optimizer = optim.Adam(ineq_net.parameters(), lr=lr)

    if lr_decay == "cosine":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)
    elif lr_decay == "step":
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(epoch / 10), gamma=0.75)
    elif lr_decay == "exp":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    else:
        lr_scheduler = None

    if resume is not False:
        checkpoint = torch.load("{}model_checkpoint.pt".format(resume))
        ineq_net = checkpoint['model']
        optimizer = checkpoint['optimizer']
        update = checkpoint['update']
        from pathlib import Path
        file_check = Path("{}training_losses.json".format(resume))
        if file_check.is_file():
            training_losses = \
                json.load(open("{}training_losses.json".format(resume), "r"))
        else:
            training_losses = list()
        lemma_acces = json.load(open("{}lemma_accuracies.json".format(resume), "r"))
        ent_acces = json.load(open("{}ent_accuracies.json".format(resume), "r"))
        name_acces = json.load(open("{}name_accuracies.json".format(resume), "r"))
        train_succ_rates = json.load(open("{}train_succ_rate.json".format(resume), "r"))
        # TODO: change this
        val_succ_rates = list()
        saving_directory = resume
    else:
        update = 0
        training_losses = list()
        timestamp = str(datetime.datetime.now()).split('.')[0].replace("-", "_").replace(":", "_").replace(" ", "_")
        train_lemmas_str = "_".join([str(n) for n in train_val_diffs])
        saving_directory = "{}/{}_lr{}_gt{}_atten{}_hi{}_sd{}_lemma{}/".format(saving_directory, timestamp,
                                                                               lr, gnn_type, attention_type, state_dim,
                                                                               hidden_layers, train_lemmas_str)
        if not os.path.exists(saving_directory):
            os.makedirs(saving_directory)
        json.dump(env_config, open(saving_directory + "env_config.json", "w"))
        lemma_acces = list()
        ent_acces = list()
        name_acces = list()
        train_succ_rates = list()
        val_succ_rates = list()

    return all_datapoints, batch_size, ineq_net, optimizer, lr_scheduler, \
           update, training_losses, lemma_acces, ent_acces, name_acces, train_succ_rates, val_succ_rates, \
           saving_directory, update_per_record


if __name__ == "__main__":
    env_config = {
        "lr": 1e-5,
        "epoch": 100000,
        "resume": False,
        "batch_size": 32,
        "train_ratio": 0.9,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "custom_objective_dir": None,
        "state_dim": 256,
        "gnn_type": "GIN",
        "combined_gt_obj": True,
        "attention_type": 0,
        "lr_decay": "exp",
        "hidden_layers": 3,
        "train_val_diffs": [3],
        "test_diffs": [3],
        "norm": None
    }

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-cec", "--custom_env_configuration", required=False, default=None,
                    help="whether to use a custom environment configuration")
    ap.add_argument("-e", "--epoch", required=False, default=None,
                    help="how many epochs")
    ap.add_argument("-l", "--lr", required=False, default=None,
                    help="learning rate")
    ap.add_argument("-r", "--resume", required=False, default=None,
                    help="whether to resume training")
    ap.add_argument("-hi", "--hidden", required=False, default=None,
                    help="how many hidden layers of nn")
    ap.add_argument("-bs", "--batchsize", required=False, default=None,
                    help="what batch size to use")
    ap.add_argument("-sd", "--statedim", required=False, default=None,
                    help="what state dimension to use")
    ap.add_argument("-gt", "--gnntype", required=False, default=None,
                    help="what type of GNN to use")
    ap.add_argument("-ti", "--timelimit", required=False, type=int, default=20,
                    help="what type of GNN to use")
    ap.add_argument("-atten", "--attentype", type=int, required=False, default=None,
                    help="attention type")
    ap.add_argument("-le", "--lemma_cost", type=float, required=False, default=1.0,
                    help="attention type")
    ap.add_argument("-ent", "--entity_cost", type=float, required=False, default=1.0,
                    help="attention type")
    ap.add_argument("-cod", "--custom_objective_dir", required=False, default=None,
                    help="what custom algos to use")
    ap.add_argument("--train_val_diffs", required=False, default=[3], type=str,
                    nargs="+", help="what norm to use")
    ap.add_argument("--test_diffs", required=False, default=[3], type=str,
                    nargs="+")
    ap.add_argument("-n", "--norm", required=False, default=None,
                    help="what norm to use")
    ap.add_argument("-bag", "--bag_of_words", required=False, default=False,
                    help="learning rate")
    ap.add_argument("--saving_dir", required=False, default="../pt_models/supervised_baseline/")
    args = vars(ap.parse_args())

    # Resolve terminal arguments
    if args['custom_env_configuration'] is not None:
        env_config = json.load(open(args['custom_env_configuration'], "r"))
    else:
        if args['resume'] is not None:
            env_config["resume"] = args['resume']
        if args["epoch"] is not None:
            env_config["epoch"] = int(args['epoch'])
        if args["lr"] is not None:
            env_config["lr"] = float(args["lr"])
        if args["hidden"] is not None:
            env_config["hidden_layers"] = int(args["hidden"])
        if args["batchsize"] is not None:
            env_config["batch_size"] = int(args["batchsize"])
        if args["statedim"] is not None:
            env_config["state_dim"] = int(args["statedim"])
        if args["gnntype"] is not None and args["gnntype"] in ["GCN", "GAT", "GIN", "GICN"]:
            env_config["gnn_type"] = args["gnntype"]
        if args["custom_objective_dir"] is not None:
            env_config["custom_objective_dir"] = args["custom_objective_dir"]
        if args["norm"] is not None:
            env_config["norm"] = args["norm"]
        if args["saving_dir"] is not None:
            env_config["saving_directory"] = args["saving_dir"]
        if args["attentype"] is not None:
            env_config["attention_type"] = args["attentype"]
        if args["entity_cost"] is not None:
            env_config["entity_cost"] = args["entity_cost"]
        if args["lemma_cost"] is not None:
            env_config["lemma_cost"] = args["lemma_cost"]
        if args["bag_of_words"] is not None:
            env_config["bag_of_words"] = args["bag_of_words"]
        if args["train_val_diffs"] is not None:
            env_config["train_val_diffs"] = args["train_val_diffs"]
        if args["test_diffs"] is not None:
            env_config["test_diffs"] = args["test_diffs"]

    epoch = env_config["epoch"]

    print("Start initializing")
    all_datapoints, batch_size, ineq_net, optimizer, lr_scheduler, update, \
    training_losses, lemma_acces, ent_acces, name_acces, train_succ_rates, val_succ_rates, \
    saving_directory, update_per_record = initialize(env_config)
    print("Start real training")

    update = train(
        all_datapoints,
        batch_size, epoch, ineq_net, optimizer,
        saving_directory=saving_directory,
        timelimit=args["timelimit"],
        update=update,
        training_losses=training_losses,
        lemma_acces=lemma_acces,
        ent_acces=ent_acces,
        name_acces=name_acces,
        lr_scheduler=lr_scheduler,
        update_per_record=update_per_record,
        bag_of_words=args["bag_of_words"]
    )
