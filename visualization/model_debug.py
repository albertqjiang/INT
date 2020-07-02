# Load a model and a dataset according to a given algos
import json
from baselines.supervised import *


def load_model(directory):
    env_config = json.load(open(directory + "env_config.json", "r"))
    env_config["resume"] = directory

    data_points, batch_size, ineq_net, optimizer, lr_scheduler, \
    update, training_losses, lemma_acces, ent_acces, name_acces, train_succ_rates, val_succ_rates, \
    saving_directory = initialize(env_config)
    return data_points, ineq_net


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--algos", required=True,
                    help="whether to retrieve model")
    args = vars(ap.parse_args())

    data_points, ineq_net = load_model(args["algos"])
    train_set = Dataset(data_points["train"])
    test1 = train_set.get_multiple([0])
    test2 = train_set.get_multiple(list(range(5)))
    batch_states1, batch_actions1, batch_name_actions1 = batch_process(test1)
    batch_states2, batch_actions2, batch_name_actions2 = batch_process(test2)
    print(batch_states1, batch_actions1, batch_name_actions1)
    print(batch_states2, batch_actions2, batch_name_actions2)
    import pdb;

    pdb.set_trace()
