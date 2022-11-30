import json
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import numpy as np
import math
import seaborn as sns
import collections

sns.set()
sns.set_style("darkgrid")


def moving_average(input_list, gamma=0):
    output_list = list()
    ma = input_list[0]
    for element in input_list:
        ma = gamma * ma + (1 - gamma) * element
        output_list.append(ma)
    return output_list


def plot_stuff(log_dir, fig_size=(8, 6)):
    config = json.load(open(os.path.join(log_dir, "env_config.json"), "r"))
    label = "bow" if config["bag_of_words"] else config["obs_mode"][:3]
    label += ", online split: {}, epd {}, lr {}, {}".format(
        config["online_train_test_split"], config["epochs_per_online_dataset"],
        config["lr"], "Forward" if not config["online_backwards"] else "Backward"
    )
    title = "Trained on {} and tested on: {}".format(config["train_dirs"], config["test_dirs"])

    record = json.load(open(os.path.join(log_dir, "record.json"), "r"))

    # training loss
    training_loss = record["train_losses"]
    quantity_to_plot = moving_average(np.array(training_loss), 0.9)
    hor_axis = np.array(range(len(quantity_to_plot)))
    f, ax = plt.subplots(figsize=fig_size)
    ax.plot(hor_axis, quantity_to_plot, label=label)
    ax.annotate(round(quantity_to_plot[-1], 2),
                xy=(hor_axis[-1], quantity_to_plot[-1]))
    ax.legend()
    plt.xlabel("Epochs trained")
    plt.ylabel("Cross entropy")
    ax.set_title("Training loss: " + title)
    plt.savefig(os.path.join(log_dir, "training_loss.jpg"))

    # Lemma accuracy
    f, ax = plt.subplots(figsize=fig_size)
    training_lemma_acc = record["train_lemma_accs"]
    quantity_to_plot = moving_average(np.array(training_lemma_acc), 0.9)
    hor_axis = np.array(range(len(quantity_to_plot)))
    ax.plot(hor_axis, quantity_to_plot, label=label + " train")
    ax.annotate(round(quantity_to_plot[-1], 2),
                xy=(hor_axis[-1], quantity_to_plot[-1]))
    test_lemma_acc = record["test_lemma_accs"]
    quantity_to_plot = moving_average(np.array(test_lemma_acc), 0.9)
    ax.plot(hor_axis, quantity_to_plot, label=label + " test")
    ax.annotate(round(quantity_to_plot[-1], 2),
                xy=(hor_axis[-1], quantity_to_plot[-1]))
    ax.legend()
    plt.xlabel("Epochs trained")
    plt.ylabel("Lemma accuracy")
    ax.set_title("Lemma accuracy: " + title)
    plt.savefig(os.path.join(log_dir, "lemma_accuracy.jpg"))

    # Entity accuracy
    f, ax = plt.subplots(figsize=fig_size)
    training_ent_acc = record["train_ent_accs"]
    quantity_to_plot = moving_average(np.array(training_ent_acc), 0.9)
    hor_axis = np.array(range(len(quantity_to_plot)))
    ax.plot(hor_axis, quantity_to_plot, label=label + " train")
    ax.annotate(round(quantity_to_plot[-1], 2),
                xy=(hor_axis[-1], quantity_to_plot[-1]))
    test_ent_acc = record["test_ent_accs"]
    quantity_to_plot = moving_average(np.array(test_ent_acc), 0.9)
    ax.plot(hor_axis, quantity_to_plot, label=label + " test")
    ax.annotate(round(quantity_to_plot[-1], 2),
                xy=(hor_axis[-1], quantity_to_plot[-1]))
    ax.legend()
    plt.xlabel("Epochs trained")
    plt.ylabel("Entity accuracy")
    ax.set_title("Entity accuracy: " + title)
    plt.savefig(os.path.join(log_dir, "entity_accuracy.jpg"))

    # Success rate
    f, ax = plt.subplots(figsize=fig_size)
    train_succ_rate = record["train_first_success_rates"]
    quantity_to_plot = moving_average(np.array(train_succ_rate), 0.2)
    hor_axis = np.array(range(len(quantity_to_plot))) * config["epoch_per_case_record"]
    ax.plot(hor_axis, quantity_to_plot, label=label + " train")
    ax.annotate(round(quantity_to_plot[-1], 2),
                xy=(hor_axis[-1], quantity_to_plot[-1]))
    test_succ_rate = record["test_first_success_rates"]
    quantity_to_plot = moving_average(np.array(test_succ_rate), 0.2)
    ax.plot(hor_axis, quantity_to_plot, label=label + " test")
    ax.annotate(round(quantity_to_plot[-1], 2),
                xy=(hor_axis[-1], quantity_to_plot[-1]))
    ax.legend()
    plt.xlabel("Epochs trained")
    plt.ylabel("Success rate")
    ax.set_title("Success rate: " + title)
    plt.savefig(os.path.join(log_dir, "success_rate.jpg"))
    return ax


def plot_against_kl(model_dir, eval_dir):
    eval_record_train = collections.defaultdict(dict)
    eval_record_test = collections.defaultdict(dict)
    for time_stamp in os.listdir(model_dir):
        if os.path.isdir(os.path.join(model_dir, time_stamp)):
            config = json.load(open(os.path.join(model_dir, time_stamp, "env_config.json"), "r"))
            signature = "Trained on {}, atten {}, lr {}, sd {}, hidden {}, {}".format(
                config["train_dirs"], config["atten_type"], config["lr"],
                config["state_dim"], config["hidden"],
                "backward" if config["online_backwards"] else "forward",

            )
            eval_rewards = json.load(open(os.path.join(eval_dir, time_stamp, "eval_rewards.json"), "r"))
            for key, value in eval_rewards.items():
                eval_record_train[key][signature] = value["train"]
                eval_record_test[key][signature] = value["test"]
    ax_train = plot_eval_bar_chart(
        eval_record_train,
        "Success rate on training combos"
    )
    ax_test = plot_eval_bar_chart(
        eval_record_test,
        "Success rate on test combos"
    )
    ax_train.savefig(os.path.join(model_dir, "train_kl.jpg"))
    ax_test.savefig(os.path.join(model_dir, "test_kl.jpg"))
    return ax_train, ax_test


def plot_eval_bar_chart(eval_record, title=''):
    performances = [eval_record[criterion] for criterion in sorted(eval_record.keys())]
    criteria = list(sorted(eval_record.keys()))
    curricula = sorted(performances[0].keys(), key=lambda x: -len(x))

    df_list = list()
    for i, criterion in enumerate(criteria):
        for j, curriculum in enumerate(curricula):
            df_list.append(
                {
                    "curriculum": curriculum,
                    "eval criterion": criterion,
                    "success rate": performances[i][curriculum]
                }
            )

    df = pd.DataFrame(df_list)
    print(df.to_string())
    ax = sns.catplot(x="eval criterion", y="success rate", hue="curriculum", data=df,
                     height=12, kind="bar", palette="Paired", )
    plt.title(title)
    return ax


def step_to_latex(step):
    step_string = ""
    step_string += "The observation is: \n"
    step_string += "Ground truth:\n"
    for gt in step["gt"]:
        step_string += "\t{}\n".format(gt)
    step_string += "Objective:\n"
    step_string += "\t{}\n".format(step["obj"][0])
    step_string += "Lemma name is: {}\n".format(step["lemma"])
    for i, in_ent in enumerate(step["input_entities"]):
        step_string += "The {}th chosen input entity is {}\n".format(i + 1, in_ent)
    return step_string


def trajs_to_str(trajectories):
    all_traj_strs = []
    for traj in trajectories:
        single_traj_str = ""
        for step in traj:
            single_traj_str += ("*" * 100 + "\n")
            single_traj_str += (step_to_latex(step) + "\n")
        all_traj_strs.append(single_traj_str)
    return all_traj_strs
