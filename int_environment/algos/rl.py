import collections
import json
import os
import os.path as osp
import pickle
import random
import time
from collections import deque
from datetime import datetime
import numpy as np

import torch

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.storage import RolloutStorage
from algos.lib.arguments import get_args
from algos.eval import eval_agent
from algos.lib.envs import make_thm_vec_envs
from algos.lib.obs import nodename2index, thm2index
from legacy.data_generation.random_numerical_specified import load_online_combo_and_length
from algos.lib.ops import turn_grad_on_off
from algos.model.thm_model import ThmNet
from data_generation.utils import Dataset

torch.manual_seed(123)

timestamp = str(datetime.fromtimestamp(time.time())).replace(" ", "_").replace(":", "_").replace("-", "_").replace(".", "_")
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

def load_checkpoint(model, filename, optimizer=None):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['extra']


def resume_checkpoint(model, resume_dir, optimizer=None):
    name = 'last_epoch.pth'
    fname = os.path.join(resume_dir, name)
    if os.path.exists(fname):
        extra = load_checkpoint(model, fname, optimizer=optimizer)
        update = extra['update']
        best_val_succ = extra['best_val_succ']
        return update, best_val_succ
    return 0, np.inf


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


def _eval(model, args, test_dataset=None, combos_and_lengths=None):
    indices = list(range(len(test_dataset)))
    test_indices = random.choices(indices, k=100)
    test_starting_points = [test_dataset.trajectories[ind] for ind in test_indices]
    eval_env_config = {
        # Backplay shouldn't be used in evaluation
        "backplay": False,
        "bag_of_words": False,
        "batch_eval": False,
        "max_theorems": 25,
        "max_ground_truth": 50,
        "max_objectives": 1,
        "max_operands": 4,
        "max_edges": 200,
        "max_nodes": 50,
        "max_node_types": 40,
        "max_configuration": 10,
        "eval_dataset": test_starting_points,
        "online": False,
        "online_backwards": args.online_backwards,
        "combos_and_lengths": combos_and_lengths,
        "mode": "eval",
        "num_online_evals": 100,
        "verbo": True,
        "obs_mode": args.obs_mode,
        "time_limit": args.time_limit,
    }

    return eval_agent(model, env_config=eval_env_config)


def main():
    args = get_args()
    log_dir = os.path.expanduser(args.dump)
    log_dir = osp.join(log_dir, str(timestamp))
    utils.cleanup_log_dir(log_dir)
    args_dict = vars(args)
    json.dump(args_dict, open(osp.join(log_dir, "env_config.json"), "w"))


    if args.online:
        # train_combos_and_lengths, _ = load_online_combo_and_length(args.combo_path, args.train_dirs)
        # test_combos_and_lengths = None
        # train_dirs = None
        # test_dirs = [os.path.join(args.combo_path, test_dir, "test_first.pkl") for test_dir in args.test_dirs]
        pass
    else:
        train_dirs = [os.path.join(args.path_to_data, train_dir, "train_first.pkl") for train_dir in args.train_dirs]
        test_dirs = [os.path.join(args.path_to_data, test_dir, "test_first.pkl") for test_dir in args.test_dirs]
        train_combos_and_lengths = None
        test_combos_and_lengths = None
    test_datasets = dict()
    # for proof_dir in test_dirs:
    #     test_dataset = Dataset([])
    #     test_dataset.merge(pickle.load(open(proof_dir, "rb")))
    #     test_datasets[proof_dir.split("/")[-2]] = test_dataset

    kl_dict = json.load(open(args.combo_path, "r"))
    env_config = {
        "bag_of_words": args.bag_of_words,
        "max_theorems": 25,
        "mode": args.mode,
        "online": args.online,
        "obs_mode": "geometric",
        "proof_dir": args.train_sets,
        "time_limit": args.time_limit,
        "verbo": args.verbo,
        "kl_dict": kl_dict 
    }
    model_config = {
        "state_dim": args.state_dim,
        "gnn_type": args.gnn_type,
        "combined_gt_obj": True,
        "attention_type": args.atten_type,
        "hidden_layers": args.hidden,
        "entity_cost": 1.0,
        "lemma_cost": 1.0,
        "num_nodes": len(nodename2index),
        "num_lemmas": len(thm2index),
        "norm": None,
        "cuda": use_gpu 
    }

    torch.manual_seed(args.seed)
    if use_gpu:
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(0)
    random.seed(args.seed)
    np.random.seed(args.seed)


    # torch.set_num_threads(1)
    envs = make_thm_vec_envs(env_config, args.num_processes, False, log_dir=log_dir)


    actor_critic = ThmNet(**model_config)
    if args.pretrain_dir is not None:
        load_checkpoint(
            actor_critic, osp.join(args.pretrain_dir, "best_epoch.pth"))
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.resume_dir:
        update, best_vf_loss = resume_checkpoint(
            actor_critic, args.resume_dir, optimizer=agent.optimizer)
    else:
        update, best_vf_loss = 0, np.inf

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.action_space)

    obs = envs.reset()
    rollouts.obs.append(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    value_losses, action_losses = [], []
    eval_rewards = collections.defaultdict(list)

    if args.fix_policy:
        # Turn all the gradients off except for the value function
        for param in actor_critic.parameters():
            param.requires_grad = False
        turn_grad_on_off(actor_critic, "vf_net", "on")

    all_trajectories = {
        "train": [],
        "test": {
            "right": None,
            "wrong": None
        }
    }

    for j in range(update, num_updates):
        all_trajectories["train"] = []
        # Turn the gradients back on mid-training
        if j == int(5000):
            for param in actor_critic.parameters():
                param.requires_grad = True
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        start_time = time.time()
        compute_time = 0
        env_step_time = 0
        env_reset_time = 0
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                tt0 = time.time()
                action, value = actor_critic.forward(
                    rollouts.obs[step], actions=None)
                tt1 = time.time()
                compute_time += tt1 - tt0


                # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            tt2 = time.time()
            env_step_time += tt2 - tt1

            # if done:
            #     obs = envs.reset()
            tt3 = time.time()
            env_reset_time += tt3 - tt2

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
            all_trajectories["train"].append(infos)

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, torch.from_numpy(action),
                            value, torch.from_numpy(reward).view(-1, 1), masks, bad_masks)

            # rollouts.insert(obs, torch.from_numpy(action).view(1, -1),
            #                 value.view(1, 1), torch.FloatTensor([[reward]]), masks, bad_masks)

        t2 = time.time()
        print("rollout_time: %f" % (t2 - start_time))
        print("compute percentage in rollout: %f" % (compute_time / (t2 - start_time)))
        print("env step percentage in rollout: %f" % (env_step_time / (t2 - start_time)))
        print("env reset percentage in rollout: %f" % (env_reset_time / (t2 - start_time)))

        with torch.no_grad():
            next_value = actor_critic.vf(
                rollouts.obs[-1]).detach()


        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        print(value_loss)
        value_losses.append(value_loss)
        action_losses.append(action_loss)
        last_100_value_loss = np.mean(value_losses[:-1000])

        t3 = time.time()
        print("train_time: %f" % (t3 - t2))

        # print([len(ob[0][1]) + len(ob[1][1]) for ob in rollouts.obs])
        # print(sum([len(ob[0][1]) + len(ob[1][1]) for ob in rollouts.obs]))
        rollouts.after_update()

        t4 = time.time()
        print("rollout_update_time: %f" % (t4 - t3))

        # save for every interval-th episode or for the last epoch
        if last_100_value_loss < best_vf_loss:
            best_vf_loss = last_100_value_loss
            save_checkpoint(actor_critic, agent.optimizer, ckpt_dir=log_dir, epoch=j,
                            extra=dict(update=j, best_val_succ=best_vf_loss), is_best=True)

        if args.resume_dir:
            save_checkpoint(actor_critic, agent.optimizer, ckpt_dir=args.resume_dir, epoch=j,
                            extra=dict(update=j, best_val_succ=best_vf_loss), is_last=True)

        if (j % args.save_interval == 0
            or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(log_dir, "models")
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            save_checkpoint(actor_critic, agent.optimizer, ckpt_dir=log_dir, epoch=j,
                            extra=dict(update=j, best_val_succ=best_vf_loss), is_last=True)
            json.dump(value_losses, open(os.path.join(log_dir, "value_losses.json"), "w"))
            json.dump(action_losses, open(os.path.join(log_dir, "action_losses.json"), "w"))

        t5 = time.time()
        print("saving_time: %f" % (t5 - t4))

        if j % 2 == 1:  # and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            # print(
            #     "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
            #     .format(j, total_num_steps,
            #             int(total_num_steps / (end - start)),
            #             len(episode_rewards), np.mean(episode_rewards),
            #             np.median(episode_rewards), np.min(episode_rewards),
            #             np.max(episode_rewards), dist_entropy, value_loss,
            #             action_loss))
            print(
                "Updates {}, num timesteps {}, FPS {} \n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            dist_entropy, value_loss,
                            action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
            and j % args.eval_interval == 0):
            print(j)
            # eval_envs.reset()
            # ob_rms = utils.get_vec_normalize(eval_envs).ob_rms
            # TODO: validate this
            for key, dataset in test_datasets.items():
                success_rate, wrong_cases, right_cases = _eval(actor_critic, args, test_dataset=dataset)
                eval_rewards[key].append(success_rate)
                all_trajectories["test"]["right"] = right_cases
                all_trajectories["test"]["wrong"] = wrong_cases
            json.dump(eval_rewards, open(os.path.join(log_dir, "eval_rewards.json"), "w"))
            json.dump(all_trajectories, open(os.path.join(log_dir, "all_trajectories.json"), "w"))
        t6 = time.time()
        print("eval_time: %f" % (t6 - t5))
        print("rollout time percentage: %f\n" % ((t2 - start_time) / (t6 - start_time)))


if __name__ == "__main__":
    main()
