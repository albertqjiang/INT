import random

import numpy as np
import torch

from algos.lib.envs import make_thm_env
random.seed(0)


def eval_agent(agent, env_config, log_dir=None):
    # Evaluate policy rollout success rates and record right and wrong cases
    # TODO: get multi-processing working

    env = make_thm_env(env_config, log_dir=log_dir)()

    actor_critic = agent
    obs = env.reset(index=0)
    successes = []
    right_cases = []
    wrong_cases = []
    previous_steps = []
    prob_index = 0
    num_steps = []
    while not env.eval_finish:
        try:
            # Sample actions
            with torch.no_grad():
                action, value = actor_critic.forward([obs])
                # action, value = actor_critic.compute_action(obs)
                # Obser reward and next obs
            obs, reward, done, infos = env.step(action[0])
        except RuntimeError:
            reward, done, infos = 0, True, "CUDA OUT OF MEMORY"
        # obs, reward, done, infos = env.step(action)
        previous_steps.append(infos)
        if done:
            prob_index += 1
            successes.extend([reward])
            if reward:
                right_cases.append(previous_steps)
                print("prob {} success!".format(prob_index))
            else:
                print("prob {} fail!".format(prob_index))
                wrong_cases.append(previous_steps)
            num_steps.append(len(previous_steps))
            obs = env.reset()
            # if done:
            #     successes.append(reward)
            previous_steps = []
    return np.mean(successes), wrong_cases, right_cases, np.mean(num_steps)


def test_rollout(model, test_dataset, env_config=None, log_dir=None, test_points=100):
    model.eval()

    if env_config is None:
        indices = list(range(len(test_dataset)))
        test_indices = random.choices(indices, k=test_points)
        test_starting_points = [test_dataset.trajectories[ind] for ind in test_indices]
        env_config = {
            "max_theorems": 25,
            "max_ground_truth": 50,
            "max_objectives": 1,
            "max_operands": 4,
            "max_edges": 200,
            "max_nodes": 50,
            "max_node_types": 40,
            "max_configuration": 10,
            "backplay": False,
            "mode": "eval",
            "online": False,
            "eval_dataset": test_starting_points,
            "batch_eval": False,
            "verbo": True,
            "obs_mode": "geometric",
            "bag_of_words": False,
            "time_limit": 10
        }

    success_rate, wrong_cases, success_cases = \
        eval_agent(model, env_config=env_config, log_dir=log_dir)
    model.train()
    return success_rate, wrong_cases, success_cases
