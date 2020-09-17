import gym
from gym import spaces
from gym.utils import seeding

from proof_system.all_axioms import all_axioms_to_prove
from proof_system.prover import Prover
from algos.lib.obs import obs_to_graphs, index2thm, obs_to_graphs_dgl
from visualization.latex_parse import logic_statement_to_latex, entity_to_latex
from data_generation.generate_problems import generate_problem
import random

random.seed(123)

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def one_hot(batch, depth):
    ones = torch.sparse.torch.eye(depth)
    return ones.index_select(0, batch)


class TheoremProver(gym.Env):
    """Theorem proving environment

    The observation is a 3 tuple of: the adjacency matrices of the entity graphs,
    the adjacency matrices of the ground truth logic statement graphs,
    and the adjacency matrices of the objective logic statement graph.

    The action is a tuple of: the index of the lemma chosen,
    and the indices of the lemma operands chosen.
    The first entity in any proof must be named NOOP.

    Reward scheme:
    proof completed: 10,
    proof proceeded: 1,
    otherwise: 0.
    """
    def __init__(self, env_config=None):
        if env_config is None:
            env_config = {
                "mode": "solve",
                "bag_of_words": False,
                "obs_mode": "geometric"
            }
        self.env_config = env_config
        self.mode = env_config["mode"]
        self.bag_of_words = env_config["bag_of_words"]
        self.verbo = False
        self.obs_mode = "geometric"
        self.online = env_config["online"]
        if "kl_dict" in env_config:
            self.kl_dict = env_config["kl_dict"]
        if "verbo" in env_config:
            self.verbo = env_config["verbo"]
        self.obs_mode = env_config["obs_mode"]
        self.eval_finish = False
        if self.mode == "eval":
            self.eval_finish = False
            if self.online:
                self.num_online_evals = env_config["num_online_evals"]
        if not self.online:
            if self.mode == "eval":
                self.dataset = env_config["eval_dataset"]
        self.action_space = spaces.MultiDiscrete(
            [25] +
            [1000] * 4
        )
        self.observation_space = spaces.Box(low=0, high=1, shape=(84, 84))
        self.spaces = None

        # Seeding
        self.rng = None
        self.seed()

        # Start the first game
        self.proof = None
        self.ind_to_ent = dict()
        self.proof_index = 0
        self.reset(index=0)

    def seed(self, seed=None):
        self.rng, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if self.mode == "eval":
            if self.eval_finish:
                return self._get_obs(), 0, False, {"eval_finish": True}

        if self.obs_mode == "seq":
            lemma, input_entities = self.proof.parser.find_action(self.proof.get_observation(), action)
        else:
            lemma = all_axioms_to_prove[index2thm[action[0]]]
            input_entities = list()
            for input_entity_index in action[1:]:
                input_entity_index -= 1
                if input_entity_index != -1:
                    try:
                        entity = self.ind_to_ent[input_entity_index][0]
                    except:
                        print(self.ind_to_ent)
                        print(input_entity_index)
                    input_entities.append(entity)

        if self.verbo:
            info = {
                "lemma": lemma.name,
                "input_entities": [entity_to_latex(ent) for ent in input_entities],
                "gt": [logic_statement_to_latex(gt) for gt in self.proof.get_ground_truth()],
                "obj": [logic_statement_to_latex(gt) for gt in self.proof.get_objectives()]
            }
        else:
            info = {}

        if lemma.input_no != len(input_entities):
            # Operand size mismatch
            print(lemma.name)
            print(action)
            print("lemma input no %i is not %i"%(lemma.input_no, len(input_entities)))
            raise ValueError("REWARD_OPERAND_SIZE_MISMATCH")
        else:
            result = self.proof.apply_theorem(lemma, input_entities)
        info_string = self.proof.interpret_result(result)
        if info_string == "REWARD_THEOREM_PROCEEDED" and False:
            print("REWARD_THEOREM_PROCEEDED")
            print(index2thm[action[0]])
            print([ent.name for ent in input_entities])
            print(action)
        if self.mode == "solve" or self.mode == "eval":
            reward, done = {
                "REWARD_PROOF_COMPLETE": (1, True),
                "REWARD_THEOREM_PROCEEDED": (0, False),
                "REWARD_ASSUMPTION_INVALID": (0, False),
                "REWARD_DUPLICATED_RESULTS": (0, False),
            }[info_string]
        elif self.mode == "discover":
            if result is None:
                reward = 0
            else:
                reward = {
                    "REWARD_PROOF_COMPLETE":
                        sum([self.proof.ls_id2ls[ls_id].degree for ls_id in result["conclusion_ids"]]),
                    "REWARD_THEOREM_PROCEEDED":
                        sum([self.proof.ls_id2ls[ls_id].degree for ls_id in result["conclusion_ids"]]),
                    "REWARD_ASSUMPTION_INVALID":
                        0,
                    "REWARD_DUPLICATED_RESULTS":
                        0
                }[info_string]
            done = False
        else:
            raise NotImplementedError

        self.done = done
        self.reward = reward
        if self.mode == "eval":
            info.update({"eval_finish": self.eval_finish})
        # print(self.index)
        if done:
            return (None, None, None, None), reward, done, info
        else:
            return self._get_obs(), reward, done, info

    def _get_obs(self):
        if self.obs_mode == "seq":
            return self.parser.observation_to_source(self.proof.get_observation())

        obs = {
            "objectives": self.proof.get_objectives(),
            "ground_truth": self.proof.get_ground_truth()
        }
        if self.obs_mode == "dgl":
            obj_graph, gt_graph, node_ent, node_name, ent_dic, name_dic = obs_to_graphs_dgl(obs)
        else:
            obj_graph, gt_graph, node_ent, node_name, ent_dic, name_dic = obs_to_graphs(obs, bag=self.bag_of_words)
        self.ind_to_ent = {v[0]: (k, v[1]) for k, v in ent_dic.items()}

        return obj_graph, gt_graph, node_ent, node_name

    def reset(self, index=None):
        import time
        t0 = time.time()
        if self.online:
            if self.mode == "eval":
                if self.index == self.num_online_evals - 1:
                    self.eval_finish = True
                else:
                    self.index += 1

            kl = random.choice(self.env_config["proof_dir"])
            combos, prob_length = int(kl[2]), int(kl[6])
            # take first step
            step = generate_problem(combos, prob_length, "train", backwards=True, orders=self.kl_dict)[0]
            conditions = step["observation"]["ground_truth"]
            objectives = step["observation"]["objectives"]
            self.proof = Prover(axioms=all_axioms_to_prove, conditions=conditions,
                                objectives=objectives, prove_direction="backward")

        else:
            if index is not None:
                self.index = index
            else:
                if self.mode == "eval":
                    if self.index == len(self.dataset) - 1:
                        self.eval_finish = True
                    else:
                        self.index += 1
                else:
                    self.index = random.choices(range(len(self.dataset)))[0]
            step = self.dataset[self.index]
            conditions = step["observation"]["ground_truth"]
            objectives = step["observation"]["objectives"]
            self.proof = Prover(axioms=all_axioms_to_prove, conditions=conditions,
                                objectives=objectives, prove_direction="backward")

        obs = self._get_obs()
        return obs

    def render(self, mode='human'):
        raise NotImplemented
