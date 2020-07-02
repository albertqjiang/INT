from visualization.latex_parse import logic_statement_to_latex, traj_path_to_str
from logic.logic import Entity
from logic.utils import standard_logic_functions, standard_numerical_functions
from legacy.data_generation.random_numerical_specified import specify_problem
from proof_system.all_axioms import all_axioms
from algos.lib.envs import make_thm_env
from algos.eval import eval_agent
import torch

add = standard_numerical_functions["add"]
mul = standard_numerical_functions["mul"]
sqr = standard_numerical_functions["sqr"]
inv = standard_numerical_functions["inv"]
BiggerOrEqual = standard_logic_functions["BiggerOrEqual"]
Equivalent = standard_logic_functions["Equivalent"]


def gen_problem():
    AMDL = 'AdditionMultiplicationLeftDistribution'
    AMDR = 'AdditionMultiplicationRightDistribution'
    AA = "AdditionAssociativity"
    AC = "AdditionCommutativity"
    MA = "MultiplicationAssociativity"
    MC = "MultiplicationCommutativity"
    eval_data = specify_problem(axiom_list={AC: all_axioms[AC],
                                            # AA:all_axioms[AA],
                                            MC: all_axioms[MC],
                                            # MA:all_axioms[MA],
                                            }, length=3)[0]
    return eval_data


def gen_true_gt():
    a = Entity(name="a")
    b = Entity(name="b")
    c = Entity(name="c")
    b_a = mul.execute_nf([b, a])
    c_c = add.execute_nf([c, c])
    b_a_c = mul.execute_nf([b_a, inv.execute_nf([c])])
    gt = Equivalent.execute_lf([add.execute_nf([b_a_c, c_c]),
                                add.execute_nf([c_c, b_a_c])])
    return gt


def gen_fake_gt():
    a = Entity(name="a")
    b = Entity(name="b")
    c = Entity(name="c")
    b_a = add.execute_nf([b, a])
    c_c = mul.execute_nf([c, c])
    b_a_c = mul.execute_nf([b_a, sqr.execute_nf([c])])
    gt = Equivalent.execute_lf([add.execute_nf([b_a_c, c_c]),
                                add.execute_nf([c_c, b_a_c])])
    return gt


def gen_fake_gt2():
    a = Entity(name="a")
    b = Entity(name="b")
    c = Entity(name="c")
    b_a = add.execute_nf([b, a])
    c_c = mul.execute_nf([c, c])
    b_a_c = mul.execute_nf([b_a, sqr.execute_nf([c])])
    gt = Equivalent.execute_lf([add.execute_nf([c_c, c_c]),
                                add.execute_nf([c_c, c_c])])
    return gt


def gen_fake_gt3():
    a = Entity(name="a")
    b = Entity(name="b")
    c = Entity(name="c")
    b_a = add.execute_nf([b, a])
    c_c = mul.execute_nf([c, c])
    b_a_c = mul.execute_nf([b_a, sqr.execute_nf([c])])
    gt = Equivalent.execute_lf([add.execute_nf([b_a_c, b]),
                                add.execute_nf([b_a_c, b])])
    return gt


eval_data = gen_problem()
model = torch.load("../../results/2019_11_29_07_10_24_196992/model_checkpoint.pt", map_location=torch.device('cpu'))


def run_one_step(agent, env_config, log_dir=None):
    # Evaluate policy rollout success rates and record right and wrong cases
    # TODO: get multi-processing working

    env = make_thm_env(env_config, log_dir=log_dir)()

    actor_critic = agent
    obs = env.reset(index=0)
    # Sample actions
    with torch.no_grad():
        action, value = actor_critic.forward([obs])
        # action, value = actor_critic.compute_action(obs)
        # Obser reward and next obs
    env.step(action[0])
    return env.proof


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
    "eval_dataset": [eval_data],
    "batch_eval": False,
    "verbo": True,
    "obs_mode": "geometric",
    "bag_of_words": False,
    "time_limit": 10}
after_one_step = run_one_step(model, env_config)
objectives = after_one_step.get_objectives()
# conditions = gen_true_gt()
# conditions = gen_fake_gt2()
conditions = gen_fake_gt3()
print(logic_statement_to_latex(conditions))

obs = {"objectives": objectives,
       "ground_truth": [conditions]}
eval_data = {"observation": obs}
env_config["eval_dataset"] = [eval_data]
succ, wrong, right = eval_agent(model, env_config)

try:
    print(traj_path_to_str(wrong)[0])
except:
    print(traj_path_to_str(right)[0])
print(succ)
