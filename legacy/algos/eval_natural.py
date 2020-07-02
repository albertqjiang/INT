from logic.logic import Entity
from logic.utils import standard_logic_functions, standard_numerical_functions
from legacy.data_generation.random_numerical_specified import specify_problem
from proof_system.all_axioms import all_axioms_to_prove
from proof_system.utils import is_structured
from legacy.connection_prover_exp.connection_prover_backward import ConnectionProverBack as Proof
from proof_system.field_axioms import AdditionCommutativity, AdditionAssociativity
from visualization.latex_parse import traj_path_to_str
import random
import torch
from algos.eval import eval_agent

add = standard_numerical_functions["add"]
mul = standard_numerical_functions["mul"]
sqr = standard_numerical_functions["sqr"]
BiggerOrEqual = standard_logic_functions["BiggerOrEqual"]
Equivalent = standard_logic_functions["Equivalent"]


def complete_square():
    a = Entity(name="a")
    b = Entity(name="b")
    a_sqr = sqr.execute_nf([a])
    b_sqr = sqr.execute_nf([b])
    a_mul_b = mul.execute_nf([a, b])
    a_mul_b_add_a_mul_b = add.execute_nf([a_mul_b, a_mul_b])
    a_sqr_add_b_sqr = add.execute_nf([a_sqr, b_sqr])
    objectives = [BiggerOrEqual.execute_lf([a_sqr_add_b_sqr, a_mul_b_add_a_mul_b])]
    assumptions = [BiggerOrEqual.execute_lf([a, b]),
                   BiggerOrEqual.execute_lf([a_sqr, a])]
    return assumptions, objectives


def expand_square():
    a = Entity(name="a")
    b = Entity(name="b")

    a_add_b = add.execute_nf([a, b])
    # (a+b)*(a+b)
    # a_add_b_sqr = mul.execute_nf([a_add_b, a_add_b])
    # (a+b)^2
    # a_add_b_sqr = sqr.execute_nf([a_add_b])
    a_add_b_mul_a_add_b = mul.execute_nf([a_add_b, a_add_b])
    # (a+b)*a + (a+b)*b
    s1 = add.execute_nf([mul.execute_nf([a_add_b, a]),
                         mul.execute_nf([a_add_b, b])])
    # (a+b)*a + (a+b)*b
    s2 = add.execute_nf([add.execute_nf([mul.execute_nf([a, a]), mul.execute_nf([b, a])]),
                         mul.execute_nf([a_add_b, b])])

    a_sqr = sqr.execute_nf([a])
    a_mul_a = mul.execute_nf([a, a])
    b_sqr = sqr.execute_nf([b])
    b_mul_b = mul.execute_nf([b, b])
    a_mul_b = mul.execute_nf([a, b])
    a_mul_b_add_a_mul_b = add.execute_nf([a_mul_b, a_mul_b])
    a_sqr_add_2ab_add_b_sqr = add.execute_nf([add.execute_nf([a_sqr, b_sqr]), a_mul_b_add_a_mul_b])
    a_mul_a_add_2ab_add_b_mul_b = add.execute_nf([add.execute_nf([a_mul_a, b_mul_b]), a_mul_b_add_a_mul_b])
    # a_sqr_add_2ab_add_b_sqr = add.execute_nf([add.execute_nf([a_sqr, a_mul_b_add_a_mul_b]), b_sqr])
    # objectives = [Equivalent.execute_lf([a_add_b_sqr, a_sqr_add_2ab_add_b_sqr])]
    # objectives = [Equivalent.execute_lf([s1, s2])]
    objectives = [Equivalent.execute_lf([a_add_b_mul_a_add_b, a_mul_a_add_2ab_add_b_mul_b])]
    conditions = []
    return conditions, objectives


def simple_distractor():
    a = Entity(name="a")
    b = Entity(name="b")
    c = Entity(name="c")

    a_add_b = add.execute_nf([a, b])
    b_add_a = add.execute_nf([b, a])
    a_add_b_add_c = add.execute_nf([a_add_b, c])
    b_add_a_add_c = add.execute_nf([b_add_a, c])
    objectives = [Equivalent.execute_lf([a_add_b_add_c, b_add_a_add_c])]
    conditions = []
    return conditions, objectives


def search_problem():
    a = Entity(name="a")
    b = Entity(name="b")
    c = Entity(name="c")

    a_add_b = add.execute_nf([a, b])
    c_add_a = add.execute_nf([c, a])
    lhs = add.execute_nf([a_add_b, c])
    rhs = add.execute_nf([c_add_a, b])
    objectives = [Equivalent.execute_lf([lhs, rhs])]
    conditions = []
    return conditions, objectives


def gen_problem():
    amdl = 'AdditionMultiplicationLeftDistribution'
    amdr = all_axioms_to_prove['AdditionMultiplicationRightDistribution']
    eval_data = specify_problem(axiom_list={amdl: all_axioms_to_prove[amdl]}, length=5)[0]
    return eval_data


conditions, objectives = expand_square()
# conditions, objectives = simple_distractor()
# conditions, objectives = search_problem()
obs = {"objectives": objectives,
       "ground_truth": conditions}
eval_data = {"observation": obs}
model = torch.load(
    "/scratch/hdd001/home/ajiang/pt_models/sl_backward_basic/2020_01_26_04_10_44_016325/model_checkpoint.pt"
)

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
    "time_limit": 20,
    "degree": 0
}
# amdl = all_axioms['AdditionMultiplicationLeftDistribution']

# import pdb;pdb.set_trace()
# print(proof_path_to_str(step))

succ, wrong, right, num_steps = eval_agent(model, env_config)
print(succ)

try:
    print(traj_path_to_str(wrong)[0])
except:
    print(traj_path_to_str(right)[0])


# Random pattern matching baseline
def give_valid_entities(ls, axiom):
    all_entities = []
    if axiom.name == "AdditionCommutativity":
        for ent in ls.ent_dic.values():
            if isinstance(ent, Entity) and is_structured(ent, "add"):
                all_entities.append(ent)
    elif axiom.name == "AdditionAssociativity":
        all_entities.append(ls.operands[0])
        all_entities.append(ls.operands[1])
    else:
        raise NotImplementedError
    return all_entities


def random_agent():
    conditions, objectives = search_problem()
    proof = Proof(all_axioms, conditions, objectives)
    step = 0
    AC, AA = AdditionCommutativity(), AdditionAssociativity()
    while not proof.is_proved():
        step += 1
        axiom = random.choice([AC, AA])
        all_entities = give_valid_entities(proof.get_objectives()[0], axiom)
        proof.apply_theorem(axiom, [random.choice(all_entities)])
    return step
