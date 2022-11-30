import re

from logic.logic import Entity
from logic.utils import standard_numerical_functions


def split_count(s, count):
    chunks = list()
    chunk_index = -1
    for i in range(len(s)):
        if int(i / count) != chunk_index:
            chunk_index = int(i / count)
            chunks.append(list(s[i]))
        else:
            chunks[-1].append(s[i])
    for i in range(len(chunks)):
        chunks[i] = "".join(chunks[i])
    return chunks


def extract_two_operands(entity_name):
    parenthesis_stack = list()
    # +10 to ensure the error gets reported if an anomaly happens
    separation_index = len(entity_name) + 10
    for index, character in enumerate(entity_name):
        if character == "(":
            parenthesis_stack.append(1)
        elif character == ")":
            parenthesis_stack.pop()
        elif character == "," and len(parenthesis_stack) == 0:
            separation_index = index
            break
    # print([entity_name[:separation_index], entity_name[separation_index+1:]])
    return [entity_name[:separation_index], entity_name[separation_index + 1:]]


def parse(entity_name):
    if entity_name.startswith("add"):
        outmost_numerical_function = "add"
        entity_name = entity_name.lstrip(outmost_numerical_function)
        entity_name = entity_name.lstrip("(")
        entity_name = entity_name.rstrip(")")
        two_operands = extract_two_operands(entity_name)
        two_operands_latex = [parse(operand) for operand in two_operands]
        return "(" + "+".join(two_operands_latex) + ")"
    elif entity_name.startswith("sub"):
        outmost_numerical_function = "sub"
        entity_name = entity_name.lstrip(outmost_numerical_function)
        entity_name = entity_name.lstrip("(")
        entity_name = entity_name.rstrip(")")
        two_operands = extract_two_operands(entity_name)
        two_operands_latex = [parse(operand) for operand in two_operands]
        return "(" + "-".join(two_operands_latex) + ")"
    elif entity_name.startswith("mul"):
        outmost_numerical_function = "mul"
        entity_name = entity_name.lstrip(outmost_numerical_function)
        entity_name = entity_name.lstrip("(")
        entity_name = entity_name.rstrip(")")
        two_operands = extract_two_operands(entity_name)
        two_operands_latex = [parse(operand) for operand in two_operands]
        return "(" + "*".join(two_operands_latex) + ")"
    elif entity_name.startswith("opp"):
        outmost_numerical_function = "opp"
        entity_name = entity_name.lstrip(outmost_numerical_function)
        entity_name = entity_name.lstrip("(")
        entity_name = entity_name.rstrip(")")
        return "(-" + parse(entity_name) + ")"
    elif entity_name.startswith("sqr"):
        outmost_numerical_function = "sqr"
        entity_name = entity_name.lstrip(outmost_numerical_function)
        entity_name = entity_name.lstrip("(")
        entity_name = entity_name.rstrip(")")
        return "(" + parse(entity_name) + "^2" + ")"
    elif entity_name.startswith("sqrt"):
        outmost_numerical_function = "sqrt"
        entity_name = entity_name.lstrip(outmost_numerical_function)
        entity_name = entity_name.lstrip("(")
        entity_name = entity_name.rstrip(")")
        return "(" + r"\sqrt(" + parse(entity_name) + "))"
    elif entity_name.startswith("inv"):
        outmost_numerical_function = "inv"
        entity_name = entity_name.lstrip(outmost_numerical_function)
        entity_name = entity_name.lstrip("(")
        entity_name = entity_name.rstrip(")")
        return "(" + r"\frac{1}{" + parse(entity_name) + "})"
    else:
        return entity_name


def entity_to_latex(entity, string=False):
    if string:
        entity_name = entity
    else:
        entity_name = entity.name
    entity_name = re.sub(" ", "", entity_name)
    entity_name = re.sub("input1", "a", entity_name)
    entity_name = re.sub("input2", "b", entity_name)
    entity_name = re.sub("input3", "c", entity_name)
    return parse(entity_name)


def logic_statement_to_latex(logic_statement, string=False):
    if string:
        logic_statement_name = logic_statement
    else:
        logic_statement_name = logic_statement.name
    logic_statement_name = re.sub(" ", "", logic_statement_name)
    logic_statement_name = re.sub("input1", "a", logic_statement_name)
    logic_statement_name = re.sub("input2", "b", logic_statement_name)
    logic_statement_name = re.sub("input3", "c", logic_statement_name)
    if logic_statement_name.startswith("Real"):
        logic_function_name = "Real"
        logic_statement_name = logic_statement_name.lstrip(logic_function_name)
        logic_statement_name = logic_statement_name.lstrip("(")
        logic_statement_name = logic_statement_name.rstrip(")")
        return parse(logic_statement_name) + " is real."
    elif logic_statement_name.startswith("NonNegative"):
        logic_function_name = "NonNegative"
        logic_statement_name = logic_statement_name.lstrip(logic_function_name)
        logic_statement_name = logic_statement_name.lstrip("(")
        logic_statement_name = logic_statement_name.rstrip(")")
        return parse(logic_statement_name) + " is non-negative."
    elif logic_statement_name.startswith("BiggerOrEqual"):
        logic_function_name = "BiggerOrEqual"
        logic_statement_name = logic_statement_name.lstrip(logic_function_name)
        logic_statement_name = logic_statement_name.lstrip("(")
        logic_statement_name = logic_statement_name.rstrip(")")
        two_operands = extract_two_operands(logic_statement_name)
        two_operands_latex = [parse(operand) for operand in two_operands]
        return r"\geq ".join(two_operands_latex)
    elif logic_statement_name.startswith("SmallerOrEqual"):
        logic_function_name = "SmallerOrEqual"
        logic_statement_name = logic_statement_name.lstrip(logic_function_name)
        logic_statement_name = logic_statement_name.lstrip("(")
        logic_statement_name = logic_statement_name.rstrip(")")
        two_operands = extract_two_operands(logic_statement_name)
        two_operands_latex = [parse(operand) for operand in two_operands]
        return r"\leq ".join(two_operands_latex)
    elif logic_statement_name.startswith("Equivalent"):
        logic_function_name = "Equivalent"
        logic_statement_name = logic_statement_name.lstrip(logic_function_name)
        logic_statement_name = logic_statement_name.lstrip("(")
        logic_statement_name = logic_statement_name.rstrip(")")
        two_operands = extract_two_operands(logic_statement_name)
        two_operands_latex = [parse(operand) for operand in two_operands]
        return r"=".join(two_operands_latex)
    else:
        raise NotImplementedError


def step_to_latex(step):
    step_string = ""
    step_string += "The observation: \n"
    step_string += "Ground truth:\n"
    for gt in step["observation"]["ground_truth"]:
        step_string += "\t{}\n".format(logic_statement_to_latex(gt))
    step_string += "Objective:\n"
    step_string += "\t{}\n".format(logic_statement_to_latex(step["observation"]["objectives"][0]))
    step_string += "Lemma name is: {}\n".format(step["lemma"].name)
    for i, in_ent in enumerate(step["input_entities"]):
        step_string += "The {}th chosen input entity is {}\n".format(i + 1, entity_to_latex(in_ent))
    return step_string


def proof_path_to_str(steps):
    proof_str = ""
    for step in steps:
        proof_str += ("*" * 100 + "\n")
        proof_str += (step_to_latex(step) + "\n")
    return proof_str


def step_to_latex2(step):
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


def traj_path_to_str(trajectories):
    all_traj_strs = []
    for traj in trajectories:
        single_traj_str = ""
        for step in traj:
            single_traj_str += ("*" * 100 + "\n")
            single_traj_str += (step_to_latex2(step) + "\n")
        all_traj_strs.append(single_traj_str)
    return all_traj_strs


def decorate_string(s):
    if len(s) > 50:
        return s.replace(r"=", r"= \\ ").replace(r"\geq", r"\geq \\ ").replace(r"\leq", r"\leq \\ ")
    return s


if __name__ == "__main__":
    a = Entity(name="a")
    b = Entity(name="b")
    c = Entity(name="c")
    a_and_b = standard_numerical_functions["add"].execute_nf([a, b])
    a_and_b_sub_c = standard_numerical_functions["sub"].execute_nf([a_and_b, c])
    entity = standard_numerical_functions["sqr"].execute_nf([a_and_b_sub_c])

    import pickle
    from legacy.connection_prover_exp.connection_prover import ConnectionProver
    from proof_system.all_axioms import all_axioms
    from pprint import pprint

    steps_by_objective = dict()
    steps = pickle.load(open("../data/expansion_dataset/steps_1.p", "rb"))
    print("Total steps: {}".format(len(steps)))

    proof = ConnectionProver(axioms=all_axioms, conditions=[], objectives=steps[0]["observation"]["objectives"])
    pprint([logic_statement_to_latex(gt) for gt in proof.get_observation()["ground_truth"]])
    for i, step in enumerate(steps):
        pprint(step["lemma"].name)
        pprint([entity_to_latex(ent) for ent in step["input_entities"]])
        result = proof.apply_theorem(theorem=step["lemma"], operands=step["input_entities"])
        pprint(result)
        pprint([logic_statement_to_latex(gt) for gt in proof.get_observation()["ground_truth"]])
    assert proof.is_proved()
