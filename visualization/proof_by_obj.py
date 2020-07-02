import pickle
from legacy.connection_prover_exp.connection_prover_lean import ConnectionProverLean
from proof_system.all_axioms import all_axioms
from visualization.latex_parse import logic_statement_to_latex, entity_to_latex


def display_proof(directory):
    steps = pickle.load(open(directory, "rb"))
    first_step = steps[0]
    obj_string = logic_statement_to_latex(first_step["observation"]["objectives"][0])

    proof_string = []
    proof = ConnectionProverLean(axioms=all_axioms,
                                 conditions=first_step["observation"]["ground_truth"],
                                 objectives=first_step["observation"]["objectives"])
    for step in steps:
        obs = proof.get_observation()
        step_dict = {
            "ground_truth": [logic_statement_to_latex(gt) for gt in obs["ground_truth"]],
            "objective": logic_statement_to_latex(obs["objectives"][0]),
            "lemma_chosen": step["lemma"].name,
            "operands": [entity_to_latex(ent) for ent in step["input_entities"]]
        }
        proof_string.append(step_dict)
        proof.apply_theorem(theorem=step["lemma"], operands=step["input_entities"])

    return proof_string, obj_string


if __name__ == "__main__":
    from pprint import pprint

    pprint(display_proof("../data/random_combination_dataset/1/steps_0_0.p"))
