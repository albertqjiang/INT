import random

random.seed(0)
from visualization.latex_parse import logic_statement_to_latex


def random_walk_on_proof(base_proof, random_walk_iterations=100):
    axioms = list(base_proof.axioms.values())
    random_walk_destinations = list()
    for _ in range(random_walk_iterations):
        lemma = random.choice(axioms)
        operands = random.choices(base_proof.get_entities(), k=lemma.input_no)
        result = base_proof.apply_theorem(lemma, operands)
        if result is not None and len(result["conclusion_ids"]) > 0:
            random_walk_destinations.append(logic_statement_to_latex(base_proof.ls_id2ls[result["conclusion_ids"][0]]))
    return random_walk_destinations


if __name__ == "__main__":
    import pickle
    import os

    configuration_counter = 0
    directory = "../data/expansion_dataset/"
    for file in os.listdir(directory):
        if file.startswith("proof_"):
            configuration_counter += 1
            print("This is configuration {} \n".format(configuration_counter))
            base_proof = pickle.load(open(directory + file, "rb"))
            print("Base logic statement is:")
            print(r"\begin{equation}")
            print(logic_statement_to_latex(base_proof.objectives[0]))
            print(r"\end{equation}")
            print()
            destinations = random_walk_on_proof(base_proof)
            for i, destination in enumerate(destinations):
                print("Problem {} is:".format(i + 1))
                print(r"\begin{equation}")
                print(destination)
                print(r"\end{equation}")
                print()
