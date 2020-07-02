from proof_system.all_axioms import equal_axioms, general_axioms, unequal_axioms

if __name__ == "__main__":
    import pickle
    import os
    import random

    random.seed(0)

    directory = "../data/expansion_dataset/"
    for file in os.listdir(directory):
        if file.startswith("proof_"):
            perturbed_proof_name = file.replace("proof", "perturbed_proof")
            steps_name = file.replace("proof", "steps")
            perturbed_steps_name = file.replace("proof", "perturbed_steps")
            steps = pickle.load(open(directory + steps_name, "rb"))

            base_proof = pickle.load(open(directory + file, "rb"))
            squared_term = base_proof.objectives[0].operands[0].operands[0]
            # print(squared_term.name)
            lemma = unequal_axioms["SquareGEQZero"]
            operands = [squared_term]
            observation = base_proof.get_observation()
            step = {"observation": observation,
                    "lemma": lemma,
                    "input_entities": operands}
            steps.append(step)
            square_term = \
                base_proof.ls_id2ls[base_proof.apply_theorem(lemma, operands)["conclusion_ids"][0]].operands[0]

            lemma = equal_axioms["EquivalenceSubstitution"]
            operands = [square_term, base_proof.objectives[0].operands[1]]
            observation = base_proof.get_observation()
            step = {"observation": observation,
                    "lemma": lemma,
                    "input_entities": operands}
            steps.append(step)
            result = base_proof.apply_theorem(lemma, operands)
            ls = base_proof.ls_id2ls[result["conclusion_ids"][0]]

            sum_entity = base_proof.ls_id2ls[result["conclusion_ids"][0]].operands[0]

            entity_id_to_move = random.choice(base_proof._parse_entity_ids_from_entity(sum_entity))
            entity_to_move = base_proof.ent_id2ent[entity_id_to_move]

            lemma = general_axioms["MoveTerm"]
            operands = [entity_to_move]
            observation = base_proof.get_observation()
            result = base_proof.apply_theorem(lemma, operands)
            count = 1
            while len(result["conclusion_ids"]) == 0 and count < 100:
                # In case of infinite loop
                count += 1
                entity_id_to_move = random.choice(base_proof._parse_entity_ids_from_entity(sum_entity))
                entity_to_move = base_proof.ent_id2ent[entity_id_to_move]
                lemma = general_axioms["MoveTerm"]
                operands = [entity_to_move]
                observation = base_proof.get_observation()
                result = base_proof.apply_theorem(general_axioms["MoveTerm"], [entity_to_move])
            # print(result)
            # print(entity_to_move.root.name)
            step = {"observation": observation,
                    "lemma": lemma,
                    "input_entities": operands}
            steps.append(step)
            # print(logic_statement_to_latex(base_proof.ls_id2ls[result["conclusion_ids"][0]]))

            base_proof.objectives[0] = base_proof.ls_id2ls[result["conclusion_ids"][0]]

            pickle.dump(base_proof, open(directory + perturbed_proof_name, "wb"))
            pickle.dump(steps, open(directory + perturbed_steps_name, "wb"))
