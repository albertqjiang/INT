from proof_system.all_axioms import all_axioms_to_prove
from proof_system.prover import Prover
from visualization.latex_parse import logic_statement_to_latex


def generate_valid_steps(steps):
    # This assembles operands in the current graphs
    # As long as the roots of the operands are in the graphs, we can synthesize valid steps
    valid_steps = list()
    if len(steps) == 0:
        return list()
    test_proof = Prover(axioms=all_axioms_to_prove,
                        conditions=steps[0]["observation"]["ground_truth"],
                        objectives=steps[0]["observation"]["objectives"],
                        prove_direction="backward")
    # assert not test_proof.is_proved()
    for step in steps:
        if test_proof.is_proved():
            break
        for op in step["input_entities"]:
            # Make sure the root of each operand is in the current graphs
            # assert op.root in step["observation"]["objectives"] + step["observation"]["ground_truth"]
            assert op.root is not None
            # assert op.root.name in [ls.name for ls in test_proof.get_ground_truth() + test_proof.get_objectives()]
        if step["lemma"].name == "EquivalenceSubstitution":
            assembled_operands = list()
            op1, op2 = step["input_entities"]
            print(logic_statement_to_latex(op1.root))
            print(logic_statement_to_latex(op2.root))
            replacement1 = test_proof.ls_id2ls[test_proof.ls_name2id[op1.root.name]].ent_dic[op1.index]
            assembled_operands.append(replacement1)

            available_ents = []
            all_lss = test_proof.get_objectives() + test_proof.get_ground_truth()
            for ls in all_lss:
                available_ents.extend([ls.ent_dic[key] for key in ls.ent_dic if key != 0])

            for ent in available_ents:
                if ent.name == op2.name:
                    replacement2 = ent
                    assembled_operands.append(replacement2)
                    break
            assert len(assembled_operands) == 2

        else:
            assembled_operands = list()
            available_ents = []
            all_lss = test_proof.get_objectives() + test_proof.get_ground_truth()
            for ls in all_lss:
                available_ents.extend([ls.ent_dic[key] for key in ls.ent_dic if key != 0])
            for op in step["input_entities"]:
                for ent in available_ents:
                    if op.name == ent.name:
                        assembled_operands.append(ent)
                        break
            assert len(assembled_operands) == step["lemma"].input_no

        lemma = step["lemma"]
        operands = assembled_operands
        step = {
            "observation": test_proof.get_observation(),
            "lemma": lemma,
            "input_entities": operands
        }
        valid_steps.append(step)
        test_proof.apply_theorem(theorem=step["lemma"], operands=assembled_operands)

    # Make sure the proof is complete when all steps are carried out
    assert test_proof.is_proved()
    return valid_steps
