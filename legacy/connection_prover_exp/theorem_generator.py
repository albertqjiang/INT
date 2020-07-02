from proof_system.all_axioms import equal_axioms
from proof_system.numerical_functions import necessary_numerical_functions
from proof_system.logic_functions import necessary_logic_functions
from legacy.connection_prover_exp.connection_prover import ConnectionProver
from logic.logic import Entity
from visualization.latex_parse import logic_statement_to_latex, decorate_string


def get_all_premises(prover, ls):
    ls_id = prover.ls_name2id[ls.name]
    all_premises = list()
    all_premise_names = set()
    if len(prover.ls_id2input_gates[ls_id]) == 0:
        return []
    for gate in prover.ls_id2input_gates[ls_id]:
        for input_ls_id in gate.get_inputs():
            input_ls = prover.ls_id2ls[input_ls_id]
            if input_ls.name not in all_premise_names:
                all_premise_names.add(input_ls.name)
                all_premises.append(input_ls)
                # all_premises.extend(get_all_premises(prover, input_ls))
    return all_premises


if __name__ == "__main__":
    import random

    random.seed(0)
    """
    Do the backward-style proof of (a+b)^2 = a*a + ab + b*b + ab
    """
    a = Entity("input1")
    b = Entity("input2")
    a_and_b = necessary_numerical_functions["add"].execute_nf([a, b])
    lhs = necessary_numerical_functions["sqr"].execute_nf([a_and_b])
    a_mul_a = necessary_numerical_functions["mul"].execute_nf([a, a])
    b_mul_b = necessary_numerical_functions["mul"].execute_nf([b, b])
    a_mul_b = necessary_numerical_functions["mul"].execute_nf([a, b])
    rhs = necessary_numerical_functions["add"].execute_nf([
        necessary_numerical_functions["add"].execute_nf([a_mul_a, a_mul_b]),
        necessary_numerical_functions["add"].execute_nf([b_mul_b, a_mul_b])
    ])

    objective = necessary_logic_functions["Equivalent"].execute_lf([lhs, rhs])
    prover = ConnectionProver(axioms=equal_axioms, conditions=[], objectives=[objective])

    axiom_list = list(equal_axioms.values())

    step_names = set()
    entity_index_pointer = 0
    total_count = 300000
    count = 0
    max_length = 200
    # Forward random search
    while len(prover.ls_id2ls) < 1000 and count < total_count:
        # Simplify all entities
        while entity_index_pointer < len(prover.ent_id2ent):
            entity = prover.ent_id2ent[entity_index_pointer]
            entity_index_pointer += 1
            lemma_operand_tuples = simplify_entity(entity)
            for lemma, operands in lemma_operand_tuples:
                count += 1
                step_name = lemma.name + " ".join([operand.name for operand in operands])
                if step_name not in step_names:
                    prover.apply_theorem(lemma, operands)

        lemma = random.choice(axiom_list)
        count += 1
        initial_entities = list(prover.ent_id2ent.values())
        operands = random.choices(initial_entities, k=lemma.input_no)

        # Check whether the step has been taken before
        step_name = lemma.name + " ".join([operand.name for operand in operands])
        if step_name not in step_names:
            step_names.add(step_name)
            result = prover.apply_theorem(lemma, operands, length_limiting=max_length)
            if result is not None:
                assumptions = [prover.ls_id2ls[id] for id in result["assumption_ids"]]
                conclusions = [prover.ls_id2ls[id] for id in result["conclusion_ids"]]
                # if len(assumptions) == 0 and len(conclusions) > 0:
                #     print(lemma.name)

                # debug: Visualize process for debugging
                # if len(conclusions) > 0:
                #     print(r"Lemma applied is:\\ ")
                #     print(lemma.name)
                #     print(r"\\ Operands are:\\ ")
                #     for operand in operands:
                #         print('$' + entity_to_latex(operand) + '$')
                #     print(r" \\ Premises are:\\ ")
                #     for premise in assumptions:
                #         s = decorate_string(logic_statement_to_latex(premise))
                #         print(r"\begin{equation}")
                #         print(r"\begin{split}")
                #         print(s)
                #         print(r"\end{split}")
                #         print(r"\end{equation}")
                #     print(r"The conclusion is\\")
                #     for conclu in conclusions:
                #         s = decorate_string(logic_statement_to_latex(conclu))
                #         print(r"\begin{equation}")
                #         print(r"\begin{split}")
                #         print(s)
                #         print(r"\end{split}")
                #         print(r"\end{equation}")
                #
                #     print()

                # for ls in assumptions + conclusions:
                #     for lemma in axiom_list:
                #         count += 1
                #         plausible, operands = lemma.infer_entities(ls)
                #         if plausible and len(operands) != 0:
                #             step_name = lemma.name + " ".join([operand.name for operand in operands])
                #             if step_name not in step_names:
                #                 step_names.add(step_name)
                #                 result = prover.apply_theorem(lemma, operands, length_limiting=max_length)
                #         else:
                #             pass

    print(count)
    print(len(prover.ls_id2ls))

    # Visualize the result
    ls_count = 0
    for ls in prover.ls_id2ls.values():
        ls_id = prover.ls_name2id[ls.name]
        if prover.logic_statement_connected(ls_id):
            ls_count += 1
            print(r"The {}-th problem is\\ ".format(ls_count))
            for index, gate in enumerate(prover.ls_id2upstream_gates[ls_id]):
                print(r"The {}-th set of premises are:\\ ".format(index))
                print(gate.info)
                for input_ls_id in gate.get_inputs():
                    print(r"\begin{equation}")
                    print(r"\begin{split}")
                    print(decorate_string(logic_statement_to_latex(
                        prover.ls_id2ls[input_ls_id]
                    )), r"\\ ")
                    print(r"\end{split}")
                    print(r"\end{equation}")

            print(r"The conclusion is:\\ ")
            print(r"\begin{equation}")
            print(r"\begin{split}")
            print(decorate_string(logic_statement_to_latex(ls)), r"\\ ")
            print(r"\end{split}")
            print(r"\end{equation}")

    print(prover.ls_id2upstream_gates[prover.ls_name2id[objective.name]])
