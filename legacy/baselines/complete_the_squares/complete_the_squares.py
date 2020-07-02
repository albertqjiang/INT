import os
import sys

sys.path.insert(0, os.path.abspath('../../..'))

from logic.logic import Entity, Proof
from logic.utils import standard_logic_functions, standard_numerical_functions
from legacy.logic_math import real_number_axioms
import random


# Complete The Squares base generator
def cts_base():
    proof_templates = list()

    total_trials = 10

    input_no = 3
    entities = [Entity(name="input{}".format(i)) for i in range(1, 1 + input_no)] + \
               [Entity(name="0", is_constant=True), Entity(name="1", is_constant=True)]
    constant_two = Entity(name="2", is_constant=True)

    simple_proof = Proof(entities=entities + [constant_two], axioms=list(real_number_axioms.values()), assumptions=[],
                         objectives=[])
    pickle.dump(simple_proof, open("../../data/simple_proof.p", "wb"))

    for _ in range(total_trials):
        how_many_items = random.randrange(2, len(entities))
        which_items = random.sample(entities, how_many_items)

        all_products = list()

        for i in range(how_many_items):
            for j in range(i, how_many_items):
                product = standard_numerical_functions["mul"].execute_nf(
                    [which_items[i], which_items[j]]
                )
                if i != j:
                    product = standard_numerical_functions["mul"].execute_nf([constant_two, product])
                else:
                    pass
                all_products.append(product)

        proof_template = Proof(entities=entities + [constant_two] + all_products,
                               axioms=list(real_number_axioms.values()),
                               assumptions=[],
                               objectives=[])

        rhs_index = random.choice(list(range(len(all_products))))
        rhs = standard_numerical_functions["opp"].execute_nf(
            [all_products[rhs_index]]
        )
        del (all_products[rhs_index])

        # Generate an addition tree with least depth
        while len(all_products) > 1:
            item1 = all_products.pop(0)
            item2 = all_products.pop(0)
            new_item = standard_numerical_functions["add"].execute_nf([item1, item2])
            all_products.append(new_item)

        lhs = all_products[0]
        objective = standard_logic_functions["BiggerOrEqual"].execute_lf([lhs, rhs])

        proof_template.objectives.append(objective)
        proof_templates.append(proof_template)
    return proof_templates


if __name__ == "__main__":
    import pickle

    proof_templates = cts_base()
    pickle.dump(proof_templates, open("../../data/cts/proof_templates.p", "wb"))
