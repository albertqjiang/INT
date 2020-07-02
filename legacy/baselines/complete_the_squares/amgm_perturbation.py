from logic.logic import Entity
from logic.utils import standard_logic_functions, standard_numerical_functions
import random


# Complete The Squares generator with some amgm morph
def cts_base():
    objectives = list()

    # Probability of taking the AMGM transformation to a cross product term
    amgm_prob = 0.5

    total_trials = 10
    input_no = 3
    entities = [Entity(name="input{}".format(i)) for i in range(1, 1 + input_no)] + \
               [Entity(name="0", is_constant=True), Entity(name="1", is_constant=True)]
    constant_two = Entity(name="2", is_constant=True)

    for _ in range(total_trials):
        how_many_items = random.randrange(2, len(entities))
        which_items = random.sample(entities, how_many_items)

        all_products = list()
        extra_lhs_terms = list()

        for i in range(how_many_items):
            for j in range(i, how_many_items):
                if i != j:
                    if random.random() > amgm_prob:
                        # Don't do amgm
                        product = standard_numerical_functions["mul"].execute_nf(
                            [which_items[i], which_items[j]]
                        )
                        product = standard_numerical_functions["mul"].execute_nf([constant_two, product])
                        all_products.append(product)
                    else:
                        # Do amgm
                        a_square = standard_numerical_functions["sqr"].execute_nf([which_items[i]])
                        b_square = standard_numerical_functions["sqr"].execute_nf([which_items[j]])
                        a2_and_b2 = standard_numerical_functions["add"].execute_nf([a_square, b_square])
                        extra_lhs_terms.append(a2_and_b2)
                else:
                    product = standard_numerical_functions["mul"].execute_nf(
                        [which_items[i], which_items[j]]
                    )
                    all_products.append(product)

        rhs_index = random.choice(list(range(len(all_products))))
        rhs = standard_numerical_functions["opp"].execute_nf(
            [all_products[rhs_index]]
        )
        del (all_products[rhs_index])

        all_products.extend(extra_lhs_terms)
        while len(all_products) > 1:
            # print(len(all_products))
            item1 = all_products.pop(0)
            item2 = all_products.pop(0)
            new_item = standard_numerical_functions["add"].execute_nf([item1, item2])
            all_products.append(new_item)
            # print(len(all_products))

        lhs = all_products[0]
        objective = standard_logic_functions["BiggerOrEqual"].execute_lf([lhs, rhs])
        objectives.append(objective)
    return objectives


if __name__ == "__main__":
    from pprint import pprint

    objectives = cts_base()
    for obj in objectives:
        pprint(obj.name)
