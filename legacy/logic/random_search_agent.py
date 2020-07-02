from logic.logic import *

import random


class RandomSearchAgent(Agent):
    def step(self, proof):
        theorem_to_apply = random.sample(proof.lemmas, 1)[0]
        operands = random.sample(proof.entities, theorem_to_apply.input_no)
        try:
            theorem_eval = proof.apply_theorem(theorem_to_apply, operands)
            return {True: 1, False: 0, None: -1}[theorem_eval]
        except InputError:
            return -1


if __name__ == "__main__":
    r = LogicFunction("Real")
    equal = LogicFunction("Equals")
    nonNeg = LogicFunction("NonNegative")
    biggerOrEqual = LogicFunction("BiggerOrEqual")

    add = NumericalFunction("add")
    sub = NumericalFunction("sub")
    mul = NumericalFunction("mul")
    sqr = NumericalFunction("sqr")

    real_addition_closed = Theorem(name="real_addition_closed", input_no=3,
                                   input_constraints=[(add, (0, 1), (2,))], assumptions=[(r, (0,)), (r, (1,))],
                                   conclusions=[(r, (2,))])
    real_subtraction_closed = Theorem(name="real_subtraction_closed", input_no=3,
                                      input_constraints=[(sub, (0, 1), (2,))], assumptions=[(r, (0,)), (r, (1,))],
                                      conclusions=[(r, (2,))])
    real_sqr_non_neg = Theorem(name="real_sqr_non_neg", input_no=2, input_constraints=[(sqr, (0,), (1,))],
                               assumptions=[(r, (0,))], conclusions=[(nonNeg, (1,))])
    first_inequality_principle = Theorem(name="first_inequality_principle", input_no=6,
                                         input_constraints=[(add, (0, 2), (4,)), (add, (1, 3), (5,))],
                                         assumptions=[(biggerOrEqual, (0, 1)), (biggerOrEqual, (2, 3))],
                                         conclusions=[(biggerOrEqual, (4, 5))])

    x = Entity("x")
    y = Entity("y")
    x_and_y = add.execute_nf([x, y])
    x_sub_y = sub.execute_nf([x, y])
    x_sub_y_sqr = sqr.execute_nf([x_sub_y])

    proof = Proof(entities={x, y, x_sub_y, x_sub_y_sqr},
                  axioms=[real_addition_closed, real_subtraction_closed, real_sqr_non_neg],
                  assumptions=[r.execute_lf([x]), r.execute_lf([y])],
                  objectives=[nonNeg.execute_lf(inputs=[x_sub_y_sqr])])

    rsagent = RandomSearchAgent()
    for _ in range(1000):
        rsagent.step(proof)
        if proof.proved:
            break
    print(proof.print_proof_status())
