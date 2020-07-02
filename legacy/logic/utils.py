from legacy.logic.logicRL import *
from logic.logic import *


class NumericalEvaluator:
    def __init__(self, file_path=os.path.dirname(os.path.abspath(__file__)) + "/../data/3inputs1000tuples.npy"):
        self.inputs = np.load(file_path)

        def identity(x):
            return x

        def add(x, y):
            return x + y

        def mul(x, y):
            return x * y

        def sub(x, y):
            return x - y

        def geometric_mean(x, y):
            return 2 * np.sqrt(x * y)

        def sqr(x):
            return x ** 2

        def opp(x):
            return -x

        def sqrt(x):
            return np.sqrt(x)

        def inv(x):
            return 1 / x

        self.numerical_function_dict = {"add": add, "sub": sub, "mul": mul, "sqr": sqr, "sqrt": sqrt, "inv": inv,
                                        "geometric_mean": geometric_mean, "identity": identity, "opp": opp}

    def evaluate(self, entity_string, input_no=2):
        return eval(entity_string, {
            **{"input{}".format(i): self.inputs[:, i - 1] for i in range(1, 1 + input_no)},
            **self.numerical_function_dict
        })

    def equal_pair(self, entity_pair, input_no=2):
        return np.allclose(self.evaluate(entity_pair[0].name, input_no=input_no),
                           self.evaluate(entity_pair[1].name, input_no=input_no))

    def equal_string_pair(self, entity_string_pair, input_no=2):
        return np.allclose(self.evaluate(entity_string_pair[0], input_no=input_no),
                           self.evaluate(entity_string_pair[1], input_no=input_no))

    def batch_evaluate_equal_pairs(self, entity_pairs, input_no=2):
        return [self.equal_pair(pair, input_no=input_no) for pair in entity_pairs]


import torch.utils.data as data_handler


class List2Dataset(data_handler.Dataset):
    def __init__(self, d_list):
        self.dataset = d_list

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


def simple_prover():
    """

    :return: a prover for the problem (a-b)^2 >= 0 for real a, b
    """
    # Define proof
    r = LogicFunction("Real", 1)
    nonNeg = LogicFunction("NonNegative", 1)
    sub = NumericalFunction("sub")
    sqr = NumericalFunction("sqr")
    real_subtraction_closed = Theorem(name="real_subtraction_closed", input_no=3,
                                      input_constraints=[(sub, (0, 1), (2,))], assumptions=[(r, (0,)), (r, (1,))],
                                      conclusions=[(r, (2,))])
    real_sqr_non_neg = Theorem(name="real_sqr_non_neg", input_no=2, input_constraints=[(sqr, (0,), (1,))],
                               assumptions=[(r, (0,))], conclusions=[(nonNeg, (1,))])
    x = Entity("input1")
    y = Entity("input2")
    x_sub_y = sub.execute_nf([x, y])
    x_sub_y_sqr = sqr.execute_nf([x_sub_y])
    proof = Proof(entities=[x, y, x_sub_y, x_sub_y_sqr], axioms=[real_subtraction_closed, real_sqr_non_neg],
                  assumptions=[r.execute_lf([x]), r.execute_lf([y])],
                  objectives=[nonNeg.execute_lf(inputs=[x_sub_y_sqr])])

    # Define entity maxsize
    ent_maxsize = 10

    # Define ground truth maxsize
    gt_maxsize = 10

    # Define lemma maxsize
    lemma_maxsize = 5

    # Define lemma input_entity_embedding capacity
    lemma_embedding_size = 128

    # Define objective maxsize
    objective_maxsize = 1

    lgProver = LogicBasedProver(proof=proof, ent_maxsize=ent_maxsize, gt_maxsize=gt_maxsize,
                                lemma_maxsize=lemma_maxsize, lemma_embedding_size=lemma_embedding_size,
                                lemma_operand_size=5, objective_maxsize=objective_maxsize)
    return lgProver


def non_trivial_prover():
    """

    :return: a solver for the theorem y^2/x^2 + x^2 >= y + y when x and y are real
    """
    BOE = LogicFunction("BiggerOrEqual", input_no=2)
    Equal = LogicFunction("Equal", input_no=2)
    NonNeg = LogicFunction("Not negative", input_no=1)
    Real = LogicFunction("Real", input_no=1)
    add = NumericalFunction("add")
    mul = NumericalFunction("mul")
    sqr = NumericalFunction("sqr")
    inv = NumericalFunction("inv")
    gmean = NumericalFunction("geometric_mean")

    x = Entity("input1")
    y = Entity("input2")

    x_sqr = sqr.execute_nf([x])
    x_inv = inv.execute_nf([x])
    y_over_x = mul.execute_nf([y, x_inv])
    y_over_x_sqr = sqr.execute_nf([y_over_x])
    lhs = add.execute_nf([y_over_x_sqr, x_sqr])
    rhs = add.execute_nf([y, y])
    all_entities = [x, y, x_sqr, x_inv, y_over_x, y_over_x_sqr, lhs, rhs]

    # Define theorems
    real_sqr_non_neg = Theorem(name="real_sqr_non_neg", input_no=2, input_constraints=[(sqr, (0,), (1,))],
                               assumptions=[(Real, (0,))], conclusions=[(NonNeg, (1,))])
    # Define the geometric_mean inequality theorem, the inputs are x, y, x+y(lhs), 2 * sqrt(x*y)
    amgmtwo = Theorem(name="AMGM for 2 elements", input_no=4,
                      input_constraints=[(add, (0, 1), (2,)), (gmean, (0, 1), (3,))],
                      assumptions=[(NonNeg, (0,)), (NonNeg, (1,))],
                      conclusions=[(BOE, (2, 3))])
    # inequal_transitive = Theorem(name="Inequality relations are transitive", input_no=3, input_constraints=[],
    #                              assumptions=[(BOE, (0, 1)), (Equal, (1, 2))], conclusions=[(BOE, (0, 2))])

    proof = Proof(entities=all_entities,
                  axioms=[real_sqr_non_neg, amgmtwo],
                  assumptions=[Real.execute_lf([entity]) for entity in all_entities],
                  objectives=[BOE.execute_lf([lhs, rhs])])

    # Define entity maxsize
    ent_maxsize = 10

    # Define ground truth maxsize
    gt_maxsize = 20

    # Define lemma maxsize
    lemma_maxsize = 5

    # Define lemma input_entity_embedding capacity
    lemma_embedding_size = 128

    # Define objective maxsize
    objective_maxsize = 5

    lgProver = LogicBasedProver(proof=proof, ent_maxsize=ent_maxsize, gt_maxsize=gt_maxsize,
                                lemma_maxsize=lemma_maxsize, lemma_embedding_size=lemma_embedding_size,
                                lemma_operand_size=5, objective_maxsize=objective_maxsize,
                                )
    return lgProver


def end_to_end_max_q_and_action(prover, q_net):
    max_action, max_q = None, 0.
    for action in exhaust_actions(prover):
        predicted_q = q_net(obs=prover.raw_observe(), act=action)
        if predicted_q >= max_q:
            max_action = action
            max_q = predicted_q.item()
    return max_q, max_action


def entity_pair_evaluate_model(model, theorem, test_set):
    test_size = len(test_set)
    outputs = [(model(theorem=theorem, entities=[entity0, entity1]), target)
               for entity0, entity1, target in test_set]
    output_tensor = torch.cat([output[0] for output in outputs], dim=0)
    target_tensor = torch.FloatTensor([output[1] for output in outputs]).view(-1, 1)
    return float(torch.sum(torch.abs(output_tensor - target_tensor) < 0.5).float() / float(test_size))


def run_q_solver(prover, q_net, depth):
    for i in range(1, 1 + depth):
        max_q, max_action = end_to_end_max_q_and_action(prover=prover, q_net=q_net)
        _, reward, done, _ = prover.step(max_action)
        print(max_action["action"][0])
        print([ent.to_string() for ent in max_action["action"][1]])
        if done:
            pprint(prover.proof.print_proof_status())


def what_is_proved(observation, obj_observation):
    to_be_proved = list()
    already_proved_string_list = [ls.name for ls in observation['ground_truth']]
    for logic_state in obj_observation['ground_truth']:
        if logic_state.name not in already_proved_string_list:
            to_be_proved.append(logic_state)
    print(to_be_proved)
    return to_be_proved
