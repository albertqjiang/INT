from legacy.pseq.graph import *
from legacy.data_generation.num_rep import save_standard_num_tuples


def evaluate(graph: Graph, num_tuple=None, trial=1000):
    """

    :param num_tuple: Number tuple to be used
    :param graph: The graph to evaluate
    :param trial: How many number tuple trials shall be used
    :return: a very long numpy vector of the evaluation result

    """
    num_of_input = len(graph.inputs)
    if num_tuple is None:
        # Open num_tuple
        try:
            num_tuple = np.load("../data/{}inputs{}tuples.npy".format(num_of_input, trial))
        except FileNotFoundError:
            print("This many tuples are not generated.")
            print("Generating now......")
            save_standard_num_tuples(num_of_input, trial)
            num_tuple = np.load("../data/{}inputs{}tuples.npy".format(num_of_input, trial))

    trial = num_tuple.shape[0]
    evaluation = np.zeros((trial, 1))
    for i in range(num_tuple.shape[0]):
        graph.set_input_values(num_tuple[i])
        evaluation[i] = graph.g_evaluate_value()
    return evaluation


if __name__ == "__main__":
    x = [IOunit(Symbol("x", value=1., expression="x"))]
    y = [IOunit(Symbol("y", value=2., expression="y"))]
    G = Graph("testExpression", inputs=x + y)

    add1 = Addition(x + y)

    G.add_operation([add1])

    print(evaluate(G))
