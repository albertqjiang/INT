import numpy as np

np.random.seed(1)


def num_tuple_gen(graph, num_tuple_len):
    """
    Generate random num_tuple for numerical representation of the graph
    :param graph: instance of Graph
    :param num_tuple_len: how many num_tuples do we want
    :return: a numpy array of generated num_tuples
    """
    tuple_len = len(graph.inputs)
    return np.random.random_sample((num_tuple_len, tuple_len))


def save_standard_num_tuples(graph_inputs_len, num_tuple_len):
    num_tuples = np.random.random_sample((num_tuple_len, graph_inputs_len))
    np.save("../data/" + str(graph_inputs_len) + "inputs" + str(num_tuple_len) + "tuples.npy", num_tuples)


def vec_gen(graph, num_tuple):
    """
    Evaluate the results of the samples generated
    :param graph: instance of Graph
    :param num_tuple: numpy array of capacity samples_len * len(graph.inputs)
    :return: a vector as the numerical representation of the graph w.r.t. this particular num_tuple set
    """
    l = len(num_tuple)
    vector = np.zeros([l])
    for i in range(l):
        graph.set_input_values(num_tuple[i])
        vector[i] = graph.g_evaluate_output()[0].content.value
    return vector


if __name__ == "__main__":
    save_standard_num_tuples(3, 1000)
