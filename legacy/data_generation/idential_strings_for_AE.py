import pickle


def generate_graph_identical_strings(input=2, degree=5):
    file = "../data/{}input{}degree.pkl".format(input, degree)
    graphs = pickle.load(open(file, "rb"))
    with open("../data/{}input{}degree_identical.txt".format(input, degree), "w") as fh:
        for graph in graphs:
            exp = graph.g_expression()
            fh.write(exp + '\t' + exp + '\n')


if __name__ == "__main__":
    generate_graph_identical_strings()
