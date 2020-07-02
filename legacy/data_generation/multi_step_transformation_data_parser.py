from __future__ import unicode_literals, print_function, division
from io import open


def seq2dict(seq_string):
    transformations = []
    lines = seq_string.split("\n")
    initial_exp = lines[0].split("\t")[0]
    del lines[-1]
    final_exp = lines[-1].strip(";")
    for i in range(len(lines) - 1):
        transformations.append(lines[i].split("\t")[1].split(":"))
    return {"initial_exp": initial_exp, "final_exp": final_exp, "transformations": transformations}


def create_datapoints(input=2, degree=5, depth=1):
    """

    :param input: how many inputs
    :param degree: degree of the computation graphs
    :param depth: depth of the transformations
    :return: a string dataset containing all graphs in the original txt file
    """
    fh = open("../data/{}input{}degree{}depth_trans_chain.txt".format(input, degree, depth), "r")
    all_data_points = []
    data_point_string = ""
    for line in fh:

        data_point_string += line
        if ";" in line:
            all_data_points.append(data_point_string)
            data_point_string = ""
    fh.close()
    with open("../data/{}input{}degree{}depth_trans_type_formatted.txt".format(input, degree, depth), "w") as outh:
        for dp in all_data_points:
            dp_dict = seq2dict(dp)
            trans_string = " ".join([tran[0] for tran in dp_dict["transformations"]])
            line_to_write = "".join(
                [word for word in [dp_dict["initial_exp"], " ; ", dp_dict["final_exp"], "\t", trans_string, "\n"]])
            outh.write(line_to_write)


if __name__ == "__main__":
    data_points = create_datapoints(depth=3)
