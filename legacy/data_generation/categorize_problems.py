def categorize_problems(directory_name, total_categories=5):
    import os
    import pickle
    all_file_names = list()
    for file in os.listdir(directory_name):
        if file.startswith("steps"):
            steps = pickle.load(open(directory_name + file, "rb"))
            all_file_names.append((directory_name + str(file), len(steps)))

    all_file_names = sorted(all_file_names, key=lambda x: x[1])
    total_problems = len(all_file_names)
    categorized_problem_names = dict()
    for cate in range(total_categories):
        categorized_problem_names[cate] = \
            all_file_names[int(cate * total_problems / total_categories):
                           int((1 + cate) * total_problems / total_categories)]

    for cate in range(total_categories):
        if not os.path.exists(directory_name + "{}/".format(cate)):
            os.mkdir(directory_name + "{}/".format(cate))
        for i, file_name in enumerate(categorized_problem_names[cate]):
            os.rename(file_name[0], directory_name + "{}/steps_{}.p".format(cate, i))
