from legacy.data_generation.exhaust_comp_graph import exhaust_graph
from legacy.data_generation.amgm_applied_to_first_add import apply_amgm

pgs = exhaust_graph(degree=4)
print("{} possible graphs in total".format(len(pgs)))
apply_amgm()
