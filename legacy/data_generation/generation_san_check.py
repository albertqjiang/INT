import pickle
import argparse
import os
from legacy.connection_prover_exp.connection_prover_lean import ConnectionProverLean
from proof_system.all_axioms import all_axioms
from baselines.utils import gt_2_graph

ap = argparse.ArgumentParser()
ap.add_argument("-cod", "--custom_objective_dir",
                required=False,
                default="../data/random_combination_dataset",
                help="what custom algos to use")
args = vars(ap.parse_args())
directory = args["custom_objective_dir"]

for category in os.listdir(directory):
    diff_directory_name = directory + "/{}/".format(category)

    for step_name in os.listdir(diff_directory_name):
        if step_name.startswith("steps"):
            print(diff_directory_name + step_name)
            steps = pickle.load(open(diff_directory_name + step_name, "rb"))

            proof = ConnectionProverLean(
                axioms=all_axioms,
                conditions=steps[0]["observation"]["ground_truth"],
                objectives=steps[0]["observation"]["objectives"]
            )
            print(proof.is_proved())

            for step in steps:
                # print(step["lemma"].name)
                # print([entity_to_latex(ent) for ent in step["input_entities"]])
                for op in step["input_entities"]:
                    assert op.root in step["observation"]["ground_truth"] or \
                           op.root in step["observation"]["objectives"]
                result = proof.apply_theorem(theorem=step["lemma"], operands=step["input_entities"])
                # print(proof.interpret_result(result))
                print(proof.is_proved())

                gt_graph = []
                gt_gnn_ind = []
                obj_graph = []
                obj_gnn_ind = []
                ent_dic = dict()
                name_dic = dict()
                node_ent = []
                node_name = []
                g_ind = 0
                for gt in step["observation"]['ground_truth']:
                    # print(gt.name)
                    # print(gt.operands)
                    graph_data, node_op, local_poses = gt_2_graph(
                        gt, node_ent, node_name, ent_dic, name_dic
                    )

            assert proof.is_proved()
