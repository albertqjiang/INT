import json

from data_generation.generate_problems import generate_multiple_problems
from representation.action_representation_pointer import ActionRepresentationPointer, thm2index

with open('data/benchmark/field/combinations.json', 'r') as file:
    combos = json.load(file)

x = generate_multiple_problems(5, 5, 1, train_test='test', combos=combos)

sample_objective = x[0]['all'].io_tuples[0][0]['objectives'][0]
sample_axiom = thm2index[x[0]['all'].io_tuples[0][1].name]
sample_entities = [ent.index for ent in x[0]['all'].io_tuples[0][2]]
sample_action = (sample_axiom, *sample_entities)
print(f'Original action: {sample_action}')

action_rep_str = ActionRepresentationPointer.action_to_formula(sample_objective, sample_action)
print(f'Encoded action: {action_rep_str}')

decoded_action = ActionRepresentationPointer.pointer_str_to_action(sample_objective, action_rep_str)
print(f'Decoded action: {decoded_action}')

