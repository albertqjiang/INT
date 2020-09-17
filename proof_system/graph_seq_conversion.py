from visualization.seq_parse import entity_to_seq_string, logic_statement_to_seq_string
from proof_system.all_axioms import all_axioms_to_prove


compact_theorem_name = {
    "AdditionCommutativity": "add_cmmt",
    "AdditionAssociativity": "add_assc",
    "AdditionZero": "add_zero",
    "AdditionSimplification": "add_simp",
    "MultiplicationCommutativity": "mul_cmmt",
    "MultiplicationAssociativity": "mul_assc",
    "MultiplicationOne": "mul_one",
    "MultiplicationSimplification": "mul_simp",
    "AdditionMultiplicationLeftDistribution": "add_mul_l_dist",
    "AdditionMultiplicationRightDistribution": "add_mul_r_dist",
    "SquareDefinition": "sqr_def",
    "EquivalenceSymmetry": "equ_symm",
    "PrincipleOfEquality": "equ_prin",
    "EquMoveTerm": "equ_mv_tm",
    "IneqMoveTerm": "ineq_mv_tm",
    "SquareGEQZero": "sqr_geq_zero",
    "EquivalenceImpliesDoubleInequality": "equ_dbl_ineq",
    "FirstPrincipleOfInequality": "ineq_prin_one",
    "SecondPrincipleOfInequality": "ineq_prin_two"
}


class Parser:
    def __init__(self):
        self.full2compact = compact_theorem_name
        self.compact2full = {v: k for k, v in self.full2compact.items()}

    @staticmethod
    def filter_seq_string(seq_string):
        output_string = seq_string.replace("  ", " ")
        output_string = output_string.replace(r"\geq ", ">=")
        output_string = output_string.replace(r"\frac", "/")
        output_string = output_string.replace(r"ø", "emp")
        return output_string

    @staticmethod
    def index_entity_by_name_in_root(entity):
        root = entity.root
        index = 0
        # root.ent_dic is traversed dfs order
        for key in sorted(root.ent_dic):
            if root.ent_dic[key].name == entity.name:
                if root.ent_dic[key] is entity:
                    return index
                else:
                    index += 1

    def pretraining_target(self, step, next_step, is_last_step):
        # Target, the theorem applied, the goal and premises after applying the theorem
        target = ""
        theorem_name = self.full2compact[step["lemma"].name]
        target += theorem_name + " "

        # Separation
        target += "| "

        if not is_last_step:
            next_premises = next_step["observation"]["ground_truth"]
            next_premises_string = " & ".join([logic_statement_to_seq_string(premise) for premise in next_premises])
            target += next_premises_string + " to " + \
                      logic_statement_to_seq_string(next_step["observation"]["objectives"][0])
        else:
            target += "ø"
        target = self.filter_seq_string(target)
        return target

    def execution_target(self, step):
        # Target, the theorem applied, the arguments used, the goal and premises after applying the theorem
        target = ""
        theorem_name = self.full2compact[step["lemma"].name]
        target += theorem_name + " | "

        for i, argument in enumerate(step["input_entities"]):
            if argument.root.name == step["observation"]["objectives"][0].name:
                root_name = "obj"
            else:
                root_name = logic_statement_to_seq_string(argument.root)
            index_by_name = self.index_entity_by_name_in_root(argument)
            if index_by_name == 0:
                index_by_name_str = ""
            else:
                index_by_name_str = f" {index_by_name}"
            target += entity_to_seq_string(argument) + " in " + root_name + index_by_name_str

            if i != len(step["input_entities"]) - 1:
                target += ", "

        return self.filter_seq_string(target)

    def observation_to_source(self, observation):
        premises = observation["ground_truth"]
        premises_string = " & ".join([logic_statement_to_seq_string(premise) for premise in premises])
        if premises_string:
            source = premises_string + " to "
        else:
            source = "to "
        source = source + logic_statement_to_seq_string(observation["objectives"][0])
        source = self.filter_seq_string(source)
        return source

    def parse_proof_step_to_seq(self, step, next_step=None, is_last_step=None, pretraining=False):
        # if len(step["observation"]["objectives"]) != 1:
        #     return

        # Source, the goal and premises before applying the theorem
        source = self.observation_to_source(step["observation"])
        if pretraining:
            return source, self.pretraining_target(step=step, next_step=next_step, is_last_step=is_last_step)
        else:
            return source, self.execution_target(step=step)

    @staticmethod
    def single_argument_execution_string_to_argument(observation, argument_execution_string):
        argument_execution_string = argument_execution_string.strip()

        premises = observation["ground_truth"]
        goals = observation["objectives"]

        entity_string, entity_root_execution = argument_execution_string.split("in")
        entity_string, entity_root_execution = entity_string.strip(), entity_root_execution.strip()

        entity_root_execution_split = entity_root_execution.split()
        how_many_pieces = len(entity_root_execution_split)
        if how_many_pieces == 2:
            entity_root_string, entity_index = entity_root_execution_split
        elif how_many_pieces == 1:
            entity_root_string, entity_index = entity_root_execution, 0
        else:
            raise AssertionError
        entity_index = int(entity_index)

        entity_root = None
        if entity_root_string.startswith("obj"):
            entity_root = goals[0]
        else:
            for logic_statement in goals + premises:
                if entity_root_string == logic_statement_to_seq_string(logic_statement):
                    entity_root = logic_statement
                    break
        assert entity_root

        nodes_with_the_same_name = 0
        for index in sorted(entity_root.ent_dic.keys()):
            if entity_to_seq_string(entity_root.ent_dic[index]) == entity_string:
                if nodes_with_the_same_name == entity_index:
                    return entity_root.ent_dic[index]
                nodes_with_the_same_name += 1

    def find_action(self, observation, execution_string):
        theorem_string, all_argument_strings = execution_string.split("|")
        theorem_string, all_argument_strings = theorem_string.strip(), all_argument_strings.strip()
        theorem = all_axioms_to_prove[self.compact2full[theorem_string]]

        operands = [self.single_argument_execution_string_to_argument(observation, single_argument.strip())
                    for single_argument in all_argument_strings.split(",")]

        return theorem, operands

    @staticmethod
    def validate_execution_string(observation, execution_string):
        raise NotImplementedError


if __name__ == "__main__":
    import json
    from data_generation.generate_problems import generate_multiple_problems
    orders = json.load(open("/Users/qj213/Papers/My papers/INT_arXiv/INT/data/benchmark/ordered_field/orders.json"))
    dataset, problems = generate_multiple_problems(num_axioms=3, length=3,
                                                   num_probs=100, train_test="train",
                                                   orders=orders, degree=0)
    parser = Parser()
    for problem in problems:
        for step in problem:
            a, b = parser.parse_proof_step_to_seq(step)

            print(a)
            print(b)
            print(step["lemma"], [entity.name for entity in step["input_entities"]])
            theorem, operands = parser.find_action(step["observation"], b)
            print(theorem, [entity.name for entity in operands])
            assert theorem.name == step["lemma"].name
            for i in range(len(operands)):
                assert operands[i] is step["input_entities"][i]
