from copy import deepcopy


class LogicGate:
    def __init__(self, gate_type, input_no, output_no):
        self.gate_type = gate_type
        self.input_no = input_no
        self.output_no = output_no
        self._inputs = list()
        self._outputs = list()
        self.info = None

        # Boolean value
        self.value = None

    def set_inputs(self, inputs):
        assert len(inputs) == self.input_no
        self._inputs = inputs

    def set_outputs(self, outputs):
        assert len(outputs) == self.output_no
        self._outputs = outputs

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs


class ConnectionProver:
    def __init__(self, axioms, conditions, objectives):
        self.axioms = axioms
        self.conditions = conditions
        self.objectives = objectives
        self.ground_truth = [condition for condition in conditions]
        self.ground_truth_names = {gt.name for gt in self.ground_truth}
        self.ent_id2ent = dict()
        self.ls_name2id = dict()
        self.ls_id2ls = dict()
        self.ls_id2upstream_gates = dict()
        self.ls_id2downstream_gates = dict()

        self.condition_ids = self.add_logic_statements(self.conditions)
        self.objective_ids = self.add_logic_statements(self.objectives)

    def add_logic_statement(self, ls):
        ls.indexing()
        if ls.name in self.ls_name2id:
            pass
        else:
            self.ls_name2id[ls.name] = len(self.ls_name2id)
            self.ls_id2ls[self.ls_name2id[ls.name]] = ls
            self.ls_id2upstream_gates[self.ls_name2id[ls.name]] = list()
            self.ls_id2downstream_gates[self.ls_name2id[ls.name]] = list()
            self.parse_entity_nodes_from_ls(ls)
        return self.ls_name2id[ls.name]

    def add_logic_statements(self, ls_list):
        id_list = list()
        for ls in ls_list:
            id_list.append(self.add_logic_statement(ls))
        return id_list

    def degree_transaction(self, assumption_ids, conclusion_ids):
        total_assumption_degree = sum([self.ls_id2ls[ls_id].degree for ls_id in assumption_ids])
        for conclusion_id in conclusion_ids:
            self.ls_id2ls[conclusion_id].degree = total_assumption_degree + 1

    def apply_theorem(self, theorem, operands, stay_connected=True, length_limiting=None):
        """stay_connected: set to True to skip connections not on the main tree"""
        assert theorem.name in self.axioms
        results = theorem.execute_th(operands)
        assumptions, conclusions = \
            [deepcopy(x) for x in results["Assumptions"]], \
            [deepcopy(x) for x in results["Conclusions"]]
        if length_limiting is not None:
            if not (all([(len(assump.name) <= length_limiting) for assump in assumptions]) and
                    all([(len(conclu.name) <= length_limiting) for conclu in conclusions])):
                return None

        if stay_connected:
            existing_assumptions = (len(assumptions) == 0) or \
                                   all([(assump.name in self.ls_name2id) for assump in assumptions])
            existing_conclusions = all([(conclu.name in self.ls_name2id) for conclu in conclusions])
            if (not existing_assumptions) and (not existing_conclusions):
                return None
            else:
                pass

        # Configure logic gate
        add_gate = LogicGate(gate_type="AND",
                             input_no=len(assumptions),
                             output_no=len(conclusions))
        add_gate.set_inputs(self.add_logic_statements(assumptions))
        add_gate.set_outputs(self.add_logic_statements(conclusions))
        add_gate.info = theorem.name
        for assump in assumptions:
            self.ls_id2downstream_gates[self.ls_name2id[assump.name]].append(add_gate)
        for conclu in conclusions:
            self.ls_id2upstream_gates[self.ls_name2id[conclu.name]].append(add_gate)

        # Register hooks
        assumption_ids = self.add_logic_statements(assumptions)
        conclusion_ids = self.add_logic_statements(conclusions)

        self.degree_transaction(assumption_ids, conclusion_ids)

        return {
            "assumption_ids": assumption_ids,
            "conclusion_ids": conclusion_ids
        }

    def interpret_result(self, result):
        # This might be troublesome as it's not well figured out h
        if result is None:
            return "REWARD_ASSUMPTION_INVALID"
        else:
            if self.is_proved():
                return "REWARD_PROOF_COMPLETE"
            elif (not self.is_proved()) and (len(self.ls_id2ls) - 1) not in result["conclusion_ids"]:
                return "REWARD_DUPLICATED_RESULTS"
            elif (not self.is_proved()) and (len(self.ls_id2ls) - 1) in result["conclusion_ids"]:
                return "REWARD_THEOREM_PROCEEDED"
            else:
                raise NotImplementedError

    def parse_entity_nodes_from_entity(self, entity):
        entity_ids = list()

        self.ent_id2ent[len(self.ent_id2ent)] = entity
        entity_ids.append(len(self.ent_id2ent) - 1)

        if entity.recent_numerical_function is None:
            pass
        else:
            for next_level_entity in entity.operands:
                entity_ids.extend(self.parse_entity_nodes_from_entity(next_level_entity))
        entity_ids = list(set(entity_ids))
        return entity_ids

    def parse_entity_nodes_from_ls(self, logic_statement):
        all_entity_ids = []
        for entity in logic_statement.operands:
            all_entity_ids.extend(self.parse_entity_nodes_from_entity(entity))
        return all_entity_ids

    def get_entities(self):
        return list(self.ent_id2ent.values())

    def get_ground_truth(self):
        gts = list()
        for ls_id, ls in self.ls_id2ls.items():
            if self.logic_statement_connected(ls_id, []):
                gts.append(ls)
        return gts

    def get_objectives(self):
        objs = list()
        for obj_id in self.objective_ids:
            objs.append(self.ls_id2ls[obj_id])
        return objs

    def get_observation(self):
        return {
            "ground_truth": self.get_ground_truth(),
            "lemmas": self.axioms,
            "entities": self.get_entities(),
            "objectives": self.get_objectives()
        }

    def find_logic_gate_value(self, logic_gate, searched_names):
        # print("logic inputs", [logic_statement_to_latex(self.ls_id2ls[inp]) for inp in logic_gate.get_inputs()])
        # print("searched names", [logic_statement_to_latex(name, string=True) for name in searched_names])
        if len(logic_gate.get_inputs()) == 0:
            return True
        elif logic_gate.gate_type == "AND":
            return all([self.logic_statement_connected(ls_id, searched_names=searched_names)
                        for ls_id in logic_gate.get_inputs()])
        elif logic_gate.gate_type == "OR":
            return any([self.logic_statement_connected(ls_id, searched_names=searched_names)
                        for ls_id in logic_gate.get_inputs()])
        else:
            raise NotImplementedError

    def logic_statement_connected(self, ls_id, searched_names):
        if self.ls_id2ls[ls_id].name in searched_names:
            return False
        else:
            # print("added to searched names", logic_statement_to_latex(self.ls_id2ls[ls_id]))
            searched_names.append(self.ls_id2ls[ls_id].name)

        if ls_id in self.condition_ids:
            return True
        elif len(self.ls_id2upstream_gates[ls_id]) == 0:
            return False
        else:
            return any([self.find_logic_gate_value(logic_gate, searched_names=searched_names)
                        for logic_gate in self.ls_id2upstream_gates[ls_id]])

    def is_proved(self):
        ls_id = self.objective_ids[0]
        return self.logic_statement_connected(ls_id, [])


# if __name__ == "__main__":
#     from logic.logic import Entity
#     from logic.utils import standard_numerical_functions, standard_logic_functions
#     from logic_math.real_number_axioms import real_number_axioms
#     from pprint import pprint
#
#     a = Entity("input1")
#     b = Entity("input2")
#     zero = Entity("0")
#     zero_and_zero = standard_numerical_functions["add"].execute_nf([zero, zero])
#     a_and_b = standard_numerical_functions["add"].execute_nf([a, b])
#
#     # Conditions
#     a_geq_0 = standard_logic_functions["BiggerOrEqual"].execute_lf([a, zero])
#     b_geq_0 = standard_logic_functions["BiggerOrEqual"].execute_lf([b, zero])
#     conditions = [a_geq_0, b_geq_0]
#
#     # Objectives
#     a_and_b_geq_0 = standard_logic_functions["BiggerOrEqual"].execute_lf([a_and_b, zero])
#     objectives = [a_and_b_geq_0]
#
#     connection_prover = ConnectionProver(axioms=real_number_axioms,
#                                          conditions=conditions,
#                                          objectives=objectives)
#
#     print(connection_prover.is_proved())
#     connection_prover.apply_theorem(
#         theorem=real_number_axioms["FirstPrincipleOfInequality"],
#         operands=[a, zero, b, zero]
#     )
#     connection_prover.apply_theorem(
#         theorem=real_number_axioms["AdditionIdentity"],
#         operands=[zero, zero]
#     )
#     connection_prover.apply_theorem(
#         theorem=real_number_axioms["EquivalenceSymmetry"],
#         operands=[zero, zero_and_zero]
#     )
#     connection_prover.apply_theorem(
#         theorem=real_number_axioms["EquivalenceImpliesDoubleInequality"],
#         operands=[zero_and_zero, zero]
#     )
#     connection_prover.apply_theorem(
#         theorem=real_number_axioms["InequalityTransitivity"],
#         operands=[a_and_b, zero_and_zero, zero]
#     )
#     print(connection_prover.is_proved())
#     # pprint(connection_prover.get_observation())

if __name__ == "__main__":
    from logic.logic import Entity
    from proof_system.logic_functions import necessary_logic_functions

    a = Entity("input1")
    b = Entity("input2")
    ls = necessary_logic_functions["Equivalent"].execute_lf([a, b])
    print(a)
    print(ls.operands[0].root)
