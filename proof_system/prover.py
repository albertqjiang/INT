from proof_system.logic_functions import necessary_logic_functions
from proof_system.graph_seq_conversion import Parser


class Prover:
    def __init__(self, axioms: dict, conditions: list, objectives: list, prove_direction: str):
        """
        This prover is capable of doing forward generation and backward proving
        :param axioms: the axioms that can be used inside the prover
        :param conditions: the conditions to start the proof with
        :param objectives: the objectives to prove, usually only one
        :param prove_direction: either "forward" or "backward",
                                use forward for generation and backward for writing proofs
        """
        if prove_direction == "forward":
            self.mode_theorem = "generate"
        elif prove_direction == "backward":
            self.mode_theorem = "prove"
        else:
            raise NotImplementedError

        self.prove_direction = prove_direction

        self.axioms = axioms
        self.conditions = conditions
        self.objectives = objectives
        self.ent_id2ent = dict()
        self.ent_ent2id = dict()
        self.ls_name2id = dict()
        self.ls_id2ls = dict()

        # Register the premises, ground truths, and objectives
        self.condition_ids = self.add_logic_statements(self.conditions)
        self.initial_condition_ids = [con_id for con_id in self.condition_ids]
        self.ground_truth_ids = [ls_id for ls_id in self.condition_ids]
        self.objective_ids = self.add_logic_statements(self.objectives)

        self.update_conditions()

        self.parser = Parser()

    @staticmethod
    def _trivial(logic_statement):
        # Checks if a logic statement is trivially true or not.
        logic_statement.update_name()
        lhs, rhs, = logic_statement.operands
        if lhs.name == rhs.name and logic_statement.logic_function.name in necessary_logic_functions:
            return True
        return False

    def update_conditions(self):
        # Register the initial conditions as ground truth
        for con_id in self.condition_ids:
            if con_id not in self.ground_truth_ids:
                self.ground_truth_ids.append(con_id)

    def add_logic_statement(self, logic_statement):
        # Index and register a logic statement
        logic_statement.indexing()
        if logic_statement.name in self.ls_name2id:
            pass
        else:
            self.ls_name2id[logic_statement.name] = len(self.ls_name2id)
            self.ls_id2ls[self.ls_name2id[logic_statement.name]] = logic_statement
            self._parse_entity_ids_from_ls(logic_statement)
        return self.ls_name2id[logic_statement.name]

    def add_logic_statements(self, logic_statement_list):
        # Index and register multiple logic statements
        id_list = list()
        for ls in logic_statement_list:
            id_list.append(self.add_logic_statement(ls))
        return id_list

    def interpret_result(self, result):
        # Interpret the result of a theorem application
        if result is None:
            return "REWARD_ASSUMPTION_INVALID"
        else:
            if self.is_proved():
                return "REWARD_PROOF_COMPLETE"
            elif not result["progress"]:
                return "REWARD_DUPLICATED_RESULTS"
            elif result["progress"]:
                return "REWARD_THEOREM_PROCEEDED"
            else:
                raise NotImplementedError

    def _parse_entity_ids_from_entity(self, entity):
        # Get the ids of all entities that are subtrees of a given entity(including itself)
        entity_ids = [self._add_entity(entity)]

        if entity.recent_numerical_function is not None:
            for next_level_entity in entity.operands:
                entity_ids.extend(self._parse_entity_ids_from_entity(next_level_entity))
        assert len(entity_ids) == len(set(entity_ids))
        return entity_ids

    def _parse_entity_ids_from_ls(self, logic_statement):
        # Get the ids of all entities in a logic statement
        entity_nodes = [logic_statement.ent_dic[key] for key in logic_statement.ent_dic if key != 0]
        return self._add_entities(entity_nodes)

    def _add_entity(self, entity):
        # Add one entity to the register, return the id.
        if entity not in self.ent_ent2id:
            ent_id = len(self.ent_id2ent)
            self.ent_id2ent[ent_id] = entity
            self.ent_ent2id[entity] = ent_id
            return ent_id
        else:
            return self.ent_ent2id[entity]

    def _add_entities(self, entities):
        # Add multiple entities to the register, return a list of the ids of the entities
        entity_ids = list()
        for entity in entities:
            entity_ids.append(self._add_entity(entity))
        return entity_ids

    def get_entities(self):
        # Get all the entities in the prover ever registered
        return list(self.ent_ent2id.values())

    def get_ground_truth(self):
        # Get all the ground truth in the prover ever registered,
        # including the initial conditions and statements proven later
        gts = list()
        for ls_id in set(sorted(self.ls_id2ls.keys(), reverse=True) + self.condition_ids):
            ls = self.ls_id2ls[ls_id]
            if self.logic_statement_connected(ls_id):
                gts.append(ls)
        return gts

    def get_objectives(self):
        # Get the current objectives in the prover
        self.is_proved()
        self.objective_ids = sorted(set(self.objective_ids))
        return [self.ls_id2ls[obj_id] for obj_id in self.objective_ids]

    def get_observation(self):
        # Get the observation for the prover environment, this should contain everything needed to prove the theorem
        return {
            "ground_truth": self.get_ground_truth(),
            "lemmas": self.axioms,
            "entities": self.get_entities(),
            "objectives": self.get_objectives()
        }

    def logic_statement_connected(self, ls_id):
        if ls_id in self.ground_truth_ids:
            return True
        else:
            return False

    def _logic_statements_exist_and_are_proven(self, ls_list):
        # If ls_list is empty, return True
        for ls in ls_list:
            ls_id = self.ls_name2id.get(ls.name, None)
            if ls_id is not None and self.logic_statement_connected(ls_id):
                pass
            else:
                return False
        return True

    # Methods implemented differently in ProverBack and ProverLean
    def apply_theorem(self, theorem, operands):
        # Apply a theorem with operands
        results = theorem.execute_th(operands, mode=self.mode_theorem)
        assumptions, conclusions = results["Assumptions"], results["Conclusions"]

        # Prevent the scenario [0, 1] -> [0]
        assumption_names = [assump.name for assump in assumptions]
        conclusion_ids_to_delete = []
        for i in range(len(conclusions)):
            if conclusions[i].name in assumption_names:
                conclusion_ids_to_delete.append(i)
                del conclusions[i]
        conclusions = [conclusions[i] for i in range(len(conclusions)) if i not in conclusion_ids_to_delete]

        # Determine whether all assumptions are existent and true, if there are no assumptions, it is true
        all_assumptions_true = self._logic_statements_exist_and_are_proven(assumptions)
        # Determine whether all conclusions are existent and true, if there are no assumptions, it is true
        all_conclusions_true = self._logic_statements_exist_and_are_proven(conclusions)

        # In forward generation, unproven assumptions are not allowed
        if self.prove_direction == "forward" and not all_assumptions_true:
            return None

        assumption_ids = self.add_logic_statements(assumptions)
        conclusion_ids = self.add_logic_statements(conclusions)

        # Determine whether there are new conclusions
        num_gt_before = len(self.ground_truth_ids)
        if not all_conclusions_true:
            if all_assumptions_true:
                self.ground_truth_ids.extend(conclusion_ids)
        num_gt_after = len(self.ground_truth_ids)
        if num_gt_after > num_gt_before:
            new_gts = True
        else:
            new_gts = False

        # Get the assumptions that have not been proven yet and substitute the objectives
        # if the original objectives are conclusions
        unproven_assump_ids = [assump_id for assump_id in assumption_ids
                               if (not self.logic_statement_connected(assump_id))
                               and (assump_id not in conclusion_ids)]

        obj_ids_before = set(self.objective_ids)
        indices_to_delete = []
        for obj_id in self.objective_ids:
            if self.logic_statement_connected(obj_id):
                indices_to_delete.append(obj_id)
            elif obj_id in conclusion_ids:
                indices_to_delete.append(obj_id)
                self.objective_ids.extend(unproven_assump_ids)
            else:
                pass
        self.objective_ids = [i for i in self.objective_ids if i not in indices_to_delete]
        for obj_id in self.objective_ids:
            assert not self.logic_statement_connected(obj_id)

        # Determine whether the objectives are new
        obj_ids_after = set(self.objective_ids)
        if obj_ids_before != obj_ids_after:
            new_objs = True
        else:
            new_objs = False

        # Different progress metrics for forward generation and backward proving
        if self.prove_direction == "forward":
            progress = new_gts
        elif self.prove_direction == "backward":
            progress = new_gts or new_objs
        else:
            raise NotImplementedError

        return {
            "assumption_ids": assumption_ids,
            "conclusion_ids": conclusion_ids,
            "progress": progress
        }

    def apply_theorem_seq_style(self, exec_seq):
        lemma, input_entities = self.parser.find_action(self.get_observation(), exec_seq)
        return self.apply_theorem(lemma, input_entities)

    def is_proved(self):
        if self.prove_direction == "forward":
            # In forward generation, this should always be false.
            # The theorem is never really proved
            return False

        elif self.prove_direction == "backward":
            ids_to_delete = []
            for ind, obj_id in enumerate(self.objective_ids):
                # Delete the logic statement from objectives if it is proven or trivial
                if self.logic_statement_connected(obj_id) or self._trivial(self.ls_id2ls[obj_id]):
                    ids_to_delete.append(self.objective_ids[ind])

            self.objective_ids = [ind for ind in self.objective_ids if ind not in ids_to_delete]

            if len(self.objective_ids) == 0:
                return True

            for ls_id in self.objective_ids:
                if not self.logic_statement_connected(ls_id):
                    return False

            # It shouldn't get this far
            raise AssertionError
        else:
            raise NotImplementedError
