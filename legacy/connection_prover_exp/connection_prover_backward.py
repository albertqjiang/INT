class ConnectionProverBack:
    def __init__(self):
        self.root = None

    def apply_theorem(self, theorem, operands):
        results = theorem.execute_th(operands, mode="prove")

        assumptions, conclusions = results["Assumptions"], results["Conclusions"]
        # Prevent the scenario [0, 1] -> [0]
        assumption_names = [assump.name for assump in assumptions]
        for i in range(len(conclusions)):
            if conclusions[i].name in assumption_names:
                del conclusions[i]

        all_assumptions_exist = \
            (len(assumptions) == 0) or \
            all([
                (assump.name in self.root.ls_name2id and
                 self.logic_statement_connected(self.root.ls_name2id[assump.name]))
                for assump in assumptions])
        existing_conclusions = [(conclusion.name in self.root.ls_name2id and
                                 self.logic_statement_connected(self.root.ls_name2id[conclusion.name]))
                                for conclusion in conclusions]
        all_conclusions_exist = all(existing_conclusions)

        # # Check if this is a valid step, either forward or backward
        # proposed_conclusion_names = {con.name for con in conclusions}
        # step_valid = all_assumptions_exist or \
        #              any([True for obj_id in self.root.objective_ids if
        #                   self.root.ls_id2ls[obj_id].name in proposed_conclusion_names])

        assumption_ids = self.root.add_logic_statements(assumptions)
        no_con_before_add = len(self.root.ls_id2ls)
        conclusion_ids = self.root.add_logic_statements(conclusions)
        no_con_after_add = len(self.root.ls_id2ls)

        # Not duplicated conclusions
        if not all_conclusions_exist:
            if all_assumptions_exist:
                self.root.ground_truth_ids.extend(conclusion_ids)
            # for gt_id in conclusion_ids:
            #     if gt_id not in assumption_ids:
            #         if gt_id in self.root.ls_id2premises:
            #             self.root.ls_id2premises[gt_id].append(assumption_ids)
            #             self.root.ls_id2lemma[gt_id].append(theorem.name)
            #         else:
            #             self.root.ls_id2premises[gt_id] = [assumption_ids]
            #             self.root.ls_id2lemma[gt_id] = [theorem.name]
        #
        # for assump_id in assumption_ids:
        #     if assump_id not in self.root.ls_id2premises:
        #         self.root.ls_id2premises[assump_id] = []
        #     if assump_id not in self.root.ls_id2lemma:
        #         self.root.ls_id2lemma[assump_id] = []

        unproven_assump_ids = [assump_id for assump_id in assumption_ids
                               if (not self.logic_statement_connected(assump_id))
                               and (assump_id not in conclusion_ids)]

        indices_to_delete = []
        for obj_id in self.root.objective_ids:
            if self.logic_statement_connected(obj_id):
                indices_to_delete.append(obj_id)
            elif obj_id in conclusion_ids:
                indices_to_delete.append(obj_id)
                self.root.objective_ids.extend(unproven_assump_ids)
            else:
                pass
        self.root.objective_ids = [i for i in self.root.objective_ids if i not in indices_to_delete]
        for obj_id in self.root.objective_ids:
            assert not self.logic_statement_connected(obj_id)

        return {
            "assumption_ids": assumption_ids,
            "conclusion_ids": conclusion_ids,
            "new_conclusion": True
        }

    def logic_statement_connected(self, ls_id, searched_ids=None):
        # if searched_ids is None:
        #     searched_ids = []
        # if ls_id in self.root.ground_truth_ids:
        #     return True
        # elif ls_id in searched_ids:
        #     return False
        # else:
        #     searched_ids.append(ls_id)
        #
        #     premise_id_tuples = self.root.ls_id2premises[ls_id]
        #     premise_proven_tuples = []
        #
        #     for premise_tuple in premise_id_tuples:
        #         tuple_truth = []
        #         for prem_id in premise_tuple:
        #             searched_ids_copy = [s_id for s_id in searched_ids]
        #             tuple_truth.append(self.logic_statement_connected(prem_id, searched_ids=searched_ids_copy))
        #         tuple_proven = all(tuple_truth)
        #         premise_proven_tuples.append(tuple_proven)
        #
        #     return any(premise_proven_tuples)
        if ls_id in self.root.ground_truth_ids:
            return True
        else:
            return False

    # def is_proved(self):
    #     # self.root.get_ground_truth()
    #     ids_to_delete = []
    #     for ind, obj_id in enumerate(self.root.objective_ids):
    #         # Delete the logic statement from objectives if it is proven or trivial
    #         if self.logic_statement_connected(obj_id) or self.root._trivial(self.root.ls_id2ls[obj_id]):
    #             ids_to_delete.append(self.root.objective_ids[ind])
    #
    #     self.root.objective_ids = [ind for ind in self.root.objective_ids if ind not in ids_to_delete]
    #
    #     if len(self.root.objective_ids) == 0:
    #         return True
    #
    #     for ls_id in self.root.objective_ids:
    #         if not self.logic_statement_connected(ls_id):
    #             return False
    #
    #     # It shouldn't get this far
    #     raise AssertionError
