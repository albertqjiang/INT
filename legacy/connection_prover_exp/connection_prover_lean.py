class ConnectionProverLean:
    def __init__(self):
        self.root = None

    def apply_theorem(self, theorem, operands):
        results = theorem.execute_th(operands)
        # Special treatment for substitution
        # if theorem.name == "EquivalenceSubstitution":
        #     results = results[0]
        # else:
        #     pass
        assumptions, conclusions = results["Assumptions"], results["Conclusions"]

        existing_assumptions = \
            (len(assumptions) == 0) or \
            all([(assump.name in self.root.ls_name2id and
                  self.root.ls_name2id[assump.name] in self.root.ground_truth_ids)
                 for assump in assumptions])
        existing_conclusions = all([(conclusion.name in self.root.ls_name2id and
                                     self.root.ls_name2id[conclusion.name] in self.root.ground_truth_ids)
                                    for conclusion in conclusions])
        if not existing_assumptions:
            print("Hit ya")
            return None
        else:
            pass

        # Register hooks
        assumption_ids = self.root.add_logic_statements(assumptions)
        no_con_before_add = len(self.root.ls_id2ls)
        conclusion_ids = self.root.add_logic_statements(conclusions)
        no_con_after_add = len(self.root.ls_id2ls)
        if not existing_conclusions:
            self.root.ground_truth_ids.extend(conclusion_ids)
            # for gt_id in conclusion_ids:
            #     self.root.ls_id2premises[gt_id] = assumption_ids
            #     self.root.ls_id2lemma[gt_id] = theorem.name

        if no_con_after_add > no_con_before_add:
            return {
                "assumption_ids": assumption_ids,
                "conclusion_ids": conclusion_ids,
                "new_conclusion": True
            }
        else:
            return {
                "assumption_ids": assumption_ids,
                "conclusion_ids": conclusion_ids,
                "new_conclusion": False
            }

    def logic_statement_connected(self, ls_id, searched_ids=None):
        if ls_id in self.root.ground_truth_ids:
            return True
        else:
            return False

    def is_proved(self):
        if len(self.root.objective_ids) == 0:
            return False
        for ls_id in self.root.objective_ids:
            if not self.logic_statement_connected(ls_id):
                return False
        return True
