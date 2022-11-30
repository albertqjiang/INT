from abc import ABC, abstractmethod


class MetaAxiom(ABC):
    def __init__(self, input_no, assumption_size, conclusion_size, assumption_types):
        """
        The design principle of the axioms extending core ground truths:
        L = R => f_l(L, c) = f_r(L, c) => L' = f_l(R, c) = f_r(L, c) = R'
        Therefore we can always recover L from L' and recover R from R'
        The how to extend function should provide h so that h(L'=R') = [L, R]
        It should also provide g such that g(L'=R') = operands used in prove mode
        :param input_no: the number of arguments the axiom takes
        :param assumption_size: how many assumptions the axiom will produce
        :param conclusion_size: how many conclusions the axiom will produce
        :param assumption_types: whether the assumptions are equalities or inequalities
        """
        self.input_no = input_no
        self.assumption_size = assumption_size
        self.conclusion_size = conclusion_size
        self.assumption_types = assumption_types
        self.name = self.__class__.__name__

    @abstractmethod
    def execute_th(self, operands, mode):
        raise NotImplementedError

    @abstractmethod
    def extend_core_gt(self, core_gt, entities, transform_gt):
        """Extend the core ground truth with the current lemma"""
        raise NotImplementedError

    @staticmethod
    def original_coding():
        """Function h represented by coding"""
        raise NotImplementedError

    @staticmethod
    def prove_operands(new_ls):
        """Function g"""
        raise NotImplementedError
