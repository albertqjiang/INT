from abc import ABC, abstractmethod


class MetaTheorem(ABC):
    def __init__(self, input_no, assumption_size, conclusion_size, extra_entity_size):
        self.input_no = input_no
        self.assumption_size = assumption_size
        self.conclusion_size = conclusion_size
        self.extra_entity_size = extra_entity_size
        self.name = self.__class__.__name__

    @abstractmethod
    def execute_th(self, inputs):
        raise NotImplementedError

    @abstractmethod
    def check(self, ground_truths):
        """
        Check whether if it's possible to apply the lemma.
        If not, return False, []
        If yes, return True, [[possible operand set 1], [possible operand set 2], ...], if lemma requires assumptions
                return True, "many", if lemma requires assumptions, but there are too many possible sets of operands
                return True, "all", if lemma doesn't require assumptions
        """
        raise NotImplementedError
