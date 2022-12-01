import collections

TokenConsts = collections.namedtuple(
    'TokenConsts',
    [
        'num_tokens',
        'padding_token',
        'output_start_token'
    ]
)

MaskTokenConsts = collections.namedtuple(
    'TokenConsts',
    [
        'num_tokens',
        'padding_token',
        'output_start_token',
        'end_token',
        'mask_separator'
    ]
)


class Representation:
    # Subclasses should fill this attribute with TokenConsts object.
    token_consts = None

    @staticmethod
    def proof_state_to_input_formula(state):
        raise NotImplementedError()

    @staticmethod
    def proof_state_to_target_formula(state):
        raise NotImplementedError()

    @staticmethod
    def tokenize_formula(formula):
        raise NotImplementedError()

    @staticmethod
    def formula_from_tokens(tokens):
        raise NotImplementedError()


