from int_environment.data_generation.combos_and_orders import generate_combinations_and_orders
from int_environment.data_generation.generate_problems import generate_multiple_problems
from int_environment.proof_system.all_axioms import axiom_sets


def test_data_generation():
    axiom_combinations, axiom_orders = generate_combinations_and_orders(
        axiom_sets['field'],
        max_k=2,
        max_l=2,
        trial_per_kl=2,
    )

    # golden testing
    expect_axiom_combinations = {
        'k1': [('EquMoveTerm',), ('AdditionMultiplicationLeftDistribution',), ('MultiplicationOne',)],
        'k2': [('AdditionMultiplicationRightDistribution', 'MultiplicationAssociativity'),
               ('EquMoveTerm', 'MultiplicationCommutativity')]
    }
    expect_axiom_orders = {
        'k1l1': [('MultiplicationOne',), ('EquMoveTerm',)],
        'k1l2': [('MultiplicationOne', 'MultiplicationOne'), (
            'AdditionMultiplicationLeftDistribution', 'AdditionMultiplicationLeftDistribution')],
        'k2l2': [('AdditionMultiplicationRightDistribution', 'MultiplicationAssociativity'),
                 ('EquMoveTerm', 'MultiplicationCommutativity')]
    }

    assert axiom_combinations == expect_axiom_combinations
    assert axiom_orders == expect_axiom_orders

    datasets, problems = generate_multiple_problems(
        num_axioms=2,
        length=2,
        num_probs=2,
        train_test="train",
        orders=axiom_orders,
        degree=0,
    )

    # TODO: stronger checks
    assert datasets is not None
    assert problems is not None
