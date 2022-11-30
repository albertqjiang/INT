from proof_system.special_axioms import EquivalenceSubstitution
from proof_system.field_axioms import field_axioms
from proof_system.ordered_field_additional_axioms import ordered_field_additional_axioms

special_axioms = {
    "EquivalenceSubstitution": EquivalenceSubstitution(),
}

all_axioms = {**field_axioms, **ordered_field_additional_axioms, **special_axioms}
all_axioms_to_prove = {**field_axioms, **ordered_field_additional_axioms}

generation_type = {
    "AdditionCommutativity": "Equality",
    "AdditionAssociativity": "Equality",
    "AdditionZero": "Equality",
    "AdditionSimplification": "Equality",
    "MultiplicationCommutativity": "Equality",
    "MultiplicationAssociativity": "Equality",
    "MultiplicationOne": "Equality",
    "MultiplicationSimplification": "Equality",
    "AdditionMultiplicationLeftDistribution": "Equality",
    "AdditionMultiplicationRightDistribution": "Equality",
    "SquareDefinition": "Equality",
    "EquivalenceSymmetry": "Equality",
    "PrincipleOfEquality": "Equality",
    "EquMoveTerm": "Equality",
    "IneqMoveTerm": "Inequality",
    "SquareGEQZero": "Transition",
    "EquivalenceImpliesDoubleInequality": "Transition",
    "FirstPrincipleOfInequality": "Inequality",
    "SecondPrincipleOfInequality": "Inequality"
}

axiom_sets = {
    "field": field_axioms,
    "ordered_field": {**field_axioms, **ordered_field_additional_axioms}
}
