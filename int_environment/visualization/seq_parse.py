import re
from visualization.latex_parse import extract_two_operands


def rm_function_and_brackets(whole_string, function_name):
    whole_string = whole_string.lstrip(function_name)
    whole_string = whole_string.lstrip("(")
    whole_string = whole_string.rstrip(")")
    return whole_string


def _entity_name_to_seq_string(entity_name):
    if entity_name.startswith("add"):
        outermost_numerical_function = "add"
        entity_name = rm_function_and_brackets(entity_name, outermost_numerical_function)
        two_operands = extract_two_operands(entity_name)
        two_operands_latex = [_entity_name_to_seq_string(operand) for operand in two_operands]
        return "(" + "+".join(two_operands_latex) + ")"
    elif entity_name.startswith("sub"):
        outermost_numerical_function = "sub"
        entity_name = rm_function_and_brackets(entity_name, outermost_numerical_function)
        two_operands = extract_two_operands(entity_name)
        two_operands_latex = [_entity_name_to_seq_string(operand) for operand in two_operands]
        return "(" + "-".join(two_operands_latex) + ")"
    elif entity_name.startswith("mul"):
        outermost_numerical_function = "mul"
        entity_name = rm_function_and_brackets(entity_name, outermost_numerical_function)
        two_operands = extract_two_operands(entity_name)
        two_operands_latex = [_entity_name_to_seq_string(operand) for operand in two_operands]
        return "(" + "*".join(two_operands_latex) + ")"
    elif entity_name.startswith("opp"):
        outermost_numerical_function = "opp"
        entity_name = rm_function_and_brackets(entity_name, outermost_numerical_function)
        return "(-" + _entity_name_to_seq_string(entity_name) + ")"
    elif entity_name.startswith("sqr"):
        outermost_numerical_function = "sqr"
        entity_name = rm_function_and_brackets(entity_name, outermost_numerical_function)
        return "(" + _entity_name_to_seq_string(entity_name) + "^2" + ")"
    elif entity_name.startswith("sqrt"):
        outermost_numerical_function = "sqrt"
        entity_name = rm_function_and_brackets(entity_name, outermost_numerical_function)
        return "(" + r"sqrt(" + _entity_name_to_seq_string(entity_name) + "))"
    elif entity_name.startswith("inv"):
        outermost_numerical_function = "inv"
        entity_name = rm_function_and_brackets(entity_name, outermost_numerical_function)
        return "(" + r"1/" + _entity_name_to_seq_string(entity_name) + ")"
    else:
        return entity_name


def entity_name_to_seq_string(entity_name):
    entity_name = re.sub(" ", "", entity_name)
    return _entity_name_to_seq_string(entity_name)


def entity_to_seq_string(entity):
    return entity_name_to_seq_string(entity.name)


def logic_statement_name_to_seq_string(logic_statement_name):
    logic_statement_name = re.sub(" ", "", logic_statement_name)
    if logic_statement_name.startswith("BiggerOrEqual"):
        logic_function_name = "BiggerOrEqual"
        logic_statement_name = rm_function_and_brackets(logic_statement_name, logic_function_name)
        two_operands = extract_two_operands(logic_statement_name)
        two_operands_latex = [entity_name_to_seq_string(operand) for operand in two_operands]
        return r"\geq ".join(two_operands_latex)
    elif logic_statement_name.startswith("SmallerOrEqual"):
        logic_function_name = "SmallerOrEqual"
        logic_statement_name = rm_function_and_brackets(logic_statement_name, logic_function_name)
        two_operands = extract_two_operands(logic_statement_name)
        two_operands_latex = [entity_name_to_seq_string(operand) for operand in two_operands]
        return r"\leq ".join(two_operands_latex)
    elif logic_statement_name.startswith("Equivalent"):
        logic_function_name = "Equivalent"
        logic_statement_name = rm_function_and_brackets(logic_statement_name, logic_function_name)
        two_operands = extract_two_operands(logic_statement_name)
        two_operands_latex = [entity_name_to_seq_string(operand) for operand in two_operands]
        return r"=".join(two_operands_latex)
    else:
        raise NotImplementedError


def logic_statement_to_seq_string(logic_statement):
    return logic_statement_name_to_seq_string(logic_statement.name)
