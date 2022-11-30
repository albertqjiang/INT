from logic.logic import LogicFunction

BiggerOrEqual = LogicFunction("BiggerOrEqual", 2)
SmallerOrEqual = LogicFunction("SmallerOrEqual", 2)
Equivalent = LogicFunction("Equivalent", 2)

necessary_logic_functions = {
    "BiggerOrEqual": BiggerOrEqual,
    "SmallerOrEqual": SmallerOrEqual,
    "Equivalent": Equivalent
}
