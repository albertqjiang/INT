from representation.action_representation_pointer import generate_masks_for_logic_statement, CHAR_TO_AXIOM, \
    AXIOM_LENGTH, POINTER_SYMBOLS


def pointer_str_to_action(objective, action_str):
    action_raw = split_pointer_action_str(action_str)
    assert action_raw is not None, 'Improperly decoded action string'
    _, mask_to_entity = generate_masks_for_logic_statement(objective)
    action = [action_raw[0]]
    for entity_str in action_raw[1:]:
        if entity_str in mask_to_entity:
            action.append(mask_to_entity[entity_str])
        else:
            raise ValueError(f'Unrecognized entity: {entity_str}')
    return action

def split_pointer_action_str(action_str):
    if action_str[0] != '@' or action_str[-1:] != '$':
        raise ValueError('Invalid prediction format')
    prediction_str = action_str[1:-1]
    if len(prediction_str) == 0:
        return None

    if prediction_str[0] in CHAR_TO_AXIOM:
        axiom = CHAR_TO_AXIOM[prediction_str[0]]
        axiom_len = AXIOM_LENGTH[prediction_str[0]]
        input_entities_raw = [prediction_str[1:] for _ in range(axiom_len)]
        input_entities_str = []
        for num in range(len(input_entities_raw)):
            entity_str = input_entities_raw[num]
            pointer_symbol = POINTER_SYMBOLS[num]
            for different_pointer_symbol in POINTER_SYMBOLS:
                if different_pointer_symbol != pointer_symbol:
                    entity_str = entity_str.replace(different_pointer_symbol, '')
            entity_str = entity_str.replace(pointer_symbol, POINTER_SYMBOLS[0])
            input_entities_str.append(entity_str)
        return [axiom, *input_entities_str]
    else:
        return None