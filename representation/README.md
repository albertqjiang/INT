# How to use the pointer action representation?

### Pointer representation
To uniquely identify a particular entity within a algebraic expression, 
we can point out the symbol of mathematical operation associated with this entity. 
For example in the following expression marking the plus sign, identifies the entity as (a+b):
1/(a [+] b)*(b+0).

To point out the symbol in a string form, we add a special character ~ just after the symbol 
of mathematical operation, for example: (((b*c)+c)*((b*c)+~(c*b))).

## Encoding and decoding
To encode action into a pointer format one must use the function:

```ActionRepresentationPointer.action_to_formula(objective, action)```.

This function takes two arguments: objective (it must be an object of type LogicStatement) and 
action, in the format: (index_of_theorem, indices_of_entities).

To decode the action from pointer representation one must use the function:
```ActionRepresentationPointer.pointer_str_to_action(objective, action_str)```

