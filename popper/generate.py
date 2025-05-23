from . core import Literal
from collections import defaultdict
import clingo 
def gen_args(args):
    return tuple(chr(ord('A') + arg.number) for arg in args)

def generate_program(model):
    before     = defaultdict(set)
    min_clause = defaultdict(lambda: 0)
    directions = defaultdict(lambda: defaultdict(lambda: '?'))
    clause_id_to_body = defaultdict(set)
    clause_id_to_head = {}

    #print("DEBUG: Entering generate_program()")
    #print(f"DEBUG: model={model}")

    for atom in model:
        #print(f"DEBUG: Processing atom: {atom}, type={type(atom)}")
        #print(f"DEBUG: atom.arguments={atom.arguments}")

        # Ensure the atom has at least two arguments
        if len(atom.arguments) < 2:
            #print(f"WARNING: Skipping atom {atom} because it has less than 2 arguments.")
            continue

        # Extract the second argument
        second_arg = atom.arguments[1]

        # Handle cases where it's a Function
        if isinstance(second_arg, clingo.Symbol) and second_arg.type == clingo.SymbolType.Function:
            predicate = second_arg.name
        # Handle cases where it's a String (convert it to a function name)
        elif isinstance(second_arg, clingo.Symbol) and second_arg.type == clingo.SymbolType.String:
            predicate = str(second_arg)
        else:
            #print(f"WARNING: Skipping atom {atom} because arguments[1] is not a function or string (type={type(second_arg)}).")
            continue

        #print(f"DEBUG: Extracted predicate={predicate}")
        if atom.name == 'body_literal':
            clause_id = atom.arguments[0].number
            predicate = atom.arguments[1].name
            arity = atom.arguments[2].number
            arguments = gen_args(atom.arguments[3].arguments)
            body_literal = (predicate, arguments, arity)
            clause_id_to_body[clause_id].add(body_literal)

        elif atom.name == 'head_literal':
            clause_id = atom.arguments[0].number
            predicate = atom.arguments[1].name
            arity = atom.arguments[2].number
            args = atom.arguments[3].arguments
            arguments = gen_args(atom.arguments[3].arguments)
            head_literal = (predicate, arguments, arity)
            clause_id_to_head[clause_id] = head_literal

        elif atom.name == 'direction_':
            pred_name = atom.arguments[0].name
            arg_index = atom.arguments[1].number
            arg_dir_str = atom.arguments[2].name

            if arg_dir_str == 'in':
                arg_dir = '+'
            elif arg_dir_str == 'out':
                arg_dir = '-'
            else:
                raise Exception(f'Unrecognised argument direction "{arg_dir_str}"')
            directions[pred_name][arg_index] = arg_dir

        elif atom.name == 'before':
            clause1 = atom.arguments[0].number
            clause2 = atom.arguments[1].number
            before[clause1].add(clause2)

        elif atom.name == 'min_clause':
            clause = atom.arguments[0].number
            min_clause_num = atom.arguments[1].number
            min_clause[clause] = max(min_clause[clause], min_clause_num)

    clauses = []
    for clause_id in clause_id_to_head:

        (head_pred, head_args, head_arity) = clause_id_to_head[clause_id]
        head_modes = tuple(directions[head_pred][i] for i in range(head_arity))
        head = Literal(head_pred, head_args, head_modes)

        body = set()
        for (body_pred, body_args, body_arity) in clause_id_to_body[clause_id]:
            body_modes = tuple(directions[body_pred][i] for i in range(body_arity))
            body.add(Literal(body_pred, body_args, body_modes))
        body = frozenset(body)
        clauses.append((head, body))
    clauses = tuple(clauses)
    return (clauses, before, min_clause)