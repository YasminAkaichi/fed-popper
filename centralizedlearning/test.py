import logging
from popper.util import Settings, Stats
from popper.tester import Tester
from popper.constrain import Constrain
from popper.generate import generate_program
from popper.core import Clause, Literal, ConstVar
from popper.asp import ClingoSolver
from popper.util import load_kbpath
from clingo import Function, Number, String, Tuple_ # ✅ Correct Clingo types
import clingo
# 🔹 Load knowledge base paths
kbpath = "trains"
bk_file, ex_file, bias_file = load_kbpath(kbpath)

# 🔹 Define outcome decision function
def decide_outcome(conf_matrix):
    tp, fn, tn, fp = conf_matrix
    if fn == 0:
        positive_outcome = "ALL"
    elif tp == 0 and fn > 0:
        positive_outcome = "NONE"
    else:
        positive_outcome = "SOME"

    if fp == 0:
        negative_outcome = "NONE"
    else:
        negative_outcome = "SOME"

    return (positive_outcome, negative_outcome)

# 🔹 Initialize settings

settings = Settings(bias_file, ex_file, bk_file)

# 🔹 Set up logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# 🔹 Create solver, tester, and constrainer
solver = ClingoSolver(settings)
tester = Tester(settings)
constrainer = Constrain()
stats = Stats(log_best_programs=True)

# 1️⃣ **Generate Initial Rules**
log.debug("Generating initial rules...")
initial_rules, before, min_clause = generate_program(solver.get_model())
log.debug(f"Initial Rules: {initial_rules}")

# 2️⃣ **Test the Rules**
log.debug("Testing generated rules...")
conf_matrix = tester.test(initial_rules)
log.debug(f"Test Results (Confusion Matrix): {conf_matrix}")

# 3️⃣ **Generate Constraints**
log.debug("Generating constraints from errors...")
outcome = decide_outcome(conf_matrix)

# Define mapping from outcomes to constraints
OUTCOME_TO_CONSTRAINTS = {
    ("ALL", "NONE"): ("banish",),
    ("ALL", "SOME"): ("generalisation",),
    ("SOME", "NONE"): ("specialisation",),
    ("SOME", "SOME"): ("specialisation", "generalisation"),
    ("NONE", "NONE"): ("specialisation", "redundancy"),
    ("NONE", "SOME"): ("specialisation", "redundancy", "generalisation"),
}

log.debug(f"Outcome: {outcome}")

# Get the correct constraints from the table
constraints_to_apply = OUTCOME_TO_CONSTRAINTS.get(outcome, ())

log.debug(f"Constraints to Apply: {constraints_to_apply}")

constraints = set()

if "generalisation" in constraints_to_apply:
    gen_constraints = constrainer.generalisation_constraint(initial_rules, before, min_clause)
    log.debug(f"Applying Generalisation Constraints: {gen_constraints}")
    constraints.update(gen_constraints)

if "specialisation" in constraints_to_apply:
    spec_constraints = constrainer.specialisation_constraint(initial_rules, before, min_clause)
    log.debug(f"Applying Specialisation Constraints: {spec_constraints}")
    constraints.update(spec_constraints)

if "banish" in constraints_to_apply:
    banish_constraints = constrainer.banish_constraint(initial_rules)
    log.debug(f"Applying Banish Constraints: {banish_constraints}")
    constraints.update(banish_constraints)

if "redundancy" in constraints_to_apply:
    redundancy_constraints = constrainer.redundancy_constraint(initial_rules)
    log.debug(f"Applying Redundancy Constraints: {redundancy_constraints}")
    constraints.update(redundancy_constraints)

log.debug(f"Final Generated Constraints: {constraints}")




# 4️⃣ **Convert Constraints into Proper ASP Objects**
popper_model = []
fake_clause_id = 1

def convert_to_string(value):
    """ Ensure proper conversion to Clingo ASP model values. """
    if isinstance(value, ConstVar):
        return String(value.name)
    elif isinstance(value, int):  # ✅ Convert integers to Clingo Numbers
        return Number(value)
    return String(str(value))

for head, body in constraints:
    if head is None:
        head = Literal("new_rule", ("X",))  # Ensure head exists

    # ✅ Convert arguments into Clingo-friendly format
    head_args = [convert_to_string(arg) for arg in head.arguments]

    # ✅ Convert head into a valid Clingo Function
    popper_model.append(Function("head_literal", [
        Number(fake_clause_id),  # ✅ Convert clause ID to Number
        String(head.predicate),  # ✅ Convert predicate to String()
        Number(len(head.arguments)),  # ✅ Convert integer to Number
        Function("args", head_args)  # ✅ Pass a list of properly converted values
    ]))

    for lit in body:
        # ✅ Convert arguments into Clingo-friendly format
        body_args = [convert_to_string(arg) for arg in lit.arguments]

        # ✅ Convert body literals into valid Clingo Functions
        popper_model.append(Function("body_literal", [
            Number(fake_clause_id),  # ✅ Convert clause ID to Number
            String(lit.predicate),  # ✅ Convert predicate to String()
            Number(len(lit.arguments)),  # ✅ Convert integer to Number
            Function("args", body_args)  # ✅ Pass a list of properly converted values
        ]))

    fake_clause_id += 1  # Increment fake clause ID

log.debug(f"Processed Model for generate_program: {popper_model}")


def fix_arguments(arguments):
    """ Recursively fix arguments, converting strings that represent tuples. """
    fixed_args = []
    for arg in arguments:
        if isinstance(arg, clingo.Symbol) and arg.type == clingo.SymbolType.Function:
            # Fix inner function arguments recursively
            fixed_args.append(clingo.Function(arg.name, fix_arguments(arg.arguments)))
        elif isinstance(arg, clingo.Symbol) and arg.type == clingo.SymbolType.String:
            # Check if the string represents a tuple (heuristic approach)
            if arg.string.startswith("(") and arg.string.endswith(")"):
                # Convert to a tuple of symbols
                inner_elements = arg.string.strip("()").split(", ")
                fixed_args.append(clingo.Tuple_([clingo.String(e.strip()) for e in inner_elements]))
            else:
                fixed_args.append(arg)  # Keep string as is
        else:
            fixed_args.append(arg)
    return fixed_args

# 5️⃣ **Generate Improved Rules using Properly Structured Model**
converted_model = []

print("DEBUG: Entering model conversion loop")
print(f"DEBUG: popper_model={popper_model}")

for atom in popper_model:
    print(f"DEBUG: Processing atom: {atom}, type={type(atom)}")
    second_arg = atom.arguments[1]

    if isinstance(second_arg, clingo.Symbol):
        if second_arg.type == clingo.SymbolType.Function:
            predicate = second_arg.name
        elif second_arg.type == clingo.SymbolType.String:
            predicate = str(second_arg)
        elif second_arg.type == clingo.SymbolType.Number:  # Handle numbers
            predicate = str(second_arg.number)  # Convert Number to String
        else:
            print(f"WARNING: Skipping atom {atom} because arguments[1] is an unsupported type.")
            continue
    else:
        print(f"WARNING: Skipping atom {atom} because arguments[1] is not a clingo.Symbol.")
        continue

improved_rules, before, min_clause = generate_program(converted_model)

log.debug(f"Improved Rules: {improved_rules}")

# 🔹 Print final rules
print("\n===== FINAL RULES =====")
print("\n".join(Clause.to_code(rule) for rule in improved_rules))
