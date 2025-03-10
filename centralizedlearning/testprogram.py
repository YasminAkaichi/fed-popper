import logging
from popper.util import Settings, Stats
from popper.tester import Tester
from popper.constrain import Constrain
from popper.generate import generate_program
from popper.core import Clause, Literal, ConstVar
from popper.asp import ClingoSolver
from popper.util import load_kbpath
from popper.loop import build_rules
from clingo import Function, Number, String

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

# 🔹 Mapping of outcomes to constraints
OUTCOME_TO_CONSTRAINTS = {
    ("ALL", "NONE"): ("banish",),
    ("ALL", "SOME"): ("generalisation",),
    ("SOME", "NONE"): ("specialisation",),
    ("SOME", "SOME"): ("specialisation", "generalisation"),
    ("NONE", "NONE"): ("specialisation", "redundancy"),
    ("NONE", "SOME"): ("specialisation", "redundancy", "generalisation"),
}

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
log.debug(f"Outcome: {outcome}")

constraints = set()
constraint_types = OUTCOME_TO_CONSTRAINTS[outcome]
log.debug(f"Constraints to Apply: {constraint_types}")

for constraint_type in constraint_types:
    if constraint_type == "generalisation":
        constraints.update(constrainer.generalisation_constraint(initial_rules, before, min_clause))
        log.debug(f"Applying Generalisation Constraints: {constraints}")
    elif constraint_type == "specialisation":
        constraints.update(constrainer.specialisation_constraint(initial_rules, before, min_clause))
        log.debug(f"Applying Specialisation Constraints: {constraints}")
    elif constraint_type == "redundancy":
        constraints.update(constrainer.redundancy_constraint(initial_rules, before, min_clause))
        log.debug(f"Applying Redundancy Constraints: {constraints}")
    elif constraint_type == "banish":
        constraints.update(constrainer.banish_constraint(initial_rules, before, min_clause))
        log.debug(f"Applying Banish Constraints: {constraints}")

log.debug(f"Final Generated Constraints: {constraints}")

# 4️⃣ **Generate Improved Rules using Constraints**
log.debug("Generating improved rules...")
popper_model = []

fake_clause_id = 1

def convert_to_string(value):
    """ Convert values to Clingo-compatible format """
    if isinstance(value, ConstVar):
        return String(value.name)
    elif isinstance(value, int):
        return Number(value)
    return String(str(value))

for head, body in constraints:
    if head is None:
        head = Literal("new_rule", ("X",))

    head_args = [convert_to_string(arg) for arg in head.arguments]

    popper_model.append(Function("head_literal", [
        Number(fake_clause_id),
        String(head.predicate),
        Number(len(head.arguments)),
        Function("args", head_args)
    ]))

    for lit in body:
        body_args = [convert_to_string(arg) for arg in lit.arguments]

        popper_model.append(Function("body_literal", [
            Number(fake_clause_id),
            String(lit.predicate),
            Number(len(lit.arguments)),
            Function("args", body_args)
        ]))

    fake_clause_id += 1

log.debug(f"Processed Model for generate_program: {popper_model}")

# 5️⃣ **Run the Learning Process Again with New Constraints**
converted_model = []

log.debug("Entering model conversion loop")
for atom in popper_model:
    log.debug(f"Processing atom: {atom}, type={type(atom)}")

improved_rules, before, min_clause = generate_program(solver.get_model())
log.debug(f"Improved Rules: {improved_rules}")

# 🔹 Print final rules
print("\n===== FINAL RULES =====")
print("\n".join(Clause.to_code(rule) for rule in improved_rules))
