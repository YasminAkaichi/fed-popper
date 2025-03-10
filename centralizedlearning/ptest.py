import logging
from popper.util import Settings, Stats
from popper.tester import Tester
from popper.constrain import Constrain
from popper.generate import generate_program
from popper.core import Clause, Literal, ConstVar
from popper.asp import ClingoGrounder, ClingoSolver
from popper.util import load_kbpath
from popper.loop import build_rules, decide_outcome, ground_rules
from clingo import Function, Number, String

# 🔹 Load knowledge base paths
kbpath = "trains"
bk_file, ex_file, bias_file = load_kbpath(kbpath)


# 🔹 Initialize settings
settings = Settings(bias_file, ex_file, bk_file)

# 🔹 Set up logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# 🔹 Create solver, tester, and constrainer
solver = ClingoSolver(settings)
grounder = ClingoGrounder()
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

# 4️⃣ **Use build_rules from loop.py**
constraints = build_rules(settings, stats, constrainer, tester, initial_rules, before, min_clause, outcome)
log.debug(f"Generated Constraints using build_rules: {constraints}")
for constraint in constraints:
    log.debug(f"Constraint structure: {constraint}")
# ✅ Step 1: Convert tuples back into Literal objects
# ✅ Step 1: Convert tuples back into Literal objects
proper_constraints = set()

for head, body in constraints:
    # Convert head if it's not None
    if head is None:
        head_lit = None  # No head for some constraints
    else:
        head_lit = Literal(head.predicate, head.arguments)  # Convert tuple back to Literal

    # Convert each body item back to Literal and store as frozenset
    body_lits = frozenset(Literal(lit.predicate, lit.arguments) for lit in body)

    # ✅ Store as a tuple with a frozenset to be hashable
    proper_constraints.add((head_lit, body_lits))

# ✅ Step 2: Ground the Rules
grounded_constraints = ground_rules(stats, grounder, solver.max_clauses, solver.max_vars, proper_constraints)

# ✅ Step 3: Add Grounded Constraints to the Solver
solver.add_ground_clauses(grounded_constraints)

# ✅ Step 4: Generate New Improved Rules After Adding Constraints
improved_rules, before, min_clause = generate_program(solver.get_model())
log.debug(f"Improved Rules: {improved_rules}")

# ✅ Step 5: Print the Final Rules
print("\n===== FINAL RULES =====")
print("\n".join(Clause.to_code(rule) for rule in improved_rules))
