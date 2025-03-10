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

# 🔹 Load multiple datasets
dataset_names = ["trains", "reverse"]  # Modify with your dataset names
datasets = {}

for name in dataset_names:
    bk_file, ex_file, bias_file = load_kbpath(name)
    datasets[name] = {
        "settings": Settings(bias_file, ex_file, bk_file),
        "solver": ClingoSolver(Settings(bias_file, ex_file, bk_file)),
        "tester": Tester(Settings(bias_file, ex_file, bk_file)),
        "grounder": ClingoGrounder(),
        "constrainer": Constrain(),
    }

# 🔹 Logging Setup
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
stats = Stats(log_best_programs=True)

# 1️⃣ **Generate Initial Rules (No Constraints Yet)**
log.debug("🌀 Generating Initial Rules...")
initial_rules_by_dataset = {}

for name, data in datasets.items():
    log.debug(f"Generating rules for dataset: {name}")
    initial_rules, before, min_clause = generate_program(data["solver"].get_model())
    initial_rules_by_dataset[name] = initial_rules
    log.debug(f"Initial Rules for {name}: {initial_rules}")

# 2️⃣ **Test Rules on Each Dataset**
log.debug("🧪 Testing Rules on Each Dataset...")
outcomes = {}

for name, data in datasets.items():
    log.debug(f"Testing on dataset: {name}")
    conf_matrix = data["tester"].test(initial_rules_by_dataset[name])
    outcome = decide_outcome(conf_matrix)
    outcomes[name] = outcome
    log.debug(f"Test Results (Confusion Matrix) on {name}: {conf_matrix}")
    log.debug(f"Outcome on {name}: {outcome}")

# 3️⃣ **Aggregate Outcomes**
log.debug("🔹 Starting Outcome Aggregation 🔹")
outcome_list = list(outcomes.values())
log.debug(f"Initial Outcomes: {outcome_list}")

aggregated_outcome = outcome_list[0]
for outcome in outcome_list[1:]:
    aggregated_outcome = (max(aggregated_outcome[0], outcome[0]), max(aggregated_outcome[1], outcome[1]))

log.info(f"✅ Final Aggregated Outcome: {aggregated_outcome}")
print(f"\n🔹 Aggregated Outcome: {aggregated_outcome} 🔹")

# 4️⃣ **Generate Constraints Based on Aggregated Outcome**
constraints_by_dataset = {}

for name, data in datasets.items():
    log.debug(f"Generating constraints for dataset: {name}")
    constraints = build_rules(
        data["settings"], stats, data["constrainer"], data["tester"],
        initial_rules_by_dataset[name], before, min_clause, aggregated_outcome
    )
    constraints_by_dataset[name] = constraints
    log.debug(f"Constraints for {name}: {constraints}")

# 5️⃣ **Use Constraints to Generate New Improved Rules (One Time)**
log.debug("🚀 Generating Improved Rules...")

for name, data in datasets.items():
    log.debug(f"Generating improved rules for dataset: {name}")

    # ✅ Convert constraints into ground rules before adding them to the solver
    proper_constraints = set()
    for head, body in constraints_by_dataset[name]:
        head_lit = None if head is None else Literal(head.predicate, head.arguments)
        body_lits = frozenset(Literal(lit.predicate, lit.arguments) for lit in body)
        proper_constraints.add((head_lit, body_lits))

    # ✅ Ground constraints
    grounded_constraints = ground_rules(stats, data["grounder"], data["solver"].max_clauses, data["solver"].max_vars, proper_constraints)
    data["solver"].add_ground_clauses(grounded_constraints)

    # ✅ Generate the improved rules (ONE iteration)
    improved_rules, before, min_clause = generate_program(data["solver"].get_model())
    log.debug(f"Improved Rules for {name}: {improved_rules}")

# 6️⃣ **Print Final Rules**
print("\n===== FINAL RULES =====")
for name, data in datasets.items():
    print(f"\n🔹 Dataset: {name}")
    print("\n".join(Clause.to_code(rule) for rule in improved_rules))
