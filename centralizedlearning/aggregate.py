import logging
from popper.util import Settings, Stats
from popper.tester import Tester
from popper.constrain import Constrain
from popper.generate import generate_program
from popper.core import Clause, Literal
from popper.asp import ClingoGrounder, ClingoSolver
from popper.util import load_kbpath
from popper.loop import Outcome, build_rules, decide_outcome, ground_rules
from clingo import Function, Number, String

# 🔹 Load Multiple Knowledge Bases (trains, trains2)
datasets = ["trains", "reverse"]

# 🔹 Store each dataset's components
dataset_components = {}

for dataset in datasets:
    bk_file, ex_file, bias_file = load_kbpath(dataset)
    settings = Settings(bias_file, ex_file, bk_file)

    dataset_components[dataset] = {
        "solver": ClingoSolver(settings),
        "grounder": ClingoGrounder(),
        "tester": Tester(settings),
        "constrainer": Constrain(),
        "settings": settings,
        "stats": Stats(log_best_programs=True),
    }

# 🔹 Mapping of Outcomes to Constraints
OUTCOME_TO_CONSTRAINTS = {
    ("ALL", "NONE"): ("banish",),
    ("ALL", "SOME"): ("generalisation",),
    ("SOME", "NONE"): ("specialisation",),
    ("SOME", "SOME"): ("specialisation", "generalisation"),
    ("NONE", "NONE"): ("specialisation", "redundancy"),
    ("NONE", "SOME"): ("specialisation", "redundancy", "generalisation"),
}

# 🔹 Aggregation Table for Outcomes

AGGREGATION_TABLE = {
    (Outcome.ALL, Outcome.ALL): Outcome.ALL,
    (Outcome.ALL, Outcome.SOME): Outcome.ALL,
    (Outcome.ALL, Outcome.NONE): Outcome.ALL,
    (Outcome.SOME, Outcome.ALL): Outcome.ALL,
    (Outcome.SOME, Outcome.SOME): Outcome.SOME,
    (Outcome.SOME, Outcome.NONE): Outcome.SOME,
    (Outcome.NONE, Outcome.ALL): Outcome.ALL,
    (Outcome.NONE, Outcome.SOME): Outcome.SOME,
    (Outcome.NONE, Outcome.NONE): Outcome.NONE,
}

# 🔹 Function to Aggregate Outcomes
def aggregate_outcomes(outcomes):
    """Aggregates multiple outcomes into a single outcome using the aggregation table."""
    
    log.debug("🔹 Starting Outcome Aggregation 🔹")
    log.debug(f"Initial Outcomes: {outcomes}")

    aggregated_E_plus = outcomes[0][0]  # ✅ These should now be Outcome.ALL, Outcome.SOME, or Outcome.NONE
    aggregated_E_minus = outcomes[0][1]

    for i, (E_plus, E_minus) in enumerate(outcomes[1:], start=1):
        prev_E_plus, prev_E_minus = aggregated_E_plus, aggregated_E_minus

        aggregated_E_plus = AGGREGATION_TABLE[(aggregated_E_plus, E_plus)]
        aggregated_E_minus = AGGREGATION_TABLE[(aggregated_E_minus, E_minus)]

        log.debug(f"Step {i}: E⁺: ({prev_E_plus} + {E_plus}) → {aggregated_E_plus},  E⁻: ({prev_E_minus} + {E_minus}) → {aggregated_E_minus}")

    log.info(f"✅ Final Aggregated Outcome: ({aggregated_E_plus}, {aggregated_E_minus})")
    print(f"\n🔹 Aggregated Outcome: ({aggregated_E_plus}, {aggregated_E_minus}) 🔹")

    return aggregated_E_plus, aggregated_E_minus

# 🔹 Set up logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# ✅ Iterative Learning Until Convergence
converged = False
iteration = 0
max_iterations = 10  # Set a limit to prevent infinite loops

while not converged and iteration < max_iterations:
    log.debug(f"🌀 Iteration {iteration + 1} - Generating Initial Rules...")

    # 🔹 1️⃣ Generate Initial Rules for Each Dataset
    dataset_rules = {}
    for dataset, components in dataset_components.items():
        log.debug(f"Generating rules for dataset: {dataset}")
        solver = components["solver"]
        rules, before, min_clause = generate_program(solver.get_model())
        dataset_rules[dataset] = (rules, before, min_clause)
        log.debug(f"Initial Rules for {dataset}: {rules}")

    # 🔹 2️⃣ Test on Each Dataset
    outcomes = {}
    for dataset, components in dataset_components.items():
        tester = components["tester"]
        log.debug(f"Testing on dataset: {dataset}")
        conf_matrix = tester.test(dataset_rules[dataset][0])  # Test the rules
        log.debug(f"Test Results (Confusion Matrix) on {dataset}: {conf_matrix}")
        outcome = decide_outcome(conf_matrix)
        outcomes[dataset] = outcome

    # 🔹 3️⃣ Aggregate Outcomes Across Datasets
    aggregated_outcome = aggregate_outcomes(list(outcomes.values()))
    log.debug(f"Aggregated Outcome Across Datasets: {aggregated_outcome}")

    # 🔹 4️⃣ Generate Constraints for Each Dataset
    dataset_constraints = {}
    for dataset, components in dataset_components.items():
        log.debug(f"Generating constraints for dataset: {dataset}")
        settings = components["settings"]
        stats = components["stats"]
        constrainer = components["constrainer"]
        tester = components["tester"]
        rules, before, min_clause = dataset_rules[dataset]

        constraints = build_rules(settings, stats, constrainer, tester, rules, before, min_clause, aggregated_outcome)
        dataset_constraints[dataset] = constraints
        log.debug(f"Constraints for {dataset}: {constraints}")

    # 🔹 5️⃣ Ground Constraints & Apply to Each Dataset’s Solver
    for dataset, components in dataset_components.items():
        solver = components["solver"]
        grounder = components["grounder"]
        stats = components["stats"]

        # Convert constraints to proper format
        proper_constraints = set()
        for head, body in dataset_constraints[dataset]:
            head_lit = None if head is None else Literal(head.predicate, head.arguments)
            body_lits = frozenset(Literal(lit.predicate, lit.arguments) for lit in body)
            proper_constraints.add((head_lit, body_lits))

        # Ground Constraints
        grounded_constraints = ground_rules(stats, grounder, solver.max_clauses, solver.max_vars, proper_constraints)
        solver.add_ground_clauses(grounded_constraints)

    # 🔹 6️⃣ Generate Improved Rules for Each Dataset
    improved_rules = {}
    for dataset, components in dataset_components.items():
        solver = components["solver"]
        log.debug(f"Generating improved rules for dataset: {dataset}")
        rules, before, min_clause = generate_program(solver.get_model())
        improved_rules[dataset] = rules
        log.debug(f"Improved Rules for {dataset}: {rules}")

    # 🔹 7️⃣ Check for Convergence
    all_converged = all(improved_rules[ds] == dataset_rules[ds][0] for ds in datasets)
    if all_converged:
        converged = True
        log.debug("✅ Converged: No further changes in rules.")

    iteration += 1

# 🔹 Print Final Learned Rules for Each Dataset
for dataset, rules in improved_rules.items():
    print(f"\n===== FINAL RULES for {dataset} =====")
    print("\n".join(Clause.to_code(rule) for rule in rules))
