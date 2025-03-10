import logging
from collections import Counter
from popper.util import Settings, Stats, load_kbpath
from popper.tester import Tester
from popper.constrain import Constrain
from popper.generate import generate_program
from popper.core import Literal, Clause
from popper.asp import ClingoGrounder, ClingoSolver
from popper.loop import decide_outcome


# ---------------------- 🔹 CONFIG & SETUP ---------------------- #

# Define Paths for Two Clients (Simulating Federated Learning)
kbpath1 = "trains"
kbpath2 = "trains2"

# Load Knowledge Base Files
bk_file1, ex_file1, bias_file1 = load_kbpath(kbpath1)
bk_file2, ex_file2, bias_file2 = load_kbpath(kbpath2)

# Create Settings for Each Client
settings_1 = Settings(bias_file1, ex_file1, bk_file1)
settings_2 = Settings(bias_file2, ex_file2, bk_file2)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# Outcome Encoding
OUTCOME_ENCODING = {"ALL": 1, "SOME": 2, "NONE": 3}
OUTCOME_DECODING = {v: k for k, v in OUTCOME_ENCODING.items()}  # Reverse mapping

# Aggregation Table
AGGREGATION_TABLE = {
    (1, 1): 1, (1, 2): 2, (1, 3): 3,
    (2, 1): 2, (2, 2): 2, (2, 3): 3,
    (3, 1): 3, (3, 2): 3, (3, 3): 3,
}

# Mapping from Outcomes to Constraints
OUTCOME_TO_CONSTRAINTS = {
    ("ALL", "NONE"): ("banish",),
    ("ALL", "SOME"): ("generalisation",),
    ("SOME", "NONE"): ("specialisation",),
    ("SOME", "SOME"): ("specialisation", "generalisation"),
    ("NONE", "NONE"): ("specialisation", "redundancy"),
    ("NONE", "SOME"): ("specialisation", "redundancy", "generalisation"),
}

# ---------------------- 🔹 FUNCTION: TEST RULES ---------------------- #

def test_rules(rules, tester):
    """Tests a given set of rules using a tester and returns the encoded outcome."""
    conf_matrix = tester.test(rules)
    outcome = decide_outcome(conf_matrix)
    encoded_outcome = (
        OUTCOME_ENCODING[outcome[0].upper()],
        OUTCOME_ENCODING[outcome[1].upper()]
    )
    return encoded_outcome

# ---------------------- 🔹 FUNCTION: AGGREGATE OUTCOMES ---------------------- #

def aggregate_outcomes(outcomes):
    """Aggregates multiple outcomes into a single outcome using the aggregation table."""
    aggregated_E_plus = outcomes[0][0]
    aggregated_E_minus = outcomes[0][1]

    for E_plus, E_minus in outcomes[1:]:
        aggregated_E_plus = AGGREGATION_TABLE[(aggregated_E_plus, E_plus)]
        aggregated_E_minus = AGGREGATION_TABLE[(aggregated_E_minus, E_minus)]

    return aggregated_E_plus, aggregated_E_minus

# ---------------------- 🔹 FUNCTION: APPLY CONSTRAINTS ---------------------- #

def apply_constraints(aggregated_outcome, constrainer):
    """Generates constraints based on the aggregated outcome."""
    positive_outcome = OUTCOME_DECODING[aggregated_outcome[0]]
    negative_outcome = OUTCOME_DECODING[aggregated_outcome[1]]

    constraints_to_apply = OUTCOME_TO_CONSTRAINTS.get((positive_outcome, negative_outcome), [])
    log.debug(f"Selected Constraints: {constraints_to_apply}")

    constraints = set()
    for constraint_type in constraints_to_apply:
        log.debug(f"Applying constraint: {constraint_type}")
        if constraint_type == "generalisation":
            constraints.update(constrainer.generalisation_constraint([], {}, {}))
        elif constraint_type == "specialisation":
            constraints.update(constrainer.specialisation_constraint([], {}, {}))
        elif constraint_type == "redundancy":
            constraints.update(constrainer.redundancy_constraint([], {}, {}))
        elif constraint_type == "banish":
            constraints.update(constrainer.banish_constraint([], {}, {}))

    log.debug(f"Total Constraints Generated: {len(constraints)}")
    return constraints

# ---------------------- 🔹 FUNCTION: GENERATE RULES ---------------------- #

def generate_rules(constraints):
    """Generates new rules based on the given constraints."""
    processed_constraints = {body for _, body in constraints}
    flattened_constraints = {literal for body in processed_constraints for literal in body}

    log.debug(f"Flattened Constraints for generate_program: {flattened_constraints}")

    # Ensure proper rule format before passing to generate_program
    processed_constraints = { (Literal("f", ("A",)), (literal,)) for literal in flattened_constraints }



    log.debug(f"Processed Constraints for generate_program: {processed_constraints}")

    # Call generate_program to generate new rules
    program, before, min_clause = generate_program(processed_constraints)

    log.debug(f"Generated Rules: {program}")
    return program

# ---------------------- 🔹 MAIN EXECUTION PIPELINE ---------------------- #

def main():
    """Main pipeline to iteratively generate, test, and refine rules."""

    # Initialize Tester Objects for Each Client
    tester_1 = Tester(settings_1)
    tester_2 = Tester(settings_2)
    
    # Initialize Constrainer
    constrainer = Constrain()

    # Initialize Stats
    stats = Stats(log_best_programs=True)

    # Step 1️⃣: Generate Initial Rules
    log.debug("Generating initial rules...")
    initial_rules, _, _ = generate_program([])  # Start with no constraints
    log.debug(f"Initial Rules: {initial_rules}")

    iteration = 0
    while iteration < 5:  # Max Iterations (adjustable)
        log.debug(f"\n===== ITERATION {iteration + 1} =====")

        # Step 2️⃣: Test Rules on Both Clients
        log.debug("Testing rules on client 1...")
        outcome_1 = test_rules(initial_rules, tester_1)
        log.debug(f"Client 1 Encoded Outcome: {outcome_1}")

        log.debug("Testing rules on client 2...")
        outcome_2 = test_rules(initial_rules, tester_2)
        log.debug(f"Client 2 Encoded Outcome: {outcome_2}")

        # Step 3️⃣: Aggregate Outcomes
        aggregated_outcome = aggregate_outcomes([outcome_1, outcome_2])
        log.debug(f"Aggregated Outcome: {aggregated_outcome}")

        # Step 4️⃣: Generate Constraints
        constraints = apply_constraints(aggregated_outcome, constrainer)

        # Step 5️⃣: Generate New Rules
        new_rules = generate_rules(constraints)

        # Stop if no new rules are generated
        if not new_rules:
            log.debug("No new rules generated. Convergence reached.")
            break

        # Update rules for the next iteration
        initial_rules = new_rules
        iteration += 1

    log.debug("\nFinal Learned Rules:")
    for rule in initial_rules:
        log.debug(rule)

# Run the Pipeline
if __name__ == "__main__":
    main()
