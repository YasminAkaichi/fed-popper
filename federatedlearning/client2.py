import logging
from popper.util import Settings, Stats
from popper.tester import Tester
from popper.core import Clause, Literal
from popper.util import load_kbpath, format_program
from popper.loop import decide_outcome, Outcome, calc_score
import flwr as fl
import numpy as np

# 🔹 Outcome Encoding
OUTCOME_ENCODING = {"ALL": 1, "SOME": 2, "NONE": 3}
OUTCOME_DECODING = {1: "ALL", 2: "SOME", 3: "NONE"}
# 🔹 Logging Setup
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# 🔹 Load dataset
kbpath = "part2"
bk_file, ex_file, bias_file = load_kbpath(kbpath)

# 🔹 Initialize ILP settings
settings = Settings(bias_file, ex_file, bk_file)
tester = Tester(settings)
stats = Stats(log_best_programs=settings.info)
settings.num_pos, settings.num_neg = len(tester.pos), len(tester.neg)
best_score = None
import re

def parse_clause(code: str):
    """Convert a Prolog-style rule back into (head, body) tuple."""
    
    # 1️⃣ Remove the final period if present
    code = code.strip()
    if code.endswith('.'):
        code = code[:-1]
    
    # 2️⃣ Split head and body
    if ":-" in code:
        head, body = code.split(":-")
        body_literals = tuple(lit.strip() for lit in body.split(","))
    else:
        head = code.strip()
        body_literals = ()  # No body literals (a fact)
    
    return head.strip(), body_literals
from popper.core import Clause, Literal

def transform_rule_to_tester_format(rule_str):
    log.debug(f"🔍 Transforming rule: {rule_str}")

    try:
        # ✅ Split head and body correctly
        head_body = rule_str.split(":-")
        if len(head_body) != 2:
            raise ValueError(f"Invalid rule format: {rule_str}")

        head_str = head_body[0].strip()
        body_str = head_body[1].strip()

        # ✅ **Fix: Properly extract body literals using regex**
        body_literals = re.findall(r'\w+\(.*?\)', body_str)

        log.debug(f"🔹 Parsed head: {head_str}")
        log.debug(f"🔹 Parsed body literals: {body_literals}")

        # ✅ Convert to Literal objects (assuming `Literal.from_string` exists)
        head = Literal.from_string(head_str)
        body = tuple(Literal.from_string(lit) for lit in body_literals)

        formatted_rule = (head, body)
        log.debug(f"✅ Formatted rule: {formatted_rule}")

        return formatted_rule
    except Exception as e:
        log.error(f"❌ Error transforming rule: {rule_str} → {e}")
        return None  # Return None to indicate failure

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, tester):
        """Initialize the Flower client with its ILP components."""
        self.tester = tester  # Tester for ILP evaluation
        self.current_rules = None  # Store current hypothesis
        self.encoded_outcome = None  # Store encoded outcome as (E+, E-)

    def encode_outcome(self, outcome):
        """Convert ('ALL', 'SOME', 'NONE') to (1,2,3) encoding."""
        normalized_outcome = (outcome[0].upper(), outcome[1].upper())  # Convert to uppercase
        return (OUTCOME_ENCODING[normalized_outcome[0]], OUTCOME_ENCODING[normalized_outcome[1]])

    def decode_outcome(self, encoded_outcome):
        """Convert (1,2,3) encoding back to ('ALL', 'SOME', 'NONE')."""
        return (OUTCOME_DECODING[encoded_outcome[0]], OUTCOME_DECODING[encoded_outcome[1]])

    def get_parameters(self, config):
        """Retrieve and send the last computed (E+, E-) outcome to the server."""
        if self.encoded_outcome is None:
            log.warning("⚠️ No computed outcome yet, sending empty array.")
            return [np.array([])]
        return [np.array(self.encoded_outcome, dtype ="<U100")]  # ✅ Send encoded outcome

    def set_parameters(self, parameters):
        """Receive and store the new hypothesis (rules) from the server."""
        log.debug(f"📥 Raw received parameters: {parameters}")

        if parameters[0].size == 0:
            log.debug("🚨 No rules received, skipping update.")
            self.current_rules = []
            return 
        parameters = np.array(parameters, dtype="<U100") 
        received_rules = parameters[0].tolist()
        log.debug(f"🔹 Converted received rules to list: {received_rules}")

        try:
            # ✅ Convert received rules into Clause objects
            parsed_rules = [transform_rule_to_tester_format(rule) for rule in received_rules]

            # 🔹 Remove any None values (failed transformations)
            self.current_rules = [rule for rule in parsed_rules if rule is not None]

            log.debug(f"✅ Updated client hypothesis: {self.current_rules}")
        except Exception as e:
            log.error(f"❌ Error processing received rules: {e}")
            self.current_rules = []  # Reset to empty if parsing fails



    def fit(self, parameters, config):
        """Test the received rules, compute outcomes, and send them back."""
        self.set_parameters(parameters)  # ✅ Update rules before testing
        log.debug(f"🔹 Current rules: {self.current_rules}")

        if not self.current_rules:
            log.warning("🚨 No rules available! Sending default outcome (NONE, NONE).")
            self.encoded_outcome = (OUTCOME_ENCODING["NONE"], OUTCOME_ENCODING["NONE"])
            return [np.array(self.encoded_outcome)], len(self.encoded_outcome), {}

        # 1️⃣ **Test the Rules**
        log.debug("Testing received rules...")
        with stats.duration('test'):
            conf_matrix = self.tester.test(self.current_rules)
        log.debug(f"Test Results (Confusion Matrix): {conf_matrix}")

        # 2️⃣ **Generate Constraints**
        log.debug("Generating constraints from errors...")
        outcome = decide_outcome(conf_matrix)
        log.debug(f"Outcome: {outcome}")
        score = calc_score(conf_matrix)
        stats.register_program(self.current_rules, conf_matrix)
        # UPDATE BEST PROGRAM
        best_score = None
        if best_score == None or score > best_score:
            best_score = score
            if outcome == (Outcome.ALL, Outcome.NONE):
                stats.register_solution(self.current_rules, conf_matrix)
            stats.register_best_program(self.current_rules, conf_matrix)
        # 3️⃣ **Encode outcome before sending**
        
        self.encoded_outcome = self.encode_outcome(outcome)
        log.info(f"🔹 Computed Outcome: {outcome} → Encoded: {self.encoded_outcome}")
        
        return [np.array(self.encoded_outcome)], len(self.encoded_outcome), {}


    def evaluate(self, parameters, config):
        """Evaluate the hypothesis and return accuracy."""
        log.info(f"📥 Received parameters for evaluation: {parameters}")

        try:
            self.set_parameters(parameters)
        except Exception as e:
            log.error(f"❌ Error processing received rules: {e}")
            self.current_rules = []
        
        if not self.current_rules:
            log.warning("🚨 No rules to evaluate! Skipping evaluation.")
            return 1.0, 0, {"accuracy": 0.0}

        conf_matrix = self.tester.test(self.current_rules)
        accuracy = (conf_matrix[0] + conf_matrix[2]) / sum(conf_matrix)

        log.info(f"✅ Evaluation results: {conf_matrix}, Accuracy: {accuracy}")
        return float(1 - accuracy), len(conf_matrix), {"accuracy": float(accuracy)}




# 🔹 Start the client
fl.client.start_client(
    server_address="localhost:8080",
    client=FlowerClient(tester).to_client(),  # ✅ Fixed Flower API usage
)