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
kbpath = "part1"
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
        self.best_score = None  # <- track across rounds if you want

    def encode_outcome(self, outcome):
        norm = (outcome[0].upper(), outcome[1].upper())
        return (OUTCOME_ENCODING[norm[0]], OUTCOME_ENCODING[norm[1]])

    def decode_outcome(self, enc):
        return (OUTCOME_DECODING[int(enc[0])], OUTCOME_DECODING[int(enc[1])])

    def get_parameters(self, config):
        # Send last computed (E+,E-) (encoded as ints)
        if self.encoded_outcome is None:
            log.warning("⚠️ No computed outcome yet, sending empty array.")
            return [np.array([], dtype=np.int64)]
        return [np.array(self.encoded_outcome, dtype=np.int64)]

    def set_parameters(self, parameters):
        """Receive rules from server and parse to Popper (Clause, Literal)."""
        log.debug(f"📥 Raw received parameters: {parameters}")

        # (1) Pas de paramètres → aucune règle
        if not parameters or parameters[0].size == 0:
            log.debug("🚨 No rules received, skipping update.")
            self.current_rules = []
            return

        arr = parameters[0]

        # (2) Si ce ne sont PAS des strings → ce ne sont PAS des règles
        if arr.dtype.kind not in ["U", "S", "O"]: 
            log.debug("🚨 Received parameters are NOT rules (maybe outcomes). Skipping update.")
            self.current_rules = []
            return

        try:
            # (3) Convertir les règles (strings)
            received_rules = arr.tolist()
            log.debug(f"🔹 Received rules: {received_rules}")

            # (4) Convertir en structure Popper
            parsed = [transform_rule_to_tester_format(r) for r in received_rules]
            self.current_rules = [p for p in parsed if p is not None]

            log.debug(f"✅ Parsed hypothesis: {self.current_rules}")

        except Exception as e:
            log.error(f"❌ Error processing received rules: {e}")
            self.current_rules = []




    
    def fit(self, parameters, config):
        """Test rules locally, compute local outcome (E+,E-), send encoded."""
        self.set_parameters(parameters)

        if not self.current_rules:
            log.warning("🚨 No rules available! Sending default outcome (NONE,NONE).")
            self.encoded_outcome = (OUTCOME_ENCODING["NONE"], OUTCOME_ENCODING["NONE"])
            num_examples = settings.num_pos + settings.num_neg
            return [np.array(self.encoded_outcome, dtype=np.int64)], num_examples, {}

        with stats.duration('test'):
            conf_matrix = self.tester.test(self.current_rules)
        log.debug(f"Confusion matrix: {conf_matrix}")

        outcome = decide_outcome(conf_matrix)
        score = calc_score(conf_matrix)
        stats.register_program(self.current_rules, conf_matrix)

        if self.best_score is None or score > self.best_score:
            self.best_score = score
            if outcome == (Outcome.ALL, Outcome.NONE):
                stats.register_solution(self.current_rules, conf_matrix)
            stats.register_best_program(self.current_rules, conf_matrix)

        # Encode and return as ints
        self.encoded_outcome = self.encode_outcome(outcome)
        num_examples = settings.num_pos + settings.num_neg
        log.info(f"🔹 Outcome: {outcome} → Encoded: {self.encoded_outcome}")
        return [np.array(self.encoded_outcome, dtype=np.int64)], num_examples, {}


    def evaluate(self, parameters, config):
        """Return loss, num_examples, metrics."""
        self.set_parameters(parameters)
        if not self.current_rules:
            log.warning("🚨 No rules to evaluate! Skipping.")
            return 1.0, 0, {"accuracy": 0.0}

        conf_matrix = self.tester.test(self.current_rules)
        total = sum(conf_matrix) if sum(conf_matrix) > 0 else 1
        accuracy = (conf_matrix[0] + conf_matrix[2]) / total
        num_examples = sum(conf_matrix)

        log.info(f"✅ Eval: cm={conf_matrix}, acc={accuracy:.4f}")
        return float(1 - accuracy), num_examples, {"accuracy": float(accuracy)}




# 🔹 Start the client
fl.client.start_client(
    server_address="localhost:8080",
    client=FlowerClient(tester).to_client(),  # ✅ Fixed Flower API usage
)