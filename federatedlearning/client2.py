import logging
from popper.util import Settings, Stats
from popper.tester import Tester
from popper.constrain import Constrain
from popper.generate import generate_program
from popper.core import Clause, Literal
from popper.asp import ClingoGrounder, ClingoSolver
from popper.util import load_kbpath
from popper.loop import decide_outcome
import flwr as fl
import numpy as np

# 🔹 Outcome Encoding
OUTCOME_ENCODING = {"ALL": 1, "SOME": 2, "NONE": 3}
OUTCOME_DECODING = {1: "ALL", 2: "SOME", 3: "NONE"} 

# 🔹 Logging Setup
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# 🔹 Load dataset
kbpath = "trains"
bk_file, ex_file, bias_file = load_kbpath(kbpath)

# 🔹 Initialize ILP settings
settings = Settings(bias_file, ex_file, bk_file)
solver = ClingoSolver(settings)
tester = Tester(settings)

def set_parameters(tester, parameters):
    """Receive and store the new hypothesis (rules) from the server."""
    if parameters[0].size == 0:
        log.debug("🚨 No rules received, skipping update.")
        return 

    # Ajoute un log pour voir ce qu'on reçoit
    log.debug(f"🚀 Received raw rules: {parameters[0].tolist()}")

    # Vérifie si les règles sont bien sous forme de chaînes de caractères
    try:
        received_rules = [Clause.to_code(str(rule)) for rule in parameters[0].tolist()]
    except Exception as e:
        log.error(f"❌ Error converting received rules: {e}")
        return
    
    tester.current_program = received_rules  # ✅ Stocker les règles dans le tester
    log.info(f"📥 Updated client hypothesis: {received_rules}")


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, tester, solver):
        """Initialize the Flower client with its ILP components."""
        self.tester = tester  # Tester for ILP evaluation
        self.solver = solver  # Solver for rule generation
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
        return [np.array(self.encoded_outcome)]  # ✅ Send encoded outcome

    def set_parameters(self, parameters):
        """Receive and store the new hypothesis (rules) from the server."""
        if parameters[0].size == 0:
            log.debug("🚨 No rules received, skipping update.")
            return 
        received_rules = self.convert_numpy_to_prolog(parameters[0])
        self.current_rules = received_rules  # ✅ Store received rules

    def convert_numpy_to_prolog(self, rules_array):
        """Convert NumPy array back to Prolog-compatible rules."""
        rules_list = rules_array.tolist()
        return [Clause.to_code(rule) for rule in rules_list]  

    def fit(self, parameters, config):
        """Test the received rules, compute outcomes, and send them back."""
        self.set_parameters(parameters)  # ✅ Update rules before testing
        
        # 1️⃣ **Generate Initial Rules**
        log.debug("Generating initial rules...")
        initial_rules, before, min_clause = generate_program(self.solver.get_model())
        log.debug(f"Initial Rules: {initial_rules}")
        
        # 2️⃣ **Test the Rules**
        log.debug("Testing generated rules...")
        conf_matrix = self.tester.test(initial_rules)
        log.debug(f"Test Results (Confusion Matrix): {conf_matrix}")

        # 3️⃣ **Generate Constraints**
        log.debug("Generating constraints from errors...")
        outcome = decide_outcome(conf_matrix)
        log.debug(f"Outcome: {outcome}")

        # 4️⃣ **Encode outcome before sending**
        self.encoded_outcome = self.encode_outcome(outcome)
        log.info(f"🔹 Computed Outcome: {outcome} → Encoded: {self.encoded_outcome}")

        return [np.array(self.encoded_outcome)], len(self.encoded_outcome), {}

    def evaluate(self, parameters, config):
        """Evaluate the hypothesis and return accuracy."""
        log.info(f"📥 Received parameters for evaluation: {parameters}")  # Ajout du log

        set_parameters(self.tester, parameters)  # Mise à jour des règles
        current_program = self.tester.current_program

        if not current_program:
            log.warning("🚨 No rules to evaluate! Skipping evaluation.")
            return 1.0, 0, {"accuracy": 0.0}  # Évite de planter en renvoyant un résultat bidon

        conf_matrix = self.tester.test(current_program)
        accuracy = (conf_matrix[0] + conf_matrix[2]) / sum(conf_matrix)  # (TP + TN) / Total

        log.info(f"✅ Evaluation results: {conf_matrix}, Accuracy: {accuracy}")  # Log des résultats
        return 1 - accuracy, len(conf_matrix), {"accuracy": float(accuracy)}


# 🔹 Start the client
fl.client.start_client(
    server_address="localhost:8080",
    client=FlowerClient(tester, solver).to_client(),  # ✅ Fixed Flower API usage
)
