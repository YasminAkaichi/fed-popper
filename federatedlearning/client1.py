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
import flwr as fl
import json
import numpy as np

OUTCOME_ENCODING = {"ALL": 1, "SOME": 2, "NONE": 3}
OUTCOME_DECODING = {1: "ALL", 2: "SOME", 3: "NONE"} 


# 🔹 Logging Setup
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
stats = Stats(log_best_programs=True)

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

def get_parameters(self, config):
    """Retrieve the last computed outcome pairs (E+, E-) and send them to the server."""
    if self.encoded_outcome is None:
        log.warning("⚠️ No computed outcome yet, sending empty array.")
        return [np.array([])]  # No computed outcome yet
    return [np.array(self.encoded_outcome)]  

 
def convert_numpy_to_prolog(rules_array):
    """Convert NumPy array back to Prolog-compatible rules."""
    rules_list = rules_array.tolist()  # Convert NumPy array to list
    return [Clause.to_code(rule) for rule in rules_list]  # Convert to Prolog format

def set_parameters(tester, parameters):
    """Receive and store the new hypothesis (rules) from the server."""
    if parameters[0].size == 0:
        log.debug(f"🚨 No rules received for")
        return 
    received_rules = convert_numpy_to_prolog(parameters[0])  # Convert back to Prolog format
    tester.current_program = received_rules  # ✅ Store received rules in the tester
    #decoded_rules = [Clause.to_code(rule) for rule in parameters[0].tolist()]
    #log.debug(f"📥 Received Hypothesis for {self.dataset_name}: {decoded_rules}")

    #self.current_rules = decoded_rules
def encode_outcome(self, outcome):
        """Convert symbolic outcome ('ALL', 'SOME', 'NONE') to numerical encoding."""
        return (OUTCOME_ENCODING[outcome[0]], OUTCOME_ENCODING[outcome[1]])

def decode_outcome(self, encoded_outcome):
    """Convert numerical encoding (1, 2, 3) back to symbolic outcome ('ALL', 'SOME', 'NONE')."""
    return (OUTCOME_DECODING[encoded_outcome[0]], OUTCOME_DECODING[encoded_outcome[1]])

class FlowerClient(fl.client.NumPyClient):  # ✅ Hérite de NumPyClient, pas de tester
    def __init__(self, tester,solver):
        """Initialisation du client Flower avec le tester Popper."""
        self.tester = tester
        self.solver = solver
        self.current_rules = None  # Store current hypothesis
        self.encoded_outcome = None  # Store outcome as (E+, E-)

    def get_parameters(self, config):
        """Retrieve the last computed outcome pairs (E+, E-) and send them to the server."""
        if self.encoded_outcome is None:
            log.warning("⚠️ No computed outcome yet, sending empty array.")
            return [np.array([])]  # No computed outcome yet
        return [np.array(self.encoded_outcome)]  # ✅ Send encoded outcome

    def fit(self, parameters, config):
        """Teste les règles reçues, calcule les outcomes et les encode."""
        set_parameters(self.tester, parameters)  # Update rules before testing
        
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
        self.encoded_outcome = self.encode_outcome(outcome)
        log.info(f"🔹 Client {self.dataset_name}: Computed Outcome {outcome} (Encoded: {self.encoded_outcome})")
        return [self.get_parameters(config)], len(self.tester.encoded_outcome), {}

    def evaluate(self, parameters, config):
        """Evaluate the hypothesis and return accuracy."""
        set_parameters(self.tester, parameters)  # Update rules before testing
        current_program = self.tester.current_program
        conf_matrix = self.tester.test(current_program)
        accuracy = (conf_matrix[0] + conf_matrix[2]) / sum(conf_matrix)  # (TP + TN) / Total
        return 1 - accuracy, len(conf_matrix), {"accuracy": float(accuracy)}


# Démarrer le client
fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient(tester,solver))

