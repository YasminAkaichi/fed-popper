import flwr as fl
import json
import numpy as np
import helper
from popper.tester import Tester  # Importation du testeur existant
from popper.util import Settings
from popper.loop import decide_outcome 
from popper.core import Clause 

OUTCOME_ENCODING = {
    "ALL": 1,
    "SOME": 2,
    "NONE": 3
}

settings = Settings(
    bias_file="examples/trains/bias.pl",   # Fichier contenant les biais d'apprentissage
    ex_file="examples/trains/exs.pl",  # Fichier des exemples
    bk_file="examples/trains/bk.pl",  # Fichier des connaissances de base
    max_rules=5,  # Par exemple, définir une limite pour les règles
    max_body=3,
    max_vars=5,
)

tester = Tester(settings)

def get_parameters(tester):
    """Retrieve the last computed outcome pairs (E⁺, E⁻) and send them to the server."""
    if tester.encoded_outcome is None:
        return [np.array([])]  # No computed outcome yet
    return [np.array(tester.encoded_outcome)]  # Send computed outcome to the server

 
def convert_numpy_to_prolog(rules_array):
    """Convert NumPy array back to Prolog-compatible rules."""
    rules_list = rules_array.tolist()  # Convert NumPy array to list
    return [Clause.to_code(rule) for rule in rules_list]  # Convert to Prolog format

def set_parameters(tester, parameters):
    """Receive and store the new hypothesis (rules) from the server."""
    received_rules = convert_numpy_to_prolog(parameters[0])  # Convert back to Prolog format
    tester.current_program = received_rules  # ✅ Store received rules in the tester

class FlowerClient(fl.client.NumPyClient):  # ✅ Hérite de NumPyClient, pas de tester
    def __init__(self, tester):
        """Initialisation du client Flower avec le tester Popper."""
        self.tester = tester

    def get_parameters(self, config):
        """Retrieve the last computed outcome pairs (E⁺, E⁻) and send them to the server."""
        return [np.array(self.tester.encoded_outcome) if self.tester.encoded_outcome else np.array([])]


    def fit(self, parameters, config):
        """Teste les règles reçues, calcule les outcomes et les encode."""
        set_parameters(self.tester, parameters)  # Update rules before testing
        
        # Retrieve the updated rules
        current_program = self.tester.current_program
        
        # Evaluate hypothesis and compute confusion matrix
        conf_matrix = self.tester.test(current_program)
        
        # Compute the outcome per example
        outcome = decide_outcome(conf_matrix)
        
        # Encode the outcome (convert to numbers)
        self.tester.encoded_outcome = (OUTCOME_ENCODING[outcome[0]], OUTCOME_ENCODING[outcome[1]])
        
        return [np.array(self.tester.encoded_outcome)], len(self.tester.encoded_outcome), {}

    def evaluate(self, parameters, config):
        """Evaluate the hypothesis and return accuracy."""
        set_parameters(self.tester, parameters)  # Update rules before testing
        current_program = self.tester.current_program
        conf_matrix = self.tester.test(current_program)
        accuracy = (conf_matrix[0] + conf_matrix[2]) / sum(conf_matrix)  # (TP + TN) / Total
        return 1 - accuracy, len(conf_matrix), {"accuracy": float(accuracy)}




# Démarrer le client
fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient())