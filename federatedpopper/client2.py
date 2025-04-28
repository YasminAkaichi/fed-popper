import logging
from popper.util import Settings, Stats
from popper.tester import Tester
from popper.core import Clause, Literal
from popper.util import load_kbpath, format_program, parse_settings
from popper.loop import decide_outcome, calc_score, Outcome, popper, learn_solution
import flwr as fl
import numpy as np

from popper.asp import ClingoGrounder, ClingoSolver

from popper.loop import Outcome, build_rules, decide_outcome, ground_rules, Con,calc_score
from popper.constrain import Constrain
from popper.generate import generate_program
import re

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

settings = parse_settings()
# 🔹 Load dataset
kbpath = "trains"
bk_file, ex_file, bias_file = load_kbpath(kbpath)
#settings = Settings(bias_file, ex_file, bk_file)

stats = Stats(log_best_programs=settings.info)
tester = Tester(settings)
solver = ClingoSolver(settings)
grounder = ClingoGrounder()
constrainer = Constrain()
settings.num_pos, settings.num_neg = len(tester.pos), len(tester.neg)

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
    def __init__(self, settings, stats):
        """Initialize the Flower client with its ILP components."""
        
        self.settings = settings
        self.tester = tester
        self.solver = solver
        self.grounder = grounder 
        self.constrainer = constrainer
        self.stats = stats
        self.current_rules = None  # Store current hypothesis
        

    def get_parameters(self, config):
        """Retourner les règles courantes sous forme de chaînes Prolog encodées en np.array."""
        if not self.current_rules:
            log.warning("⚠️ Aucune règle actuelle, envoi d'un tableau vide.")
            return [np.array([])]
        # Utilisation propre de Clause to code pour sérialiser 
        rule_strings = [Clause.to_code(rule) for rule in self.current_rules]
        rule_array = np.array(rule_strings, dtype="<U1000")
        log.debug(f"Envoi des règles : {rule_strings}")
        return [rule_array]

    def set_parameters(self, parameters):
        """Receive and store the new hypothesis (rules) from the server."""
        log.debug(f"Raw received parameters: {parameters}")

        if not parameters or parameters[0].size == 0:
            log.debug(" No rules received, skipping update.")
            self.current_rules = []
            return 
        try:
            received_rules = parameters[0].tolist()
            log.debug(f"🔹 Converted received rules to list: {received_rules}")
            # ✅ Convert received rules into Clause objects
            parsed_rules = [transform_rule_to_tester_format(rule) for rule in received_rules]

            # 🔹 Remove any None values (failed transformations)
            self.current_rules = [rule for rule in parsed_rules if rule is not None]
            log.debug(f"✅ Updated client hypothesis: {self.current_rules}")
        except Exception as e:
            log.error(f"❌ Error processing received rules: {e}")
            self.current_rules = []  # Reset to empty if parsing fails

    def fit(self, parameters, config):
        """Génère des règles avec POPPER, les stocke et les retourne au serveur."""
        print("🔥 FIT() called!")
        log.info("🚀 Lancement de POPPER dans fit()...")
        #self.set_parameters(parameters)
        # learning 
        best_score = None
        for size in range(1, settings.max_literals + 1):
            
            self.stats.update_num_literals(size)
            self.solver.update_number_of_literals(size)
            while True:
                # GENERATE HYPOTHESIS
                with self.stats.duration('generate'):
                    model = self.solver.get_model() 
                    if not model:
                        break
                    (program, before, min_clause) = generate_program(model)
                # TEST HYPOTHESIS
                with self.stats.duration('test'):
                    conf_matrix = self.tester.test(program)
                    outcome = decide_outcome(conf_matrix)
                    score = calc_score(conf_matrix)
                self.stats.register_program(program, conf_matrix)

                # UPDATE BEST PROGRAM
                if best_score == None or score > best_score:
                    best_score = score

                    if outcome == (Outcome.ALL, Outcome.NONE):
                        self.stats.register_solution(program, conf_matrix)
                        return self.stats.solution.code

                    self.stats.register_best_program(program, conf_matrix)
                    learned_rules = self.stats.best_program.code if self.stats.best_program else None
                    
                # BUILD RULES
                with self.stats.duration('build'):
                    rules = build_rules(self.settings, self.stats, self.constrainer, self.tester, program, before, min_clause, outcome)

                # GROUND RULES
                with self.stats.duration('ground'):
                    rules = ground_rules(self.stats, self.grounder, self.solver.max_clauses, self.solver.max_vars, rules)

                # UPDATE SOLVER
                with self.stats.duration('add'):
                    self.solver.add_ground_clauses(rules)

        self.stats.register_completion()
        learned_rules = self.stats.best_program.code if self.stats.best_program else None

        log.info(f"learned rules {learned_rules}")
        if not learned_rules:
            log.warning("🚨 Aucune règle générée. Envoi d’un tableau vide.")
            self.current_rules = []
            return [np.array([])], 0, {}
         
        rule_strings = learned_rules if isinstance(learned_rules, list) else [learned_rules]

        # 🧠 On utilise set_parameters pour stocker les règles
        rules_array = np.array(rule_strings, dtype="<U1000")
        self.set_parameters([rules_array])  # 💡 Stocke proprement les règles parsées

        # 🧼 Et on les récupère via get_parameters
        return self.get_parameters(config), len(self.current_rules), {}


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
        
        #score = calc_score(conf_matrix)
        accuracy = (conf_matrix[0] + conf_matrix[2]) / sum(conf_matrix)

        log.info(f"✅ Evaluation results: {conf_matrix}, Accuracy: {accuracy}")
        return 1 - accuracy, len(conf_matrix), {"accuracy": float(accuracy)}




# 🔹 Start the client
fl.client.start_client(
    server_address="localhost:8080",
    client=FlowerClient(settings,stats).to_client(),  # ✅ Fixed Flower API usage
)
if __name__ == "__main__":
    settings = parse_settings()
    stats = Stats(log_best_programs=settings.info)
    client = FlowerClient(settings, stats)
    fl.client.start_client(
        server_address="localhost:8080",
        client=client.to_client()
    )