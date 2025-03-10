import flwr as fl
import os
from popper.util import parse_settings
from popper.loop import learn_solution

class PopperClient(fl.client.NumPyClient):
    def __init__(self, kbpath):
        self.settings = parse_settings()
        self.settings.kbpath = kbpath  # Assigner le chemin correct dans Settings
        self.settings.bk_file, self.settings.ex_file, self.settings.bias_file = self.load_kbpath(kbpath)

    def get_parameters(self, config):
        """Renvoie les règles actuelles apprises par Popper."""
        return self.load_rules()

    def fit(self, parameters, config):
        """Exécute Popper pour générer de nouvelles règles et les retourne."""
        _prog, stats = learn_solution(self.settings)
        rules = self.load_rules()
        return rules, len(rules), {}

    def evaluate(self, parameters, config):
        """Évalue les règles générées sur des exemples tests."""
        self.save_rules(parameters)
        _prog, stats = learn_solution(self.settings)
        return stats.total_exec_time(), len(parameters), {}

    def load_rules(self):
        """Charge les règles générées par Popper."""
        rules_file = os.path.join(self.settings.kbpath, "rules.pl")
        if os.path.exists(rules_file):
            with open(rules_file, "r") as f:
                rules = f.readlines()
        else:
            rules = []
        return rules

    def save_rules(self, rules):
        """Sauvegarde les règles pour la prochaine exécution."""
        rules_file = os.path.join(self.settings.kbpath, "rules.pl")
        with open(rules_file, "w") as f:
            f.writelines(rules)

    def load_kbpath(self, kbpath):
        """Charge les fichiers de connaissance depuis le chemin donné."""
        return (
            os.path.join(kbpath, "bk.pl"),
            os.path.join(kbpath, "exs.pl"),
            os.path.join(kbpath, "bias.pl")
        )

# Lancer le client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=PopperClient(kbpath="/mnt/data"))
