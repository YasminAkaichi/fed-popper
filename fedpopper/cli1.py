import logging
import os
import re
import flwr as fl
import numpy as np

from popper.util import Settings, Stats, load_kbpath
from popper.tester import Tester
from popper.core import Literal
from popper.loop import decide_outcome, calc_score

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

OUTCOME_ENCODING = {"all": 1, "some": 2, "none": 3}
CLIENT_ID = 1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "trains_part1")   # à changer par client

bk_file, ex_file, bias_file = load_kbpath(DATASET_PATH)
settings = Settings(bias_file, ex_file, bk_file)
tester = Tester(settings)
stats = Stats(log_best_programs=settings.info)
settings.num_pos, settings.num_neg = len(tester.pos), len(tester.neg)


def transform_rule_to_tester_format(rule_str):
    head_body = rule_str.split(":-")
    if len(head_body) != 2:
        raise ValueError(f"Invalid rule format: {rule_str}")

    head_str = head_body[0].strip()
    body_str = head_body[1].strip()

    body_literals = re.findall(r'\w+\(.*?\)', body_str)

    head = Literal.from_string(head_str)
    body = tuple(Literal.from_string(lit) for lit in body_literals)

    return (head, body)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, tester, stats):
        self.tester = tester
        self.stats = stats
        self.current_rules = []

    def get_parameters(self, config):
        return [np.array([], dtype=np.int64)]

    def set_parameters(self, parameters):
        if not parameters or len(parameters) == 0 or parameters[0].size == 0:
            self.current_rules = []
            return

        arr = parameters[0]
        if arr.dtype.kind not in ["U", "S", "O"]:
            self.current_rules = []
            return

        received_rules = arr.tolist()
        parsed = [transform_rule_to_tester_format(r) for r in received_rules]
        self.current_rules = [p for p in parsed if p is not None]

    def fit(self, parameters, config):
        round_id = config.get("round", -1)
        print(f"\nCLIENT {CLIENT_ID} — ROUND {round_id}")

        self.set_parameters(parameters)

        if not self.current_rules:
            payload = np.array(
                [OUTCOME_ENCODING["none"], OUTCOME_ENCODING["none"], 0],
                dtype=np.int64,
            )
            return [payload], 1, {}

        cm = self.tester.test(self.current_rules)
        tp, fn, tn, fp = cm

        print(f"Local Result: TP={tp} FN={fn} TN={tn} FP={fp}")

        eps_plus, eps_minus = decide_outcome(cm)
        score = calc_score(cm)

        print(f"Feedback: e+={eps_plus}, e-={eps_minus}, score={score}")

        payload = np.array(
            [
                OUTCOME_ENCODING[str(eps_plus).lower()],
                OUTCOME_ENCODING[str(eps_minus).lower()],
                int(score),
            ],
            dtype=np.int64,
        )
        return [payload], 1, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        if not self.current_rules:
            return 1.0, 0, {"accuracy": 0.0}

        cm = self.tester.test(self.current_rules)
        tp, fn, tn, fp = cm
        total = tp + fn + tn + fp
        accuracy = (tp + tn) / total if total > 0 else 0.0

        return float(1 - accuracy), total, {"accuracy": float(accuracy)}


fl.client.start_client(
    server_address="localhost:8080",
    client=FlowerClient(tester, stats).to_client(),
)