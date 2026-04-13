import flwr as fl
import sys
import numpy as np
from flwr.server.strategy.fedpopper import FedPopper
from flwr.common.logger import log
from flwr.common import parameters_to_ndarrays
from logging import DEBUG, INFO
from popper.util import Settings, Stats
from popper.core import Clause
from popper.util import load_kbpath
from popper.structural_tester import StructuralTester
from popper.constrain import Constrain
from popper.asp import ClingoGrounder, ClingoSolver

# Load ILP settings
kbpath = "/Users/yasmineakaichi/fed-popper/fedpopper/trains"

_, _, bias_file = load_kbpath(kbpath)
settings = Settings(bias_file, None, None)
mytester = StructuralTester()

mystats = Stats(log_best_programs=settings.info)
mysolver = ClingoSolver(settings)
mygrounder = ClingoGrounder()
myconstrainer = Constrain()

strategy = FedPopper(
    settings=settings,
    stats=mystats,
    solver=mysolver,
    grounder=mygrounder,
    tester=mytester,
    constrainer=myconstrainer,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_available_clients=3,
    min_evaluate_clients=3,
    fit_metrics_aggregation_fn=None,
)

log(DEBUG, "Starting Flower server with FedILP strategy.")

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=15000),
    strategy=strategy,
)

log(DEBUG, "Flower server has stopped.")

# ========== AFFICHER LA SOLUTION ==========
print("\n========== FINAL SOLUTION ==========")
if strategy.solution_params:
    arrs = parameters_to_ndarrays(strategy.solution_params)
    if arrs and arrs[0].size > 0:
        print("Solution found:")
        for rule in arrs[0].tolist():
            print(f"  {rule}")
    else:
        print("Solution params empty.")
elif strategy.best_hypothesis:
    print("No perfect solution — best hypothesis:")
    for r in strategy.best_hypothesis:
        print(f"  {Clause.to_code(r)}")
else:
    print("No solution found.")