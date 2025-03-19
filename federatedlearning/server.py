import flwr as fl;
import sys; 
import numpy as np;
from flwr.server.strategy.fedpopper import FedPopper
from flwr.common.logger import log
from logging import DEBUG
from popper.util import Settings, Stats
from popper.tester import Tester
from popper.core import Clause, Literal
from popper.util import load_kbpath

from popper.asp import ClingoGrounder, ClingoSolver
# Create strategy and run server
# Create strategy with configuration

# ✅ Load ILP settings
#kbpath = "trains"
#_, _, bias_file = load_kbpath(kbpath)
#settings = Settings(bias_file,"" , "")

kbpath = "trains"
bk_file, ex_file, bias_file = load_kbpath(kbpath)

# 🔹 Initialize ILP settings
settings = Settings(bias_file, ex_file, bk_file)
stats = Stats(log_best_programs=settings.info)
tester = Tester(settings)  # ✅ Create the tester instance
solver= ClingoSolver(settings)
settings.num_pos, settings.num_neg = len(tester.pos), len(tester.neg)

strategy = FedPopper(
    settings = settings,
    stats = stats,
    solver = solver,
    fraction_fit=0.5,  # Use 50% of clients in each round
    min_fit_clients=2,
    min_available_clients=2,
    fit_metrics_aggregation_fn=None,
)

log(DEBUG, "Starting Flower server with FedILP strategy.")
# Start Flower server for three rounds of federated learning
fl.server.start_server(
        server_address = "0.0.0.0:8080" , 
        config=fl.server.ServerConfig(num_rounds=45),
        strategy = strategy,
)
log(DEBUG, "Flower server has stopped.")
