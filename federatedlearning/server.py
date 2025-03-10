import flwr as fl;
import sys; 
import numpy as np;
from flwr.server.strategy.fedpopper import FedPopper
from flwr.common.logger import log
from logging import DEBUG
from popper.util import Settings
from popper.tester import Tester
from popper.util import load_kbpath
# Create strategy and run server
# Create strategy with configuration

# ✅ Load ILP settings
kbpath = "trains"
bk_file, ex_file, bias_file = load_kbpath(kbpath)
settings = Settings(bias_file, ex_file, bk_file)
tester = Tester(settings)  # ✅ Create the tester instance

strategy = FedPopper(
    settings = settings,
    tester = tester,
    fraction_fit=0.5,  # Use 50% of clients in each round
    min_fit_clients=2,
    min_available_clients=2,
    fit_metrics_aggregation_fn=None,  # Custom metrics aggregation if needed
)

log(DEBUG, "Starting Flower server with FedILP strategy.")
# Start Flower server for three rounds of federated learning
fl.server.start_server(
        server_address = "0.0.0.0:8080" , 
        config=fl.server.ServerConfig(num_rounds=2),
        strategy = strategy,
)
log(DEBUG, "Flower server has stopped.")
