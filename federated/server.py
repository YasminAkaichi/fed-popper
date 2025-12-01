import flwr as fl;
import sys; 
import numpy as np;
from flwr.server.strategy.fedrules import FedRules

from flwr.common.logger import log
from logging import DEBUG
from popper.util import Settings, Stats


strategy = FedRules(
    fraction_fit=1,  # Use 50% of clients in each round
    min_fit_clients=2,
    min_available_clients=2,
    fit_metrics_aggregation_fn=None,
)

log(DEBUG, "Starting Flower server with FedILP strategy.")
# Start Flower server for three rounds of federated learning
fl.server.start_server(
        server_address = "0.0.0.0:8080" , 
        config=fl.server.ServerConfig(num_rounds=1),
        strategy = strategy,
)


log(DEBUG, "Flower server has stopped.")