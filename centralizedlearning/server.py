import flwr as fl;
import sys; 
import numpy as np;
from flwr.server.strategy.fedconstraints import FedConstraint
from flwr.common.logger import log
from logging import DEBUG

# Create strategy and run server
# Create strategy with configuration
strategy = FedILP(
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
