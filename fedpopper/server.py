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
from popper.structural_tester import StructuralTester
#from popper.tester import Tester
from popper.constrain import Constrain

from popper.asp import ClingoGrounder, ClingoSolver
# Create strategy and run server
# Create strategy with configuration

# Load ILP settings
kbpath = "/Users/yasmineakaichi/fed-popper/fedpopper/trains"
# on lit seulement le BIAS.pl pas les données
_, _, bias_file = load_kbpath(kbpath)
settings = Settings(bias_file,None, None)
mytester = StructuralTester()
#kbpath = "trains"
#bk_file, ex_file, bias_file = load_kbpath(kbpath)

# Initialize ILP settings
mystats = Stats(log_best_programs=settings.info)
#mytester = Tester()  # Create the tester instance
mysolver= ClingoSolver(settings)
mygrounder = ClingoGrounder()
myconstrainer = Constrain()

#ajouter nommbre examples pos et neg

#settings.num_pos, settings.num_neg = (5,5)
#len(mytester.pos), len(mytester.neg)

number_rounds = 15000
strategy = FedPopper(
    settings = settings,
    stats = mystats,
    solver = mysolver,
    grounder = mygrounder,
    tester = mytester,
    constrainer= myconstrainer,
    fraction_fit=1,  # Use 50% of clients in each round
    fraction_evaluate= 1,
    min_fit_clients=3,
    min_available_clients=3,
    min_evaluate_clients= 3,
    fit_metrics_aggregation_fn=None,
    total_rounds = 15000,
)


log(DEBUG, "Starting Flower server with FedILP strategy.")
# Start Flower server for three rounds of federated learning
fl.server.start_server(
        server_address = "0.0.0.0:8080" , 
        config=fl.server.ServerConfig(num_rounds=15000),
        strategy = strategy,
)
log(DEBUG, "Flower server has stopped.")