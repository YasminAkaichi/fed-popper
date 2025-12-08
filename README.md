# Reproducibility and Execution Instructions

To run FedPopper, follow the steps below:

# 1. Clone this repository

git clone https://github.com/<your-user>/fed-popper.git
cd fed-popper


# 2. Install the modified Flower version
FedPopper relies on a custom extension of Flower adapted for symbolic learning.
Install it directly from our GitHub fork:

pip install git+https://github.com/<your-user>/flower-fedpopper.git


(No additional cloning of Flower is required.)

# 3. Navigate to the FedPopper directory

cd fedpopper


# 4. Run the system using four terminals

Terminal 1 — Server

python3 server.py


Terminals 2–4 — Clients

python3 client1.py
python3 client2.py
python3 client3.py


Each client script specifies the path to its local dataset.
If you wish to test different datasets, you only need to update the DATA_PATH variable inside the corresponding client file.

The default configuration uses the Trains dataset partitioned into three balanced local subsets.

# Running FedPopper with Two Clients (optional)

For smaller experiments or quick validation (on two clients):

cd fedpopper-2clients


Launch three terminals:

Server:

python3 server.py


Two clients:

python3 client1.py
python3 client2.py


This setup is pre-configured with the Trains dataset partitioned into two balanced parts.

📂 Dataset Structure

All datasets are located inside the repository under:

fedpopper/trains 
/trains_part1
/trains_part2
/trains_part3 


Each dataset folder contains:

exs.pl — positive & negative examples

bk.pl — background knowledge

bias.pl — predicate declarations and structural limits

You may change any dataset into any client configuration by adjusting the kbpath in the client code.

# Notes

Always start the server first, then launch clients in parallel.

The system displays progress round-by-round until a globally correct hypothesis is found


# ⚠️ Reproducibility Notice

FedPopper inherits Popper’s operational non-determinism:
the underlying ASP solver may explore different valid search branches across executions.
For this reason, you may occasionally need to run FedPopper 2–3 times before obtaining the same final hypothesis reported in our experiments.

This does not indicate a configuration error — it simply reflects
the solver’s internal heuristics and the dependency of the search trajectory on the order of failures encountered during learning.