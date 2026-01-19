# Reproducibility and Execution Instructions

This artefact provides a self-contained implementation of FedPopper together
with the exact customized version of the Flower framework used in the experiments.

All instructions below have been tested on Python ≥ 3.9.


##  1. System requirements

FedPopper relies on Popper and Answer Set Programming solvers.
The following system dependencies must be installed before running the artefact:

- SWI-Prolog (version ≥ 9.2.0)
- Clingo (version ≥ 5.6.2)

These tools are required by Popper and are not Python packages.


## 2. Environment setup

Create and activate a virtual environment:
```bash
python3 -m venv fedpopper-env
source fedpopper-env/bin/activate
pip install --upgrade pip
```
## 3. Install dependencies
```bash
pip install -r requirements.txt
```
## 4. Install Popper v-1.1.0

```bash
pip install -e . 
```

## 5. Install the customized Flower framework

The customized Flower framework required to run FedPopper is included
in this repository under `external/flower/` and must be installed locally.

Install it in editable mode:
```bash
cd external/flower
pip install -e .
cd ../../
```

## 6. Run FedPopper (3 clients)
```bash
cd fedpopper
```

Open four terminals.

Terminal 1 (Server)
```bash
python server.py
```
Terminals 2–4 (Clients)
```bash
python client1.py
python client2.py
python client3.py
```

Each client file contains a DATA_PATH (or equivalent) variable pointing to its local dataset partition.
To test another dataset/partition, modify that variable in the corresponding client file.

Default setup: Trains dataset partitioned into three balanced subsets.

## 7. Run FedPopper (2 clients) — optional
cd fedpopper-2clients


Open three terminals.

Server:
```bash
python server.py
```

Clients:
```bash
python client1.py
python client2.py
```
## 8. Dataset structure

Datasets are located under:

fedpopper/trains/
  trains_part1/
  trains_part2/
  trains_part3/


Each folder contains:

exs.pl — positive & negative examples

bk.pl — background knowledge

bias.pl — declarations and structural limits

## Dataset selection and configuration

Each client file (client1.py, client2.py, client3.py) contains a dataset path variable (e.g., DATA_PATH or kbpath) pointing to its local dataset partition.

Similarly, the server file (server.py) contains a dataset path pointing to the corresponding bias file.

To evaluate a different dataset or partitioning, users should:

locate the predefined dataset paths already present in the code,

comment the currently active path, and

uncomment the desired dataset path.

No additional configuration is required.

Default setup:
The Trains dataset, partitioned into three balanced subsets

## 9. Execution notes

The server must be started before launching the clients.

Clients should be started in parallel.

The system reports progress round by round until a globally valid hypothesis is found.

## 10. Reproducibility notice

FedPopper inherits Popper’s operational non-determinism: the underlying ASP solver may explore different valid search branches across runs.
It may occasionally be necessary to run the system multiple times to obtain the same final hypothesis. This reflects solver heuristics rather than configuration issues.
This behaviour is expected and reflects solver heuristics and distributed search dynamics rather than configuration issues or implementation errors.

