from typing import Callable, Dict, List, Optional, Tuple, Union
from logging import WARNING, INFO, DEBUG
from flwr.common.logger import log
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg, aggregate_outcomes
from .strategy import Strategy

from popper.loop import Outcome, build_rules, ground_rules, decide_outcome, calc_score
from popper.constrain import Constrain
from popper.core import Clause, Literal, ConstVar
from popper.asp import ClingoGrounder, ClingoSolver
from popper.generate import generate_program
from popper.structural_tester import StructuralTester
from popper.util import Settings, Stats

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTCOME_DECODING = {1: "all", 2: "some", 3: "none"}
OUTCOME_ENCODING = {"all": 1, "some": 2, "none": 3}


class FedPopper(Strategy):

    def __init__(
        self,
        settings,
        stats=None,
        solver=None,
        grounder=None,
        constrainer=None,
        tester=None,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        super().__init__()

        # --- Popper core ---
        self.settings    = settings
        self.solver      = solver      if solver      is not None else ClingoSolver(settings)
        self.grounder    = grounder    if grounder    is not None else ClingoGrounder()
        self.constrainer = constrainer if constrainer is not None else Constrain()
        self.tester      = tester      if tester      is not None else StructuralTester()
        self.stats       = stats       if stats       is not None else Stats(log_best_programs=settings.info)

        # --- Popper state (mirrors srvpopper FILPServerState) ---
        self.current_hypothesis  = None
        self.current_before      = None
        self.current_min_clause  = 0
        self.current_clause_size = 1

        # --- Best tracking ---
        self.best_score      = None
        self.best_hypothesis = None

        # --- Flower ---
        self.fraction_fit              = fraction_fit
        self.fraction_evaluate         = fraction_evaluate
        self.min_fit_clients           = min_fit_clients
        self.min_evaluate_clients      = min_evaluate_clients
        self.min_available_clients     = min_available_clients
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn

        self.solution_params: Optional[Parameters] = None
        self.early_stop = False

    def __repr__(self) -> str:
        return "FedPopper"

    # ------------------------------------------------------------------
    # initialize_parameters
    # Called ONCE at startup — generates H0 from bias.pl
    # Mirrors: first get_model() in srvpopper before the loop
    # ------------------------------------------------------------------
    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        log(INFO, "Generating H0 from bias.pl...")

        # Taille 1 déjà initialisée dans server.py
        # On essaie toutes les tailles jusqu'à trouver un modèle
        for size in range(1, self.settings.max_literals + 1):
            self.solver.update_number_of_literals(size)
            self.stats.update_num_literals(size)
            self.current_clause_size = size

            with self.stats.duration('generate'):
                model = self.solver.get_model()

            if model:
                break
        else:
            log(WARNING, "No H0 found at any size — check bias.pl")
            return ndarrays_to_parameters([np.array([], dtype="<U1000")])

        program, before, min_clause = generate_program(model)

        self.current_hypothesis = program
        self.current_before     = before
        self.current_min_clause = min_clause

        log(INFO, f"H0 ({len(program)} clause(s)):")
        for r in program:
            log(INFO, f"  {Clause.to_code(r)}")

        rules_arr = np.array(
            [Clause.to_code(r) for r in program],
            dtype="<U1000"
        )
        return ndarrays_to_parameters([rules_arr])
    # ------------------------------------------------------------------
    # configure_fit
    # Just sends whatever parameters aggregate_fit produced last round.
    # parameters = H0 on round 1 (from initialize_parameters),
    #            = HN on round N+1 (returned by aggregate_fit round N)
    # ------------------------------------------------------------------
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:

        if self.early_stop:
            return []

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )

        return [
            (c, FitIns(parameters, {"round": server_round}))
            for c in clients
        ]

    # ------------------------------------------------------------------
    # aggregate_fit
    # Exactly run_server() from srvpopper.
    # federated_test() is replaced by collecting Flower client results.
    # The for+while loop is INSIDE, identical to srvpopper.
    # ------------------------------------------------------------------
    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        if not results:
            log(WARNING, f"[Round {server_round}] No results received.")
            return None, {}

        # ----------------------------------------------------------------
        # federated_test() → collect client feedback
        # Mirrors: outcome, score, rules_str = federated_test(...)
        # ----------------------------------------------------------------
        epsilons = []
        scores   = []

        for _, res in results:
            arrs = parameters_to_ndarrays(res.parameters)
            if not arrs or len(arrs[0]) < 2:
                log(WARNING, f"[Round {server_round}] Invalid payload, skipping.")
                continue
            vals  = arrs[0].tolist()
            e_pos = int(vals[0])
            e_neg = int(vals[1])
            score = float(vals[2]) if len(vals) >= 3 else 0.0
            epsilons.append((OUTCOME_DECODING[e_pos], OUTCOME_DECODING[e_neg]))
            scores.append(score)

        if not epsilons:
            log(WARNING, f"[Round {server_round}] No valid payloads.")
            return None, {}

        # Mirrors: outcome = aggregate_outcomes(eps_pairs)
        #          fed_score = sum(scores)
        outcome   = aggregate_outcomes(epsilons)
        fed_score = sum(scores)

        self.stats.total_programs += 1
        log(INFO, f"[Program #{server_round}] outcome={outcome}, score={fed_score}")

        # ---- UPDATE BEST
        # Mirrors: if best_score is None or score > best_score
        if self.best_score is None or fed_score > self.best_score:
            self.best_score      = fed_score
            self.best_hypothesis = self.current_hypothesis

        # ---- STOP CONDITION
        # Mirrors: if outcome == ("all", "none"): found_solution = True; break
        if outcome == ("all", "none"):
            log(INFO, f"[Round {server_round}] Solution found (ALL, NONE)!")
            self.early_stop = True
            rules_arr = np.array(
                [Clause.to_code(r) for r in self.current_hypothesis],
                dtype="<U1000"
            )
            self.solution_params = ndarrays_to_parameters([rules_arr])
            return self.solution_params, {"fed_score": fed_score}

        # ----------------------------------------------------------------
        # THE for+while LOOP — exactly srvpopper
        # BUILD / GROUND / ADD + GENERATE are INSIDE
        # ----------------------------------------------------------------
        for size in range(self.current_clause_size, self.settings.max_literals + 1):

            self.current_clause_size = size
            self.stats.update_num_literals(size)
            self.solver.update_number_of_literals(size)

            while True:
                if self.current_hypothesis is None:
                    log(WARNING, "No hypothesis — skipping build_rules.")
                    break
                # ---- BUILD / GROUND / ADD
                # Mirrors: build_rules → ground_rules → add_ground_clauses
                with self.stats.duration('build'):
                    rules = build_rules(
                        self.settings, self.stats, self.constrainer,
                        self.tester, self.current_hypothesis,
                        self.current_before, self.current_min_clause,
                        outcome
                    )
                with self.stats.duration('ground'):
                    rules = ground_rules(
                        self.stats, self.grounder,
                        self.solver.max_clauses, self.solver.max_vars,
                        rules
                    )
                with self.stats.duration('add'):
                    self.solver.add_ground_clauses(rules)

                # ---- GENERATE
                # Mirrors: model = solver.get_model()
                #          if not model: break  → next size
                with self.stats.duration('generate'):
                    model = self.solver.get_model()

                if not model:
                    break  # inner while → try next size

                program, before, min_clause = generate_program(model)

                # ---- Store next hypothesis + return to clients
                # Mirrors: round_id += 1 → loop back → federated_test
                self.current_hypothesis = program
                self.current_before     = before
                self.current_min_clause = min_clause

                log(INFO, f"[Round {server_round}] Next hypothesis ({len(program)} clause(s)):")
                for r in program:
                    log(INFO, f"  {Clause.to_code(r)}")

                rules_arr = np.array(
                    [Clause.to_code(r) for r in program],
                    dtype="<U1000"
                )
                # Return to Flower → configure_fit sends it to clients next round
                return ndarrays_to_parameters([rules_arr]), {"fed_score": fed_score}

        # ---- for exhausted → search complete (all sizes tried)
        # Mirrors: for size loop ends without solution
        log(INFO, "Search exhausted — no solution found.")
        self.early_stop = True
        rules_arr = (
            np.array(
                [Clause.to_code(r) for r in self.best_hypothesis],
                dtype="<U1000"
            )
            if self.best_hypothesis
            else np.array([], dtype="<U1000")
        )
        self.solution_params = ndarrays_to_parameters([rules_arr])
        return self.solution_params, {"fed_score": fed_score}

    # ------------------------------------------------------------------
    # configure_evaluate
    # ------------------------------------------------------------------
    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:

        if self.early_stop:
            log(DEBUG, f"Early stop at round {server_round}: skipping evaluation.")
            return []

        if self.fraction_evaluate == 0.0:
            return []

        params_for_eval = (
            self.solution_params
            if self.solution_params is not None
            else parameters
        )

        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )

        return [(c, EvaluateIns(params_for_eval, {})) for c in clients]

    # ------------------------------------------------------------------
    # aggregate_evaluate
    # ------------------------------------------------------------------
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        if not results:
            log(WARNING, "No evaluation results received.")
            return None, {}

        num_examples = sum(r.num_examples for _, r in results)
        if num_examples == 0:
            log(WARNING, "No valid examples for evaluation.")
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [(r.num_examples, r.loss) for _, r in results]
        )

        return loss_aggregated, {}

    # ------------------------------------------------------------------
    # evaluate (server-side — unused)
    # ------------------------------------------------------------------
    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients