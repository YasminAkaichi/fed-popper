#!/usr/bin/env python3

import logging
import sys
from . util import Settings, Stats, timeout, parse_settings, format_program
from . asp import ClingoGrounder, ClingoSolver
from . tester import Tester
from . constrain import Constrain
from . generate import generate_program
from . core import Grounding, Clause
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
class Outcome:
    ALL = 'all'
    SOME = 'some'
    NONE = 'none'

class Con:
    GENERALISATION = 'generalisation'
    SPECIALISATION = 'specialisation'
    REDUNDANCY = 'redundancy'
    BANISH = 'banish'

OUTCOME_TO_CONSTRAINTS = {
        (Outcome.ALL, Outcome.NONE)  : (Con.BANISH,),
        (Outcome.ALL, Outcome.SOME)  : (Con.GENERALISATION,),
        (Outcome.SOME, Outcome.NONE) : (Con.SPECIALISATION,),
        (Outcome.SOME, Outcome.SOME) : (Con.SPECIALISATION, Con.GENERALISATION),
        (Outcome.NONE, Outcome.NONE) : (Con.SPECIALISATION, Con.REDUNDANCY),
        (Outcome.NONE, Outcome.SOME) : (Con.SPECIALISATION, Con.REDUNDANCY, Con.GENERALISATION)
    }

def ground_rules(stats, grounder, max_clauses, max_vars, clauses):
    out = set()
    for clause in clauses:
        head, body = clause
        # find bindings for variables in the constraint
        assignments = grounder.find_bindings(clause, max_clauses, max_vars)

        # keep only standard literals
        body = tuple(literal for literal in body if not literal.meta)

        # ground the clause for each variable assignment
        for assignment in assignments:
            out.add(Grounding.ground_clause((head, body), assignment))
    
    stats.register_ground_rules(out)

    return out

def decide_outcome(conf_matrix):
    tp, fn, tn, fp = conf_matrix
    if fn == 0:
        positive_outcome = Outcome.ALL # complete
    elif tp == 0 and fn > 0:
        positive_outcome = Outcome.NONE # totally incomplete
    else:
        positive_outcome = Outcome.SOME # incomplete

    if fp == 0:
        negative_outcome = Outcome.NONE  # consistent
    # elif FP == self.num_neg:     # AC: this line may not work with minimal testing
        # negative_outcome = Outcome.ALL # totally inconsistent
    else:
        negative_outcome = Outcome.SOME # inconsistent

    return (positive_outcome, negative_outcome)

def build_rules(settings, stats, constrainer, tester, program, before, min_clause, outcome):

    (positive_outcome, negative_outcome) = outcome
    # RM: If you don't use these two lines you need another three entries in the OUTCOME_TO_CONSTRAINTS table (one for every positive outcome combined with negative outcome ALL).
    if negative_outcome == Outcome.ALL:
         negative_outcome = Outcome.SOME
    for rule in program:
        log.debug(f"🛠 Checking rule in `build_rules()`: {rule}")
        head, body = rule
        log.debug(f" Parsed heard {head}")
        log.debug(f" Parsed body {body}")
    rules = set()
    for constraint_type in OUTCOME_TO_CONSTRAINTS[(positive_outcome, negative_outcome)]:
        if constraint_type == Con.GENERALISATION:
            #new_rules = constrainer.generalisation_constraint(program, before, min_clause)
            new_rules = list(constrainer.generalisation_constraint([rule], before, min_clause))
            log.debug(f"✅ New generalized rules after inconsistency: {new_rules}")

            #log.debug(f"🔹 GENERALISATION generated: {new_rules}")
            rules.update(new_rules)
            
        elif constraint_type == Con.SPECIALISATION:
            new_rules = constrainer.specialisation_constraint(program, before, min_clause)
            log.debug(f"🔹 SPECIALISATION generated: {new_rules}")
            rules.update(new_rules)
            
        elif constraint_type == Con.REDUNDANCY:
            new_rules = constrainer.redundancy_constraint(program, before, min_clause)
            log.debug(f"🔹 REDUNDANCY generated: {new_rules}")
            rules.update(new_rules)
            
        elif constraint_type == Con.BANISH:
            new_rules = constrainer.banish_constraint(program, before, min_clause)
            log.debug(f"🔹 BANISH generated: {new_rules}")
            rules.update(new_rules)

        # 🔎 Functional Test Constraint
    if settings.functional_test and tester.is_non_functional(program):
        new_rules = constrainer.generalisation_constraint(program, before, min_clause)
        log.debug(f"🔹 FUNCTIONAL TEST generated: {new_rules}")
        rules.update(new_rules)

    # 🔎 Redundant Literal Constraint
    for rule in tester.check_redundant_literal(program):
        new_rules = constrainer.redundant_literal_constraint(rule, before, min_clause)
        log.debug(f"🔹 REDUNDANT LITERAL generated: {new_rules}")
        rules.update(new_rules)

    # 🔎 Redundant Clause Constraint
    if tester.check_redundant_clause(program):
        new_rules = constrainer.generalisation_constraint(program, before, min_clause)
        log.debug(f"🔹 REDUNDANT CLAUSE generated: {new_rules}")
        rules.update(new_rules)

    for rule in program:
        if Clause.is_separable(rule):
            log.debug(f"🔎 Checking if rule is inconsistent: {rule}")
            if tester.is_inconsistent(rule):
                log.debug(f"🚨 Rule is inconsistent! {rule}")
                for x in constrainer.generalisation_constraint([rule], before, min_clause):
                    rules.add(x)
        
            log.debug(f"🔎 Checking if rule is totally incomplete: {rule}")
            if tester.is_totally_incomplete(rule):
                log.debug(f"🚨 Rule is totally incomplete! {rule}")
                for x in constrainer.redundancy_constraint([rule], before, min_clause):
                    rules.add(x)
    if len(program) > 1:
        # evaluate inconsistent sub-clauses
        for rule in program:
            if Clause.is_separable(rule) and tester.is_inconsistent(rule):
                for x in constrainer.generalisation_constraint([rule], before, min_clause):
                    rules.add(x)

        # eliminate totally incomplete rules
        if all(Clause.is_separable(rule) for rule in program):
            for rule in program:
                if tester.is_totally_incomplete(rule):
                    for x in constrainer.redundancy_constraint([rule], before, min_clause):
                        rules.add(x)

    stats.register_rules(rules)

    return rules

PROG_KEY = 'prog'

def calc_score(conf_matrix):
    tp, fn, tn, fp = conf_matrix
    return tp + tn

def popper(settings, stats):
    solver = ClingoSolver(settings)
    tester = Tester(settings)
    settings.num_pos, settings.num_neg = len(tester.pos), len(tester.neg)
    grounder = ClingoGrounder()
    constrainer = Constrain()
    best_score = None

    for size in range(1, settings.max_literals + 1):
        
        stats.update_num_literals(size)
        solver.update_number_of_literals(size)
        log.debug(f"size is:{size}")
        while True:
            # GENERATE HYPOTHESIS
            with stats.duration('generate'):
                model = solver.get_model()
                if not model:
                    break
                (program, before, min_clause) = generate_program(model)
                log.debug(f"before {before}")
                log.debug(f"min_clause{min_clause}")
            # TEST HYPOTHESIS
            with stats.duration('test'):
                conf_matrix = tester.test(program)
                outcome = decide_outcome(conf_matrix)
                score = calc_score(conf_matrix)
            log.debug(f"🔹 After test: {stats}")
            stats.register_program(program, conf_matrix)

            # UPDATE BEST PROGRAM
            if best_score == None or score > best_score:
                best_score = score

                if outcome == (Outcome.ALL, Outcome.NONE):
                    stats.register_solution(program, conf_matrix)
                    return stats.solution.code

                stats.register_best_program(program, conf_matrix)

            # BUILD RULES
            with stats.duration('build'):
                rules = build_rules(settings, stats, constrainer, tester, program, before, min_clause, outcome)

            # GROUND RULES
            with stats.duration('ground'):
                rules = ground_rules(stats, grounder, solver.max_clauses, solver.max_vars, rules)

            # UPDATE SOLVER
            with stats.duration('add'):
                solver.add_ground_clauses(rules)

    stats.register_completion()
    return stats.best_program.code if stats.best_program else None

def show_hspace(settings):
    f = lambda i, m: print(f'% program {i}\n{format_program(generate_program(m)[0])}')
    ClingoSolver.get_hspace(settings, f)

def learn_solution(settings):
    stats = Stats(log_best_programs=settings.info)
    log_level = logging.DEBUG if settings.debug else logging.INFO
    logging.basicConfig(level=log_level, stream=sys.stderr, format='%(message)s')
    timeout(popper, (settings, stats), timeout_duration=int(settings.timeout))

    if stats.solution:
        prog_stats = stats.solution
    elif stats.best_programs:
        prog_stats = stats.best_programs[-1]
    else:
        return None, stats

    return prog_stats.code, stats
