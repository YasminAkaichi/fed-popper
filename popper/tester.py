from pyswip import Prolog

import os
import sys
import time
import pkg_resources
from contextlib import contextmanager
from . core import Clause, Literal
from datetime import datetime
import logging

from logging import DEBUG
# 🔹 Logging Setup
logging.basicConfig(level=logging.DEBUG)


log = logging.getLogger(__name__)

class Tester():
    def __init__(self, settings):
        self.settings = settings
        self.prolog = Prolog()
        self.eval_timeout = settings.eval_timeout
        self.already_checked_redundant_literals = set()
        self.seen_tests = {}
        self.seen_prog = {}
        self.current_program = None  # ✅ Store the current rules (hypothesis)
        self.encoded_outcome = None  # ✅ Store the last computed outcome
        bk_pl_path = self.settings.bk_file if self.settings.bk_file else None
        exs_pl_path = self.settings.ex_file if self.settings.ex_file else None
        test_pl_path = pkg_resources.resource_filename(__name__, "lp/test.pl")
        

        # List of files to consult in Prolog, skipping None or empty files
        files_to_consult = [test_pl_path]  # Always consult test.pl
        if bk_pl_path:  
            files_to_consult.append(bk_pl_path)  # Add bk.pl only if it exists
        if exs_pl_path:
            files_to_consult.append(exs_pl_path)  # Add exs.pl only if it exists

        for x in files_to_consult:
            if os.name == 'nt': # if on Windows, SWI requires escaped directory separators
                x = x.replace('\\', '\\\\')
            self.prolog.consult(x)

        if exs_pl_path:
            list(self.prolog.query('load_examples'))
            self.pos = [x['I'] for x in self.prolog.query('current_predicate(pos_index/2),pos_index(I,_)')]
            self.neg = [x['I'] for x in self.prolog.query('current_predicate(neg_index/2),neg_index(I,_)')]
        else:
            self.pos = []
            self.neg = []
        self.prolog.assertz(f'timeout({self.eval_timeout})')

    def first_result(self, q):
        return list(self.prolog.query(q))[0]

    @contextmanager
    def using(self, rules):
        current_clauses = set()
        try:
            for rule in rules:
                (head, body) = rule
                self.prolog.assertz(Clause.to_code(Clause.to_ordered(rule)))
                current_clauses.add((head.predicate, head.arity))
            yield
        finally:
            for predicate, arity in current_clauses:
                args = ','.join(['_'] * arity)
                self.prolog.retractall(f'{predicate}({args})')

    def check_redundant_literal(self, program):
        for clause in program:
            k = Clause.clause_hash(clause)
            if k in self.already_checked_redundant_literals:
                continue
            self.already_checked_redundant_literals.add(k)
            (head, body) = clause
            C = f"[{','.join(('not_'+ Literal.to_code(head),) + tuple(Literal.to_code(lit) for lit in body))}]"
            res = list(self.prolog.query(f'redundant_literal({C})'))
            if res:
                yield clause

    def check_redundant_clause(self, program):
        # AC: if the overhead of this call becomes too high, such as when learning programs with lots of clauses, we can improve it by not comparing already compared clauses
        prog = []
        for (head, body) in program:
            C = f"[{','.join(('not_'+ Literal.to_code(head),) + tuple(Literal.to_code(lit) for lit in body))}]"
            prog.append(C)
        prog = f"[{','.join(prog)}]"
        return list(self.prolog.query(f'redundant_clause({prog})'))

    def is_non_functional(self, program):
        with self.using(program):
            return list(self.prolog.query(f'non_functional.'))

    def success_set(self, rules):
        prog_hash = frozenset(rule for rule in rules)
        if prog_hash not in self.seen_prog:
            with self.using(rules):
                self.seen_prog[prog_hash] = set(next(self.prolog.query('success_set(Xs)'))['Xs'])
        return self.seen_prog[prog_hash]

    def test(self, rules):
        log.debug(f"📥 Received rules for testing: {rules}")

        self.current_program = rules
        if all(Clause.is_separable(rule) for rule in rules):
            covered = set()
            for rule in rules:
                log.debug(f"🔹 Testing rule: {Clause.to_code(rule)}")
                covered.update(self.success_set([rule]))
        else:
            covered = self.success_set(rules)

        tp, fn, tn, fp = 0, 0, 0, 0
        for p in self.pos:
            if p in covered:
                tp +=1
            else:
                fn +=1
        for n in self.neg:
            if n in covered:
                fp +=1
            else:
                tn +=1

        return tp, fn, tn, fp

    def is_totally_incomplete(self, rule):
        if not Clause.is_separable(rule):
            return False
        return not any(x in self.success_set([rule]) for x in self.pos)

    def is_inconsistent(self, rule):
        if not Clause.is_separable(rule):
            return False
        return any(x in self.success_set([rule]) for x in self.neg)
