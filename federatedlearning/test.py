
import logging
from popper.util import Settings, Stats
from popper.tester import Tester
from popper.core import Clause, Literal
from popper.util import load_kbpath, format_program
from popper.loop import decide_outcome, Outcome, calc_score
import flwr as fl
import numpy as np

kbpath = "trains"
bk_file, ex_file, bias_file = load_kbpath(kbpath)
settings = Settings(bias_file, ex_file, bk_file)
tester = Tester(settings)

rule_str = "f(A):-has_car(A,B),has_car(A,C),three_wheels(C),roof_closed(C),long(B)"
rule = Clause.from_string(rule_str)
conf_matrix = tester.test([rule])
print("Conf matrix:", conf_matrix)
