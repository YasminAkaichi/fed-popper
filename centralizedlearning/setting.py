import logging
from popper.util import Settings, Stats
from popper.tester import Tester
from popper.constrain import Constrain
from popper.generate import generate_program
from popper.core import Clause, Literal, ConstVar
from popper.asp import ClingoSolver
from popper.util import load_kbpath
from clingo import Function, Number, String, Tuple_ # ✅ Correct Clingo types
import clingo
# 🔹 Load knowledge base paths

kbpath = "trains"
bias_file = load_kbpath(kbpath)
settings = Settings(bias_file, "","")
