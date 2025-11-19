# structural_tester.py

"""
StructuralTester: a Popper-compatible tester for Federated ILP (FILP).

This tester performs ONLY structural checks needed by Popper's constraint
generation components. It does NOT evaluate rules against examples,
does NOT load Prolog files, and must NEVER access BK or data.

This tester is safe to use on the server in a federated setting.
"""

from popper.core import Clause, Literal

class StructuralTester:
    def __init__(self):
        # The real Popper Tester keeps caches for efficiency.
        # We keep these empty caches only to preserve API compatibility.
        self.already_checked_redundant_literals = set()
        self.seen_tests = {}
        self.seen_prog = {}

    # ----------------------------------------------------------
    # ❗PURELY STRUCTURAL CHECKS (SAFE FOR SERVER)
    # ----------------------------------------------------------

    def check_redundant_literal(self, program):
        """
        In Popper, this uses Prolog to check if a literal is redundant.
        On the server we cannot evaluate semantics, so we ONLY return an empty list.
        """
        return []

    def check_redundant_clause(self, program):
        """
        Normally detects redundant clauses using Prolog queries.
        Not allowed in FILP, so always return empty.
        """
        return []

    def is_non_functional(self, program):
        """
        In classic Popper, checks whether the program violates functional constraints.
        Requires evaluating rules on data -> forbidden here.
        """
        return False

    def is_inconsistent(self, rule):
        """
        Normally checks if rule derives negative examples -> forbidden.
        """
        return False

    def is_totally_incomplete(self, rule):
        """
        Normally checks if rule derives no positives -> forbidden.
        """
        return False

    # ----------------------------------------------------------
    # ❗THE SERVER MUST NEVER TEST RULES ON EXAMPLES
    # ----------------------------------------------------------

    def test(self, rules):
        """
        In FILP, testing is performed ONLY on clients.
        Calling server.test(...) would violate privacy assumptions.
        """
        raise RuntimeError(
            "StructuralTester.test() called on server. "
            "In FILP, only clients may test rules on examples."
        )

    # ----------------------------------------------------------
    # Helper: API compatibility with original Tester
    # ----------------------------------------------------------
    @property
    def pos(self):
        """The server cannot know pos examples."""
        return []

    @property
    def neg(self):
        """The server cannot know neg examples."""
        return []
