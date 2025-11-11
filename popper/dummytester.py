# dummy_tester.py

class DummyTester:
    """
    Dummy tester used on the server side in Federated Popper.
    It replaces Popper's Tester, because the server cannot access
    examples or run any evaluation.
    
    Only structural checks are preserved, and anything involving
    pos/neg coverage is disabled.
    """

    def __init__(self):
        pass

    # -------------------------------
    # ✅ Structural checks (safe)
    # -------------------------------

    def check_redundant_literal(self, program):
        # Server cannot compute redundancy wrt examples → return empty
        return []

    def check_redundant_clause(self, program):
        # No test possible → no redundant clause detection
        return False

    # -------------------------------
    # ✅ Functions requiring examples (disabled)
    # -------------------------------

    def is_non_functional(self, program):
        # Cannot know without examples
        return False

    def is_totally_incomplete(self, rule):
        # Requires coverage of positives → forbidden on server
        return False

    def is_inconsistent(self, rule):
        # Requires negative coverage → forbidden on server
        return False

    # -------------------------------
    # ✅ Dummy container to mimic Tester API
    # -------------------------------

    pos = []
    neg = []
