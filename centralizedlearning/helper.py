import pickle
import numpy as np
from flwr.common import ndarrays_to_parameters

def get_parameters(self, config) -> Parameters:
    """Extracts specialization (ε⁺) and generalization (ε⁻) constraints and sends them."""

    # Test current rules against local examples
    pos_covered, neg_covered, inconsistent, _, _ = self.tester.test_prog_noisy(
        self.tester.settings.current_program, 0)

    # Convert bitarrays to sets (indices of examples)
    epsilon_plus = set(np.where(pos_covered)[0])   # Indices of uncovered positive examples
    epsilon_minus = set(np.where(neg_covered)[0])  # Indices of covered negative examples

    # ✅ Serialize (ε⁺, ε⁻) for transmission
    serialized_constraints = pickle.dumps((epsilon_plus, epsilon_minus))

    # ✅ Convert to NumPy array and wrap in a list for Flower
    constraints_as_ndarray = np.frombuffer(serialized_constraints, dtype=np.uint8)

    return ndarrays_to_parameters([constraints_as_ndarray])  # ✅ Ensure it's a list