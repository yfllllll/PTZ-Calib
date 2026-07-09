"""Test pyceres API to understand how to use it."""

import numpy as np
import pyceres


class SimpleCostFunction(pyceres.CostFunction):
    """Simple quadratic cost function: f(x) = (x - 2)^2."""
    
    def __init__(self):
        super().__init__()
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1])
    
    def Evaluate(self, parameters, residuals, jacobians):
        x = parameters[0][0]
        residuals[0] = x - 2.0
        
        if jacobians is not None and jacobians[0] is not None:
            jacobians[0][0] = 1.0
        
        return True


def test_simple_optimization():
    """Test simple optimization with pyceres."""
    # Create problem
    problem = pyceres.Problem()
    
    # Initial guess
    x = np.array([0.0])
    
    # Add residual block
    cost = SimpleCostFunction()
    problem.add_residual_block(cost, None, [x])
    
    # Configure solver
    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
    options.minimizer_progress_to_stdout = True
    
    # Solve
    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)
    
    print("Summary:")
    print(summary.BriefReport())
    print(f"\nOptimal x: {x[0]:.6f} (expected: 2.0)")
    print(f"Final cost: {summary.final_cost:.6e}")


if __name__ == "__main__":
    test_simple_optimization()
