import torch
import torch.nn as nn


def cox_de_boor(t, i, k, knots):
    """
    Vectorized Cox-de Boor recursion for the B-spline basis function N_{i,k}(t).
    """
    if k == 0:
        left = (knots[i] <= t)
        # Special handling at the endpoint: assign 1 when t equals the last knot.
        if i == len(knots) - 2:
            right = (t <= knots[i+1])
        else:
            right = (t < knots[i+1])
        return (left & right).float()
    
    denom1 = knots[i + k] - knots[i]
    denom2 = knots[i + k + 1] - knots[i + 1]
    
    term1 = 0.0
    if denom1 != 0.0:
        term1 = (t - knots[i]) / denom1 * cox_de_boor(t, i, k - 1, knots)
    
    term2 = 0.0
    if denom2 != 0.0:
        term2 = (knots[i + k + 1] - t) / denom2 * cox_de_boor(t, i + 1, k - 1, knots)
    
    return term1 + term2

class BSplineEvaluator(nn.Module):
    def __init__(self, num_ctrl_points, num_eval_points, degree=3):
        """
        Creates a B-spline evaluator.
        Parameters:
          num_ctrl_points: Number of control points.
          num_eval_points: Number of points at which to evaluate the spline.
          degree: Degree of the spline (3 for cubic).
        """
        super().__init__()
        self.num_ctrl_points = num_ctrl_points
        self.num_eval_points = num_eval_points
        self.degree = degree
        
        # Construct a clamped uniform knot vector over [0, 1]
        num_knots = num_ctrl_points + degree + 1
        knots = torch.cat([
            torch.zeros(degree),
            torch.linspace(0, 1, num_knots - 2 * degree),
            torch.ones(degree)
        ])
        self.register_buffer("knots", knots)
        
        # Evaluation parameters (equally spaced in [0, 1])
        t_values = torch.linspace(0, 1, num_eval_points)
        self.register_buffer("t_values", t_values)
        
        # Precompute the basis matrix B with shape (num_eval_points, num_ctrl_points)
        basis_list = []
        for ti in self.t_values:
            row = []
            for i in range(num_ctrl_points):
                # Compute the basis function value at ti for control point i
                val = cox_de_boor(ti.unsqueeze(0), i, degree, self.knots)  # shape (1,)
                row.append(val)
            row = torch.cat(row, dim=0)  # shape (num_ctrl_points,)
            basis_list.append(row)
        basis_matrix = torch.stack(basis_list, dim=0)
        self.register_buffer("basis_matrix", basis_matrix)
    
    def forward(self, ctrl_points):
        """
        Evaluate the B-spline for given control points.
        ctrl_points: (num_ctrl_points, D) tensor.
        Returns: (num_eval_points, D) tensor.
        """
        return self.basis_matrix @ ctrl_points

def compute_natural_cubic_m(x, y):
    """
    Given fixed nodes x and data y, compute the second derivatives m for a natural cubic spline.
    x, y: 1D tensors of length n (with x sorted).
    Returns m: 1D tensor of length n, with m[0]=m[-1]=0.
    """
    n = x.shape[0]
    if n < 3:
        return torch.zeros_like(y)
    h = x[1:] - x[:-1]  # shape: (n-1,)
    # Build the tridiagonal system for m[1:-1] (n-2 unknowns)
    A = torch.zeros(n-2, n-2, dtype=x.dtype, device=x.device)
    d = torch.zeros(n-2, dtype=x.dtype, device=x.device)
    for i in range(n-2):
        if i > 0:
            A[i, i-1] = h[i] / 6
        A[i, i] = (h[i] + h[i+1]) / 3
        if i < n-3:
            A[i, i+1] = h[i+1] / 6
        d[i] = (y[i+2] - y[i+1]) / h[i+1] - (y[i+1] - y[i]) / h[i]
    m_internal = torch.linalg.solve(A, d)
    m = torch.zeros(n, dtype=x.dtype, device=x.device)
    m[1:-1] = m_internal
    return m

def evaluate_natural_cubic(x, y, x_eval):
    """
    Evaluate the natural cubic spline at points x_eval.
    
    x, y: 1D tensors of length n (the interpolation nodes and their y-values).
    x_eval: 1D tensor of evaluation points.
    
    Returns:
      y_eval: tensor of the same shape as x_eval containing the interpolated values.
    """
    n = x.shape[0]
    m = compute_natural_cubic_m(x, y)
    y_eval = torch.empty_like(x_eval)
    for j, t in enumerate(x_eval):
        # Find interval: t in [x[i], x[i+1]]
        i = torch.searchsorted(x, t, right=False).item()
        if i == 0:
            i = 0
        elif i >= n:
            i = n - 2
        else:
            i = i - 1

        h = x[i+1] - x[i]
        # Cubic spline basis functions for the interval
        A = (x[i+1] - t) / h
        B = (t - x[i]) / h
        C = (A**3 - A) * (h**2) / 6
        D = (B**3 - B) * (h**2) / 6
        y_eval[j] = A * y[i] + B * y[i+1] + C * m[i] + D * m[i+1]
    return y_eval

class CubicInterpolationSpline(nn.Module):
    def __init__(self, n_inpoints, n_outpoints):
        """
        Constructs an interpolation object for a natural cubic spline.
        
        Parameters:
          x_nodes: 1D tensor of fixed x-coordinates for the interpolation nodes.
          x_eval: 1D tensor of x-coordinates where the spline is to be evaluated.
        
        Precomputes an interpolation matrix M (of shape (len(x_eval), len(x_nodes)))
        such that for any y_nodes, the interpolated values are given by:
          y_eval = M @ y_nodes
        """
        super().__init__()
        x_nodes = torch.linspace(0,1,n_inpoints)
        x_eval = torch.linspace(0,1,n_outpoints)
        self.register_buffer("x_nodes", x_nodes)
        self.register_buffer("x_eval", x_eval)
        n = x_nodes.shape[0]
        n_eval = x_eval.shape[0]
        # Precompute the interpolation matrix M
        M = torch.empty(n_eval, n, dtype=x_nodes.dtype, device=x_nodes.device)
        for j in range(n):
            # Use a one-hot vector at node j
            y_onehot = torch.zeros(n, dtype=x_nodes.dtype, device=x_nodes.device)
            y_onehot[j] = 1.0
            # Evaluate the spline for this one-hot input
            M[:, j] = evaluate_natural_cubic(x_nodes, y_onehot, x_eval)
        self.register_buffer("M", M)
    
    def forward(self, y_nodes):
        """
        Given new y-values at the fixed x_nodes, returns the interpolated
        y-values at x_eval.
        
        Parameters:
          y_nodes: Tensor of shape (M,), (M, D) or (N, M, D)
                  - (M,) corresponds to 1D data.
                  - (M, D) corresponds to one example with D-dimensional y-values.
                  - (N, M, D) corresponds to a batch of N examples.
        
        Returns:
          y_eval: Tensor of shape (len(x_eval),), (len(x_eval), D) or (N, len(x_eval), D)
        """
        # If input is 1D (M,), treat it as (M, 1)
        if y_nodes.dim() == 1:
            y_nodes = y_nodes.unsqueeze(1)  # (M, 1)
        
        if y_nodes.dim() == 2:
            # y_nodes shape: (M, D)
            # Output: (len(x_eval), D)
            return self.M @ y_nodes
        elif y_nodes.dim() == 3:
            # y_nodes shape: (N, M, D)
            N = y_nodes.size(0)
            # Expand the interpolation matrix to shape (N, len(x_eval), M)
            M_batch = self.M.unsqueeze(0).expand(N, -1, -1)
            # Use batch matrix multiplication: (N, len(x_eval), M) x (N, M, D) -> (N, len(x_eval), D)
            return torch.bmm(M_batch, y_nodes)
        else:
            raise ValueError("Input y_nodes must be 1D, 2D, or 3D.")

# Example usage for batched input:
"""
if __name__ == "__main__":
    # Fixed x-coordinates (nodes) and evaluation points.
    x_nodes = torch.linspace(0, 10, steps=5)      # e.g., [0, 2.5, 5, 7.5, 10]
    x_eval = torch.linspace(0, 10, steps=50)        # 50 evaluation points in [0,10]
    
    # Create the cubic interpolation spline evaluator.
    spline = CubicInterpolationSpline(5, 50)
    
    # Suppose we have a batch of 3 examples (N=3), each with 5 nodes (M=5) and 2-dimensional y-values (D=2).
    N, M, D = 3, x_nodes.shape[0], 2
    # Create random y-values for each example.
    y_nodes_batch = torch.randn(N, M, D)
    
    # Compute the interpolated y-values for the batch.
    y_interp_batch = spline(y_nodes_batch)  # Shape will be (N, len(x_eval), D)
    
    print("Input y_nodes_batch shape:", y_nodes_batch.shape)       # (3, 5, 2)
    print("Interpolated y values shape:", y_interp_batch.shape)      # (3, 50, 2)
    
    # (Optional) Visualize one example if matplotlib is available.
    try:
        import matplotlib.pyplot as plt
        example_idx = 1  # Plot the first example
        plt.figure(figsize=(8, 5))
        plt.plot(x_eval.numpy(), y_interp_batch[example_idx,:,0].numpy(), 'b-', label='Interpolated y0')
        plt.plot(x_eval.numpy(), y_interp_batch[example_idx,:,1].numpy(), 'g-', label='Interpolated y1')
        plt.plot(x_nodes.numpy(), y_nodes_batch[example_idx,:,0].numpy(), 'ro', label='Node y0')
        plt.plot(x_nodes.numpy(), y_nodes_batch[example_idx,:,1].numpy(), 'ko', label='Node y1')
        plt.title("Cubic Interpolation Spline (Example 0)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()
    except ImportError:
        print("matplotlib not installed, skipping plot.")

"""

def compute_akima_derivatives(x, y):
    """
    Compute the derivatives at the nodes for Akima interpolation.
    
    Parameters:
      x : 1D tensor of nodes (assumed sorted)
      y : 1D tensor of values at the nodes
    
    Returns:
      d : 1D tensor of computed derivatives at each node.
      
    The method computes the slopes between nodes:
        m_i = (y[i+1]-y[i])/(x[i+1]-x[i])
    and then for interior nodes (i=2,...,n-2) uses a weighted average:
        d[i] = (|m[i] - m[i-1]| * m[i-1] + |m[i-1] - m[i-2]| * m[i]) /
               (|m[i] - m[i-1]| + |m[i-1] - m[i-2]|)
    For endpoints, we set:
        d[0] = m[0]   and   d[1] = m[0]
        d[n-1] = m[n-2]
    """
    n = x.shape[0]
    if n == 0:
        return torch.tensor([])
    if n == 1:
        return torch.zeros_like(y)
    
    # Compute slopes between nodes.
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    d = torch.empty(n, dtype=x.dtype, device=x.device)
    
    if n == 2:
        d[0] = m[0]
        d[1] = m[0]
        return d

    # For the first two nodes, use the first slope.
    d[0] = m[0]
    d[1] = m[0]
    
    # For interior nodes i = 2,..., n-2, use Akima’s weighted average.
    for i in range(2, n-1):
        diff1 = torch.abs(m[i] - m[i-1])
        diff2 = torch.abs(m[i-1] - m[i-2])
        if diff1 + diff2 > 1e-6:
            d[i] = (diff1 * m[i-1] + diff2 * m[i]) / (diff1 + diff2)
        else:
            d[i] = 0.5 * (m[i-1] + m[i])
    
    # For the last node, use the last slope.
    d[n-1] = m[n-2]
    return d

def evaluate_akima(x, y, x_eval):
    """
    Evaluate the Akima cubic interpolation spline at points in x_eval.
    
    Parameters:
      x      : 1D tensor of nodes (must be sorted)
      y      : 1D tensor of values at the nodes
      x_eval : 1D tensor of evaluation points
      
    Returns:
      y_eval : 1D tensor of interpolated values at x_eval.
    
    The interpolation uses the Hermite cubic formula:
    
      S_i(x) = h00(t) * y_i + h10(t)*h*d_i + h01(t)*y_{i+1} + h11(t)*h*d_{i+1},
    
    where t = (x - x_i)/h and h = x[i+1]-x[i], and the basis functions are:
      h00(t) = 2t³ - 3t² + 1
      h10(t) = t³ - 2t² + t
      h01(t) = -2t³ + 3t²
      h11(t) = t³ - t².
    """
    n = x.shape[0]
    d = compute_akima_derivatives(x, y)
    y_eval = torch.empty_like(x_eval)
    
    for j, t in enumerate(x_eval):
        # Find interval: choose i such that x[i] <= t < x[i+1].
        # If t equals the last node, use the last interval.
        i = torch.searchsorted(x, t, right=False).item()
        if i == 0:
            i = 0
        elif i >= n:
            i = n - 2
        else:
            i = i - 1
        
        h = x[i+1] - x[i]
        t_norm = (t - x[i]) / h
        # Hermite basis functions
        h00 = 2 * t_norm**3 - 3 * t_norm**2 + 1
        h10 = t_norm**3 - 2 * t_norm**2 + t_norm
        h01 = -2 * t_norm**3 + 3 * t_norm**2
        h11 = t_norm**3 - t_norm**2
        
        y_eval[j] = h00 * y[i] + h10 * h * d[i] + h01 * y[i+1] + h11 * h * d[i+1]
    
    return y_eval

class AkimaInterpolationSpline(nn.Module):
    def __init__(self, n_inpoints, n_outpoints):
        """
        Constructs an Akima interpolation object for cubic interpolation.
        
        Parameters:
          x_nodes : 1D tensor of fixed x-coordinates for the nodes.
          x_eval  : 1D tensor of x-coordinates at which the spline is evaluated.
        
        Precomputes an interpolation matrix M of shape (len(x_eval), len(x_nodes))
        so that for any y_nodes the interpolation is:
            y_eval = M @ y_nodes.
        """
        super().__init__()
        x_nodes = torch.linspace(0,1,n_inpoints)
        x_eval = torch.linspace(0,1,n_outpoints)
        self.register_buffer("x_nodes", x_nodes)
        self.register_buffer("x_eval", x_eval)
        n = x_nodes.shape[0]
        n_eval = x_eval.shape[0]
        
        # Precompute the interpolation matrix M.
        M = torch.empty(n_eval, n, dtype=x_nodes.dtype, device=x_nodes.device)
        for j in range(n):
            # Create a one-hot vector for the j-th node.
            y_onehot = torch.zeros(n, dtype=x_nodes.dtype, device=x_nodes.device)
            y_onehot[j] = 1.0
            M[:, j] = evaluate_akima(x_nodes, y_onehot, x_eval)
        self.register_buffer("M", M)
    
    def forward(self, y_nodes):
        """
        Given new y-values at the fixed nodes, returns the interpolated y-values at x_eval.
        
        Parameters:
          y_nodes : Tensor of shape (M,), (M, D) or (N, M, D), where
                    M is the number of nodes and D is the dimensionality.
        
        Returns:
          y_eval : Tensor of shape (len(x_eval),), (len(x_eval), D) or (N, len(x_eval), D)
        """
        # If input is 1D, treat it as (M, 1)
        if y_nodes.dim() == 1:
            y_nodes = y_nodes.unsqueeze(1)
        
        if y_nodes.dim() == 2:
            # y_nodes shape: (M, D) -> output: (len(x_eval), D)
            return self.M @ y_nodes
        elif y_nodes.dim() == 3:
            # y_nodes shape: (N, M, D) -> output: (N, len(x_eval), D)
            N = y_nodes.size(0)
            M_batch = self.M.unsqueeze(0).expand(N, -1, -1)
            return torch.bmm(M_batch, y_nodes)
        else:
            raise ValueError("Input y_nodes must be 1D, 2D, or 3D.")

class DummyInterpolator(nn.Module):
    def __init__(self, *args):
        """
        Returns whatever you give it
        """
        super().__init__()
    
    def forward(self, y_nodes):
        # Returns its input
        return y_nodes

# Example usage for batched input:
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Define fixed nodes and evaluation points.
    x_nodes = torch.linspace(0, 10, steps=5)      # e.g., [0, 2.5, 5, 7.5, 10]
    x_eval = torch.linspace(0, 10, steps=100)       # 100 evaluation points between 0 and 10
    
    # Create the Akima spline interpolator.
    spline = AkimaInterpolationSpline(x_nodes, x_eval)
    
    # Suppose we have a batch of 3 examples (N=3), each with 5 nodes (M=5) and 2-dimensional y-values (D=2).
    N, M, D = 3, x_nodes.shape[0], 2
    y_nodes_batch = torch.randn(N, M, D)
    
    # Compute the interpolated y-values for the batch.
    y_interp_batch = spline(y_nodes_batch)  # Shape: (N, len(x_eval), D)
    
    print("Input y_nodes_batch shape:", y_nodes_batch.shape)   # (3, 5, 2)
    print("Interpolated y values shape:", y_interp_batch.shape)  # (3, 100, 2)
    
    # (Optional) Plot the interpolation for the first example and first dimension.
    plt.figure(figsize=(8, 5))
    plt.plot(x_eval.numpy(), y_interp_batch[0, :, 0].numpy(), 'b-', label="Interpolated")
    plt.plot(x_nodes.numpy(), y_nodes_batch[0, :, 0].numpy(), 'ro', label="Nodes")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Akima Interpolation Spline (Example 0, Dimension 0)")
    plt.legend()
    plt.grid(True)
    plt.show()