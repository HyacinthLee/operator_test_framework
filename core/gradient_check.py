"""
Gradient checking utilities for operator testing.
"""

import torch
from typing import Callable, Union, Tuple


def numerical_jacobian(
    func: Callable,
    inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Compute Jacobian matrix numerically using finite differences.
    
    Args:
        func: Function to compute Jacobian for
        inputs: Input tensor(s)
        eps: Finite difference epsilon
        
    Returns:
        Numerical Jacobian matrix
    """
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)
    
    # Flatten all inputs
    original_shapes = [inp.shape for inp in inputs]
    flat_inputs = [inp.flatten() for inp in inputs]
    n_params = sum(len(f) for f in flat_inputs)
    
    # Compute output at base point
    with torch.no_grad():
        output = func(*inputs)
        output_flat = output.flatten()
        n_outputs = len(output_flat)
    
    # Compute numerical Jacobian
    jacobian = torch.zeros(n_outputs, n_params)
    
    param_idx = 0
    for inp_idx, inp in enumerate(inputs):
        for i in range(len(flat_inputs[inp_idx])):
            # Create perturbed inputs
            inputs_plus = [inp.clone() for inp in inputs]
            inputs_minus = [inp.clone() for inp in inputs]
            
            # Flatten view for perturbation
            flat_plus = inputs_plus[inp_idx].flatten()
            flat_minus = inputs_minus[inp_idx].flatten()
            
            flat_plus[i] += eps
            flat_minus[i] -= eps
            
            # Reshape back
            inputs_plus[inp_idx] = flat_plus.view(original_shapes[inp_idx])
            inputs_minus[inp_idx] = flat_minus.view(original_shapes[inp_idx])
            
            # Compute function values
            with torch.no_grad():
                output_plus = func(*inputs_plus).flatten()
                output_minus = func(*inputs_minus).flatten()
            
            # Central difference
            jacobian[:, param_idx] = (output_plus - output_minus) / (2 * eps)
            param_idx += 1
    
    return jacobian


def verify_gradient(
    func: Callable,
    inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    rtol: float = 1e-4,
    atol: float = 1e-6,
    eps: float = 1e-5
) -> bool:
    """
    Verify that analytical gradient matches numerical gradient.
    
    Args:
        func: Function to test
        inputs: Input tensor(s) with requires_grad=True
        rtol: Relative tolerance
        atol: Absolute tolerance
        eps: Finite difference epsilon
        
    Returns:
        True if gradients match
        
    Raises:
        AssertionError: If gradients don't match
    """
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)
    
    # Ensure inputs require grad
    for inp in inputs:
        if isinstance(inp, torch.Tensor):
            inp.requires_grad_(True)
    
    # Analytical gradient
    output = func(*inputs)
    output.sum().backward()
    
    analytical_grads = []
    for inp in inputs:
        if isinstance(inp, torch.Tensor) and inp.grad is not None:
            analytical_grads.append(inp.grad.flatten())
        elif isinstance(inp, torch.Tensor):
            analytical_grads.append(torch.zeros_like(inp).flatten())
    
    analytical_grad = torch.cat(analytical_grads)
    
    # Clear gradients for numerical computation
    for inp in inputs:
        if isinstance(inp, torch.Tensor) and inp.grad is not None:
            inp.grad.zero_()
    
    # Numerical gradient
    numerical_grad = numerical_jacobian(func, inputs, eps)
    # For scalar output, we want the gradient w.r.t. sum
    numerical_grad = numerical_grad.sum(dim=0)
    
    # Compare
    if not torch.allclose(analytical_grad, numerical_grad, rtol=rtol, atol=atol):
        max_diff = (analytical_grad - numerical_grad).abs().max()
        mean_diff = (analytical_grad - numerical_grad).abs().mean()
        raise AssertionError(
            f"Gradient mismatch: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}"
        )
    
    return True


def check_gradient_numerical_stability(
    func: Callable,
    inputs: torch.Tensor,
    num_trials: int = 10
) -> dict:
    """
    Check gradient numerical stability across multiple random inputs.
    
    Args:
        func: Function to test
        inputs: Template input tensor (used for shape/dtype)
        num_trials: Number of random trials
        
    Returns:
        Dictionary with stability statistics
    """
    results = {
        "passed": 0,
        "failed": 0,
        "max_diffs": [],
        "mean_diffs": []
    }
    
    for _ in range(num_trials):
        # Random input
        rand_input = torch.randn_like(inputs)
        rand_input.requires_grad_(True)
        
        try:
            verify_gradient(func, rand_input)
            results["passed"] += 1
        except AssertionError as e:
            results["failed"] += 1
            # Extract diff from error message
            if "max_diff=" in str(e):
                diff_str = str(e).split("max_diff=")[1].split(",")[0]
                results["max_diffs"].append(float(diff_str))
    
    if results["max_diffs"]:
        results["mean_max_diff"] = sum(results["max_diffs"]) / len(results["max_diffs"])
    
    return results
