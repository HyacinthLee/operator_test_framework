"""
Gradient stability testing for operators.
Detects gradient underflow, overflow, and numerical instability.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Callable, Optional
import numpy as np


class GradientStabilityTester:
    """
    Test gradient numerical stability during backpropagation.
    
    Key checks:
    - Gradient underflow (values too small, ~0)
    - Gradient overflow (values too large, Inf/NaN)
    - Gradient vanishing/exploding across layers
    - Loss scaling effectiveness
    """
    
    def __init__(
        self,
        underflow_threshold: float = 1e-7,
        overflow_threshold: float = 1e3,
        vanishing_threshold: float = 1e-6,
        exploding_threshold: float = 1e3
    ):
        """
        Args:
            underflow_threshold: Values below this are considered underflow
            overflow_threshold: Values above this are considered overflow
            vanishing_threshold: Gradient norm below this is vanishing
            exploding_threshold: Gradient norm above this is exploding
        """
        self.underflow_threshold = underflow_threshold
        self.overflow_threshold = overflow_threshold
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold
    
    def analyze_gradient_stats(self, grad: torch.Tensor) -> Dict[str, float]:
        """
        Compute comprehensive gradient statistics.
        
        Args:
            grad: Gradient tensor
            
        Returns:
            Dictionary of statistics
        """
        with torch.no_grad():
            stats = {
                "min": grad.min().item(),
                "max": grad.max().item(),
                "mean": grad.mean().item(),
                "std": grad.std().item(),
                "norm": grad.norm().item(),
                "numel": grad.numel(),
            }
            
            # Count special values
            stats["nan_count"] = torch.isnan(grad).sum().item()
            stats["inf_count"] = torch.isinf(grad).sum().item()
            stats["zero_count"] = (grad == 0).sum().item()
            
            # Underflow/overflow detection
            stats["underflow_count"] = (grad.abs() < self.underflow_threshold).sum().item()
            stats["overflow_count"] = (grad.abs() > self.overflow_threshold).sum().item()
            
            # Ratios
            stats["underflow_ratio"] = stats["underflow_count"] / stats["numel"]
            stats["overflow_ratio"] = stats["overflow_count"] / stats["numel"]
            stats["zero_ratio"] = stats["zero_count"] / stats["numel"]
            
            # Percentiles
            grad_flat = grad.abs().flatten()
            stats["p1"] = torch.quantile(grad_flat, 0.01).item()
            stats["p50"] = torch.quantile(grad_flat, 0.50).item()
            stats["p99"] = torch.quantile(grad_flat, 0.99).item()
        
        return stats
    
    def test_gradient_health(
        self,
        operator: Callable,
        inputs: torch.Tensor,
        retain_graph: bool = False
    ) -> Dict[str, any]:
        """
        Test overall gradient health for an operator.
        
        Args:
            operator: Operator to test
            inputs: Input tensor with requires_grad=True
            retain_graph: Whether to retain computation graph
            
        Returns:
            Gradient health assessment
        """
        # Forward pass
        output = operator(inputs)
        
        # Compute loss (scalar)
        if output.numel() > 1:
            loss = output.sum()
        else:
            loss = output
        
        # Backward pass - only retain graph if explicitly requested
        # and if we need to do multiple backwards
        loss.backward(retain_graph=retain_graph)
        
        # Analyze gradient
        grad = inputs.grad
        if grad is None:
            return {
                "healthy": False,
                "issues": ["No gradient computed"],
                "stats": {},
                "recommendations": ["Check if inputs have requires_grad=True"]
            }
        
        stats = self.analyze_gradient_stats(grad)
        
        # Health assessment
        issues = []
        
        if stats["nan_count"] > 0:
            issues.append("NaN detected in gradient")
        
        if stats["inf_count"] > 0:
            issues.append("Inf detected in gradient")
        
        if stats["underflow_ratio"] > 0.5:
            issues.append(f"Severe underflow: {stats['underflow_ratio']:.2%} of values")
        
        if stats["overflow_ratio"] > 0.01:
            issues.append(f"Overflow detected: {stats['overflow_ratio']:.2%} of values")
        
        if stats["norm"] < self.vanishing_threshold:
            issues.append(f"Vanishing gradient: norm={stats['norm']:.2e}")
        
        if stats["norm"] > self.exploding_threshold:
            issues.append(f"Exploding gradient: norm={stats['norm']:.2e}")
        
        if stats["zero_ratio"] > 0.9:
            issues.append(f"Mostly zero gradients: {stats['zero_ratio']:.2%}")
        
        healthy = len(issues) == 0
        
        return {
            "healthy": healthy,
            "issues": issues,
            "stats": stats,
            "recommendations": self._generate_recommendations(stats, issues)
        }
    
    def _generate_recommendations(
        self,
        stats: Dict[str, float],
        issues: List[str]
    ) -> List[str]:
        """Generate recommendations based on issues."""
        recommendations = []
        
        if any("NaN" in i or "Inf" in i for i in issues):
            recommendations.append(
                "Check for numerical instability in forward pass. "
                "Consider gradient clipping or using more stable algorithms."
            )
        
        if any("underflow" in i.lower() for i in issues):
            recommendations.append(
                "Gradient underflow detected. Use loss scaling (GradScaler) "
                "or increase to BF16/FP32 for sensitive layers."
            )
        
        if any("overflow" in i.lower() for i in issues):
            recommendations.append(
                "Gradient overflow detected. Apply gradient clipping "
                "(max_norm=1.0) and check for unstable operations."
            )
        
        if any("vanishing" in i.lower() for i in issues):
            recommendations.append(
                "Vanishing gradients. Check weight initialization, "
                "use residual connections, or consider layer normalization."
            )
        
        if any("exploding" in i.lower() for i in issues):
            recommendations.append(
                "Exploding gradients. Apply gradient clipping and "
                "check for recurrent operations without gating."
            )
        
        if not recommendations:
            recommendations.append("Gradients look healthy. No action needed.")
        
        return recommendations
    
    def test_loss_scaling_compatibility(
        self,
        operator: Callable,
        inputs: torch.Tensor,
        loss_scaler: Optional[torch.cuda.amp.GradScaler] = None
    ) -> Dict[str, any]:
        """
        Test if operator works well with loss scaling.
        
        This is critical for FP16 training stability.
        """
        if not torch.cuda.is_available():
            return {
                "skipped": True,
                "reason": "CUDA not available",
            }
        
        if loss_scaler is None:
            loss_scaler = torch.cuda.amp.GradScaler()
        
        initial_scale = loss_scaler.get_scale()
        
        # Forward with autocast
        with torch.cuda.amp.autocast():
            output = operator(inputs)
            loss = output.sum()
        
        # Scale loss and backward
        scaled_loss = loss_scaler.scale(loss)
        scaled_loss.backward()
        
        # Check if scaling worked
        grad = inputs.grad
        grad_has_values = (grad.abs() > 1e-8).any().item()
        
        # Update scale (simulating optimizer step)
        old_scale = loss_scaler.get_scale()
        loss_scaler.step(lambda: None)  # Dummy optimizer
        loss_scaler.update()
        new_scale = loss_scaler.get_scale()
        
        # Analyze scale change
        scale_decreased = new_scale < old_scale
        
        return {
            "initial_scale": initial_scale,
            "final_scale": new_scale,
            "scale_changed": old_scale != new_scale,
            "scale_decreased": scale_decreased,
            "grad_has_values": grad_has_values,
            "grad_stats": self.analyze_gradient_stats(grad),
            "compatible": grad_has_values and not scale_decreased,
        }
    
    def run_full_gradient_test_suite(
        self,
        operator: Callable,
        input_shape: Tuple[int, ...],
        test_dtypes: List[torch.dtype] = [torch.float32, torch.float16],
        verbose: bool = True
    ) -> Dict[str, List[Dict]]:
        """
        Run complete gradient stability test suite.
        """
        results = {
            "operator": operator.__name__ if hasattr(operator, '__name__') else str(operator),
            "input_shape": input_shape,
            "dtype_tests": [],
        }
        
        for dtype in test_dtypes:
            if verbose:
                print(f"\nTesting gradient stability with {dtype}...")
            
            # Create input
            x = torch.randn(input_shape, requires_grad=True)
            if dtype != torch.float32:
                x = x.to(dtype)
                x.requires_grad_(True)
            
            # Test gradient health
            health_result = self.test_gradient_health(operator, x, retain_graph=True)
            
            # Test loss scaling (only for FP16)
            scaling_result = None
            if dtype == torch.float16 and torch.cuda.is_available():
                x.grad = None
                scaling_result = self.test_loss_scaling_compatibility(operator, x)
            
            result = {
                "dtype": str(dtype),
                "healthy": health_result["healthy"],
                "issues": health_result["issues"],
                "stats": health_result["stats"],
                "recommendations": health_result["recommendations"],
            }
            
            if scaling_result:
                result["loss_scaling"] = scaling_result
            
            results["dtype_tests"].append(result)
            
            if verbose:
                status = "✓" if health_result["healthy"] else "✗"
                print(f"  Health: {status}")
                for issue in health_result["issues"]:
                    print(f"    - {issue}")
        
        # Overall assessment
        all_healthy = all(t["healthy"] for t in results["dtype_tests"])
        results["all_healthy"] = all_healthy
        
        return results


# Example usage
if __name__ == "__main__":
    tester = GradientStabilityTester()
    
    # Test an operator
    layernorm = torch.nn.LayerNorm(64)
    results = tester.run_full_gradient_test_suite(
        layernorm,
        input_shape=(4, 64),
        verbose=True
    )
    
    print(f"\nOverall: {'HEALTHY' if results['all_healthy'] else 'ISSUES DETECTED'}")
