"""
Four-layer validation system for operator testing.

This module implements the four validation layers:
1. Mathematical Correctness - Verify algorithmic implementation
2. Numerical Stability - Check precision and edge cases
3. Functional Compliance - Validate against reference implementations
4. Performance Benchmarking - Measure throughput and memory
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import time
import math

from .test_utils import gradient_check, compare_tensors, benchmark_performance


class BaseValidator(ABC):
    """Abstract base class for all validators."""
    
    def __init__(self, config):
        """Initialize validator with configuration.
        
        Args:
            config: TestConfig instance
        """
        self.config = config
        self.results: Dict[str, Any] = {}
    
    @abstractmethod
    def validate(self, adapter) -> Dict[str, Any]:
        """Run validation and return results.
        
        Args:
            adapter: OperatorTestAdapter instance
            
        Returns:
            Dictionary with validation results
        """
        pass
    
    def _log(self, message: str):
        """Log validation message."""
        print(f"  [{self.__class__.__name__}] {message}")


class MathematicalValidator(BaseValidator):
    """Validates mathematical correctness of operator implementation.
    
    Checks include:
    - Gradient correctness via finite differences
    - Output shape validation
    - Output type validation
    - Basic functionality tests
    """
    
    def validate(self, adapter) -> Dict[str, Any]:
        """Run mathematical validation tests.
        
        Args:
            adapter: OperatorTestAdapter instance
            
        Returns:
            Dictionary with test results
        """
        self._log(f"Starting mathematical validation for {adapter.name}")
        
        results = {
            'passed': True,
            'tests': {}
        }
        
        # Test 1: Basic forward pass
        try:
            results['tests']['forward_pass'] = self._test_forward_pass(adapter)
        except Exception as e:
            results['tests']['forward_pass'] = {'passed': False, 'error': str(e)}
            results['passed'] = False
        
        # Test 2: Gradient correctness
        try:
            results['tests']['gradient_check'] = self._test_gradient_check(adapter)
        except Exception as e:
            results['tests']['gradient_check'] = {'passed': False, 'error': str(e)}
            results['passed'] = False
        
        # Test 3: Output shape validation
        try:
            results['tests']['output_shape'] = self._test_output_shape(adapter)
        except Exception as e:
            results['tests']['output_shape'] = {'passed': False, 'error': str(e)}
            results['passed'] = False
        
        # Test 4: Determinism check
        try:
            results['tests']['determinism'] = self._test_determinism(adapter)
        except Exception as e:
            results['tests']['determinism'] = {'passed': False, 'error': str(e)}
            results['passed'] = False
        
        # Overall pass status
        results['passed'] = all(t.get('passed', False) for t in results['tests'].values())
        
        status = "PASSED" if results['passed'] else "FAILED"
        self._log(f"Mathematical validation {status}")
        
        return results
    
    def _test_forward_pass(self, adapter) -> Dict[str, Any]:
        """Test basic forward pass execution."""
        inputs = adapter.generate_inputs(batch_size=4)
        inputs = {k: v.to(self.config.device).to(self.config.dtype) 
                 for k, v in inputs.items()}
        
        outputs = adapter.forward(inputs)
        
        passed = (
            outputs is not None and
            adapter.validate_output_type(outputs) and
            adapter.validate_output_shape(inputs, outputs)
        )
        
        return {
            'passed': passed,
            'output_shape': tuple(outputs.shape) if outputs is not None else None,
            'output_dtype': str(outputs.dtype) if outputs is not None else None
        }
    
    def _test_gradient_check(self, adapter) -> Dict[str, Any]:
        """Test gradient correctness using finite differences."""
        inputs = adapter.generate_inputs(batch_size=2)
        inputs = {k: v.to(self.config.device).to(self.config.dtype).requires_grad_(True)
                 for k, v in inputs.items()}
        
        # Forward pass
        outputs = adapter.forward(inputs)
        
        # Compute analytical gradients
        analytical_grads = adapter.backward(outputs, inputs)
        
        # Compute numerical gradients using finite differences
        numerical_grads = self._compute_numerical_gradients(adapter, inputs, outputs)
        
        # Compare gradients
        max_diff = 0.0
        all_close = True
        
        for name in analytical_grads.keys():
            if analytical_grads[name] is not None and numerical_grads.get(name) is not None:
                diff = (analytical_grads[name] - numerical_grads[name]).abs().max().item()
                max_diff = max(max_diff, diff)
                if diff > self.config.tolerance * 10:  # Allow 10x tolerance for numerical grad
                    all_close = False
        
        return {
            'passed': all_close,
            'max_gradient_diff': max_diff,
            'tolerance': self.config.tolerance * 10
        }
    
    def _compute_numerical_gradients(self, adapter, inputs, outputs,
                                     epsilon: float = 1e-4) -> Dict[str, torch.Tensor]:
        """Compute numerical gradients using finite differences."""
        numerical_grads = {}
        
        for name, tensor in inputs.items():
            if not tensor.requires_grad:
                continue
                
            grad = torch.zeros_like(tensor)
            
            # Flatten for iteration
            original_shape = tensor.shape
            flat_tensor = tensor.view(-1)
            flat_grad = grad.view(-1)
            
            for i in range(flat_tensor.numel()):
                # Compute f(x + epsilon)
                flat_tensor[i] += epsilon
                outputs_plus = adapter.forward(inputs)
                loss_plus = outputs_plus.sum().item()
                
                # Compute f(x - epsilon)
                flat_tensor[i] -= 2 * epsilon
                outputs_minus = adapter.forward(inputs)
                loss_minus = outputs_minus.sum().item()
                
                # Restore original value
                flat_tensor[i] += epsilon
                
                # Finite difference
                flat_grad[i] = (loss_plus - loss_minus) / (2 * epsilon)
            
            numerical_grads[name] = grad
        
        return numerical_grads
    
    def _test_output_shape(self, adapter) -> Dict[str, Any]:
        """Test output shape for various input sizes."""
        test_cases = [
            {'batch_size': 1},
            {'batch_size': 8},
            {'batch_size': 16}
        ]
        
        all_passed = True
        shapes = []
        
        for case in test_cases:
            inputs = adapter.generate_inputs(**case)
            inputs = {k: v.to(self.config.device).to(self.config.dtype)
                     for k, v in inputs.items()}
            outputs = adapter.forward(inputs)
            
            passed = adapter.validate_output_shape(inputs, outputs)
            all_passed = all_passed and passed
            shapes.append({
                'input_batch': case['batch_size'],
                'output_shape': tuple(outputs.shape),
                'passed': passed
            })
        
        return {
            'passed': all_passed,
            'shapes_tested': shapes
        }
    
    def _test_determinism(self, adapter) -> Dict[str, Any]:
        """Test that operator produces deterministic results."""
        # Set seed for reproducibility
        torch.manual_seed(42)
        inputs1 = adapter.generate_inputs(batch_size=4)
        inputs1 = {k: v.to(self.config.device).to(self.config.dtype)
                  for k, v in inputs1.items()}
        
        torch.manual_seed(42)
        inputs2 = adapter.generate_inputs(batch_size=4)
        inputs2 = {k: v.to(self.config.device).to(self.config.dtype)
                  for k, v in inputs2.items()}
        
        outputs1 = adapter.forward(inputs1)
        outputs2 = adapter.forward(inputs2)
        
        max_diff = (outputs1 - outputs2).abs().max().item()
        passed = max_diff < self.config.tolerance
        
        return {
            'passed': passed,
            'max_diff': max_diff
        }


class NumericalValidator(BaseValidator):
    """Validates numerical stability of operator implementation.
    
    Checks include:
    - Different data types (fp16, fp32, fp64)
    - Extreme input values (very small, very large)
    - NaN and Inf handling
    - Numerical precision preservation
    """
    
    def validate(self, adapter) -> Dict[str, Any]:
        """Run numerical validation tests."""
        self._log(f"Starting numerical validation for {adapter.name}")
        
        results = {
            'passed': True,
            'tests': {}
        }
        
        # Test 1: Different data types
        try:
            results['tests']['dtype_stability'] = self._test_dtype_stability(adapter)
        except Exception as e:
            results['tests']['dtype_stability'] = {'passed': False, 'error': str(e)}
        
        # Test 2: Extreme values
        try:
            results['tests']['extreme_values'] = self._test_extreme_values(adapter)
        except Exception as e:
            results['tests']['extreme_values'] = {'passed': False, 'error': str(e)}
        
        # Test 3: NaN/Inf handling
        try:
            results['tests']['nan_inf_handling'] = self._test_nan_inf_handling(adapter)
        except Exception as e:
            results['tests']['nan_inf_handling'] = {'passed': False, 'error': str(e)}
        
        # Overall pass status
        results['passed'] = all(t.get('passed', False) for t in results['tests'].values())
        
        status = "PASSED" if results['passed'] else "FAILED"
        self._log(f"Numerical validation {status}")
        
        return results
    
    def _test_dtype_stability(self, adapter) -> Dict[str, Any]:
        """Test operator with different data types."""
        dtypes = [torch.float32, torch.float64]
        
        # Add float16 if CUDA is available
        if self.config.device == 'cuda':
            dtypes.append(torch.float16)
        
        results = {'passed': True, 'dtypes': {}}
        
        for dtype in dtypes:
            try:
                inputs = adapter.generate_inputs(batch_size=4)
                inputs = {k: v.to(self.config.device).to(dtype)
                         for k, v in inputs.items()}
                
                outputs = adapter.forward(inputs)
                
                # Check for NaN or Inf
                has_nan = torch.isnan(outputs).any().item()
                has_inf = torch.isinf(outputs).any().item()
                
                results['dtypes'][str(dtype)] = {
                    'passed': not (has_nan or has_inf),
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                    'output_range': [outputs.min().item(), outputs.max().item()]
                }
                
                if has_nan or has_inf:
                    results['passed'] = False
                    
            except Exception as e:
                results['dtypes'][str(dtype)] = {'passed': False, 'error': str(e)}
                results['passed'] = False
        
        return results
    
    def _test_extreme_values(self, adapter) -> Dict[str, Any]:
        """Test operator with extreme input values."""
        test_cases = [
            ('very_small', 1e-7),
            ('small', 1e-4),
            ('large', 1e4),
            ('very_large', 1e7),
        ]
        
        results = {'passed': True, 'cases': {}}
        
        for case_name, scale in test_cases:
            try:
                inputs = adapter.generate_inputs(batch_size=4)
                inputs = {k: v.to(self.config.device).to(self.config.dtype) * scale
                         for k, v in inputs.items()}
                
                outputs = adapter.forward(inputs)
                
                has_nan = torch.isnan(outputs).any().item()
                has_inf = torch.isinf(outputs).any().item()
                
                results['cases'][case_name] = {
                    'passed': not (has_nan or has_inf),
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                    'scale': scale
                }
                
                if has_nan or has_inf:
                    results['passed'] = False
                    
            except Exception as e:
                results['cases'][case_name] = {'passed': False, 'error': str(e)}
                results['passed'] = False
        
        return results
    
    def _test_nan_inf_handling(self, adapter) -> Dict[str, Any]:
        """Test operator behavior with NaN and Inf inputs."""
        results = {'passed': True, 'tests': {}}
        
        # Test with NaN inputs
        try:
            inputs = adapter.generate_inputs(batch_size=2)
            inputs = {k: v.to(self.config.device).to(self.config.dtype)
                     for k, v in inputs.items()}
            # Inject NaN
            first_key = list(inputs.keys())[0]
            inputs[first_key][0, 0] = float('nan')
            
            outputs = adapter.forward(inputs)
            
            results['tests']['nan_input'] = {
                'passed': True,  # Just check it doesn't crash
                'output_has_nan': torch.isnan(outputs).any().item()
            }
        except Exception as e:
            results['tests']['nan_input'] = {'passed': False, 'error': str(e)}
            results['passed'] = False
        
        # Test with Inf inputs
        try:
            inputs = adapter.generate_inputs(batch_size=2)
            inputs = {k: v.to(self.config.device).to(self.config.dtype)
                     for k, v in inputs.items()}
            # Inject Inf
            first_key = list(inputs.keys())[0]
            inputs[first_key][0, 0] = float('inf')
            
            outputs = adapter.forward(inputs)
            
            results['tests']['inf_input'] = {
                'passed': True,
                'output_has_inf': torch.isinf(outputs).any().item()
            }
        except Exception as e:
            results['tests']['inf_input'] = {'passed': False, 'error': str(e)}
            results['passed'] = False
        
        return results


class FunctionalValidator(BaseValidator):
    """Validates functional compliance against reference implementations.
    
    Checks include:
    - Comparison with PyTorch native implementations
    - Numerical equivalence within tolerance
    - Edge case behavior matching
    """
    
    def validate(self, adapter) -> Dict[str, Any]:
        """Run functional validation tests."""
        self._log(f"Starting functional validation for {adapter.name}")
        
        results = {
            'passed': True,
            'tests': {}
        }
        
        # Test 1: Reference implementation comparison (if available)
        if hasattr(adapter, 'reference_forward'):
            try:
                results['tests']['reference_match'] = self._test_reference_match(adapter)
            except Exception as e:
                results['tests']['reference_match'] = {'passed': False, 'error': str(e)}
        else:
            results['tests']['reference_match'] = {
                'passed': True,
                'skipped': True,
                'reason': 'No reference implementation provided'
            }
        
        # Test 2: Consistency across multiple runs
        try:
            results['tests']['consistency'] = self._test_consistency(adapter)
        except Exception as e:
            results['tests']['consistency'] = {'passed': False, 'error': str(e)}
        
        # Overall pass status
        results['passed'] = all(
            t.get('passed', False) or t.get('skipped', False)
            for t in results['tests'].values()
        )
        
        status = "PASSED" if results['passed'] else "FAILED"
        self._log(f"Functional validation {status}")
        
        return results
    
    def _test_reference_match(self, adapter) -> Dict[str, Any]:
        """Compare with reference implementation."""
        inputs = adapter.generate_inputs(batch_size=4)
        inputs = {k: v.to(self.config.device).to(self.config.dtype)
                 for k, v in inputs.items()}
        
        # Get outputs from both implementations
        test_outputs = adapter.forward(inputs)
        ref_outputs = adapter.reference_forward(inputs)
        
        # Compare
        max_diff = (test_outputs - ref_outputs).abs().max().item()
        passed = max_diff < self.config.tolerance
        
        return {
            'passed': passed,
            'max_diff': max_diff,
            'tolerance': self.config.tolerance
        }
    
    def _test_consistency(self, adapter) -> Dict[str, Any]:
        """Test consistency across multiple runs with same input."""
        inputs = adapter.generate_inputs(batch_size=4)
        inputs = {k: v.to(self.config.device).to(self.config.dtype)
                 for k, v in inputs.items()}
        
        # Run multiple times
        outputs_list = []
        for _ in range(5):
            outputs = adapter.forward(inputs)
            outputs_list.append(outputs.clone())
        
        # Check all outputs are identical
        max_diff = 0.0
        for i in range(1, len(outputs_list)):
            diff = (outputs_list[0] - outputs_list[i]).abs().max().item()
            max_diff = max(max_diff, diff)
        
        passed = max_diff < self.config.tolerance
        
        return {
            'passed': passed,
            'max_diff': max_diff,
            'num_runs': len(outputs_list)
        }


class PerformanceValidator(BaseValidator):
    """Validates performance characteristics of operator implementation.
    
    Metrics include:
    - Forward pass latency
    - Backward pass latency
    - Memory usage
    - Throughput (operations per second)
    """
    
    def validate(self, adapter) -> Dict[str, Any]:
        """Run performance validation tests."""
        self._log(f"Starting performance validation for {adapter.name}")
        
        results = {
            'passed': True,
            'tests': {}
        }
        
        # Test 1: Forward pass performance
        try:
            results['tests']['forward_performance'] = self._test_forward_performance(adapter)
        except Exception as e:
            results['tests']['forward_performance'] = {'passed': False, 'error': str(e)}
        
        # Test 2: Backward pass performance
        try:
            results['tests']['backward_performance'] = self._test_backward_performance(adapter)
        except Exception as e:
            results['tests']['backward_performance'] = {'passed': False, 'error': str(e)}
        
        # Test 3: Memory usage
        try:
            results['tests']['memory_usage'] = self._test_memory_usage(adapter)
        except Exception as e:
            results['tests']['memory_usage'] = {'passed': False, 'error': str(e)}
        
        # Test 4: Scalability
        try:
            results['tests']['scalability'] = self._test_scalability(adapter)
        except Exception as e:
            results['tests']['scalability'] = {'passed': False, 'error': str(e)}
        
        status = "COMPLETED"
        self._log(f"Performance validation {status}")
        
        return results
    
    def _test_forward_performance(self, adapter) -> Dict[str, Any]:
        """Measure forward pass performance."""
        inputs = adapter.generate_inputs(batch_size=32)
        inputs = {k: v.to(self.config.device).to(self.config.dtype)
                 for k, v in inputs.items()}
        
        # Warmup
        for _ in range(self.config.warmup_iterations):
            _ = adapter.forward(inputs)
        
        if self.config.device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(self.config.num_iterations):
            _ = adapter.forward(inputs)
        
        if self.config.device == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.time() - start_time
        avg_time_ms = (elapsed / self.config.num_iterations) * 1000
        
        return {
            'passed': True,
            'avg_time_ms': avg_time_ms,
            'total_time_ms': elapsed * 1000,
            'iterations': self.config.num_iterations
        }
    
    def _test_backward_performance(self, adapter) -> Dict[str, Any]:
        """Measure backward pass performance."""
        inputs = adapter.generate_inputs(batch_size=32)
        inputs = {k: v.to(self.config.device).to(self.config.dtype).requires_grad_(True)
                 for k, v in inputs.items()}
        
        # Warmup
        for _ in range(self.config.warmup_iterations):
            outputs = adapter.forward(inputs)
            adapter.backward(outputs, inputs)
        
        if self.config.device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(self.config.num_iterations):
            outputs = adapter.forward(inputs)
            adapter.backward(outputs, inputs)
        
        if self.config.device == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.time() - start_time
        avg_time_ms = (elapsed / self.config.num_iterations) * 1000
        
        return {
            'passed': True,
            'avg_time_ms': avg_time_ms,
            'total_time_ms': elapsed * 1000,
            'iterations': self.config.num_iterations
        }
    
    def _test_memory_usage(self, adapter) -> Dict[str, Any]:
        """Measure memory usage."""
        if self.config.device != 'cuda':
            return {
                'passed': True,
                'skipped': True,
                'reason': 'Memory tracking only available on CUDA'
            }
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        inputs = adapter.generate_inputs(batch_size=32)
        inputs = {k: v.to(self.config.device).to(self.config.dtype).requires_grad_(True)
                 for k, v in inputs.items()}
        
        # Forward + backward
        outputs = adapter.forward(inputs)
        adapter.backward(outputs, inputs)
        
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        return {
            'passed': True,
            'peak_memory_mb': peak_memory_mb
        }
    
    def _test_scalability(self, adapter) -> Dict[str, Any]:
        """Test performance scalability with different batch sizes."""
        batch_sizes = [1, 4, 16, 64]
        results = {'passed': True, 'scaling': {}}
        
        for bs in batch_sizes:
            if bs > self.config.max_batch_size:
                continue
                
            inputs = adapter.generate_inputs(batch_size=bs)
            inputs = {k: v.to(self.config.device).to(self.config.dtype)
                     for k, v in inputs.items()}
            
            # Warmup
            for _ in range(5):
                _ = adapter.forward(inputs)
            
            if self.config.device == 'cuda':
                torch.cuda.synchronize()
            
            # Measure
            start_time = time.time()
            for _ in range(20):
                _ = adapter.forward(inputs)
            
            if self.config.device == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = time.time() - start_time
            avg_time_ms = (elapsed / 20) * 1000
            throughput = bs / (avg_time_ms / 1000)  # samples/sec
            
            results['scaling'][f'batch_{bs}'] = {
                'avg_time_ms': avg_time_ms,
                'throughput_samples_per_sec': throughput
            }
        
        return results
