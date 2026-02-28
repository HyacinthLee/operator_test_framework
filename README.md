# Operator Test Framework

A unified testing framework for deep learning operators, supporting correctness verification, numerical stability testing, and performance benchmarking.

## Features

- **Four-layer verification**: Mathematical, numerical, functional, and performance correctness
- **Modular design**: Easy to extend with new operators
- **Comprehensive testing**: Gradient checking, numerical stability, and performance benchmarking
- **PyTorch integration**: Native support for PyTorch tensors and autograd

## Installation

```bash
# Clone the repository
git clone https://github.com/HyacinthLee/operator_test_framework.git
cd operator_test_framework

# Install dependencies
pip install -r requirements.txt

# Optional: Install for flash attention testing
pip install flash-attn
```

## Quick Start

### 1. Test an existing operator

```python
from operator_test_framework.adapters import AttentionAdapter

# Create adapter
adapter = AttentionAdapter(dim=64, num_heads=8)

# Run full test suite
results = adapter.run_full_test_suite()
print(f"Passed: {results['passed']}/{results['total']}")
```

### 2. Create a custom operator adapter

```python
from operator_test_framework.core import OperatorTestAdapter
import torch

class MyCustomOpAdapter(OperatorTestAdapter):
    def __init__(self):
        super().__init__("MyOp", "custom")
    
    def reference_implementation(self, x):
        # PyTorch native implementation
        return torch.nn.functional.relu(x)
    
    def test_implementation(self, x):
        # Your custom implementation
        return my_custom_relu(x)
    
    def generate_test_cases(self):
        return [
            (torch.randn(4, 64),),
            (torch.randn(8, 128, requires_grad=True),),
        ]
    
    def verify_properties(self, output, inputs):
        # Check output is non-negative (ReLU property)
        return (output >= 0).all()
```

### 3. Run tests with pytest

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_operators.py::TestAttentionOperator -v

# Run with coverage
pytest tests/ --cov=operator_test_framework --cov-report=html
```

## Framework Structure

```
operator_test_framework/
├── core/                       # Core testing functionality
│   ├── adapter.py             # Base adapter interface
│   ├── gradient_check.py      # Gradient verification
│   ├── numerical_stability.py # Numerical stability testing
│   └── performance_benchmark.py # Performance benchmarking
├── adapters/                   # Operator-specific adapters
│   ├── attention_adapter.py   # Attention operators
│   └── normalization_adapter.py # Normalization operators
├── tests/                      # Test suite
│   └── test_operators.py      # Example tests
└── examples/                   # Usage examples
```

## Core Components

### OperatorTestAdapter

Base class for implementing operator tests:

```python
class MyAdapter(OperatorTestAdapter):
    def reference_implementation(self, *inputs):
        # Ground truth implementation
        pass
    
    def test_implementation(self, *inputs):
        # Your optimized implementation
        pass
    
    def generate_test_cases(self):
        # Return list of test inputs
        pass
    
    def verify_properties(self, output, inputs):
        # Verify operator-specific properties
        pass
```

### Gradient Checking

```python
from operator_test_framework.core import verify_gradient

# Verify gradient correctness
verify_gradient(
    func=my_operator,
    inputs=test_input,
    rtol=1e-4,
    atol=1e-6
)
```

### Numerical Stability Testing

```python
from operator_test_framework.core import NumericalStabilityTester

tester = NumericalStabilityTester()
results = tester.test_operator(
    operator=my_operator,
    input_shape=(4, 64)
)

tester.print_results(results)
```

### Performance Benchmarking

```python
from operator_test_framework.core import PerformanceBenchmark

benchmark = PerformanceBenchmark(
    warmup_iters=10,
    benchmark_iters=100
)

# Compare implementations
results = benchmark.compare_implementations({
    "implementation_a": op_a,
    "implementation_b": op_b,
})

benchmark.print_comparison(results)
```

## Supported Operators

### Attention
- `AttentionAdapter`: Standard multi-head attention
- `FlashAttentionAdapter`: FlashAttention variants

### Normalization
- `LayerNormAdapter`: Layer normalization
- `RMSNormAdapter`: Root mean square normalization

### Adding New Operators

1. Create a new file in `adapters/`
2. Inherit from `OperatorTestAdapter`
3. Implement required methods
4. Add tests in `tests/`

Example:

```python
# adapters/my_operator_adapter.py
from operator_test_framework.core import OperatorTestAdapter

class MyOperatorAdapter(OperatorTestAdapter):
    def __init__(self, param):
        super().__init__("MyOperator", "custom")
        self.param = param
    
    # ... implement abstract methods
```

## Testing Methodology

### 1. Mathematical Correctness
- Compare with reference implementation
- Verify mathematical properties

### 2. Numerical Stability
- Test with extreme values (large/small)
- Test with NaN/Inf inputs
- Test near-zero variance

### 3. Functional Correctness
- Test boundary conditions
- Test different input shapes
- Test memory layouts

### 4. Performance Correctness
- Measure latency and throughput
- Compare memory usage
- Verify scaling behavior

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## References

This framework is inspired by:
- FlashAttention (Dao et al., 2022)
- PyTorch Testing Best Practices
- NVIDIA cuDNN Documentation

## Contact

For questions or suggestions, please open an issue on GitHub.
