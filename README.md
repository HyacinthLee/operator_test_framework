# Operator Test Framework

A unified testing framework for deep learning operators, supporting correctness verification, numerical stability testing, and performance benchmarking.

## Features

- **Four-layer verification**: Mathematical, numerical, functional, and performance correctness
- **Modular design**: Easy to extend with new operators
- **Comprehensive testing**: Gradient checking, numerical stability, and performance benchmarking
- **PyTorch integration**: Native support for PyTorch tensors and autograd
- **ATTest-based workflow**: Seven-stage test generation (Understand → Requirements → Planning → Generation → Execution → Analysis → Report)

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

### 1. Simple Test Case

```python
from core import SimpleTestCase, TestRunner

# Create test case
tc = SimpleTestCase(
    name='test_add',
    inputs={'x': 1, 'y': 2},
    expected_outputs={'out': 3}
)

# Run test
runner = TestRunner()
result = runner.run(tc, add_operator)
print(f"Passed: {result.passed}, Message: {result.message}")
```

### 2. Constraint Validation

```python
from core import ShapeConstraint, DTypeConstraint, DeviceConstraint

# Define constraints
shape_c = ShapeConstraint((2, -1, 3))  # -1 for dynamic dimension
dtype_c = DTypeConstraint(['float32', 'float64'])
device_c = DeviceConstraint(['cpu', 'cuda'])

# Validate
assert shape_c.validate((2, 4, 3)) == True
assert dtype_c.validate('float32') == True
assert device_c.validate('cuda:0') == True
```

### 3. Random Test Generation

```python
from core import TestCaseGenerator, OperatorSpec, TensorConstraint

# Create generator with fixed seed
gen = TestCaseGenerator(seed=42)

# Generate test cases
test_cases = gen.generate_batch(
    operator_spec=my_spec,
    num_cases=100
)
```

### 4. Property Verification

```python
from core import PropertyOracle

oracle = PropertyOracle()

# Check commutativity: op(a, b) == op(b, a)
is_commutative = oracle.check_commutative(add_op, a, b)

# Check associativity: op(op(a, b), c) == op(a, op(b, c))
is_associative = oracle.check_associative(add_op, a, b, c)
```

### 5. Test an existing operator

```python
from operator_test_framework.adapters import AttentionAdapter

# Create adapter
adapter = AttentionAdapter(dim=64, num_heads=8)

# Run full test suite
results = adapter.run_full_test_suite()
print(f"Passed: {results['passed']}/{results['total']}")
```

## Framework Structure

```
operator_test_framework/
├── core/                       # Core testing functionality
│   ├── operator_spec.py       # Operator specification (OperatorSpec, TensorConstraint, Attribute)
│   ├── constraint.py          # Constraint validation (Shape, DType, Device)
│   ├── test_case.py           # Test case definition
│   ├── test_oracle.py         # Test oracles (Numerical, Property, Gradient)
│   ├── test_runner.py         # Test execution engine
│   ├── generator.py           # Test case generator
│   ├── adapter.py             # Base adapter interface
│   ├── gradient_check.py      # Gradient verification
│   ├── numerical_stability.py # Numerical stability testing
│   ├── performance_benchmark.py # Performance benchmarking
│   └── workflow/              # Seven-stage ATTest workflow
├── adapters/                   # Operator-specific adapters
│   ├── attention_adapter.py   # Attention operators
│   └── normalization_adapter.py # Normalization operators
├── tests/                      # Test suite
│   ├── test_operator_spec.py  # Core class tests
│   ├── test_constraint.py     # Constraint tests
│   └── test_operators.py      # Example tests
├── docs/                       # Documentation
│   └── DESIGN.md              # Design philosophy
├── examples/                   # Usage examples
└── RELEASE.md                  # Release notes
```

## Core Components

### Operator Specification

Define operator specifications with constraints:

```python
from core import OperatorSpec, TensorConstraint, Attribute

spec = OperatorSpec(
    name='layer_norm',
    inputs={
        'x': TensorConstraint(shape=(-1, -1), dtype='float32', device='cuda'),
        'eps': Attribute('eps', float, 1e-5)
    }
)
```

### Constraint Validation

```python
from core import ShapeConstraint, DTypeConstraint

# Shape with dynamic dimensions
shape_c = ShapeConstraint((2, -1, 3))
assert shape_c.validate((2, 4, 3)) == True  # Valid
assert shape_c.validate((3, 4, 3)) == False  # Invalid: first dim mismatch

# Data type constraint
dtype_c = DTypeConstraint(['float32', 'float64'])
assert dtype_c.validate('float32') == True
assert dtype_c.validate('int32') == False
```

### Test Oracle

```python
from core import NumericalOracle, PropertyOracle, GradientOracle

# Numerical comparison with tolerance
numerical = NumericalOracle(rtol=1e-5, atol=1e-8)
assert numerical.compare(1.0, 1.000001) == True

# Property verification
property_oracle = PropertyOracle()
assert property_oracle.check_commutative(add_op, a, b) == True

# Gradient checking
gradient = GradientOracle()
assert gradient.check_gradient(my_op, inputs, grad_outputs) == True
```

### Test Runner

```python
from core import SimpleTestCase, TestRunner

# Single test
tc = SimpleTestCase('test', inputs, expected)
runner = TestRunner()
result = runner.run(tc, operator_func)

# Batch tests
test_cases = [tc1, tc2, tc3]
results = runner.run_batch(test_cases, operator_func)
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

## Testing Methodology

### 1. Mathematical Correctness
- Compare with reference implementation
- Verify mathematical properties (commutativity, associativity, etc.)

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

## Documentation

- [Design Philosophy](docs/DESIGN.md) - Core design principles and architecture
- [Release Notes](RELEASE.md) - Version history and changelog

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update RELEASE.md with your changes
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## References

This framework is inspired by:
- FlashAttention (Dao et al., 2022)
- PyTorch Testing Best Practices
- NVIDIA cuDNN Documentation
- ATTest: Automated Testing for Deep Learning Operators
