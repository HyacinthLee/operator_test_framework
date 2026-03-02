# Operator Test Framework v2

A next-generation testing framework for deep learning operators based on the ATTest paper's seven-stage workflow and agent-driven architecture.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    TestFramework (Orchestrator)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
    ┌─────────────────────────┼─────────────────────────┐
    ▼                         ▼                         ▼
┌────────┐              ┌──────────┐              ┌──────────┐
│  LLM   │              │ Workflow │              │   Core   │
│ Client │              │  Stages  │              │  Models  │
└────────┘              └──────────┘              └──────────┘
                              │
    ┌─────┬─────┬─────┬─────┼─────┬─────┬─────┐
    ▼     ▼     ▼     ▼     ▼     ▼     ▼
┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐
│Under││ Req ││Plan ││ Gen ││Exec ││Anlz ││Report│
│stand││uire ││ning ││erate││ute  ││yze  ││      │
└─────┘└─────┘└─────┘└─────┘└─────┘└─────┘└─────┘
```

## Seven-Stage Workflow

### Stage 1: Understand
Parse operator API and extract tensor constraints.

**Key Components:**
- `ConstraintExtractorAgent`: LLM-based constraint extraction
- `StaticConstraintExtractor`: Static analysis extraction

### Stage 2: Requirements
Generate comprehensive test requirements.

**Key Components:**
- `RequirementGeneratorAgent`: LLM-based requirement generation
- `ConstraintBasedGenerator`: Rule-based generation

### Stage 3: Planning
Design test strategy and resource allocation.

**Key Components:**
- `TestPlanOptimizer`: Optimize for cost/coverage
- `RequirementSelector`: Select requirements based on strategy

### Stage 4: Generation
Generate concrete test cases.

**Generators:**
- `RandomGenerator`: Random valid inputs
- `BoundaryGenerator`: Edge cases and boundaries
- `SymbolicGenerator`: Symbolic/concolic test cases

### Stage 5: Execution
Execute test cases and collect results.

**Key Components:**
- `TestExecutor`: Execute individual tests
- `OracleApplier`: Apply verification oracles

### Stage 6: Analysis
Analyze results and identify issues.

**Key Components:**
- `FailureAnalyzer`: Identify failure patterns
- `RootCauseAnalyzer`: Determine root causes
- `RepairSuggester`: Generate repair suggestions

### Stage 7: Report
Generate comprehensive reports.

**Formats:**
- Markdown
- HTML
- JSON
- JUnit XML
- PDF

## Core Data Models

### OperatorSpec
Complete operator specification including inputs, outputs, and attributes.

```python
@dataclass
class OperatorSpec:
    name: str
    domain: str
    inputs: List[InputSpec]
    outputs: List[OutputSpec]
    attributes: List[OperatorAttribute]
```

### TensorConstraint
Comprehensive constraint specification for tensors.

```python
@dataclass
class TensorConstraint:
    name: str
    shape: ShapeConstraint
    dtype: DtypeConstraint
    device: DeviceConstraint
    values: ValueConstraint
```

### TestCase
Individual test case with inputs and oracles.

```python
@dataclass
class TestCase:
    id: str
    name: str
    inputs: Dict[str, Any]
    oracles: List[TestOracle]
```

## Usage Example

```python
from operator_test_framework_v2 import TestFramework

# Initialize framework
framework = TestFramework()

# Test an operator
results = framework.test_operator(
    operator_name="torch.nn.functional.softmax",
    implementation=my_softmax_impl,
    reference_impl=torch.softmax
)

# Access results
print(results.context.report.summary)
print(f"Pass rate: {results.context.analysis_result.pass_rate}")
```

## Directory Structure

```
operator_test_framework_v2/
├── __init__.py
├── DESIGN.md                 # This design document
├── requirements.txt
├── py.typed                  # PEP 561 type marker
├── core/
│   ├── __init__.py
│   ├── framework.py          # Main TestFramework
│   ├── config.py             # Configuration classes
│   ├── models/               # Data models
│   │   ├── operator_spec.py
│   │   ├── tensor_constraint.py
│   │   └── test_case.py
│   ├── workflow/             # Seven-stage workflow
│   │   ├── base.py
│   │   ├── understand.py
│   │   ├── requirements.py
│   │   ├── planning.py
│   │   ├── generation.py
│   │   ├── execution.py
│   │   ├── analysis.py
│   │   └── report.py
│   ├── generators/           # Test generators
│   │   ├── random_generator.py
│   │   ├── boundary_generator.py
│   │   └── symbolic_generator.py
│   ├── agents/               # LLM agents
│   │   ├── base.py
│   │   ├── constraint_agent.py
│   │   ├── requirement_agent.py
│   │   ├── test_generator_agent.py
│   │   └── repair_agent.py
│   └── validators/           # Test oracles
│       ├── oracle.py
│       ├── numerical_validator.py
│       └── shape_validator.py
├── llm/                      # LLM clients
│   ├── client.py
│   ├── openai_client.py
│   └── anthropic_client.py
└── utils/                    # Utilities
    ├── tensor_utils.py
    ├── shape_utils.py
    └── logging_utils.py
```

## Key Features

1. **Agent-Driven**: LLM agents for autonomous test generation
2. **Constraint-Aware**: Tensor constraint extraction and validation
3. **Iterative Repair**: Generation-validation-repair loops
4. **Multiple Generators**: Random, boundary, symbolic
5. **Comprehensive Oracles**: Exact, approximate, property-based
6. **Type-Safe**: Full type annotations

## Extension Points

### Custom Generator
```python
from operator_test_framework_v2.core.generators import TestGenerator

class MyGenerator(TestGenerator):
    @property
    def name(self) -> str:
        return "my_generator"
    
    def generate(self, spec, constraint, count):
        # Implementation
        return test_cases
```

### Custom Oracle
```python
from operator_test_framework_v2.core.validators import TestOracle

class MyOracle(TestOracle):
    def verify(self, actual, expected, inputs):
        # Implementation
        return OracleResult(...)
```

## Configuration

```python
from operator_test_framework_v2.core.config import FrameworkConfig

config = FrameworkConfig(
    llm=LLMConfig(model="gpt-4", temperature=0.7),
    generation=GenerationConfig(num_random_cases=20),
    execution=ExecutionConfig(max_workers=4),
)

framework = TestFramework(config=config)
```

## References

- ATTest Paper: Agent-driven Testing Framework
- PyTorch Testing Best Practices
