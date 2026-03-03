# Operator Test Framework v2 - Design Document

## Overview

This document describes the architecture and design of `operator_test_framework_v2`, a next-generation testing framework for deep learning operators based on the ATTest paper's seven-stage workflow and agent-driven architecture.

## Design Goals

1. **Agent-Driven**: Leverage LLM agents for autonomous test generation and repair
2. **Seven-Stage Workflow**: Implement the complete ATTest workflow
3. **Constraint-Aware**: Extract and utilize tensor constraints from operator APIs
4. **Iterative Repair**: Support generation-validation-repair loops
5. **Extensible**: Modular design for easy extension
6. **Type-Safe**: Full type annotations and validation

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TestFramework (Orchestrator)                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌──────────────┐          ┌──────────────┐          ┌──────────────┐
│  LLM Client  │          │   Workflow   │          │   Framework  │
│              │          │   Stages     │          │   State      │
└──────────────┘          └──────────────┘          └──────────────┘
                                    │
    ┌───────────┬───────────┬───────┴───────┬───────────┬───────────┐
    ▼           ▼           ▼               ▼           ▼           ▼
┌───────┐  ┌───────┐  ┌───────┐       ┌───────┐  ┌───────┐  ┌───────┐
│ Stage │  │ Stage │  │ Stage │  ...  │ Stage │  │ Stage │  │ Stage │
│   1   │  │   2   │  │   3   │       │   5   │  │   6   │  │   7   │
│ Under-│  │ Require-│  │Planning│      │Execution│  │Analysis│  │ Report│
│ stand │  │ ments  │  │       │       │       │  │       │  │       │
└───────┘  └───────┘  └───────┘       └───────┘  └───────┘  └───────┘
    │           │           │               │           │           │
    ▼           ▼           ▼               ▼           ▼           ▼
┌───────┐  ┌───────┐  ┌───────┐       ┌───────┐  ┌───────┐  ┌───────┐
│ Agent │  │ Agent │  │  Plan │       │Executor│  │Failure│  │Report │
│Extract│  │Generate│  │Optimizer│     │       │  │Analyzer│  │Renderer│
└───────┘  └───────┘  └───────┘       └───────┘  └───────┘  └───────┘
```

### Data Flow

```
Operator API → Understand → Requirements → Planning → Generation → Execution → Analysis → Report
                   │              │             │            │           │           │        │
                   ▼              ▼             ▼            ▼           ▼           ▼        ▼
              OperatorSpec   TestReqs    TestPlan    TestSuite   TestResults  Analysis  TestReport
```

## Core Components

### 1. Data Models (`core/models/`)

#### OperatorSpec

Central data model describing operator interfaces:

```python
@dataclass
class OperatorSpec:
    name: str                    # Operator name
    domain: str                  # Framework (pytorch, onnx, etc.)
    inputs: List[InputSpec]      # Input specifications
    outputs: List[OutputSpec]    # Output specifications
    attributes: List[OperatorAttribute]  # Parameters
```

#### TensorConstraint

Comprehensive constraint specification:

```python
@dataclass
class TensorConstraint:
    name: str
    shape: ShapeConstraint       # Shape constraints
    dtype: DtypeConstraint       # Data type constraints
    device: DeviceConstraint     # Device constraints
    values: ValueConstraint      # Value constraints
```

#### TestCase & TestOracle

Test case definition and verification criteria:

```python
@dataclass
class TestCase:
    id: str
    inputs: Dict[str, Any]
    oracles: List[TestOracle]
    
@dataclass
class TestOracle:
    oracle_type: OracleType
    tolerance: Dict[str, float]
```

### 2. Seven-Stage Workflow (`core/workflow/`)

#### Stage 1: Understand

**Purpose**: Parse operator API and extract constraints

**Inputs**: Operator API (function, docstring, type hints)
**Outputs**: `OperatorSpec`, shape relationships, constraints

**Key Components**:
- `ConstraintExtractorAgent`: LLM-based extraction
- `StaticConstraintExtractor`: Static analysis extraction
- `ShapeRelationshipAnalyzer`: Infer shape relationships

**Process**:
1. Extract signature from API
2. Parse docstring for constraints
3. Analyze type hints
4. Infer shape relationships
5. Build OperatorSpec

#### Stage 2: Requirements

**Purpose**: Generate comprehensive test requirements

**Inputs**: `OperatorSpec`
**Outputs**: `TestRequirements`

**Requirement Types**:
- Functional correctness
- Numerical stability
- Boundary conditions
- Error handling
- Performance characteristics

**Key Components**:
- `RequirementGeneratorAgent`: LLM-based generation
- `ConstraintBasedGenerator`: Rule-based generation
- `HistoricalPatternMatcher`: Apply known bug patterns

#### Stage 3: Planning

**Purpose**: Design test strategy

**Inputs**: `TestRequirements`
**Outputs**: `TestPlan`

**Strategies**:
- `COMPREHENSIVE`: Test everything
- `RISK_BASED`: Focus on high-risk areas
- `QUICK_SMOKE`: Fast sanity check
- `PERFORMANCE`: Focus on performance

**Key Components**:
- `TestPlanOptimizer`: Optimize plan for cost/coverage
- `RequirementSelector`: Select requirements based on strategy
- `StrategyAssigner`: Assign generation strategies

#### Stage 4: Generation

**Purpose**: Generate concrete test cases

**Inputs**: `TestPlan`, `OperatorSpec`
**Outputs**: `TestSuite`

**Generation Strategies**:
- `RandomGenerator`: Random valid inputs
- `BoundaryGenerator`: Edge cases and boundaries
- `SymbolicGenerator`: Symbolic/concolic test cases
- `LLMTestGenerator`: LLM-based generation

**Iterative Repair**:
```
Generate → Validate → (Invalid?) → Repair → Validate
```

#### Stage 5: Execution

**Purpose**: Execute test cases

**Inputs**: `TestSuite`, implementation
**Outputs**: `TestResults`

**Features**:
- Parallel execution
- Timeout handling
- Resource limits
- Oracle verification

**Key Components**:
- `TestExecutor`: Execute individual tests
- `OracleApplier`: Apply verification oracles
- `MetricsCollector`: Collect execution metrics

#### Stage 6: Analysis

**Purpose**: Analyze results and identify issues

**Inputs**: `TestResults`
**Outputs**: `AnalysisResult`

**Analysis Types**:
- Failure pattern clustering
- Root cause analysis
- Coverage assessment
- Risk evaluation

**Key Components**:
- `FailureAnalyzer`: Identify failure patterns
- `RootCauseAnalyzer`: Determine root causes
- `RepairSuggester`: Generate repair suggestions

#### Stage 7: Report

**Purpose**: Generate comprehensive reports

**Inputs**: All previous outputs
**Outputs**: `TestReport`

**Formats**:
- Markdown
- HTML
- JSON
- JUnit XML
- PDF

**Sections**:
- Executive summary
- Detailed results
- Failure analysis
- Coverage metrics
- Recommendations

### 3. Agents (`core/agents/`)

Agents are LLM-based autonomous components:

#### ConstraintExtractorAgent
- Extracts constraints from API definitions
- Uses docstring analysis
- Applies domain knowledge

#### RequirementGeneratorAgent
- Generates comprehensive test requirements
- Uses historical bug patterns
- Prioritizes by risk

#### TestGeneratorAgent
- Generates concrete test cases
- Combines LLM and programmatic generation
- Validates and repairs cases

#### RepairAgent
- Diagnoses test failures
- Generates repairs for tests/implementations
- Supports iterative repair loops

### 4. Generators (`core/generators/`)

#### RandomGenerator
Generates random inputs satisfying constraints:
- Configurable distributions
- Shape sampling
- Value generation

#### BoundaryGenerator
Generates boundary value test cases:
- Dimension boundaries (0, 1, small, large)
- Value boundaries (min, max, zero, inf, nan)
- Shape boundaries (empty, single element)

#### SymbolicGenerator
Uses symbolic execution:
- Path exploration
- Constraint solving
- SMT solver integration

### 5. Validators & Oracles (`core/validators/`)

Oracles verify test correctness:

- `ExactMatchOracle`: Bit-exact comparison
- `ApproximateMatchOracle`: Within tolerance
- `ReferenceImplOracle`: Compare with reference
- `PropertyBasedOracle`: Verify properties
- `ShapeOracle`: Verify output shape
- `GradientOracle`: Verify gradients

## Key Design Patterns

### 1. Generation-Validation-Repair Loop

```python
for test_case in generated_cases:
    if not validate(test_case):
        repaired = repair(test_case)
        if validate(repaired):
            use(repaired)
```

### 2. ReAct Agent Pattern

```python
while not done:
    thought = agent.think(observation)
    action = agent.act(thought)
    observation = environment.execute(action)
```

### 3. Workflow Context

Shared context maintains state across stages:

```python
@dataclass
class WorkflowContext:
    operator_spec: OperatorSpec
    test_requirements: TestRequirements
    test_plan: TestPlan
    test_suite: TestSuite
    test_results: TestResults
    analysis_result: AnalysisResult
    report: TestReport
```

### 4. Strategy Pattern

Pluggable strategies for:
- Test generation
- Test planning
- Constraint solving
- Failure analysis

## Extension Points

### Adding a New Generator

```python
class MyGenerator(TestGenerator):
    @property
    def name(self) -> str:
        return "my_generator"
    
    def generate(self, spec, constraint, count):
        # Implementation
        return test_cases
```

### Adding a New Workflow Stage

```python
class MyStage(WorkflowStage):
    @property
    def name(self) -> str:
        return "my_stage"
    
    def execute(self, context):
        # Implementation
        return StageResult(...)
```

### Adding a New Oracle

```python
class MyOracle(TestOracle):
    def verify(self, actual, expected, inputs):
        # Implementation
        return OracleResult(...)
```

## Type Safety

All components use type annotations:

```python
def generate(
    self,
    operator_spec: OperatorSpec,
    constraint: TensorConstraint,
    count: int,
) -> GenerationResult:
    ...
```

Type checking with mypy is enforced.

## Configuration

Configuration hierarchy:
1. Default config
2. Config file
3. Environment variables
4. Runtime parameters

```python
@dataclass
class FrameworkConfig:
    llm: LLMConfig
    generation: GenerationConfig
    execution: ExecutionConfig
    reporting: ReportingConfig
```

## Testing Philosophy

1. **Comprehensive Coverage**: All code paths, edge cases, and error conditions
2. **Numerical Stability**: Test with extreme values, NaN, Inf
3. **Shape Flexibility**: Test various input shapes
4. **Dtype Coverage**: Test all supported data types
5. **Property-Based**: Verify mathematical properties

## Future Enhancements

1. **Fuzzing Integration**: Property-based fuzzing with Hypothesis
2. **Mutation Testing**: Evaluate test suite quality
3. **Continuous Learning**: Learn from historical results
4. **Multi-Framework**: Support TensorFlow, JAX, ONNX
5. **Distributed Execution**: Scale test execution

## References

- ATTest Paper: Agent-driven Testing Framework
- PyTorch Testing Best Practices
- Property-Based Testing (Hypothesis)
- Symbolic Execution Techniques
