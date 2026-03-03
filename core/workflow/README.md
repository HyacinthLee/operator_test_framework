# 七阶段工作流引擎

基于 ATTest 思想的七阶段测试工作流实现。

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    WorkflowEngine                               │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │Understand│→│Requirements│→│Planning│→│Generation│→│Execution│   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
│                                              ↓                  │
│  ┌─────────┐ ┌─────────┐                    ┌─────────┐        │
│  │  Report │←│ Analysis │←───────────────────│  Results │        │
│  └─────────┘ └─────────┘                    └─────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

## 七阶段详解

### 1. Understand (算子理解)

**职责**: 解析算子签名，提取 tensor 约束，识别边界条件

**核心类**:
- `UnderstandPhase` - 阶段主类
- `SignatureParser` - 签名解析器基类
- `PyTorchSignatureParser` - PyTorch 签名解析实现
- `ConstraintExtractor` - 约束提取器

**输出**:
- `OperatorSignature` - 算子签名
- `TensorConstraint` - Tensor 约束
- `boundary_conditions` - 识别的边界条件

### 2. Requirements (需求生成)

**职责**: 生成测试需求（正常、边界、异常），确定覆盖率目标

**核心类**:
- `RequirementsPhase` - 阶段主类
- `TestRequirementGenerator` - 需求生成器基类
- `NormalRequirementGenerator` - 正常测试需求生成器
- `BoundaryRequirementGenerator` - 边界测试需求生成器
- `ExceptionRequirementGenerator` - 异常测试需求生成器

**输出**:
- `List[TestRequirement]` - 测试需求列表

### 3. Planning (测试计划)

**职责**: 设计测试策略，选择生成方法

**核心类**:
- `PlanningPhase` - 阶段主类
- `StrategySelector` - 策略选择器基类
- `HeuristicStrategySelector` - 启发式策略选择器
- `AdaptiveStrategySelector` - 自适应策略选择器
- `TestPlanner` - 测试计划器

**输出**:
- `List[TestPlan]` - 测试计划列表

**策略类型**:
- `RANDOM` - 随机生成
- `BOUNDARY` - 边界值生成
- `SYMBOLIC` - 符号执行
- `HEURISTIC` - 启发式生成
- `ADAPTIVE` - 自适应生成

### 4. Generation (测试生成)

**职责**: 生成具体测试输入，确保约束满足

**核心类**:
- `GenerationPhase` - 阶段主类
- `ConstraintSatisfier` - 约束满足器基类
- `DefaultConstraintSatisfier` - 默认约束满足器
- `InputGenerator` - 输入生成器基类
- `RandomInputGenerator` - 随机输入生成器
- `BoundaryInputGenerator` - 边界输入生成器
- `SymbolicInputGenerator` - 符号输入生成器

**输出**:
- `List[TestCase]` - 测试用例列表

### 5. Execution (执行测试)

**职责**: 运行测试，收集结果和日志

**核心类**:
- `ExecutionPhase` - 阶段主类
- `TestExecutor` - 测试执行器基类
- `DefaultTestExecutor` - 默认测试执行器
- `GradientTestExecutor` - 梯度测试执行器
- `ResultCollector` - 结果收集器

**输出**:
- `List[TestResult]` - 测试结果列表

### 6. Analysis (结果分析)

**职责**: 分类失败原因，生成修复建议

**核心类**:
- `AnalysisPhase` - 阶段主类
- `FailureAnalyzer` - 失败分析器基类
- `RuleBasedFailureAnalyzer` - 基于规则的分析器
- `FixSuggester` - 修复建议器基类
- `RuleBasedFixSuggester` - 基于规则的建议器

**输出**:
- `List[FailureAnalysis]` - 失败分析列表

**失败分类**:
- `SHAPE_MISMATCH` - 形状不匹配
- `DTYPE_MISMATCH` - 数据类型不匹配
- `NAN_INF` - NaN/Inf 错误
- `GRADIENT_ERROR` - 梯度错误
- `MEMORY_ERROR` - 内存错误
- `CRASH` - 崩溃
- `TIMEOUT` - 超时

### 7. Report (报告生成)

**职责**: 生成测试报告，统计覆盖率

**核心类**:
- `ReportPhase` - 阶段主类
- `ReportGenerator` - 报告生成器基类
- `DefaultReportGenerator` - 默认报告生成器
- `CoverageCalculator` - 覆盖率计算器
- `ReportFormatter` - 报告格式化器

**输出**:
- `TestReport` - 测试报告

**覆盖率指标**:
- 行覆盖率
- 分支覆盖率
- 边界覆盖率
- 数据类型覆盖率
- 形状覆盖率
- 整体覆盖率

## 数据传输格式

各阶段间通过 `WorkflowState` 传递数据:

```python
@dataclass
class WorkflowState:
    operator: Any                    # 算子适配器
    signature: OperatorSignature     # 算子签名 (阶段1输出)
    requirements: List[TestRequirement]  # 测试需求 (阶段2输出)
    plans: List[TestPlan]            # 测试计划 (阶段3输出)
    test_cases: List[TestCase]       # 测试用例 (阶段4输出)
    results: List[TestResult]        # 测试结果 (阶段5输出)
    analyses: List[FailureAnalysis]  # 失败分析 (阶段6输出)
    report: TestReport               # 测试报告 (阶段7输出)
```

## 使用方式

### 基础用法

```python
from core import WorkflowEngine, WorkflowState, run_full_workflow
from adapter import OperatorTestAdapter

# 创建适配器
adapter = MyOperatorAdapter()

# 方式1: 使用便捷函数
state = run_full_workflow(adapter)

# 方式2: 使用引擎
engine = WorkflowEngine()
state = WorkflowState(operator=adapter)
final_state = engine.run(state)

# 查看报告
print(engine.export_report(final_state, format_type="text"))
```

### 自定义配置

```python
from core import WorkflowConfig

config = WorkflowConfig(
    enable_understand=True,
    enable_requirements=True,
    enable_planning=True,
    enable_generation=True,
    enable_execution=True,
    enable_analysis=True,
    enable_report=True,
    stop_on_error=False,  # 错误时继续执行
    collect_metrics=True,
)

engine = WorkflowEngine(config=config)
```

### 增量执行

```python
from core import IncrementalWorkflowEngine

engine = IncrementalWorkflowEngine()

# 执行前几个阶段
state = WorkflowState(operator=adapter)
for phase in ["understand", "requirements", "planning"]:
    state = engine.run_phase(phase, state)

# 缓存状态
engine.cache_phase_state("planning", state)

# 从缓存恢复继续执行
restored = engine.restore_phase_state("planning")
final_state = engine.run_from_phase(restored, "generation")
```

### 部分执行

```python
# 跳过某些阶段
config = WorkflowConfig(
    enable_understand=True,
    enable_requirements=True,
    enable_planning=False,  # 跳过
    enable_generation=True,
    enable_execution=True,
    enable_analysis=False,  # 跳过
    enable_report=True,
)

engine = WorkflowEngine(config=config)
```

### 进度回调

```python
def progress_callback(phase_name, current, total):
    print(f"[{current}/{total}] 执行阶段: {phase_name}")

engine = WorkflowEngine(progress_callback=progress_callback)
```

## 扩展开发

### 自定义阶段

```python
from core.workflow import PhaseExecutor, WorkflowState, PhaseContext

class CustomPhase(PhaseExecutor):
    def execute(self, state: WorkflowState) -> Tuple[WorkflowState, PhaseContext]:
        context = PhaseContext(phase_name="custom")
        context.start_time = datetime.now()
        
        # 自定义逻辑
        # ...
        
        context.end_time = datetime.now()
        context.success = True
        return state, context

# 使用自定义阶段
phases = {
    "understand": UnderstandPhase(),
    "requirements": RequirementsPhase(),
    "custom": CustomPhase(),
    # ...
}
engine = WorkflowEngine(phases=phases)
```

### 自定义策略选择器

```python
from core.workflow import StrategySelector, GenerationStrategy

class MyStrategySelector(StrategySelector):
    def select(self, requirement, signature):
        # 自定义策略选择逻辑
        if requirement.priority >= 4:
            return GenerationStrategy.BOUNDARY
        return GenerationStrategy.RANDOM

# 使用自定义选择器
planning_phase = PlanningPhase(strategy_selector=MyStrategySelector())
```

## 文件结构

```
core/workflow/
├── __init__.py          # 模块导出
├── data_models.py       # 数据传输格式定义
├── understand.py        # 阶段1: 算子理解
├── requirements.py      # 阶段2: 需求生成
├── planning.py          # 阶段3: 测试计划
├── generation.py        # 阶段4: 测试生成
├── execution.py         # 阶段5: 执行测试
├── analysis.py          # 阶段6: 结果分析
├── report.py            # 阶段7: 报告生成
├── engine.py            # 工作流引擎主类
└── README.md            # 本文档
```

## 与四层验证体系的关系

七阶段工作流引擎与原有的四层验证体系（validators.py）可以协同工作:

- **理解阶段** → 为数学验证提供算子签名
- **需求阶段** → 为数值验证生成边界条件
- **执行阶段** → 调用四层验证器进行测试
- **分析阶段** → 整合四层验证的结果

```python
# 在 execution 阶段使用四层验证
from validators import MathematicalValidator, NumericalValidator

def execute_with_validators(test_case, adapter):
    # 数学验证
    math_validator = MathematicalValidator(config)
    math_result = math_validator.validate(adapter)
    
    # 数值验证
    num_validator = NumericalValidator(config)
    num_result = num_validator.validate(adapter)
    
    # 整合结果
    return combine_results(math_result, num_result)
```

## 设计原则

1. **单一职责**: 每个阶段只负责一个明确的任务
2. **可扩展性**: 通过抽象基类支持自定义实现
3. **可配置性**: 通过配置对象灵活控制行为
4. **可追溯性**: 完整记录各阶段的执行状态和结果
5. **容错性**: 支持错误处理和继续执行

## 未来扩展

- [ ] 支持并行执行测试用例
- [ ] 添加更多符号执行支持
- [ ] 集成机器学习模型进行智能分析
- [ ] 支持分布式测试执行
- [ ] 添加可视化报告生成功能
