# 设计思路

## 核心设计理念

### 1. 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│  应用层 (Application)                                        │
│  - 测试脚本                                                  │
│  - CI/CD 集成                                                │
│  - 报告生成                                                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  核心层 (Core)                                               │
│  - operator_spec: 算子规格定义                               │
│  - constraint: 约束验证                                      │
│  - test_case: 测试用例                                       │
│  - test_oracle: 测试验证器                                   │
│  - test_runner: 测试执行器                                   │
│  - generator: 测试生成器                                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  适配层 (Adapter)                                            │
│  - OperatorTestAdapter: 统一接口                             │
│  - Megatron-LM Adapter                                       │
│  - TransformerEngine Adapter                                 │
│  - PyTorch Native Adapter                                    │
└─────────────────────────────────────────────────────────────┘
```

### 2. 四层验证体系

| 层级 | 验证内容 | 核心类 |
|------|----------|--------|
| 数学正确性 | 算法逻辑、数学等价性 | PropertyOracle |
| 数值正确性 | 精度、溢出、下溢 | NumericalOracle |
| 功能正确性 | 输入输出、边界条件 | ShapeConstraint, DTypeConstraint |
| 性能正确性 | 内存、速度、并行效率 | PerformanceBenchmark |

### 3. 七阶段工作流 (ATTest 思想)

基于 ATTest 论文的七阶段测试生成方法：

1. **Understand (理解)** - 解析算子签名，提取约束
2. **Requirements (需求)** - 生成测试需求，确定覆盖率目标
3. **Planning (计划)** - 设计测试策略，选择生成方法
4. **Generation (生成)** - 生成测试输入，确保约束满足
5. **Execution (执行)** - 运行测试，收集结果
6. **Analysis (分析)** - 分类失败原因，生成修复建议
7. **Report (报告)** - 生成测试报告，统计覆盖率

### 4. 关键设计决策

#### 4.1 类型注解
- 所有公共 API 都有完整的类型注解
- 支持 Python 3.10+ 的新特性（如 `dict[str, Any]`）

#### 4.2 可扩展性
- Adapter 模式支持不同框架（PyTorch, Megatron-LM, TransformerEngine）
- Oracle 模式支持不同验证策略（数值、属性、梯度）

#### 4.3 随机性与可复现性
- TestCaseGenerator 支持随机种子设置
- 确保测试结果可复现

#### 4.4 错误处理
- 详细的错误信息和上下文
- 异常不会导致整个测试流程中断

## 代码组织

```
core/
├── operator_spec.py     # 算子规格定义
├── constraint.py        # 约束验证
├── test_case.py         # 测试用例
├── test_oracle.py       # 测试验证器
├── test_runner.py       # 测试执行器
├── generator.py         # 测试生成器
└── workflow/            # 七阶段工作流引擎
```

## 使用模式

### 模式 1: 简单测试
```python
from core import SimpleTestCase, TestRunner

tc = SimpleTestCase('test_add', {'x': 1, 'y': 2}, {'out': 3})
runner = TestRunner()
result = runner.run(tc, add_operator)
```

### 模式 2: 约束验证
```python
from core import ShapeConstraint, DTypeConstraint

shape_c = ShapeConstraint((2, -1, 3))  # -1 表示动态维度
dtype_c = DTypeConstraint(['float32', 'float64'])
```

### 模式 3: 随机测试生成
```python
from core import TestCaseGenerator

gen = TestCaseGenerator(seed=42)
test_cases = gen.generate_batch(
    operator_spec=my_spec,
    num_cases=100
)
```

### 模式 4: 属性验证
```python
from core import PropertyOracle

oracle = PropertyOracle()
is_commutative = oracle.check_commutative(add_op, a, b)
```

## 未来扩展

- [ ] 支持更多算子类型（稀疏算子、图算子）
- [ ] 集成更多框架（JAX, MindSpore）
- [ ] 分布式测试支持
- [ ] 可视化测试报告
