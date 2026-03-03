# Release Notes

## [0.2.0] - 2026-03-03

### 新增

#### 核心模块
- **operator_spec.py** - 算子规格定义
  - `OperatorSpec`: 算子规格类
  - `TensorConstraint`: 张量约束（shape, dtype, device）
  - `Attribute`: 算子属性定义

- **constraint.py** - 约束验证
  - `ShapeConstraint`: 形状约束验证（支持动态维度 -1）
  - `DTypeConstraint`: 数据类型约束验证
  - `DeviceConstraint`: 设备约束验证（支持 cuda:0 格式）

- **test_case.py** - 测试用例
  - `SimpleTestCase`: 简化版测试用例类

- **test_oracle.py** - 测试验证器
  - `NumericalOracle`: 数值比较（支持 rtol, atol）
  - `PropertyOracle`: 属性验证（交换律、结合律、单位元、分配律）
  - `GradientOracle`: 梯度验证（数值梯度检查、Jacobian 验证）

- **test_runner.py** - 测试执行器
  - `TestResult`: 测试结果类
  - `TestRunner`: 测试运行器（支持批量执行）

- **generator.py** - 测试生成器
  - `TestCaseGenerator`: 测试用例生成器（支持随机生成、批量生成）

#### 测试
- **tests/test_operator_spec.py** - 14 个测试
- **tests/test_constraint.py** - 24 个测试
- 所有 38 个测试通过

#### 文档
- **docs/DESIGN.md** - 设计思路文档（分层架构、四层验证、七阶段工作流）
- **RELEASE.md** - 本文件
- **README.md** - 更新使用示例和框架结构

### 变更
- 更新 `core/__init__.py` 统一导出所有新类
- 修复根目录 `__init__.py` 相对导入问题
- 添加 `SimpleTestCase` 别名避免与 workflow 的 `TestCase` 冲突

### 技术细节
- 完整类型注解支持
- 详细文档字符串
- 符合 PEP 8 规范
- 支持 Python 3.10+

## [0.1.0] - 2026-03-02

### 初始版本
- 基础框架结构
- Adapter 接口定义
- 七阶段工作流引擎（workflow/）
- 基础测试工具（gradient_check, memory_leak_detector 等）

---

## 更新指南

当代码发生变化时：

1. **新增模块** - 在 [Unreleased] 或新版本下添加 "新增" 部分
2. **API 变更** - 在 "变更" 部分记录破坏性变更
3. **Bug 修复** - 添加 "修复" 部分
4. **更新版本号** - 遵循语义化版本（MAJOR.MINOR.PATCH）

### 版本号规则
- **MAJOR** - 不兼容的 API 变更
- **MINOR** - 向后兼容的功能添加
- **PATCH** - 向后兼容的问题修复
