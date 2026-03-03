"""
七阶段工作流的数据传输格式定义

定义了各阶段间传递的数据结构，确保类型安全和数据一致性。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Tuple, Callable, Union, Set
import torch
import torch.nn as nn


class TestType(Enum):
    """测试类型枚举"""
    NORMAL = auto()           # 正常测试
    BOUNDARY = auto()         # 边界测试
    EXCEPTION = auto()        # 异常测试
    RANDOM = auto()           # 随机测试
    SYMBOLIC = auto()         # 符号测试
    STRESS = auto()           # 压力测试
    REGRESSION = auto()       # 回归测试


class GenerationStrategy(Enum):
    """输入生成策略枚举"""
    RANDOM = auto()           # 随机生成
    BOUNDARY = auto()         # 边界值生成
    SYMBOLIC = auto()         # 符号执行
    HEURISTIC = auto()        # 启发式生成
    EXHAUSTIVE = auto()       # 穷举生成
    ADAPTIVE = auto()         # 自适应生成


class FailureCategory(Enum):
    """失败分类枚举"""
    SHAPE_MISMATCH = auto()       # 形状不匹配
    DTYPE_MISMATCH = auto()       # 数据类型不匹配
    DEVICE_MISMATCH = auto()      # 设备不匹配
    VALUE_ERROR = auto()          # 数值错误
    NAN_INF = auto()              # NaN/Inf 错误
    GRADIENT_ERROR = auto()       # 梯度错误
    PERFORMANCE_REGRESSION = auto()  # 性能回归
    MEMORY_ERROR = auto()         # 内存错误
    CRASH = auto()                # 崩溃
    TIMEOUT = auto()              # 超时
    UNKNOWN = auto()              # 未知错误


@dataclass
class TensorConstraint:
    """Tensor 约束定义
    
    Attributes:
        name: Tensor 名称
        shape: 形状约束 (可以是具体值、范围或通配符)
        dtype: 数据类型约束
        device: 设备约束
        requires_grad: 是否需要梯度
        min_value: 最小值约束
        max_value: 最大值约束
        additional_constraints: 额外约束条件
    """
    name: str
    shape: Union[Tuple[int, ...], List[int], str] = "*"
    dtype: Optional[torch.dtype] = None
    device: Optional[str] = None
    requires_grad: Optional[bool] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    additional_constraints: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        return {
            "name": self.name,
            "shape": self.shape if isinstance(self.shape, (list, tuple, str)) else list(self.shape),
            "dtype": str(self.dtype) if self.dtype else None,
            "device": self.device,
            "requires_grad": self.requires_grad,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "additional_constraints": self.additional_constraints,
        }


@dataclass
class OperatorSignature:
    """算子签名定义
    
    Attributes:
        name: 算子名称
        inputs: 输入 Tensor 约束列表
        outputs: 输出 Tensor 约束列表
        parameters: 算子参数定义
        attributes: 算子属性定义
        is_in_place: 是否为原地操作
        supports_autograd: 是否支持自动微分
        backend: 后端类型 (pytorch, custom, etc.)
    """
    name: str
    inputs: List[TensorConstraint] = field(default_factory=list)
    outputs: List[TensorConstraint] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    is_in_place: bool = False
    supports_autograd: bool = True
    backend: str = "pytorch"
    
    # 边界条件识别
    boundary_conditions: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        return {
            "name": self.name,
            "inputs": [i.to_dict() for i in self.inputs],
            "outputs": [o.to_dict() for o in self.outputs],
            "parameters": self.parameters,
            "attributes": self.attributes,
            "is_in_place": self.is_in_place,
            "supports_autograd": self.supports_autograd,
            "backend": self.backend,
            "boundary_conditions": self.boundary_conditions,
        }


@dataclass
class TestRequirement:
    """测试需求定义
    
    Attributes:
        id: 需求唯一标识
        test_type: 测试类型
        description: 需求描述
        priority: 优先级 (1-5, 5为最高)
        coverage_target: 覆盖率目标 (0-1)
        constraints: 约束条件
        derived_from: 来源 (如边界条件索引)
    """
    id: str
    test_type: TestType
    description: str
    priority: int = 3
    coverage_target: float = 0.0
    constraints: Dict[str, Any] = field(default_factory=dict)
    derived_from: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        return {
            "id": self.id,
            "test_type": self.test_type.name,
            "description": self.description,
            "priority": self.priority,
            "coverage_target": self.coverage_target,
            "constraints": self.constraints,
            "derived_from": self.derived_from,
        }


@dataclass
class TestPlan:
    """测试计划定义
    
    Attributes:
        id: 计划唯一标识
        strategy: 生成策略
        test_cases_count: 测试用例数量目标
        requirements: 关联的需求 ID 列表
        strategy_config: 策略配置参数
        estimated_time: 估计执行时间 (秒)
    """
    id: str
    strategy: GenerationStrategy
    test_cases_count: int
    requirements: List[str] = field(default_factory=list)
    strategy_config: Dict[str, Any] = field(default_factory=dict)
    estimated_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        return {
            "id": self.id,
            "strategy": self.strategy.name,
            "test_cases_count": self.test_cases_count,
            "requirements": self.requirements,
            "strategy_config": self.strategy_config,
            "estimated_time": self.estimated_time,
        }


@dataclass
class TestCase:
    """测试用例定义
    
    Attributes:
        id: 用例唯一标识
        plan_id: 关联的计划 ID
        inputs: 输入 Tensor 数据
        parameters: 算子参数
        expected_outputs: 预期输出 (可选)
        metadata: 元数据
    """
    id: str
    plan_id: str
    inputs: Dict[str, torch.Tensor] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_outputs: Optional[Dict[str, torch.Tensor]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示 (不包含实际 tensor 数据)"""
        return {
            "id": self.id,
            "plan_id": self.plan_id,
            "input_shapes": {k: list(v.shape) for k, v in self.inputs.items()},
            "input_dtypes": {k: str(v.dtype) for k, v in self.inputs.items()},
            "parameters": self.parameters,
            "has_expected_outputs": self.expected_outputs is not None,
            "metadata": self.metadata,
        }


@dataclass
class TestResult:
    """测试结果定义
    
    Attributes:
        case_id: 测试用例 ID
        passed: 是否通过
        actual_outputs: 实际输出
        execution_time: 执行时间 (秒)
        memory_usage: 内存使用 (MB)
        error_message: 错误信息
        stack_trace: 堆栈跟踪
    """
    case_id: str
    passed: bool
    actual_outputs: Optional[Dict[str, torch.Tensor]] = None
    execution_time: Optional[float] = None
    memory_usage: Optional[float] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "output_shapes": {k: list(v.shape) for k, v in self.actual_outputs.items()} if self.actual_outputs else None,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
        }


@dataclass
class FailureAnalysis:
    """失败分析定义
    
    Attributes:
        result_id: 测试结果 ID
        category: 失败分类
        root_cause: 根本原因分析
        suggestions: 修复建议列表
        related_cases: 相关失败用例 ID 列表
        confidence: 分析置信度 (0-1)
    """
    result_id: str
    category: FailureCategory
    root_cause: str
    suggestions: List[str] = field(default_factory=list)
    related_cases: List[str] = field(default_factory=list)
    confidence: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        return {
            "result_id": self.result_id,
            "category": self.category.name,
            "root_cause": self.root_cause,
            "suggestions": self.suggestions,
            "related_cases": self.related_cases,
            "confidence": self.confidence,
        }


@dataclass
class CoverageStats:
    """覆盖率统计"""
    line_coverage: float = 0.0
    branch_coverage: float = 0.0
    boundary_coverage: float = 0.0
    dtype_coverage: float = 0.0
    shape_coverage: float = 0.0
    overall_coverage: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "line_coverage": self.line_coverage,
            "branch_coverage": self.branch_coverage,
            "boundary_coverage": self.boundary_coverage,
            "dtype_coverage": self.dtype_coverage,
            "shape_coverage": self.shape_coverage,
            "overall_coverage": self.overall_coverage,
        }


@dataclass
class TestReport:
    """测试报告定义
    
    Attributes:
        report_id: 报告唯一标识
        operator_name: 算子名称
        timestamp: 报告生成时间
        summary: 测试摘要
        results: 详细结果列表
        analyses: 失败分析列表
        coverage: 覆盖率统计
        recommendations: 建议列表
    """
    report_id: str
    operator_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    summary: Dict[str, Any] = field(default_factory=dict)
    results: List[TestResult] = field(default_factory=list)
    analyses: List[FailureAnalysis] = field(default_factory=list)
    coverage: CoverageStats = field(default_factory=CoverageStats)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        return {
            "report_id": self.report_id,
            "operator_name": self.operator_name,
            "timestamp": self.timestamp.isoformat(),
            "summary": self.summary,
            "results": [r.to_dict() for r in self.results],
            "analyses": [a.to_dict() for a in self.analyses],
            "coverage": self.coverage.to_dict(),
            "recommendations": self.recommendations,
        }


@dataclass
class WorkflowState:
    """工作流状态 - 在各阶段间传递
    
    Attributes:
        operator: 算子适配器
        signature: 算子签名
        requirements: 测试需求列表
        plans: 测试计划列表
        test_cases: 测试用例列表
        results: 测试结果列表
        analyses: 失败分析列表
        report: 测试报告
        metadata: 元数据
        context: 上下文信息
    """
    # 输入
    operator: Optional[Any] = None
    
    # 各阶段输出
    signature: Optional[OperatorSignature] = None
    requirements: List[TestRequirement] = field(default_factory=list)
    plans: List[TestPlan] = field(default_factory=list)
    test_cases: List[TestCase] = field(default_factory=list)
    results: List[TestResult] = field(default_factory=list)
    analyses: List[FailureAnalysis] = field(default_factory=list)
    report: Optional[TestReport] = None
    
    # 元数据和上下文
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        return {
            "signature": self.signature.to_dict() if self.signature else None,
            "requirements": [r.to_dict() for r in self.requirements],
            "plans": [p.to_dict() for p in self.plans],
            "test_cases_count": len(self.test_cases),
            "results_count": len(self.results),
            "analyses_count": len(self.analyses),
            "report": self.report.to_dict() if self.report else None,
            "metadata": self.metadata,
            "context": self.context,
        }


class PhaseOutput(ABC):
    """阶段输出抽象基类"""
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        pass
    
    @abstractmethod
    def is_valid(self) -> bool:
        """检查输出是否有效"""
        pass


@dataclass
class PhaseContext:
    """阶段执行上下文
    
    Attributes:
        phase_name: 阶段名称
        start_time: 开始时间
        end_time: 结束时间
        success: 是否成功
        error: 错误信息
    """
    phase_name: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    success: bool = True
    error: Optional[str] = None
    
    @property
    def duration(self) -> Optional[float]:
        """获取执行时长 (秒)"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
