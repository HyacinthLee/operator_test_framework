"""
阶段 4: 测试生成 (Generation)

生成具体测试输入，确保约束满足。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Callable
import uuid
import torch
import numpy as np

from .data_models import (
    OperatorSignature,
    TestPlan,
    TestCase,
    TensorConstraint,
    GenerationStrategy,
    WorkflowState,
    PhaseContext,
)


class ConstraintSatisfier(ABC):
    """约束满足器抽象基类"""
    
    @abstractmethod
    def satisfy(
        self,
        constraint: TensorConstraint,
        strategy: GenerationStrategy,
    ) -> torch.Tensor:
        """生成满足约束的 tensor
        
        Args:
            constraint: Tensor 约束
            strategy: 生成策略
            
        Returns:
            满足约束的 tensor
        """
        pass


class DefaultConstraintSatisfier(ConstraintSatisfier):
    """默认约束满足器"""
    
    def satisfy(
        self,
        constraint: TensorConstraint,
        strategy: GenerationStrategy,
    ) -> torch.Tensor:
        """生成满足约束的 tensor"""
        # 解析形状
        shape = self._resolve_shape(constraint.shape)
        
        # 解析数据类型
        dtype = constraint.dtype or torch.float32
        
        # 解析设备
        device = constraint.device or 'cpu'
        
        # 根据策略生成数据
        if strategy == GenerationStrategy.BOUNDARY:
            data = self._generate_boundary_data(shape, constraint)
        elif strategy == GenerationStrategy.RANDOM:
            data = self._generate_random_data(shape, constraint)
        elif strategy == GenerationStrategy.HEURISTIC:
            data = self._generate_heuristic_data(shape, constraint)
        else:
            data = self._generate_random_data(shape, constraint)
        
        # 创建 tensor
        tensor = torch.tensor(data, dtype=dtype, device=device)
        
        # 设置 requires_grad
        if constraint.requires_grad is not None:
            tensor.requires_grad_(constraint.requires_grad)
        
        return tensor
    
    def _resolve_shape(self, shape: Any) -> Tuple[int, ...]:
        """解析形状"""
        if shape == "*" or shape is None:
            return (2, 64)  # 默认形状
        if isinstance(shape, (list, tuple)):
            return tuple(shape)
        return (2, 64)
    
    def _generate_boundary_data(
        self,
        shape: Tuple[int, ...],
        constraint: TensorConstraint,
    ) -> np.ndarray:
        """生成边界数据"""
        # 确定数值范围
        min_val = constraint.min_value if constraint.min_value is not None else -1.0
        max_val = constraint.max_value if constraint.max_value is not None else 1.0
        
        # 生成边界值
        data = np.zeros(shape)
        
        # 填充边界值
        boundary_values = [
            min_val,
            max_val,
            min_val + (max_val - min_val) * 0.5,  # 中点
            min_val + 1e-7 if min_val < 0 else 1e-7,  # 极小值
            max_val - 1e-7 if max_val > 0 else -1e-7,  # 极大值
        ]
        
        # 循环填充
        flat_size = np.prod(shape)
        for i in range(flat_size):
            data.flat[i] = boundary_values[i % len(boundary_values)]
        
        return data
    
    def _generate_random_data(
        self,
        shape: Tuple[int, ...],
        constraint: TensorConstraint,
    ) -> np.ndarray:
        """生成随机数据"""
        min_val = constraint.min_value if constraint.min_value is not None else -1.0
        max_val = constraint.max_value if constraint.max_value is not None else 1.0
        
        return np.random.uniform(min_val, max_val, shape)
    
    def _generate_heuristic_data(
        self,
        shape: Tuple[int, ...],
        constraint: TensorConstraint,
    ) -> np.ndarray:
        """生成启发式数据"""
        data = np.zeros(shape)
        
        # 添加一些特定的测试模式
        # 全零
        if np.random.random() < 0.2:
            return data
        
        # 全一
        if np.random.random() < 0.2:
            return np.ones(shape)
        
        # 线性递增
        if np.random.random() < 0.2:
            flat = np.linspace(-1, 1, np.prod(shape))
            return flat.reshape(shape)
        
        # 随机
        return self._generate_random_data(shape, constraint)


class InputGenerator(ABC):
    """输入生成器抽象基类"""
    
    @abstractmethod
    def generate(
        self,
        plan: TestPlan,
        signature: OperatorSignature,
        constraint_satisfier: ConstraintSatisfier,
    ) -> List[TestCase]:
        """生成测试用例
        
        Args:
            plan: 测试计划
            signature: 算子签名
            constraint_satisfier: 约束满足器
            
        Returns:
            测试用例列表
        """
        pass


class RandomInputGenerator(InputGenerator):
    """随机输入生成器"""
    
    def generate(
        self,
        plan: TestPlan,
        signature: OperatorSignature,
        constraint_satisfier: ConstraintSatisfier,
    ) -> List[TestCase]:
        """生成随机测试用例"""
        test_cases = []
        
        # 设置随机种子
        seed = plan.strategy_config.get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        for i in range(plan.test_cases_count):
            inputs = {}
            
            # 为每个输入约束生成数据
            for constraint in signature.inputs:
                tensor = constraint_satisfier.satisfy(
                    constraint,
                    GenerationStrategy.RANDOM
                )
                inputs[constraint.name] = tensor
            
            test_case = TestCase(
                id=f"tc_{plan.id}_{i}",
                plan_id=plan.id,
                inputs=inputs,
                parameters={},
                metadata={
                    "generation_strategy": "random",
                    "index": i,
                }
            )
            test_cases.append(test_case)
        
        return test_cases


class BoundaryInputGenerator(InputGenerator):
    """边界输入生成器"""
    
    def generate(
        self,
        plan: TestPlan,
        signature: OperatorSignature,
        constraint_satisfier: ConstraintSatisfier,
    ) -> List[TestCase]:
        """生成边界测试用例"""
        test_cases = []
        
        boundary_types = plan.strategy_config.get(
            'boundary_types',
            ['min', 'max', 'epsilon', 'zero']
        )
        
        for i in range(plan.test_cases_count):
            inputs = {}
            
            for constraint in signature.inputs:
                # 修改约束以生成边界值
                boundary_constraint = self._create_boundary_constraint(
                    constraint, boundary_types[i % len(boundary_types)]
                )
                
                tensor = constraint_satisfier.satisfy(
                    boundary_constraint,
                    GenerationStrategy.BOUNDARY
                )
                inputs[constraint.name] = tensor
            
            test_case = TestCase(
                id=f"tc_boundary_{plan.id}_{i}",
                plan_id=plan.id,
                inputs=inputs,
                parameters={},
                metadata={
                    "generation_strategy": "boundary",
                    "boundary_type": boundary_types[i % len(boundary_types)],
                    "index": i,
                }
            )
            test_cases.append(test_case)
        
        return test_cases
    
    def _create_boundary_constraint(
        self,
        constraint: TensorConstraint,
        boundary_type: str,
    ) -> TensorConstraint:
        """创建边界约束"""
        new_constraint = TensorConstraint(
            name=constraint.name,
            shape=constraint.shape,
            dtype=constraint.dtype,
            device=constraint.device,
            requires_grad=constraint.requires_grad,
        )
        
        if boundary_type == 'min':
            new_constraint.min_value = -1e6
            new_constraint.max_value = -1e-6
        elif boundary_type == 'max':
            new_constraint.min_value = 1e-6
            new_constraint.max_value = 1e6
        elif boundary_type == 'epsilon':
            new_constraint.min_value = -1e-7
            new_constraint.max_value = 1e-7
        elif boundary_type == 'zero':
            new_constraint.min_value = 0
            new_constraint.max_value = 0
        
        return new_constraint


class SymbolicInputGenerator(InputGenerator):
    """符号输入生成器"""
    
    def generate(
        self,
        plan: TestPlan,
        signature: OperatorSignature,
        constraint_satisfier: ConstraintSatisfier,
    ) -> List[TestCase]:
        """生成符号测试用例 (简化实现)"""
        test_cases = []
        
        # 符号执行通常需要求解器，这里简化处理
        # 生成具有特定模式的输入
        patterns = [
            "linear",
            "quadratic",
            "periodic",
            "sparse",
        ]
        
        for i in range(min(plan.test_cases_count, len(patterns) * 10)):
            inputs = {}
            pattern = patterns[i % len(patterns)]
            
            for constraint in signature.inputs:
                tensor = self._generate_patterned_input(
                    constraint, pattern
                )
                inputs[constraint.name] = tensor
            
            test_case = TestCase(
                id=f"tc_symbolic_{plan.id}_{i}",
                plan_id=plan.id,
                inputs=inputs,
                parameters={},
                metadata={
                    "generation_strategy": "symbolic",
                    "pattern": pattern,
                    "index": i,
                }
            )
            test_cases.append(test_case)
        
        return test_cases
    
    def _generate_patterned_input(
        self,
        constraint: TensorConstraint,
        pattern: str,
    ) -> torch.Tensor:
        """生成有模式的输入"""
        shape = constraint.shape if constraint.shape != "*" else (4, 16)
        if isinstance(shape, str):
            shape = (4, 16)
        
        dtype = constraint.dtype or torch.float32
        device = constraint.device or 'cpu'
        
        if pattern == "linear":
            data = torch.linspace(-1, 1, np.prod(shape)).reshape(shape)
        elif pattern == "quadratic":
            x = torch.linspace(-1, 1, np.prod(shape))
            data = (x ** 2).reshape(shape)
        elif pattern == "periodic":
            x = torch.linspace(0, 4 * np.pi, np.prod(shape))
            data = torch.sin(x).reshape(shape)
        elif pattern == "sparse":
            data = torch.zeros(shape)
            # 10% 非零
            num_nonzero = np.prod(shape) // 10
            indices = torch.randperm(np.prod(shape))[:num_nonzero]
            data.view(-1)[indices] = torch.randn(num_nonzero)
        else:
            data = torch.randn(shape)
        
        tensor = data.to(dtype=dtype, device=device)
        
        if constraint.requires_grad is not None:
            tensor.requires_grad_(constraint.requires_grad)
        
        return tensor


class GenerationPhase:
    """测试生成阶段
    
    职责：
    1. 生成具体测试输入
    2. 确保约束满足
    3. 管理生成策略
    """
    
    def __init__(
        self,
        constraint_satisfier: Optional[ConstraintSatisfier] = None,
        generators: Optional[Dict[GenerationStrategy, InputGenerator]] = None,
    ):
        """初始化
        
        Args:
            constraint_satisfier: 约束满足器
            generators: 各策略的输入生成器
        """
        self.constraint_satisfier = constraint_satisfier or DefaultConstraintSatisfier()
        self.generators = generators or {
            GenerationStrategy.RANDOM: RandomInputGenerator(),
            GenerationStrategy.BOUNDARY: BoundaryInputGenerator(),
            GenerationStrategy.SYMBOLIC: SymbolicInputGenerator(),
            GenerationStrategy.HEURISTIC: RandomInputGenerator(),  # 复用随机
            GenerationStrategy.ADAPTIVE: RandomInputGenerator(),   # 复用随机
        }
    
    def execute(self, state: WorkflowState) -> Tuple[WorkflowState, PhaseContext]:
        """执行测试生成阶段
        
        Args:
            state: 当前工作流状态
            
        Returns:
            (更新后的状态, 阶段上下文)
        """
        import datetime
        context = PhaseContext(phase_name="generation")
        context.start_time = datetime.datetime.now()
        
        try:
            if not state.plans:
                raise ValueError("No test plans available. Run planning phase first.")
            
            if state.signature is None:
                raise ValueError("No operator signature available.")
            
            all_test_cases = []
            
            # 为每个计划生成测试用例
            for plan in state.plans:
                generator = self.generators.get(plan.strategy)
                if generator is None:
                    # 使用默认生成器
                    generator = self.generators[GenerationStrategy.RANDOM]
                
                test_cases = generator.generate(
                    plan,
                    state.signature,
                    self.constraint_satisfier,
                )
                all_test_cases.extend(test_cases)
            
            # 更新状态
            state.test_cases = all_test_cases
            state.metadata['test_cases_count'] = len(all_test_cases)
            state.metadata['test_cases_by_plan'] = self._count_by_plan(all_test_cases)
            
            context.success = True
            
        except Exception as e:
            context.success = False
            context.error = str(e)
            state.metadata['generation_error'] = str(e)
        
        context.end_time = datetime.datetime.now()
        return state, context
    
    def _count_by_plan(self, test_cases: List[TestCase]) -> Dict[str, int]:
        """统计各计划的测试用例数量"""
        counts = {}
        for tc in test_cases:
            plan_id = tc.plan_id
            counts[plan_id] = counts.get(plan_id, 0) + 1
        return counts
    
    def validate_constraints(
        self,
        test_case: TestCase,
        signature: OperatorSignature,
    ) -> Tuple[bool, List[str]]:
        """验证测试用例是否满足约束
        
        Args:
            test_case: 测试用例
            signature: 算子签名
            
        Returns:
            (是否满足, 违反的约束列表)
        """
        violations = []
        
        # 检查输入数量
        if len(test_case.inputs) != len(signature.inputs):
            violations.append(
                f"Input count mismatch: expected {len(signature.inputs)}, "
                f"got {len(test_case.inputs)}"
            )
        
        # 检查每个输入
        for constraint in signature.inputs:
            if constraint.name not in test_case.inputs:
                violations.append(f"Missing input: {constraint.name}")
                continue
            
            tensor = test_case.inputs[constraint.name]
            
            # 检查形状 (如果约束是具体值)
            if constraint.shape != "*" and isinstance(constraint.shape, (tuple, list)):
                if tuple(tensor.shape) != tuple(constraint.shape):
                    violations.append(
                        f"Shape mismatch for {constraint.name}: "
                        f"expected {constraint.shape}, got {tuple(tensor.shape)}"
                    )
            
            # 检查数据类型
            if constraint.dtype is not None and tensor.dtype != constraint.dtype:
                violations.append(
                    f"Dtype mismatch for {constraint.name}: "
                    f"expected {constraint.dtype}, got {tensor.dtype}"
                )
        
        return len(violations) == 0, violations
