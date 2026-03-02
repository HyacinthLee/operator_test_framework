"""
阶段 2: 需求生成 (Requirements)

生成测试需求（正常、边界、异常），确定覆盖率目标。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import uuid

from .data_models import (
    OperatorSignature,
    TestRequirement,
    TestType,
    WorkflowState,
    PhaseContext,
)


class TestRequirementGenerator(ABC):
    """测试需求生成器抽象基类"""
    
    @abstractmethod
    def generate(
        self,
        signature: OperatorSignature,
        test_type: TestType,
    ) -> List[TestRequirement]:
        """生成测试需求
        
        Args:
            signature: 算子签名
            test_type: 测试类型
            
        Returns:
            测试需求列表
        """
        pass


class NormalRequirementGenerator(TestRequirementGenerator):
    """正常测试需求生成器"""
    
    def generate(
        self,
        signature: OperatorSignature,
        test_type: TestType,
    ) -> List[TestRequirement]:
        """生成正常测试需求"""
        requirements = []
        
        # 基本功能测试
        requirements.append(TestRequirement(
            id=f"{signature.name}_normal_basic",
            test_type=TestType.NORMAL,
            description=f"验证 {signature.name} 基本功能",
            priority=5,
            coverage_target=1.0,
            constraints={"batch_size": [2, 4, 8, 16]},
        ))
        
        # 多批次测试
        requirements.append(TestRequirement(
            id=f"{signature.name}_normal_batch",
            test_type=TestType.NORMAL,
            description=f"验证 {signature.name} 不同 batch size",
            priority=4,
            coverage_target=0.9,
            constraints={"batch_sizes": [1, 2, 4, 8, 16, 32, 64]},
        ))
        
        # 不同数据类型测试
        if signature.supports_autograd:
            requirements.append(TestRequirement(
                id=f"{signature.name}_normal_dtype",
                test_type=TestType.NORMAL,
                description=f"验证 {signature.name} 不同数据类型",
                priority=4,
                coverage_target=0.85,
                constraints={"dtypes": ["float32", "float64"]},
            ))
        
        return requirements


class BoundaryRequirementGenerator(TestRequirementGenerator):
    """边界测试需求生成器"""
    
    def generate(
        self,
        signature: OperatorSignature,
        test_type: TestType,
    ) -> List[TestRequirement]:
        """生成边界测试需求"""
        requirements = []
        
        # 从签名中提取边界条件
        for i, boundary in enumerate(signature.boundary_conditions):
            req_id = f"{signature.name}_boundary_{boundary.get('name', i)}"
            requirements.append(TestRequirement(
                id=req_id,
                test_type=TestType.BOUNDARY,
                description=boundary.get('description', f"边界测试: {boundary.get('name', i)}"),
                priority=5 if boundary.get('severity') == 'high' else 3,
                coverage_target=1.0,
                constraints=boundary.get('condition', {}),
                derived_from=f"boundary_condition_{i}",
            ))
        
        # 添加通用边界测试
        requirements.append(TestRequirement(
            id=f"{signature.name}_boundary_shape",
            test_type=TestType.BOUNDARY,
            description=f"验证 {signature.name} 边界形状",
            priority=4,
            coverage_target=0.9,
            constraints={"test_shape_variants": True},
        ))
        
        return requirements


class ExceptionRequirementGenerator(TestRequirementGenerator):
    """异常测试需求生成器"""
    
    def generate(
        self,
        signature: OperatorSignature,
        test_type: TestType,
    ) -> List[TestRequirement]:
        """生成异常测试需求"""
        requirements = []
        
        # 无效输入测试
        requirements.append(TestRequirement(
            id=f"{signature.name}_exception_invalid_input",
            test_type=TestType.EXCEPTION,
            description=f"验证 {signature.name} 对无效输入的处理",
            priority=4,
            coverage_target=0.8,
            constraints={"test_invalid_inputs": True},
        ))
        
        # 类型不匹配测试
        requirements.append(TestRequirement(
            id=f"{signature.name}_exception_type_mismatch",
            test_type=TestType.EXCEPTION,
            description=f"验证 {signature.name} 对类型不匹配的处理",
            priority=3,
            coverage_target=0.7,
            constraints={"test_type_mismatch": True},
        ))
        
        # 形状不匹配测试
        requirements.append(TestRequirement(
            id=f"{signature.name}_exception_shape_mismatch",
            test_type=TestType.EXCEPTION,
            description=f"验证 {signature.name} 对形状不匹配的处理",
            priority=4,
            coverage_target=0.8,
            constraints={"test_shape_mismatch": True},
        ))
        
        # NaN/Inf 输入测试
        requirements.append(TestRequirement(
            id=f"{signature.name}_exception_nan_inf",
            test_type=TestType.EXCEPTION,
            description=f"验证 {signature.name} 对 NaN/Inf 输入的处理",
            priority=5,
            coverage_target=0.9,
            constraints={"test_nan_inf": True},
        ))
        
        return requirements


class RandomRequirementGenerator(TestRequirementGenerator):
    """随机测试需求生成器"""
    
    def __init__(self, num_random_tests: int = 100):
        self.num_random_tests = num_random_tests
    
    def generate(
        self,
        signature: OperatorSignature,
        test_type: TestType,
    ) -> List[TestRequirement]:
        """生成随机测试需求"""
        requirements = []
        
        requirements.append(TestRequirement(
            id=f"{signature.name}_random_fuzz",
            test_type=TestType.RANDOM,
            description=f"对 {signature.name} 进行模糊测试",
            priority=3,
            coverage_target=0.6,
            constraints={"num_tests": self.num_random_tests},
        ))
        
        return requirements


class RequirementsPhase:
    """需求生成阶段
    
    职责：
    1. 根据算子签名生成测试需求
    2. 覆盖正常、边界、异常场景
    3. 确定覆盖率目标
    """
    
    def __init__(
        self,
        generators: Optional[Dict[TestType, TestRequirementGenerator]] = None,
        coverage_target: float = 0.8,
    ):
        """初始化
        
        Args:
            generators: 各类型测试需求生成器
            coverage_target: 默认覆盖率目标
        """
        self.generators = generators or {
            TestType.NORMAL: NormalRequirementGenerator(),
            TestType.BOUNDARY: BoundaryRequirementGenerator(),
            TestType.EXCEPTION: ExceptionRequirementGenerator(),
            TestType.RANDOM: RandomRequirementGenerator(),
        }
        self.coverage_target = coverage_target
    
    def execute(self, state: WorkflowState) -> Tuple[WorkflowState, PhaseContext]:
        """执行需求生成阶段
        
        Args:
            state: 当前工作流状态
            
        Returns:
            (更新后的状态, 阶段上下文)
        """
        import datetime
        context = PhaseContext(phase_name="requirements")
        context.start_time = datetime.datetime.now()
        
        try:
            if state.signature is None:
                raise ValueError("No operator signature available. Run understand phase first.")
            
            all_requirements = []
            
            # 为每种测试类型生成需求
            for test_type, generator in self.generators.items():
                reqs = generator.generate(state.signature, test_type)
                all_requirements.extend(reqs)
            
            # 去重和排序
            all_requirements = self._deduplicate_requirements(all_requirements)
            all_requirements = self._sort_by_priority(all_requirements)
            
            # 更新状态
            state.requirements = all_requirements
            state.metadata['requirements_count'] = len(all_requirements)
            state.metadata['requirements_by_type'] = self._count_by_type(all_requirements)
            state.metadata['coverage_target'] = self._calculate_overall_coverage(all_requirements)
            
            context.success = True
            
        except Exception as e:
            context.success = False
            context.error = str(e)
            state.metadata['requirements_error'] = str(e)
        
        context.end_time = datetime.datetime.now()
        return state, context
    
    def _deduplicate_requirements(
        self, 
        requirements: List[TestRequirement]
    ) -> List[TestRequirement]:
        """去重需求"""
        seen_ids = set()
        unique = []
        for req in requirements:
            if req.id not in seen_ids:
                seen_ids.add(req.id)
                unique.append(req)
        return unique
    
    def _sort_by_priority(
        self, 
        requirements: List[TestRequirement]
    ) -> List[TestRequirement]:
        """按优先级排序"""
        return sorted(requirements, key=lambda r: (-r.priority, r.id))
    
    def _count_by_type(
        self, 
        requirements: List[TestRequirement]
    ) -> Dict[str, int]:
        """统计各类型需求数量"""
        counts = {}
        for req in requirements:
            type_name = req.test_type.name
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts
    
    def _calculate_overall_coverage(
        self, 
        requirements: List[TestRequirement]
    ) -> float:
        """计算整体覆盖率目标"""
        if not requirements:
            return 0.0
        total = sum(r.coverage_target for r in requirements)
        return total / len(requirements)
    
    def add_custom_requirement(
        self,
        state: WorkflowState,
        requirement: TestRequirement,
    ) -> None:
        """添加自定义需求
        
        Args:
            state: 工作流状态
            requirement: 自定义需求
        """
        state.requirements.append(requirement)
        state.requirements = self._sort_by_priority(state.requirements)
    
    def get_requirements_by_type(
        self,
        state: WorkflowState,
        test_type: TestType,
    ) -> List[TestRequirement]:
        """获取特定类型的需求
        
        Args:
            state: 工作流状态
            test_type: 测试类型
            
        Returns:
            该类型的需求列表
        """
        return [r for r in state.requirements if r.test_type == test_type]
    
    def get_high_priority_requirements(
        self,
        state: WorkflowState,
        min_priority: int = 4,
    ) -> List[TestRequirement]:
        """获取高优先级需求
        
        Args:
            state: 工作流状态
            min_priority: 最低优先级
            
        Returns:
            高优先级需求列表
        """
        return [r for r in state.requirements if r.priority >= min_priority]
