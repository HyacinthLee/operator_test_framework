"""
阶段 3: 测试计划 (Planning)

设计测试策略，选择生成方法（随机、边界、符号）。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import uuid

from .data_models import (
    OperatorSignature,
    TestRequirement,
    TestPlan,
    TestType,
    GenerationStrategy,
    WorkflowState,
    PhaseContext,
)


class StrategySelector(ABC):
    """策略选择器抽象基类"""
    
    @abstractmethod
    def select(
        self,
        requirement: TestRequirement,
        signature: OperatorSignature,
    ) -> GenerationStrategy:
        """为需求选择生成策略
        
        Args:
            requirement: 测试需求
            signature: 算子签名
            
        Returns:
            选择的生成策略
        """
        pass


class HeuristicStrategySelector(StrategySelector):
    """基于启发式的策略选择器"""
    
    def __init__(self):
        self.strategy_map = {
            TestType.NORMAL: GenerationStrategy.RANDOM,
            TestType.BOUNDARY: GenerationStrategy.BOUNDARY,
            TestType.EXCEPTION: GenerationStrategy.HEURISTIC,
            TestType.RANDOM: GenerationStrategy.RANDOM,
            TestType.SYMBOLIC: GenerationStrategy.SYMBOLIC,
            TestType.STRESS: GenerationStrategy.ADAPTIVE,
        }
    
    def select(
        self,
        requirement: TestRequirement,
        signature: OperatorSignature,
    ) -> GenerationStrategy:
        """选择生成策略"""
        # 基于测试类型选择
        strategy = self.strategy_map.get(requirement.test_type, GenerationStrategy.RANDOM)
        
        # 根据约束条件调整
        constraints = requirement.constraints
        
        # 如果有明确的边界条件，使用边界策略
        if 'inject_nan' in constraints or 'inject_inf' in constraints:
            return GenerationStrategy.BOUNDARY
        
        # 如果需要大量随机测试
        if constraints.get('num_tests', 0) > 1000:
            return GenerationStrategy.RANDOM
        
        # 如果是符号测试需求
        if requirement.test_type == TestType.SYMBOLIC:
            return GenerationStrategy.SYMBOLIC
        
        return strategy


class AdaptiveStrategySelector(StrategySelector):
    """自适应策略选择器 - 根据历史表现选择"""
    
    def __init__(self):
        self.performance_history: Dict[GenerationStrategy, float] = {}
    
    def select(
        self,
        requirement: TestRequirement,
        signature: OperatorSignature,
    ) -> GenerationStrategy:
        """自适应选择策略"""
        # 优先选择历史表现好的策略
        if self.performance_history:
            best_strategy = max(
                self.performance_history.items(),
                key=lambda x: x[1]
            )[0]
            return best_strategy
        
        # 默认使用启发式选择
        heuristic = HeuristicStrategySelector()
        return heuristic.select(requirement, signature)
    
    def update_performance(
        self,
        strategy: GenerationStrategy,
        success_rate: float,
    ) -> None:
        """更新策略表现历史
        
        Args:
            strategy: 策略
            success_rate: 成功率 (0-1)
        """
        self.performance_history[strategy] = success_rate


class TestPlanner(ABC):
    """测试计划器抽象基类"""
    
    @abstractmethod
    def create_plan(
        self,
        requirements: List[TestRequirement],
        signature: OperatorSignature,
        strategy_selector: StrategySelector,
    ) -> List[TestPlan]:
        """创建测试计划
        
        Args:
            requirements: 测试需求列表
            signature: 算子签名
            strategy_selector: 策略选择器
            
        Returns:
            测试计划列表
        """
        pass


class DefaultTestPlanner(TestPlanner):
    """默认测试计划器"""
    
    def __init__(
        self,
        base_test_count: int = 10,
        max_test_count: int = 1000,
    ):
        self.base_test_count = base_test_count
        self.max_test_count = max_test_count
    
    def create_plan(
        self,
        requirements: List[TestRequirement],
        signature: OperatorSignature,
        strategy_selector: StrategySelector,
    ) -> List[TestPlan]:
        """创建测试计划"""
        plans = []
        
        # 按策略分组需求
        strategy_groups: Dict[GenerationStrategy, List[TestRequirement]] = {}
        for req in requirements:
            strategy = strategy_selector.select(req, signature)
            if strategy not in strategy_groups:
                strategy_groups[strategy] = []
            strategy_groups[strategy].append(req)
        
        # 为每个策略创建计划
        for strategy, reqs in strategy_groups.items():
            plan = self._create_plan_for_strategy(
                strategy, reqs, signature
            )
            plans.append(plan)
        
        return plans
    
    def _create_plan_for_strategy(
        self,
        strategy: GenerationStrategy,
        requirements: List[TestRequirement],
        signature: OperatorSignature,
    ) -> TestPlan:
        """为特定策略创建计划"""
        # 计算测试用例数量
        test_count = self._calculate_test_count(strategy, requirements)
        
        # 估算执行时间
        estimated_time = self._estimate_execution_time(
            strategy, test_count, signature
        )
        
        # 构建策略配置
        config = self._build_strategy_config(strategy, requirements)
        
        return TestPlan(
            id=f"plan_{strategy.name.lower()}_{uuid.uuid4().hex[:8]}",
            strategy=strategy,
            test_cases_count=test_count,
            requirements=[r.id for r in requirements],
            strategy_config=config,
            estimated_time=estimated_time,
        )
    
    def _calculate_test_count(
        self,
        strategy: GenerationStrategy,
        requirements: List[TestRequirement],
    ) -> int:
        """计算测试用例数量"""
        base_count = self.base_test_count
        
        # 根据策略调整
        multipliers = {
            GenerationStrategy.RANDOM: 10,
            GenerationStrategy.BOUNDARY: 2,
            GenerationStrategy.SYMBOLIC: 1,
            GenerationStrategy.HEURISTIC: 5,
            GenerationStrategy.EXHAUSTIVE: 50,
            GenerationStrategy.ADAPTIVE: 10,
        }
        multiplier = multipliers.get(strategy, 1)
        
        # 根据需求数量调整
        count = base_count * multiplier * max(1, len(requirements) // 5)
        
        return min(count, self.max_test_count)
    
    def _estimate_execution_time(
        self,
        strategy: GenerationStrategy,
        test_count: int,
        signature: OperatorSignature,
    ) -> float:
        """估算执行时间 (秒)"""
        # 基础时间 (假设每个测试 10ms)
        base_time = test_count * 0.01
        
        # 根据策略调整
        factors = {
            GenerationStrategy.RANDOM: 1.0,
            GenerationStrategy.BOUNDARY: 1.2,
            GenerationStrategy.SYMBOLIC: 2.0,
            GenerationStrategy.HEURISTIC: 1.5,
            GenerationStrategy.EXHAUSTIVE: 5.0,
            GenerationStrategy.ADAPTIVE: 1.5,
        }
        factor = factors.get(strategy, 1.0)
        
        # 考虑输入复杂度
        input_complexity = len(signature.inputs)
        complexity_factor = 1 + (input_complexity * 0.1)
        
        return base_time * factor * complexity_factor
    
    def _build_strategy_config(
        self,
        strategy: GenerationStrategy,
        requirements: List[TestRequirement],
    ) -> Dict[str, Any]:
        """构建策略配置"""
        config = {
            "strategy": strategy.name,
            "target_requirements": len(requirements),
        }
        
        if strategy == GenerationStrategy.RANDOM:
            config["seed"] = 42
            config["distribution"] = "uniform"
        
        elif strategy == GenerationStrategy.BOUNDARY:
            config["boundary_types"] = ["min", "max", "epsilon"]
            config["include_corners"] = True
        
        elif strategy == GenerationStrategy.SYMBOLIC:
            config["solver_timeout"] = 30
            config["max_path_length"] = 10
        
        elif strategy == GenerationStrategy.ADAPTIVE:
            config["feedback_loop"] = True
            config["adaptation_rate"] = 0.1
        
        return config


class PlanningPhase:
    """测试计划阶段
    
    职责：
    1. 设计测试策略
    2. 选择生成方法（随机、边界、符号）
    3. 估算资源需求
    """
    
    def __init__(
        self,
        strategy_selector: Optional[StrategySelector] = None,
        test_planner: Optional[TestPlanner] = None,
    ):
        """初始化
        
        Args:
            strategy_selector: 策略选择器
            test_planner: 测试计划器
        """
        self.strategy_selector = strategy_selector or HeuristicStrategySelector()
        self.test_planner = test_planner or DefaultTestPlanner()
    
    def execute(self, state: WorkflowState) -> Tuple[WorkflowState, PhaseContext]:
        """执行测试计划阶段
        
        Args:
            state: 当前工作流状态
            
        Returns:
            (更新后的状态, 阶段上下文)
        """
        import datetime
        context = PhaseContext(phase_name="planning")
        context.start_time = datetime.datetime.now()
        
        try:
            if not state.requirements:
                raise ValueError("No test requirements available. Run requirements phase first.")
            
            if state.signature is None:
                raise ValueError("No operator signature available. Run understand phase first.")
            
            # 创建测试计划
            plans = self.test_planner.create_plan(
                state.requirements,
                state.signature,
                self.strategy_selector,
            )
            
            # 更新状态
            state.plans = plans
            state.metadata['plans_count'] = len(plans)
            state.metadata['plans_by_strategy'] = self._count_by_strategy(plans)
            state.metadata['estimated_total_time'] = sum(
                p.estimated_time or 0 for p in plans
            )
            
            context.success = True
            
        except Exception as e:
            context.success = False
            context.error = str(e)
            state.metadata['planning_error'] = str(e)
        
        context.end_time = datetime.datetime.now()
        return state, context
    
    def _count_by_strategy(self, plans: List[TestPlan]) -> Dict[str, int]:
        """统计各策略的计划数量"""
        counts = {}
        for plan in plans:
            strategy_name = plan.strategy.name
            counts[strategy_name] = counts.get(strategy_name, 0) + 1
        return counts
    
    def get_plan_by_strategy(
        self,
        state: WorkflowState,
        strategy: GenerationStrategy,
    ) -> Optional[TestPlan]:
        """获取特定策略的测试计划
        
        Args:
            state: 工作流状态
            strategy: 生成策略
            
        Returns:
            对应的测试计划，如果不存在返回 None
        """
        for plan in state.plans:
            if plan.strategy == strategy:
                return plan
        return None
    
    def optimize_plans(
        self,
        state: WorkflowState,
        max_total_time: Optional[float] = None,
        max_total_tests: Optional[int] = None,
    ) -> None:
        """优化测试计划
        
        Args:
            state: 工作流状态
            max_total_time: 最大总执行时间
            max_total_tests: 最大总测试数量
        """
        for plan in state.plans:
            if max_total_time and plan.estimated_time:
                factor = max_total_time / (plan.estimated_time * len(state.plans))
                plan.test_cases_count = int(plan.test_cases_count * factor)
            
            if max_total_tests:
                per_plan = max_total_tests // len(state.plans)
                plan.test_cases_count = min(plan.test_cases_count, per_plan)
