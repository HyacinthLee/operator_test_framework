"""
七阶段工作流引擎主类

整合七个阶段，提供统一的工作流执行接口。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Tuple, Type
import uuid
from datetime import datetime

from .data_models import (
    WorkflowState,
    PhaseContext,
    TestReport,
)

from .understand import UnderstandPhase
from .requirements import RequirementsPhase
from .planning import PlanningPhase
from .generation import GenerationPhase
from .execution import ExecutionPhase
from .analysis import AnalysisPhase
from .report import ReportPhase


@dataclass
class WorkflowConfig:
    """工作流配置
    
    Attributes:
        enable_understand: 启用算子理解阶段
        enable_requirements: 启用需求生成阶段
        enable_planning: 启用测试计划阶段
        enable_generation: 启用测试生成阶段
        enable_execution: 启用执行测试阶段
        enable_analysis: 启用结果分析阶段
        enable_report: 启用报告生成阶段
        stop_on_error: 错误时停止
        collect_metrics: 收集指标
        log_level: 日志级别
    """
    enable_understand: bool = True
    enable_requirements: bool = True
    enable_planning: bool = True
    enable_generation: bool = True
    enable_execution: bool = True
    enable_analysis: bool = True
    enable_report: bool = True
    
    stop_on_error: bool = True
    collect_metrics: bool = True
    log_level: str = "INFO"
    
    # 各阶段的自定义配置
    phase_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class PhaseExecutor(ABC):
    """阶段执行器抽象基类"""
    
    @abstractmethod
    def execute(self, state: WorkflowState) -> Tuple[WorkflowState, PhaseContext]:
        """执行阶段
        
        Args:
            state: 当前工作流状态
            
        Returns:
            (更新后的状态, 阶段上下文)
        """
        pass


class WorkflowEngine:
    """七阶段工作流引擎
    
    整合七个阶段，提供完整的工作流执行能力：
    1. 算子理解 (Understand)
    2. 需求生成 (Requirements)
    3. 测试计划 (Planning)
    4. 测试生成 (Generation)
    5. 执行测试 (Execution)
    6. 结果分析 (Analysis)
    7. 报告生成 (Report)
    
    Example:
        >>> engine = WorkflowEngine()
        >>> state = WorkflowState(operator=my_adapter)
        >>> result = engine.run(state)
        >>> print(result.report.to_dict())
    """
    
    def __init__(
        self,
        config: Optional[WorkflowConfig] = None,
        phases: Optional[Dict[str, PhaseExecutor]] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ):
        """初始化工作流引擎
        
        Args:
            config: 工作流配置
            phases: 自定义阶段实现
            progress_callback: 进度回调 (phase_name, current, total)
        """
        self.config = config or WorkflowConfig()
        self.progress_callback = progress_callback
        
        # 初始化或自定义阶段
        self.phases = phases or self._create_default_phases()
        
        # 阶段顺序
        self.phase_order = [
            "understand",
            "requirements",
            "planning",
            "generation",
            "execution",
            "analysis",
            "report",
        ]
        
        # 执行历史
        self.execution_history: List[Dict[str, Any]] = []
    
    def _create_default_phases(self) -> Dict[str, PhaseExecutor]:
        """创建默认阶段实现"""
        return {
            "understand": UnderstandPhase(),
            "requirements": RequirementsPhase(),
            "planning": PlanningPhase(),
            "generation": GenerationPhase(),
            "execution": ExecutionPhase(),
            "analysis": AnalysisPhase(),
            "report": ReportPhase(),
        }
    
    def run(
        self,
        state: WorkflowState,
        phases_to_run: Optional[List[str]] = None,
    ) -> WorkflowState:
        """运行工作流
        
        Args:
            state: 初始工作流状态 (必须包含 operator)
            phases_to_run: 指定要运行的阶段 (默认运行所有启用的阶段)
            
        Returns:
            最终工作流状态
        """
        if state.operator is None:
            raise ValueError("WorkflowState must contain an operator")
        
        # 确定要运行的阶段
        phases_to_run = phases_to_run or self._get_enabled_phases()
        
        # 执行各阶段
        execution_record = {
            "run_id": str(uuid.uuid4())[:8],
            "start_time": datetime.now(),
            "phases": {},
        }
        
        total_phases = len(phases_to_run)
        
        for i, phase_name in enumerate(phases_to_run):
            # 进度回调
            if self.progress_callback:
                self.progress_callback(phase_name, i + 1, total_phases)
            
            # 执行阶段
            state, context = self._execute_phase(phase_name, state)
            
            # 记录执行结果
            execution_record["phases"][phase_name] = {
                "success": context.success,
                "duration": context.duration,
                "error": context.error,
            }
            
            # 错误处理
            if not context.success and self.config.stop_on_error:
                break
        
        execution_record["end_time"] = datetime.now()
        self.execution_history.append(execution_record)
        
        return state
    
    def _get_enabled_phases(self) -> List[str]:
        """获取启用的阶段列表"""
        enabled = []
        
        if self.config.enable_understand:
            enabled.append("understand")
        if self.config.enable_requirements:
            enabled.append("requirements")
        if self.config.enable_planning:
            enabled.append("planning")
        if self.config.enable_generation:
            enabled.append("generation")
        if self.config.enable_execution:
            enabled.append("execution")
        if self.config.enable_analysis:
            enabled.append("analysis")
        if self.config.enable_report:
            enabled.append("report")
        
        return enabled
    
    def _execute_phase(
        self,
        phase_name: str,
        state: WorkflowState,
    ) -> Tuple[WorkflowState, PhaseContext]:
        """执行单个阶段
        
        Args:
            phase_name: 阶段名称
            state: 当前状态
            
        Returns:
            (更新后的状态, 阶段上下文)
        """
        phase = self.phases.get(phase_name)
        
        if phase is None:
            raise ValueError(f"Unknown phase: {phase_name}")
        
        # 执行阶段
        return phase.execute(state)
    
    def run_phase(
        self,
        phase_name: str,
        state: WorkflowState,
    ) -> WorkflowState:
        """运行单个阶段
        
        Args:
            phase_name: 阶段名称
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        new_state, _ = self._execute_phase(phase_name, state)
        return new_state
    
    def get_report(self, state: WorkflowState) -> Optional[TestReport]:
        """获取测试报告
        
        Args:
            state: 工作流状态
            
        Returns:
            测试报告 (如果已生成)
        """
        return state.report
    
    def export_report(
        self,
        state: WorkflowState,
        format_type: str = "text",
    ) -> str:
        """导出测试报告
        
        Args:
            state: 工作流状态
            format_type: 格式类型 (text, json)
            
        Returns:
            格式化的报告字符串
        """
        if state.report is None:
            return "No report available. Run the report phase first."
        
        report_phase = self.phases.get("report")
        if isinstance(report_phase, ReportPhase):
            return report_phase.format_report(state.report, format_type)
        
        return str(state.report.to_dict())
    
    def save_report(
        self,
        state: WorkflowState,
        filepath: str,
        format_type: Optional[str] = None,
    ) -> None:
        """保存测试报告到文件
        
        Args:
            state: 工作流状态
            filepath: 文件路径
            format_type: 格式类型 (自动推断如果为 None)
        """
        if state.report is None:
            raise ValueError("No report available. Run the report phase first.")
        
        report_phase = self.phases.get("report")
        if isinstance(report_phase, ReportPhase):
            report_phase.save_report(state.report, filepath, format_type)
        else:
            content = self.export_report(state, format_type or "text")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
    
    def get_metrics(self, state: WorkflowState) -> Dict[str, Any]:
        """获取执行指标
        
        Args:
            state: 工作流状态
            
        Returns:
            指标字典
        """
        metrics = {
            "test_cases": len(state.test_cases),
            "test_results": len(state.results),
            "passed": sum(1 for r in state.results if r.passed),
            "failed": sum(1 for r in state.results if not r.passed),
            "requirements": len(state.requirements),
            "plans": len(state.plans),
            "analyses": len(state.analyses),
        }
        
        if state.report:
            metrics["coverage"] = state.report.coverage.overall_coverage
        
        return metrics
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要
        
        Returns:
            执行摘要
        """
        if not self.execution_history:
            return {"message": "No execution history"}
        
        latest = self.execution_history[-1]
        
        return {
            "total_runs": len(self.execution_history),
            "latest_run": {
                "run_id": latest["run_id"],
                "start_time": latest["start_time"].isoformat(),
                "end_time": latest["end_time"].isoformat(),
                "phases_executed": len(latest["phases"]),
                "phases_succeeded": sum(
                    1 for p in latest["phases"].values() if p["success"]
                ),
            }
        }


class IncrementalWorkflowEngine(WorkflowEngine):
    """增量式工作流引擎
    
    支持增量执行，可以复用之前阶段的结果。
    """
    
    def __init__(
        self,
        config: Optional[WorkflowConfig] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(config)
        self.cache_dir = cache_dir
        self.phase_cache: Dict[str, WorkflowState] = {}
    
    def run_from_phase(
        self,
        state: WorkflowState,
        start_phase: str,
    ) -> WorkflowState:
        """从指定阶段开始运行
        
        Args:
            state: 初始状态
            start_phase: 起始阶段名称
            
        Returns:
            最终状态
        """
        # 获取启用的阶段
        phases_to_run = self._get_enabled_phases()
        
        # 找到起始位置
        if start_phase not in phases_to_run:
            raise ValueError(f"Phase {start_phase} not in enabled phases")
        
        start_idx = phases_to_run.index(start_phase)
        phases_to_run = phases_to_run[start_idx:]
        
        return self.run(state, phases_to_run)
    
    def cache_phase_state(
        self,
        phase_name: str,
        state: WorkflowState,
    ) -> None:
        """缓存阶段状态
        
        Args:
            phase_name: 阶段名称
            state: 工作流状态
        """
        # 深拷贝状态
        import copy
        self.phase_cache[phase_name] = copy.deepcopy(state)
    
    def restore_phase_state(
        self,
        phase_name: str,
    ) -> Optional[WorkflowState]:
        """恢复阶段状态
        
        Args:
            phase_name: 阶段名称
            
        Returns:
            缓存的状态，如果不存在返回 None
        """
        import copy
        state = self.phase_cache.get(phase_name)
        return copy.deepcopy(state) if state else None


class ParallelWorkflowEngine(WorkflowEngine):
    """并行工作流引擎
    
    支持并行执行独立的测试用例。
    
    Note: 当前为预留接口，完整实现需要额外依赖。
    """
    
    def __init__(
        self,
        config: Optional[WorkflowConfig] = None,
        max_workers: int = 4,
    ):
        super().__init__(config)
        self.max_workers = max_workers
    
    def run_parallel(
        self,
        states: List[WorkflowState],
    ) -> List[WorkflowState]:
        """并行运行多个工作流
        
        Args:
            states: 工作流状态列表
            
        Returns:
            结果状态列表
        """
        # 简化实现：顺序执行
        # 完整实现可以使用 concurrent.futures.ThreadPoolExecutor
        results = []
        for state in states:
            result = self.run(state)
            results.append(result)
        return results


def create_default_engine(
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> WorkflowEngine:
    """创建默认工作流引擎
    
    Args:
        progress_callback: 进度回调函数
        
    Returns:
        配置好的工作流引擎
    """
    config = WorkflowConfig(
        enable_understand=True,
        enable_requirements=True,
        enable_planning=True,
        enable_generation=True,
        enable_execution=True,
        enable_analysis=True,
        enable_report=True,
        stop_on_error=False,  # 继续执行后续阶段
        collect_metrics=True,
    )
    
    return WorkflowEngine(config, progress_callback=progress_callback)


def run_full_workflow(
    operator: Any,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> WorkflowState:
    """便捷函数：运行完整工作流
    
    Args:
        operator: 算子适配器
        progress_callback: 进度回调函数
        
    Returns:
        最终工作流状态
    """
    engine = create_default_engine(progress_callback)
    state = WorkflowState(operator=operator)
    return engine.run(state)
