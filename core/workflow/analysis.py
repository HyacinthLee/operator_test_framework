"""
阶段 6: 结果分析 (Analysis)

分类失败原因，生成修复建议。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Set
import re
import torch

from .data_models import (
    TestResult,
    TestCase,
    FailureAnalysis,
    FailureCategory,
    WorkflowState,
    PhaseContext,
)


class FailureAnalyzer(ABC):
    """失败分析器抽象基类"""
    
    @abstractmethod
    def analyze(self, result: TestResult) -> FailureCategory:
        """分析失败类型
        
        Args:
            result: 测试结果
            
        Returns:
            失败分类
        """
        pass


class RuleBasedFailureAnalyzer(FailureAnalyzer):
    """基于规则的失败分析器"""
    
    def __init__(self):
        self.error_patterns = {
            FailureCategory.SHAPE_MISMATCH: [
                r"shape.*mismatch",
                r"size mismatch",
                r"expected.*dimension",
                r"Dimension out of range",
            ],
            FailureCategory.DTYPE_MISMATCH: [
                r"dtype.*mismatch",
                r"expected.*type",
                r"not supported for",
                r"Invalid dtype",
            ],
            FailureCategory.DEVICE_MISMATCH: [
                r"device.*mismatch",
                r"Expected.*device",
                r"must be on",
            ],
            FailureCategory.NAN_INF: [
                r"nan",
                r"inf",
                r"overflow",
                r"underflow",
            ],
            FailureCategory.GRADIENT_ERROR: [
                r"grad",
                r"backward",
                r"autograd",
            ],
            FailureCategory.MEMORY_ERROR: [
                r"out of memory",
                r"cuda out of memory",
                r"oom",
            ],
            FailureCategory.TIMEOUT: [
                r"timeout",
                r"time limit",
            ],
        }
    
    def analyze(self, result: TestResult) -> FailureCategory:
        """分析失败类型"""
        if result.passed:
            return FailureCategory.UNKNOWN
        
        error_message = result.error_message or ""
        stack_trace = result.stack_trace or ""
        combined = error_message.lower() + " " + stack_trace.lower()
        
        for category, patterns in self.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, combined, re.IGNORECASE):
                    return category
        
        # 检查输出中的 NaN/Inf
        if result.actual_outputs:
            for name, tensor in result.actual_outputs.items():
                if isinstance(tensor, torch.Tensor):
                    if torch.isnan(tensor).any():
                        return FailureCategory.NAN_INF
                    if torch.isinf(tensor).any():
                        return FailureCategory.NAN_INF
        
        return FailureCategory.UNKNOWN


class FixSuggester(ABC):
    """修复建议器抽象基类"""
    
    @abstractmethod
    def suggest(
        self,
        result: TestResult,
        category: FailureCategory,
    ) -> List[str]:
        """生成修复建议
        
        Args:
            result: 测试结果
            category: 失败分类
            
        Returns:
            修复建议列表
        """
        pass


class RuleBasedFixSuggester(FixSuggester):
    """基于规则的修复建议器"""
    
    def suggest(
        self,
        result: TestResult,
        category: FailureCategory,
    ) -> List[str]:
        """生成修复建议"""
        suggestions = {
            FailureCategory.SHAPE_MISMATCH: [
                "检查输入张量的形状是否符合算子要求",
                "验证批处理维度是否一致",
                "使用 torch.reshape 或 torch.view 调整形状",
                "考虑使用 broadcasting 规则",
            ],
            FailureCategory.DTYPE_MISMATCH: [
                "使用 tensor.to(dtype) 转换数据类型",
                "检查算子支持的数据类型列表",
                "确保所有输入具有相同的数据类型",
                "考虑使用混合精度训练策略",
            ],
            FailureCategory.DEVICE_MISMATCH: [
                "使用 tensor.to(device) 将张量移动到正确设备",
                "确保所有输入都在同一设备上",
                "检查 CUDA 是否可用",
            ],
            FailureCategory.NAN_INF: [
                "检查输入是否包含 NaN 或 Inf",
                "添加数值稳定性处理 (如梯度裁剪)",
                "使用 torch.nan_to_num 处理异常值",
                "检查除零或 log(0) 情况",
            ],
            FailureCategory.GRADIENT_ERROR: [
                "检查 requires_grad 标志",
                "验证计算图是否完整",
                "使用 torch.autograd.gradcheck 检查梯度",
                "考虑使用 create_graph=True",
            ],
            FailureCategory.MEMORY_ERROR: [
                "减小批处理大小",
                "使用 torch.cuda.empty_cache() 释放缓存",
                "检查是否有未释放的张量",
                "考虑使用梯度检查点",
            ],
            FailureCategory.TIMEOUT: [
                "优化算子实现",
                "检查是否有无限循环",
                "考虑使用更高效的算法",
            ],
            FailureCategory.UNKNOWN: [
                "查看完整错误堆栈",
                "添加更多日志信息",
                "使用调试器逐步执行",
                "参考算子文档",
            ],
        }
        
        return suggestions.get(category, ["未知错误类型，需要手动调查"])


class AnalysisPhase:
    """结果分析阶段
    
    职责：
    1. 分类失败原因
    2. 生成修复建议
    3. 识别相关失败
    """
    
    def __init__(
        self,
        failure_analyzer: Optional[FailureAnalyzer] = None,
        fix_suggester: Optional[FixSuggester] = None,
    ):
        """初始化
        
        Args:
            failure_analyzer: 失败分析器
            fix_suggester: 修复建议器
        """
        self.failure_analyzer = failure_analyzer or RuleBasedFailureAnalyzer()
        self.fix_suggester = fix_suggester or RuleBasedFixSuggester()
    
    def execute(self, state: WorkflowState) -> Tuple[WorkflowState, PhaseContext]:
        """执行结果分析阶段
        
        Args:
            state: 当前工作流状态
            
        Returns:
            (更新后的状态, 阶段上下文)
        """
        import datetime
        context = PhaseContext(phase_name="analysis")
        context.start_time = datetime.datetime.now()
        
        try:
            if not state.results:
                raise ValueError("No test results available. Run execution phase first.")
            
            analyses = []
            
            # 分析每个失败结果
            for result in state.results:
                if not result.passed:
                    analysis = self._analyze_result(result, state.results)
                    analyses.append(analysis)
            
            # 更新状态
            state.analyses = analyses
            state.metadata['analyses_count'] = len(analyses)
            state.metadata['failure_categories'] = self._count_by_category(analyses)
            
            # 计算修复建议
            all_suggestions = []
            for analysis in analyses:
                all_suggestions.extend(analysis.suggestions)
            state.metadata['unique_suggestions'] = list(set(all_suggestions))
            
            context.success = True
            
        except Exception as e:
            context.success = False
            context.error = str(e)
            state.metadata['analysis_error'] = str(e)
        
        context.end_time = datetime.datetime.now()
        return state, context
    
    def _analyze_result(
        self,
        result: TestResult,
        all_results: List[TestResult],
    ) -> FailureAnalysis:
        """分析单个失败结果"""
        # 分类失败
        category = self.failure_analyzer.analyze(result)
        
        # 生成修复建议
        suggestions = self.fix_suggester.suggest(result, category)
        
        # 查找相关失败
        related_cases = self._find_related_cases(result, category, all_results)
        
        # 分析根本原因
        root_cause = self._analyze_root_cause(result, category)
        
        return FailureAnalysis(
            result_id=result.case_id,
            category=category,
            root_cause=root_cause,
            suggestions=suggestions,
            related_cases=related_cases,
            confidence=0.8,  # 基于规则的置信度
        )
    
    def _find_related_cases(
        self,
        result: TestResult,
        category: FailureCategory,
        all_results: List[TestResult],
    ) -> List[str]:
        """查找相关失败用例"""
        related = []
        
        for other in all_results:
            if other.case_id == result.case_id:
                continue
            if other.passed:
                continue
            
            # 检查错误消息相似性
            if result.error_message and other.error_message:
                similarity = self._calculate_similarity(
                    result.error_message,
                    other.error_message,
                )
                if similarity > 0.7:  # 70% 相似度阈值
                    related.append(other.case_id)
        
        return related
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """计算字符串相似度 (简化实现)"""
        # 使用简单的词袋模型
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _analyze_root_cause(
        self,
        result: TestResult,
        category: FailureCategory,
    ) -> str:
        """分析根本原因"""
        root_causes = {
            FailureCategory.SHAPE_MISMATCH: "输入张量形状不符合算子要求",
            FailureCategory.DTYPE_MISMATCH: "数据类型不匹配或不支持",
            FailureCategory.DEVICE_MISMATCH: "设备不一致",
            FailureCategory.NAN_INF: "数值稳定性问题导致 NaN/Inf",
            FailureCategory.GRADIENT_ERROR: "梯度计算错误",
            FailureCategory.MEMORY_ERROR: "内存不足",
            FailureCategory.TIMEOUT: "执行超时",
            FailureCategory.UNKNOWN: "未知原因",
        }
        
        base_cause = root_causes.get(category, "未知原因")
        
        if result.error_message:
            return f"{base_cause}: {result.error_message[:100]}"
        
        return base_cause
    
    def _count_by_category(
        self,
        analyses: List[FailureAnalysis]
    ) -> Dict[str, int]:
        """统计各分类的失败数量"""
        counts = {}
        for analysis in analyses:
            category_name = analysis.category.name
            counts[category_name] = counts.get(category_name, 0) + 1
        return counts
    
    def get_analysis_by_category(
        self,
        state: WorkflowState,
        category: FailureCategory,
    ) -> List[FailureAnalysis]:
        """获取特定分类的分析结果
        
        Args:
            state: 工作流状态
            category: 失败分类
            
        Returns:
            该分类的分析结果列表
        """
        return [a for a in state.analyses if a.category == category]
    
    def get_critical_failures(
        self,
        state: WorkflowState,
    ) -> List[FailureAnalysis]:
        """获取关键失败（高影响）
        
        Args:
            state: 工作流状态
            
        Returns:
            关键失败分析列表
        """
        critical_categories = {
            FailureCategory.CRASH,
            FailureCategory.MEMORY_ERROR,
            FailureCategory.NAN_INF,
        }
        return [a for a in state.analyses if a.category in critical_categories]
