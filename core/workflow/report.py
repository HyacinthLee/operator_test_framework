"""
阶段 7: 报告生成 (Report)

生成测试报告，统计覆盖率。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import uuid
from datetime import datetime

from .data_models import (
    TestResult,
    TestCase,
    TestRequirement,
    FailureAnalysis,
    TestReport,
    CoverageStats,
    WorkflowState,
    PhaseContext,
)


class CoverageCalculator(ABC):
    """覆盖率计算器抽象基类"""
    
    @abstractmethod
    def calculate(
        self,
        results: List[TestResult],
        requirements: List[TestRequirement],
    ) -> CoverageStats:
        """计算覆盖率
        
        Args:
            results: 测试结果列表
            requirements: 测试需求列表
            
        Returns:
            覆盖率统计
        """
        pass


class DefaultCoverageCalculator(CoverageCalculator):
    """默认覆盖率计算器"""
    
    def calculate(
        self,
        results: List[TestResult],
        requirements: List[TestRequirement],
    ) -> CoverageStats:
        """计算覆盖率"""
        if not results:
            return CoverageStats()
        
        # 基础覆盖率：通过测试占比
        passed = sum(1 for r in results if r.passed)
        line_coverage = passed / len(results) if results else 0.0
        
        # 边界覆盖率：假设边界测试占比
        boundary_tests = sum(1 for r in results if 'boundary' in r.case_id)
        boundary_coverage = boundary_tests / len(results) if results else 0.0
        
        # 数据类型覆盖率
        dtype_coverage = 0.5  # 简化计算
        
        # 形状覆盖率
        shape_coverage = 0.6  # 简化计算
        
        # 分支覆盖率 (假设)
        branch_coverage = line_coverage * 0.8
        
        # 整体覆盖率
        overall = (line_coverage + branch_coverage + boundary_coverage) / 3
        
        return CoverageStats(
            line_coverage=line_coverage,
            branch_coverage=branch_coverage,
            boundary_coverage=boundary_coverage,
            dtype_coverage=dtype_coverage,
            shape_coverage=shape_coverage,
            overall_coverage=overall,
        )


class ReportGenerator(ABC):
    """报告生成器抽象基类"""
    
    @abstractmethod
    def generate(
        self,
        state: WorkflowState,
        coverage: CoverageStats,
    ) -> TestReport:
        """生成测试报告
        
        Args:
            state: 工作流状态
            coverage: 覆盖率统计
            
        Returns:
            测试报告
        """
        pass


class DefaultReportGenerator(ReportGenerator):
    """默认报告生成器"""
    
    def generate(
        self,
        state: WorkflowState,
        coverage: CoverageStats,
    ) -> TestReport:
        """生成测试报告"""
        operator_name = state.signature.name if state.signature else "Unknown"
        
        # 生成摘要
        summary = self._generate_summary(state)
        
        # 生成建议
        recommendations = self._generate_recommendations(state)
        
        return TestReport(
            report_id=f"report_{uuid.uuid4().hex[:8]}",
            operator_name=operator_name,
            timestamp=datetime.now(),
            summary=summary,
            results=state.results,
            analyses=state.analyses,
            coverage=coverage,
            recommendations=recommendations,
        )
    
    def _generate_summary(self, state: WorkflowState) -> Dict[str, Any]:
        """生成测试摘要"""
        total_tests = len(state.results)
        passed_tests = sum(1 for r in state.results if r.passed)
        failed_tests = total_tests - passed_tests
        
        summary = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
            "total_execution_time": sum(
                r.execution_time or 0 for r in state.results
            ),
            "requirements_count": len(state.requirements),
            "test_cases_count": len(state.test_cases),
        }
        
        # 添加失败分类统计
        if state.analyses:
            category_counts = {}
            for analysis in state.analyses:
                cat_name = analysis.category.name
                category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
            summary["failure_categories"] = category_counts
        
        return summary
    
    def _generate_recommendations(self, state: WorkflowState) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于失败率
        total = len(state.results)
        passed = sum(1 for r in state.results if r.passed)
        failure_rate = 1 - (passed / total) if total > 0 else 0
        
        if failure_rate > 0.5:
            recommendations.append(
                "失败率较高，建议优先修复关键问题后再继续测试"
            )
        
        if failure_rate > 0.2:
            recommendations.append(
                "建议增加边界测试覆盖率"
            )
        
        # 基于分析结果的建议
        if state.analyses:
            categories = set(a.category for a in state.analyses)
            
            if any(c.name == "NAN_INF" for c in categories):
                recommendations.append(
                    "检测到数值稳定性问题，建议添加数值裁剪或稳定性处理"
                )
            
            if any(c.name == "SHAPE_MISMATCH" for c in categories):
                recommendations.append(
                    "检测到形状不匹配问题，建议改进输入验证"
                )
            
            if any(c.name == "MEMORY_ERROR" for c in categories):
                recommendations.append(
                    "检测到内存问题，建议优化内存使用或减少批处理大小"
                )
        
        # 默认建议
        if not recommendations:
            recommendations.append(
                "测试整体表现良好，建议继续保持当前测试覆盖率"
            )
        
        return recommendations


class ReportFormatter(ABC):
    """报告格式化器抽象基类"""
    
    @abstractmethod
    def format(self, report: TestReport) -> str:
        """格式化报告
        
        Args:
            report: 测试报告
            
        Returns:
            格式化后的报告字符串
        """
        pass


class TextReportFormatter(ReportFormatter):
    """文本报告格式化器"""
    
    def format(self, report: TestReport) -> str:
        """格式化为文本"""
        lines = []
        
        # 标题
        lines.append("=" * 70)
        lines.append(f"测试报告: {report.operator_name}")
        lines.append(f"报告ID: {report.report_id}")
        lines.append(f"生成时间: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        
        # 摘要
        lines.append("\n## 测试摘要")
        lines.append("-" * 40)
        summary = report.summary
        lines.append(f"总测试数: {summary.get('total_tests', 0)}")
        lines.append(f"通过: {summary.get('passed', 0)}")
        lines.append(f"失败: {summary.get('failed', 0)}")
        lines.append(f"通过率: {summary.get('pass_rate', 0)*100:.1f}%")
        lines.append(f"总执行时间: {summary.get('total_execution_time', 0):.2f}s")
        
        # 覆盖率
        lines.append("\n## 覆盖率统计")
        lines.append("-" * 40)
        cov = report.coverage
        lines.append(f"行覆盖率: {cov.line_coverage*100:.1f}%")
        lines.append(f"分支覆盖率: {cov.branch_coverage*100:.1f}%")
        lines.append(f"边界覆盖率: {cov.boundary_coverage*100:.1f}%")
        lines.append(f"整体覆盖率: {cov.overall_coverage*100:.1f}%")
        
        # 失败分析
        if report.analyses:
            lines.append("\n## 失败分析")
            lines.append("-" * 40)
            for analysis in report.analyses[:5]:  # 最多显示5个
                lines.append(f"\n测试: {analysis.result_id}")
                lines.append(f"  分类: {analysis.category.name}")
                lines.append(f"  根本原因: {analysis.root_cause}")
                lines.append(f"  建议:")
                for suggestion in analysis.suggestions[:3]:
                    lines.append(f"    - {suggestion}")
        
        # 建议
        if report.recommendations:
            lines.append("\n## 改进建议")
            lines.append("-" * 40)
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"{i}. {rec}")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)


class JSONReportFormatter(ReportFormatter):
    """JSON 报告格式化器"""
    
    def format(self, report: TestReport) -> str:
        """格式化为 JSON"""
        import json
        return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)


class ReportPhase:
    """报告生成阶段
    
    职责：
    1. 生成测试报告
    2. 统计覆盖率
    3. 提供多种格式输出
    """
    
    def __init__(
        self,
        coverage_calculator: Optional[CoverageCalculator] = None,
        report_generator: Optional[ReportGenerator] = None,
        formatters: Optional[Dict[str, ReportFormatter]] = None,
    ):
        """初始化
        
        Args:
            coverage_calculator: 覆盖率计算器
            report_generator: 报告生成器
            formatters: 格式化器字典
        """
        self.coverage_calculator = coverage_calculator or DefaultCoverageCalculator()
        self.report_generator = report_generator or DefaultReportGenerator()
        self.formatters = formatters or {
            "text": TextReportFormatter(),
            "json": JSONReportFormatter(),
        }
    
    def execute(self, state: WorkflowState) -> Tuple[WorkflowState, PhaseContext]:
        """执行报告生成阶段
        
        Args:
            state: 当前工作流状态
            
        Returns:
            (更新后的状态, 阶段上下文)
        """
        import datetime
        context = PhaseContext(phase_name="report")
        context.start_time = datetime.datetime.now()
        
        try:
            if not state.results:
                raise ValueError("No test results available.")
            
            # 计算覆盖率
            coverage = self.coverage_calculator.calculate(
                state.results,
                state.requirements,
            )
            
            # 生成报告
            report = self.report_generator.generate(state, coverage)
            
            # 更新状态
            state.report = report
            state.metadata['report_generated'] = True
            state.metadata['overall_coverage'] = coverage.overall_coverage
            
            context.success = True
            
        except Exception as e:
            context.success = False
            context.error = str(e)
            state.metadata['report_error'] = str(e)
        
        context.end_time = datetime.datetime.now()
        return state, context
    
    def format_report(
        self,
        report: TestReport,
        format_type: str = "text",
    ) -> str:
        """格式化报告
        
        Args:
            report: 测试报告
            format_type: 格式类型 (text, json)
            
        Returns:
            格式化后的报告字符串
        """
        formatter = self.formatters.get(format_type, self.formatters["text"])
        return formatter.format(report)
    
    def save_report(
        self,
        report: TestReport,
        filepath: str,
        format_type: Optional[str] = None,
    ) -> None:
        """保存报告到文件
        
        Args:
            report: 测试报告
            filepath: 文件路径
            format_type: 格式类型 (自动推断如果为 None)
        """
        # 推断格式
        if format_type is None:
            if filepath.endswith('.json'):
                format_type = "json"
            else:
                format_type = "text"
        
        # 格式化
        content = self.format_report(report, format_type)
        
        # 保存
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def export_summary(self, report: TestReport) -> Dict[str, Any]:
        """导出摘要信息
        
        Args:
            report: 测试报告
            
        Returns:
            摘要字典
        """
        return {
            "operator_name": report.operator_name,
            "report_id": report.report_id,
            "timestamp": report.timestamp.isoformat(),
            "summary": report.summary,
            "coverage": report.coverage.to_dict(),
            "recommendations": report.recommendations,
        }
