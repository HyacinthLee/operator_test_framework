"""
Stage 7: Report Generation

This stage generates comprehensive test reports in various formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from typing_extensions import override
from enum import Enum, auto
from pathlib import Path

from .base import WorkflowStage, WorkflowContext, StageResult, StageStatus


class ReportFormat(Enum):
    """Supported report formats."""
    MARKDOWN = auto()
    HTML = auto()
    JSON = auto()
    JUNIT_XML = auto()
    PDF = auto()


@dataclass
class ReportSection:
    """Section in a test report.
    
    Attributes:
        title: Section title
        content: Section content
        level: Heading level
        metadata: Section metadata
    """
    title: str
    content: str
    level: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestReport:
    """Complete test report.
    
    Attributes:
        title: Report title
        operator_name: Tested operator
        summary: Executive summary
        sections: Report sections
        metrics: Key metrics
        recommendations: Recommendations
        attachments: Attached files/artifacts
        generated_at: Generation timestamp
        metadata: Additional metadata
    """
    title: str
    operator_name: str
    summary: str
    sections: List[ReportSection] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)
    generated_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_section(self, section: ReportSection) -> None:
        """Add a section to the report."""
        ...
    
    def to_format(self, fmt: ReportFormat) -> str:
        """Convert report to specified format."""
        ...


class ReportStage(WorkflowStage):
    """Stage 7: Generate test report.
    
    This stage generates comprehensive test reports including:
    - Executive summary
    - Detailed test results
    - Failure analysis
    - Coverage metrics
    - Performance statistics
    - Recommendations
    
    Reports can be generated in multiple formats (Markdown, HTML,
    JSON, JUnit XML, PDF).
    
    Example:
        >>> stage = ReportStage(
        ...     formats=[ReportFormat.MARKDOWN, ReportFormat.HTML],
        ...     include_visualizations=True
        ... )
        >>> result = stage.execute(context)
        >>> report = result.output
        >>> stage.save_report(report, "./reports/")
    """
    
    def __init__(
        self,
        formats: Optional[List[ReportFormat]] = None,
        output_dir: Optional[str] = None,
        include_visualizations: bool = True,
        include_code_snippets: bool = True,
        template_dir: Optional[str] = None,
    ):
        """Initialize the report stage.
        
        Args:
            formats: Report formats to generate
            output_dir: Output directory for reports
            include_visualizations: Whether to include charts/plots
            include_code_snippets: Whether to include code examples
            template_dir: Custom template directory
        """
        ...
    
    @property
    @override
    def name(self) -> str:
        return "report"
    
    @property
    @override
    def description(self) -> str:
        return "Generate comprehensive test report"
    
    @override
    def can_execute(self, context: WorkflowContext) -> tuple[bool, Optional[str]]:
        """Check if analysis_result is available."""
        ...
    
    @override
    def execute(self, context: WorkflowContext) -> StageResult:
        """Generate test report.
        
        Args:
            context: Workflow context with all previous outputs
            
        Returns:
            Stage result with TestReport
        """
        ...
    
    def generate_summary_section(
        self,
        analysis_result: "AnalysisResult",
        test_results: "TestResults",
    ) -> ReportSection:
        """Generate executive summary section.
        
        Args:
            analysis_result: Analysis result
            test_results: Test results
            
        Returns:
            Summary section
        """
        ...
    
    def generate_results_section(
        self,
        test_results: List["TestResult"],
    ) -> ReportSection:
        """Generate detailed results section.
        
        Args:
            test_results: Test results
            
        Returns:
            Results section
        """
        ...
    
    def generate_analysis_section(
        self,
        analysis_result: "AnalysisResult",
    ) -> ReportSection:
        """Generate failure analysis section.
        
        Args:
            analysis_result: Analysis result
            
        Returns:
            Analysis section
        """
        ...
    
    def generate_coverage_section(
        self,
        coverage_analysis: Dict[str, float],
    ) -> ReportSection:
        """Generate coverage section.
        
        Args:
            coverage_analysis: Coverage metrics
            
        Returns:
            Coverage section
        """
        ...
    
    def generate_recommendations_section(
        self,
        repair_suggestions: List["RepairSuggestion"],
    ) -> ReportSection:
        """Generate recommendations section.
        
        Args:
            repair_suggestions: Repair suggestions
            
        Returns:
            Recommendations section
        """
        ...
    
    def render_to_markdown(self, report: TestReport) -> str:
        """Render report as Markdown.
        
        Args:
            report: Test report
            
        Returns:
            Markdown string
        """
        ...
    
    def render_to_html(self, report: TestReport) -> str:
        """Render report as HTML.
        
        Args:
            report: Test report
            
        Returns:
            HTML string
        """
        ...
    
    def render_to_json(self, report: TestReport) -> str:
        """Render report as JSON.
        
        Args:
            report: Test report
            
        Returns:
            JSON string
        """
        ...
    
    def render_to_junit(self, report: TestReport) -> str:
        """Render report as JUnit XML.
        
        Args:
            report: Test report
            
        Returns:
            JUnit XML string
        """
        ...
    
    def save_report(
        self,
        report: TestReport,
        output_dir: Union[str, Path],
        formats: Optional[List[ReportFormat]] = None,
    ) -> Dict[ReportFormat, str]:
        """Save report to files.
        
        Args:
            report: Test report
            output_dir: Output directory
            formats: Formats to save (default: all)
            
        Returns:
            Mapping from format to file path
        """
        ...
    
    def generate_visualizations(
        self,
        test_results: List["TestResult"],
        analysis_result: "AnalysisResult",
    ) -> List[str]:
        """Generate visualization charts.
        
        Args:
            test_results: Test results
            analysis_result: Analysis result
            
        Returns:
            Paths to generated visualization files
        """
        ...


class ReportRenderer(ABC):
    """Abstract base for report renderers."""
    
    @abstractmethod
    def render(self, report: TestReport) -> str:
        """Render report to string."""
        ...


class MarkdownRenderer(ReportRenderer):
    """Markdown report renderer."""
    
    @override
    def render(self, report: TestReport) -> str:
        """Render as Markdown."""
        ...


class HTMLRenderer(ReportRenderer):
    """HTML report renderer."""
    
    def __init__(self, template: Optional[str] = None):
        """Initialize with optional template."""
        ...
    
    @override
    def render(self, report: TestReport) -> str:
        """Render as HTML."""
        ...


class JSONRenderer(ReportRenderer):
    """JSON report renderer."""
    
    @override
    def render(self, report: TestReport) -> str:
        """Render as JSON."""
        ...
