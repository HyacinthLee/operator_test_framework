"""
阶段 5: 执行测试 (Execution)

运行测试，收集结果和日志。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Callable
import time
import traceback
import torch

from .data_models import (
    TestCase,
    TestResult,
    WorkflowState,
    PhaseContext,
)


class ResultCollector(ABC):
    """结果收集器抽象基类"""
    
    @abstractmethod
    def collect(
        self,
        case_id: str,
        success: bool,
        outputs: Optional[Dict[str, torch.Tensor]],
        execution_time: float,
        error: Optional[Exception] = None,
    ) -> TestResult:
        """收集测试结果
        
        Args:
            case_id: 测试用例 ID
            success: 是否成功
            outputs: 输出 tensors
            execution_time: 执行时间
            error: 异常信息
            
        Returns:
            测试结果
        """
        pass


class DefaultResultCollector(ResultCollector):
    """默认结果收集器"""
    
    def collect(
        self,
        case_id: str,
        success: bool,
        outputs: Optional[Dict[str, torch.Tensor]],
        execution_time: float,
        error: Optional[Exception] = None,
    ) -> TestResult:
        """收集测试结果"""
        error_message = None
        stack_trace = None
        
        if error is not None:
            error_message = str(error)
            stack_trace = traceback.format_exc()
        
        # 估算内存使用
        memory_usage = self._estimate_memory_usage(outputs)
        
        return TestResult(
            case_id=case_id,
            passed=success,
            actual_outputs=outputs,
            execution_time=execution_time,
            memory_usage=memory_usage,
            error_message=error_message,
            stack_trace=stack_trace,
        )
    
    def _estimate_memory_usage(
        self,
        outputs: Optional[Dict[str, torch.Tensor]]
    ) -> Optional[float]:
        """估算内存使用 (MB)"""
        if outputs is None:
            return None
        
        total_bytes = 0
        for tensor in outputs.values():
            if isinstance(tensor, torch.Tensor):
                total_bytes += tensor.numel() * tensor.element_size()
        
        return total_bytes / (1024 * 1024)


class TestExecutor(ABC):
    """测试执行器抽象基类"""
    
    @abstractmethod
    def execute(
        self,
        test_case: TestCase,
        operator: Any,
        result_collector: ResultCollector,
    ) -> TestResult:
        """执行单个测试用例
        
        Args:
            test_case: 测试用例
            operator: 算子适配器
            result_collector: 结果收集器
            
        Returns:
            测试结果
        """
        pass


class DefaultTestExecutor(TestExecutor):
    """默认测试执行器"""
    
    def __init__(
        self,
        timeout: Optional[float] = None,
        catch_exceptions: bool = True,
    ):
        """初始化
        
        Args:
            timeout: 超时时间 (秒)
            catch_exceptions: 是否捕获异常
        """
        self.timeout = timeout
        self.catch_exceptions = catch_exceptions
    
    def execute(
        self,
        test_case: TestCase,
        operator: Any,
        result_collector: ResultCollector,
    ) -> TestResult:
        """执行测试用例"""
        start_time = time.time()
        
        try:
            # 准备输入
            inputs = test_case.inputs
            
            # 同步 CUDA (如果使用)
            if any(t.is_cuda for t in inputs.values() if isinstance(t, torch.Tensor)):
                torch.cuda.synchronize()
            
            # 执行前向传播
            outputs = operator.forward(inputs)
            
            # 同步 CUDA
            if any(t.is_cuda for t in inputs.values() if isinstance(t, torch.Tensor)):
                torch.cuda.synchronize()
            
            execution_time = time.time() - start_time
            
            # 包装输出
            if isinstance(outputs, torch.Tensor):
                outputs_dict = {"output": outputs}
            elif isinstance(outputs, dict):
                outputs_dict = outputs
            else:
                outputs_dict = {"output": outputs}
            
            return result_collector.collect(
                case_id=test_case.id,
                success=True,
                outputs=outputs_dict,
                execution_time=execution_time,
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            if not self.catch_exceptions:
                raise
            
            return result_collector.collect(
                case_id=test_case.id,
                success=False,
                outputs=None,
                execution_time=execution_time,
                error=e,
            )


class GradientTestExecutor(TestExecutor):
    """梯度测试执行器"""
    
    def __init__(
        self,
        base_executor: Optional[TestExecutor] = None,
        check_gradients: bool = True,
        rtol: float = 1e-4,
        atol: float = 1e-6,
    ):
        self.base_executor = base_executor or DefaultTestExecutor()
        self.check_gradients = check_gradients
        self.rtol = rtol
        self.atol = atol
    
    def execute(
        self,
        test_case: TestCase,
        operator: Any,
        result_collector: ResultCollector,
    ) -> TestResult:
        """执行测试并检查梯度"""
        # 首先执行基础测试
        result = self.base_executor.execute(
            test_case, operator, result_collector
        )
        
        if not result.passed or not self.check_gradients:
            return result
        
        try:
            # 准备带梯度的输入
            inputs = {
                k: v.requires_grad_(True) if not v.requires_grad else v
                for k, v in test_case.inputs.items()
            }
            
            # 前向传播
            outputs = operator.forward(inputs)
            
            if isinstance(outputs, torch.Tensor):
                # 反向传播
                outputs.sum().backward()
                
                # 检查梯度
                for name, tensor in inputs.items():
                    if tensor.requires_grad:
                        if tensor.grad is None:
                            result.passed = False
                            result.error_message = f"Missing gradient for {name}"
                            break
                        elif torch.isnan(tensor.grad).any():
                            result.passed = False
                            result.error_message = f"NaN gradient for {name}"
                            break
            
        except Exception as e:
            result.passed = False
            result.error_message = f"Gradient check failed: {str(e)}"
        
        return result


class ExecutionPhase:
    """执行测试阶段
    
    职责：
    1. 运行测试
    2. 收集结果和日志
    3. 处理异常
    """
    
    def __init__(
        self,
        executor: Optional[TestExecutor] = None,
        result_collector: Optional[ResultCollector] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """初始化
        
        Args:
            executor: 测试执行器
            result_collector: 结果收集器
            progress_callback: 进度回调函数 (current, total)
        """
        self.executor = executor or DefaultTestExecutor()
        self.result_collector = result_collector or DefaultResultCollector()
        self.progress_callback = progress_callback
    
    def execute(self, state: WorkflowState) -> Tuple[WorkflowState, PhaseContext]:
        """执行测试阶段
        
        Args:
            state: 当前工作流状态
            
        Returns:
            (更新后的状态, 阶段上下文)
        """
        import datetime
        context = PhaseContext(phase_name="execution")
        context.start_time = datetime.datetime.now()
        
        try:
            if not state.test_cases:
                raise ValueError("No test cases available. Run generation phase first.")
            
            if state.operator is None:
                raise ValueError("No operator provided.")
            
            results = []
            total = len(state.test_cases)
            
            for i, test_case in enumerate(state.test_cases):
                # 执行测试
                result = self.executor.execute(
                    test_case,
                    state.operator,
                    self.result_collector,
                )
                results.append(result)
                
                # 进度回调
                if self.progress_callback:
                    self.progress_callback(i + 1, total)
            
            # 更新状态
            state.results = results
            state.metadata['results_count'] = len(results)
            state.metadata['passed_count'] = sum(1 for r in results if r.passed)
            state.metadata['failed_count'] = sum(1 for r in results if not r.passed)
            state.metadata['total_execution_time'] = sum(
                r.execution_time or 0 for r in results
            )
            
            context.success = True
            
        except Exception as e:
            context.success = False
            context.error = str(e)
            state.metadata['execution_error'] = str(e)
        
        context.end_time = datetime.datetime.now()
        return state, context
    
    def execute_single(
        self,
        test_case: TestCase,
        operator: Any,
    ) -> TestResult:
        """执行单个测试用例
        
        Args:
            test_case: 测试用例
            operator: 算子适配器
            
        Returns:
            测试结果
        """
        return self.executor.execute(
            test_case,
            operator,
            self.result_collector,
        )
    
    def get_failed_results(self, state: WorkflowState) -> List[TestResult]:
        """获取失败的测试结果
        
        Args:
            state: 工作流状态
            
        Returns:
            失败的测试结果列表
        """
        return [r for r in state.results if not r.passed]
    
    def get_passed_results(self, state: WorkflowState) -> List[TestResult]:
        """获取通过的测试结果
        
        Args:
            state: 工作流状态
            
        Returns:
            通过的测试结果列表
        """
        return [r for r in state.results if r.passed]
