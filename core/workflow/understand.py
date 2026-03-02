"""
阶段 1: 算子理解 (Understand)

解析算子签名，提取 tensor 约束（shape、dtype、device），识别边界条件。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
import inspect
import torch
import torch.nn as nn

from .data_models import (
    OperatorSignature,
    TensorConstraint,
    WorkflowState,
    PhaseContext,
)


class ConstraintExtractor(ABC):
    """约束提取器抽象基类"""
    
    @abstractmethod
    def extract_shape_constraints(self, obj: Any) -> Dict[str, Any]:
        """提取形状约束"""
        pass
    
    @abstractmethod
    def extract_dtype_constraints(self, obj: Any) -> Dict[str, Any]:
        """提取数据类型约束"""
        pass
    
    @abstractmethod
    def extract_device_constraints(self, obj: Any) -> Dict[str, Any]:
        """提取设备约束"""
        pass


class SignatureParser(ABC):
    """签名解析器抽象基类"""
    
    @abstractmethod
    def parse(self, operator: Any) -> OperatorSignature:
        """解析算子签名"""
        pass


class PyTorchSignatureParser(SignatureParser):
    """PyTorch 算子签名解析器"""
    
    def parse(self, operator: Any) -> OperatorSignature:
        """解析 PyTorch 算子签名
        
        Args:
            operator: 算子适配器或 nn.Module 实例
            
        Returns:
            OperatorSignature 实例
        """
        # 获取算子名称
        name = self._extract_name(operator)
        
        # 提取输入约束
        inputs = self._extract_input_constraints(operator)
        
        # 提取输出约束
        outputs = self._extract_output_constraints(operator)
        
        # 提取参数
        parameters = self._extract_parameters(operator)
        
        # 提取属性
        attributes = self._extract_attributes(operator)
        
        # 检查是否原地操作
        is_in_place = self._check_in_place(operator)
        
        # 检查是否支持自动微分
        supports_autograd = self._check_autograd_support(operator)
        
        # 识别边界条件
        boundary_conditions = self._identify_boundary_conditions(operator, inputs)
        
        return OperatorSignature(
            name=name,
            inputs=inputs,
            outputs=outputs,
            parameters=parameters,
            attributes=attributes,
            is_in_place=is_in_place,
            supports_autograd=supports_autograd,
            backend="pytorch",
            boundary_conditions=boundary_conditions,
        )
    
    def _extract_name(self, operator: Any) -> str:
        """提取算子名称"""
        if hasattr(operator, 'name'):
            return operator.name
        elif hasattr(operator, '__class__'):
            return operator.__class__.__name__
        else:
            return str(operator)
    
    def _extract_input_constraints(self, operator: Any) -> List[TensorConstraint]:
        """提取输入 tensor 约束"""
        constraints = []
        
        # 尝试从 generate_inputs 方法推断
        if hasattr(operator, 'generate_inputs'):
            try:
                # 生成样本输入来分析
                sample_inputs = operator.generate_inputs(batch_size=2)
                for name, tensor in sample_inputs.items():
                    constraint = TensorConstraint(
                        name=name,
                        shape=tuple(tensor.shape),
                        dtype=tensor.dtype,
                        device=str(tensor.device),
                        requires_grad=tensor.requires_grad,
                    )
                    constraints.append(constraint)
            except Exception:
                pass
        
        # 如果是 nn.Module，检查 forward 签名
        if isinstance(operator, nn.Module) or hasattr(operator, 'forward'):
            try:
                sig = inspect.signature(operator.forward)
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue
                    # 如果还没有这个名称的约束，添加一个通用约束
                    if not any(c.name == param_name for c in constraints):
                        constraints.append(TensorConstraint(
                            name=param_name,
                            shape="*",  # 未知形状
                        ))
            except Exception:
                pass
        
        return constraints
    
    def _extract_output_constraints(self, operator: Any) -> List[TensorConstraint]:
        """提取输出 tensor 约束"""
        constraints = []
        
        # 尝试运行一次前向传播来分析输出
        if hasattr(operator, 'generate_inputs') and hasattr(operator, 'forward'):
            try:
                sample_inputs = operator.generate_inputs(batch_size=2)
                with torch.no_grad():
                    outputs = operator.forward(sample_inputs)
                
                if isinstance(outputs, torch.Tensor):
                    constraints.append(TensorConstraint(
                        name="output",
                        shape=tuple(outputs.shape),
                        dtype=outputs.dtype,
                        device=str(outputs.device),
                    ))
                elif isinstance(outputs, (tuple, list)):
                    for i, out in enumerate(outputs):
                        if isinstance(out, torch.Tensor):
                            constraints.append(TensorConstraint(
                                name=f"output_{i}",
                                shape=tuple(out.shape),
                                dtype=out.dtype,
                                device=str(out.device),
                            ))
                elif isinstance(outputs, dict):
                    for name, out in outputs.items():
                        if isinstance(out, torch.Tensor):
                            constraints.append(TensorConstraint(
                                name=name,
                                shape=tuple(out.shape),
                                dtype=out.dtype,
                                device=str(out.device),
                            ))
            except Exception:
                pass
        
        return constraints
    
    def _extract_parameters(self, operator: Any) -> Dict[str, Any]:
        """提取算子参数"""
        parameters = {}
        
        if isinstance(operator, nn.Module):
            # 提取模块参数
            for name, param in operator.named_parameters():
                parameters[name] = {
                    "shape": tuple(param.shape),
                    "dtype": str(param.dtype),
                    "requires_grad": param.requires_grad,
                }
        
        # 提取配置参数
        if hasattr(operator, 'config'):
            config = operator.config
            if hasattr(config, '__dict__'):
                parameters['config'] = {
                    k: str(v) if not isinstance(v, (int, float, bool, str)) else v
                    for k, v in config.__dict__.items()
                    if not k.startswith('_')
                }
        
        return parameters
    
    def _extract_attributes(self, operator: Any) -> Dict[str, Any]:
        """提取算子属性"""
        attributes = {}
        
        # 提取常见属性
        for attr in ['training', 'dtype', 'device']:
            if hasattr(operator, attr):
                try:
                    val = getattr(operator, attr)
                    attributes[attr] = str(val) if not isinstance(val, (int, float, bool, str)) else val
                except Exception:
                    pass
        
        return attributes
    
    def _check_in_place(self, operator: Any) -> bool:
        """检查是否为原地操作"""
        name = self._extract_name(operator).lower()
        in_place_suffixes = ['_', '_in_place', 'inplace']
        return any(name.endswith(suffix) for suffix in in_place_suffixes)
    
    def _check_autograd_support(self, operator: Any) -> bool:
        """检查是否支持自动微分"""
        if hasattr(operator, 'supports_autograd'):
            return operator.supports_autograd
        
        # 默认假设支持
        return True
    
    def _identify_boundary_conditions(
        self, 
        operator: Any, 
        inputs: List[TensorConstraint]
    ) -> List[Dict[str, Any]]:
        """识别边界条件"""
        boundary_conditions = []
        
        # 常见的边界条件类型
        boundary_types = [
            {
                "name": "empty_batch",
                "description": "空 batch (batch_size=0)",
                "condition": {"batch_size": 0},
                "severity": "high",
            },
            {
                "name": "single_element",
                "description": "单元素 batch (batch_size=1)",
                "condition": {"batch_size": 1},
                "severity": "medium",
            },
            {
                "name": "large_batch",
                "description": "大 batch (batch_size=1024)",
                "condition": {"batch_size": 1024},
                "severity": "medium",
            },
            {
                "name": "zero_values",
                "description": "全零输入",
                "condition": {"fill": 0},
                "severity": "high",
            },
            {
                "name": "negative_values",
                "description": "负值输入",
                "condition": {"min_value": -1.0, "max_value": 0},
                "severity": "medium",
            },
            {
                "name": "very_small_values",
                "description": "极小值 (1e-7)",
                "condition": {"scale": 1e-7},
                "severity": "medium",
            },
            {
                "name": "very_large_values",
                "description": "极大值 (1e7)",
                "condition": {"scale": 1e7},
                "severity": "high",
            },
            {
                "name": "nan_input",
                "description": "NaN 输入",
                "condition": {"inject_nan": True},
                "severity": "high",
            },
            {
                "name": "inf_input",
                "description": "Inf 输入",
                "condition": {"inject_inf": True},
                "severity": "high",
            },
        ]
        
        for boundary in boundary_types:
            # 根据输入约束调整边界条件
            applicable = self._check_boundary_applicable(boundary, inputs)
            if applicable:
                boundary_conditions.append(boundary)
        
        return boundary_conditions
    
    def _check_boundary_applicable(
        self, 
        boundary: Dict[str, Any], 
        inputs: List[TensorConstraint]
    ) -> bool:
        """检查边界条件是否适用于给定输入"""
        # 默认所有边界条件都适用
        return True


class DefaultConstraintExtractor(ConstraintExtractor):
    """默认约束提取器"""
    
    def extract_shape_constraints(self, obj: Any) -> Dict[str, Any]:
        """提取形状约束"""
        if isinstance(obj, torch.Tensor):
            return {
                "rank": len(obj.shape),
                "shape": tuple(obj.shape),
                "min_dims": len(obj.shape),
                "max_dims": len(obj.shape),
            }
        return {}
    
    def extract_dtype_constraints(self, obj: Any) -> Dict[str, Any]:
        """提取数据类型约束"""
        if isinstance(obj, torch.Tensor):
            return {
                "dtype": str(obj.dtype),
                "is_floating": obj.dtype.is_floating_point if hasattr(obj.dtype, 'is_floating_point') else False,
                "is_integer": obj.dtype in [torch.int8, torch.int16, torch.int32, torch.int64],
            }
        return {}
    
    def extract_device_constraints(self, obj: Any) -> Dict[str, Any]:
        """提取设备约束"""
        if isinstance(obj, torch.Tensor):
            return {
                "device": str(obj.device),
                "is_cuda": obj.is_cuda,
            }
        return {}


class UnderstandPhase:
    """算子理解阶段
    
    职责：
    1. 解析算子签名
    2. 提取 tensor 约束
    3. 识别边界条件
    """
    
    def __init__(
        self,
        signature_parser: Optional[SignatureParser] = None,
        constraint_extractor: Optional[ConstraintExtractor] = None,
    ):
        """初始化
        
        Args:
            signature_parser: 签名解析器
            constraint_extractor: 约束提取器
        """
        self.signature_parser = signature_parser or PyTorchSignatureParser()
        self.constraint_extractor = constraint_extractor or DefaultConstraintExtractor()
    
    def execute(self, state: WorkflowState) -> Tuple[WorkflowState, PhaseContext]:
        """执行算子理解阶段
        
        Args:
            state: 当前工作流状态
            
        Returns:
            (更新后的状态, 阶段上下文)
        """
        import datetime
        context = PhaseContext(phase_name="understand")
        context.start_time = datetime.datetime.now()
        
        try:
            if state.operator is None:
                raise ValueError("No operator provided in workflow state")
            
            # 解析算子签名
            signature = self.signature_parser.parse(state.operator)
            
            # 为每个输入提取额外约束
            for constraint in signature.inputs:
                # 提取形状约束
                shape_constraints = self.constraint_extractor.extract_shape_constraints(
                    constraint
                )
                constraint.additional_constraints.update(shape_constraints)
                
                # 提取 dtype 约束
                dtype_constraints = self.constraint_extractor.extract_dtype_constraints(
                    constraint
                )
                constraint.additional_constraints.update(dtype_constraints)
            
            # 更新状态
            state.signature = signature
            state.metadata['understand_completed'] = True
            state.metadata['signature_parsed'] = True
            
            context.success = True
            
        except Exception as e:
            context.success = False
            context.error = str(e)
            state.metadata['understand_error'] = str(e)
        
        context.end_time = datetime.datetime.now()
        return state, context
    
    def get_supported_backends(self) -> List[str]:
        """获取支持的后端类型"""
        return ["pytorch", "tensorflow", "jax"]
