"""
七阶段工作流引擎使用示例

展示如何使用 WorkflowEngine 进行完整的算子测试。
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '..')

from core import (
    OperatorTestAdapter,
    TestConfig,
    WorkflowEngine,
    WorkflowConfig,
    WorkflowState,
    run_full_workflow,
)


# 1. 定义一个简单的算子适配器
class ReLUAdapter(OperatorTestAdapter):
    """ReLU 算子适配器示例"""
    
    def __init__(self):
        super().__init__("ReLU", TestConfig(device='cpu', dtype=torch.float32))
        self.relu = nn.ReLU()
    
    def forward(self, inputs):
        return self.relu(inputs['x'])
    
    def generate_inputs(self, batch_size=2, **kwargs):
        return {
            'x': torch.randn(batch_size, 64, dtype=self.config.dtype)
        }


class LinearAdapter(OperatorTestAdapter):
    """Linear 层适配器示例"""
    
    def __init__(self, in_features=64, out_features=128):
        super().__init__("Linear", TestConfig(device='cpu', dtype=torch.float32))
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, inputs):
        return self.linear(inputs['x'])
    
    def generate_inputs(self, batch_size=2, **kwargs):
        return {
            'x': torch.randn(batch_size, self.linear.in_features, dtype=self.config.dtype)
        }
    
    def get_parameters(self):
        return {
            'weight': self.linear.weight,
            'bias': self.linear.bias,
        }


def basic_example():
    """基础使用示例"""
    print("=" * 70)
    print("示例 1: 基础工作流执行")
    print("=" * 70)
    
    # 创建算子适配器
    adapter = ReLUAdapter()
    
    # 创建进度回调
    def progress_callback(phase_name, current, total):
        print(f"  [{current}/{total}] 执行阶段: {phase_name}")
    
    # 创建工作流引擎
    engine = WorkflowEngine(progress_callback=progress_callback)
    
    # 创建初始状态
    state = WorkflowState(operator=adapter)
    
    # 运行工作流
    final_state = engine.run(state)
    
    # 输出结果
    print("\n执行完成!")
    print(f"测试用例数: {len(final_state.test_cases)}")
    print(f"通过: {sum(1 for r in final_state.results if r.passed)}")
    print(f"失败: {sum(1 for r in final_state.results if not r.passed)}")
    
    # 导出报告
    if final_state.report:
        print("\n" + "=" * 70)
        print("测试报告:")
        print("=" * 70)
        print(engine.export_report(final_state, format_type="text"))
    
    return final_state


def custom_config_example():
    """自定义配置示例"""
    print("\n" + "=" * 70)
    print("示例 2: 自定义工作流配置")
    print("=" * 70)
    
    # 创建自定义配置
    config = WorkflowConfig(
        enable_understand=True,
        enable_requirements=True,
        enable_planning=True,
        enable_generation=True,
        enable_execution=True,
        enable_analysis=True,
        enable_report=True,
        stop_on_error=False,  # 错误时继续
        collect_metrics=True,
    )
    
    # 使用自定义配置创建引擎
    engine = WorkflowEngine(config=config)
    
    # 运行工作流
    adapter = LinearAdapter(in_features=128, out_features=256)
    state = WorkflowState(operator=adapter)
    final_state = engine.run(state)
    
    # 获取指标
    metrics = engine.get_metrics(final_state)
    print("\n执行指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    return final_state


def incremental_example():
    """增量执行示例"""
    print("\n" + "=" * 70)
    print("示例 3: 增量执行")
    print("=" * 70)
    
    from core import IncrementalWorkflowEngine
    
    # 创建增量引擎
    engine = IncrementalWorkflowEngine()
    
    # 创建适配器
    adapter = ReLUAdapter()
    state = WorkflowState(operator=adapter)
    
    # 执行到某个阶段
    print("执行前三个阶段...")
    phases = ["understand", "requirements", "planning"]
    for phase in phases:
        state = engine.run_phase(phase, state)
        print(f"  {phase}: 完成")
    
    # 缓存状态
    engine.cache_phase_state("planning", state)
    print("\n已缓存 planning 阶段状态")
    
    # 从缓存恢复并从下一阶段继续
    restored_state = engine.restore_phase_state("planning")
    if restored_state:
        print("已恢复 planning 阶段状态")
        print(f"  需求数: {len(restored_state.requirements)}")
        print(f"  计划数: {len(restored_state.plans)}")
    
    return state


def partial_execution_example():
    """部分执行示例"""
    print("\n" + "=" * 70)
    print("示例 4: 部分执行（跳过某些阶段）")
    print("=" * 70)
    
    # 创建配置，禁用某些阶段
    config = WorkflowConfig(
        enable_understand=True,
        enable_requirements=True,
        enable_planning=False,  # 跳过计划阶段
        enable_generation=True,
        enable_execution=True,
        enable_analysis=False,  # 跳过分析阶段
        enable_report=True,
    )
    
    engine = WorkflowEngine(config=config)
    adapter = LinearAdapter()
    state = WorkflowState(operator=adapter)
    
    # 运行
    final_state = engine.run(state)
    
    print(f"\n跳过了 planning 和 analysis 阶段")
    print(f"测试用例数: {len(final_state.test_cases)}")
    print(f"结果数: {len(final_state.results)}")
    print(f"分析数: {len(final_state.analyses)} (应为 0)")
    
    return final_state


def custom_phases_example():
    """自定义阶段示例"""
    print("\n" + "=" * 70)
    print("示例 5: 使用便捷函数")
    print("=" * 70)
    
    # 使用便捷函数快速运行
    adapter = ReLUAdapter()
    
    def on_progress(phase, current, total):
        print(f"  进度: {phase} ({current}/{total})")
    
    final_state = run_full_workflow(adapter, progress_callback=on_progress)
    
    print("\n完成!")
    if final_state.report:
        print(f"整体覆盖率: {final_state.report.coverage.overall_coverage*100:.1f}%")
    
    return final_state


if __name__ == "__main__":
    # 运行所有示例
    print("\n" + "=" * 70)
    print("七阶段工作流引擎使用示例")
    print("=" * 70)
    
    try:
        basic_example()
    except Exception as e:
        print(f"示例 1 出错: {e}")
    
    try:
        custom_config_example()
    except Exception as e:
        print(f"示例 2 出错: {e}")
    
    try:
        incremental_example()
    except Exception as e:
        print(f"示例 3 出错: {e}")
    
    try:
        partial_execution_example()
    except Exception as e:
        print(f"示例 4 出错: {e}")
    
    try:
        custom_phases_example()
    except Exception as e:
        print(f"示例 5 出错: {e}")
    
    print("\n" + "=" * 70)
    print("所有示例执行完成!")
    print("=" * 70)
