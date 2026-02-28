"""
Performance benchmarking utilities.
"""

import torch
import time
from typing import Callable, Dict, List, Optional
from contextlib import contextmanager


class PerformanceBenchmark:
    """
    Benchmark operator performance with various metrics.
    """
    
    def __init__(
        self,
        warmup_iters: int = 10,
        benchmark_iters: int = 100,
        device: str = 'cuda'
    ):
        self.warmup_iters = warmup_iters
        self.benchmark_iters = benchmark_iters
        self.device = device
    
    @contextmanager
    def _cuda_sync(self):
        """Context manager for CUDA synchronization."""
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        yield
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def measure_latency(
        self,
        operator: Callable,
        *inputs: torch.Tensor
    ) -> Dict[str, float]:
        """
        Measure operator latency.
        
        Args:
            operator: Operator to benchmark
            *inputs: Input tensors
            
        Returns:
            Dictionary with latency statistics
        """
        # Warmup
        for _ in range(self.warmup_iters):
            _ = operator(*inputs)
        
        # Benchmark
        times = []
        with self._cuda_sync():
            for _ in range(self.benchmark_iters):
                start = time.perf_counter()
                _ = operator(*inputs)
                with self._cuda_sync():
                    end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms
        
        return {
            "mean_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5
        }
    
    def measure_throughput(
        self,
        operator: Callable,
        *inputs: torch.Tensor
    ) -> float:
        """
        Measure throughput in operations per second.
        
        Args:
            operator: Operator to benchmark
            *inputs: Input tensors
            
        Returns:
            Throughput (ops/sec)
        """
        latency = self.measure_latency(operator, *inputs)
        mean_latency_sec = latency["mean_ms"] / 1000
        return 1.0 / mean_latency_sec if mean_latency_sec > 0 else float('inf')
    
    def measure_memory(
        self,
        operator: Callable,
        *inputs: torch.Tensor
    ) -> Dict[str, int]:
        """
        Measure memory usage.
        
        Args:
            operator: Operator to benchmark
            *inputs: Input tensors
            
        Returns:
            Dictionary with memory statistics (bytes)
        """
        if self.device != 'cuda' or not torch.cuda.is_available():
            return {"device": "cpu", "peak_memory": 0}
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Measure
        _ = operator(*inputs)
        torch.cuda.synchronize()
        
        peak_memory = torch.cuda.max_memory_allocated()
        
        return {
            "device": "cuda",
            "peak_memory_bytes": peak_memory,
            "peak_memory_mb": peak_memory / (1024 ** 2)
        }
    
    def compare_implementations(
        self,
        implementations: Dict[str, Callable],
        *inputs: torch.Tensor,
        metrics: List[str] = ["latency", "memory"]
    ) -> Dict:
        """
        Compare multiple implementations.
        
        Args:
            implementations: Dictionary of {name: operator}
            *inputs: Input tensors
            metrics: List of metrics to compare
            
        Returns:
            Comparison results
        """
        results = {
            "implementations": list(implementations.keys()),
            "metrics": {}
        }
        
        for name, op in implementations.items():
            results["metrics"][name] = {}
            
            if "latency" in metrics:
                results["metrics"][name]["latency"] = self.measure_latency(op, *inputs)
            
            if "memory" in metrics:
                results["metrics"][name]["memory"] = self.measure_memory(op, *inputs)
            
            if "throughput" in metrics:
                results["metrics"][name]["throughput"] = self.measure_throughput(op, *inputs)
        
        # Compute speedups relative to first implementation
        if len(implementations) > 1:
            baseline_name = list(implementations.keys())[0]
            baseline_latency = results["metrics"][baseline_name]["latency"]["mean_ms"]
            
            results["speedups"] = {}
            for name in implementations.keys():
                if name != baseline_name:
                    latency = results["metrics"][name]["latency"]["mean_ms"]
                    results["speedups"][name] = baseline_latency / latency
        
        return results
    
    def print_comparison(self, results: Dict):
        """Pretty print comparison results."""
        print("\nPerformance Comparison")
        print("=" * 80)
        
        # Header
        impl_names = results["implementations"]
        print(f"{'Implementation':<20} {'Latency (ms)':<15} {'Memory (MB)':<15} {'Speedup':<10}")
        print("-" * 80)
        
        # Data
        baseline_name = impl_names[0]
        for name in impl_names:
            metrics = results["metrics"][name]
            latency = metrics["latency"]["mean_ms"]
            memory = metrics.get("memory", {}).get("peak_memory_mb", 0)
            
            speedup_str = "baseline"
            if name != baseline_name and "speedups" in results:
                speedup = results["speedups"][name]
                speedup_str = f"{speedup:.2f}x"
            
            print(f"{name:<20} {latency:<15.3f} {memory:<15.1f} {speedup_str:<10}")


def profile_with_pytorch_profiler(
    operator: Callable,
    *inputs: torch.Tensor,
    activities: Optional[List] = None
) -> torch.profiler.profile:
    """
    Profile operator using PyTorch profiler.
    
    Args:
        operator: Operator to profile
        *inputs: Input tensors
        activities: Profiler activities (default: CPU + CUDA)
        
    Returns:
        Profiler object with results
    """
    if activities is None:
        activities = [
            torch.profiler.ProfilerActivity.CPU,
        ]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
    
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        _ = operator(*inputs)
    
    return prof
