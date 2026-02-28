"""
Memory leak detection for GPU operators.
Monitors GPU memory usage to detect leaks during operator execution.
"""

import torch
import gc
import time
from typing import Dict, List, Callable, Tuple, Optional
from contextlib import contextmanager


class MemoryLeakDetector:
    """
    Detect GPU memory leaks in operators.
    
    A memory leak occurs when memory is allocated but not freed
    after the operation completes.
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Args:
            device: Device to monitor ('cuda' or 'cpu')
        """
        self.device = device
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
    
    @contextmanager
    def _gpu_sync(self):
        """Context manager for GPU synchronization."""
        if self.device == 'cuda':
            torch.cuda.synchronize()
        yield
        if self.device == 'cuda':
            torch.cuda.synchronize()
    
    def get_memory_stats(self) -> Dict[str, int]:
        """
        Get current memory statistics.
        
        Returns:
            Dictionary with memory stats in bytes
        """
        if self.device == 'cuda':
            return {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "active": torch.cuda.memory_stats().get("active_bytes.all.current", 0),
            }
        else:
            return {"allocated": 0, "reserved": 0, "active": 0}
    
    def detect_leak_single_run(
        self,
        operator: Callable,
        *inputs: torch.Tensor,
        warmup: int = 3,
        iterations: int = 10
    ) -> Dict[str, any]:
        """
        Detect memory leak by running operator multiple times.
        
        If memory keeps increasing across iterations, there's likely a leak.
        
        Args:
            operator: Operator to test
            *inputs: Input tensors
            warmup: Number of warmup iterations
            iterations: Number of test iterations
            
        Returns:
            Leak detection results
        """
        # Clear cache and collect garbage
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        with self._gpu_sync():
            initial_memory = self.get_memory_stats()["allocated"]
        
        # Warmup
        for _ in range(warmup):
            output = operator(*inputs)
            if isinstance(output, torch.Tensor):
                output.detach()
            del output
        
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        with self._gpu_sync():
            post_warmup_memory = self.get_memory_stats()["allocated"]
        
        # Track memory across iterations
        memory_readings = []
        
        for i in range(iterations):
            # Run operator
            output = operator(*inputs)
            
            # Handle multi-output case
            if isinstance(output, (list, tuple)):
                for o in output:
                    if isinstance(o, torch.Tensor):
                        o.detach()
            elif isinstance(output, torch.Tensor):
                output.detach()
            
            with self._gpu_sync():
                current_memory = self.get_memory_stats()["allocated"]
                memory_readings.append(current_memory)
            
            # Clean up output but keep inputs
            del output
        
        # Final cleanup
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        with self._gpu_sync():
            final_memory = self.get_memory_stats()["allocated"]
        
        # Analyze memory trend
        if len(memory_readings) > 1:
            # Check if memory is monotonically increasing
            increases = sum(
                1 for i in range(1, len(memory_readings))
                if memory_readings[i] > memory_readings[i-1]
            )
            
            # Calculate slope of memory growth
            x = list(range(len(memory_readings)))
            y = memory_readings
            n = len(x)
            slope = (n * sum(xi * yi for xi, yi in zip(x, y)) - sum(x) * sum(y)) / (
                n * sum(xi ** 2 for xi in x) - sum(x) ** 2
            ) if n > 1 else 0
            
            # Memory growth per iteration
            total_growth = memory_readings[-1] - memory_readings[0]
            avg_growth_per_iter = total_growth / iterations if iterations > 0 else 0
        else:
            increases = 0
            slope = 0
            total_growth = 0
            avg_growth_per_iter = 0
        
        # Determine if leak detected
        # Heuristic: if memory increased in >70% of iterations and slope > 0
        increase_ratio = increases / max(1, len(memory_readings) - 1) if len(memory_readings) > 1 else 0
        leak_detected = (increase_ratio > 0.7) and (slope > 1000)
        
        return {
            "leak_detected": leak_detected,
            "initial_memory_mb": initial_memory / (1024 ** 2),
            "post_warmup_memory_mb": post_warmup_memory / (1024 ** 2),
            "final_memory_mb": final_memory / (1024 ** 2),
            "total_growth_mb": total_growth / (1024 ** 2),
            "avg_growth_per_iter_mb": avg_growth_per_iter / (1024 ** 2),
            "memory_readings_mb": [m / (1024 ** 2) for m in memory_readings],
            "increase_ratio": increases / max(1, len(memory_readings) - 1),
            "slope": slope,
            "iterations": iterations,
        }
    
    def detect_leak_with_increasing_batch(
        self,
        operator: Callable,
        input_shape: Tuple[int, ...],
        batch_sizes: List[int] = [1, 2, 4, 8, 16]
    ) -> Dict[str, any]:
        """
        Detect leak by testing with increasing batch sizes.
        
        Memory should scale linearly with batch size. Super-linear growth
        indicates a potential leak.
        
        Args:
            operator: Operator to test
            input_shape: Base input shape (without batch dimension)
            batch_sizes: List of batch sizes to test
            
        Returns:
            Leak detection results
        """
        memory_per_batch = []
        
        for batch_size in batch_sizes:
            # Create input with this batch size
            shape = (batch_size,) + input_shape
            x = torch.randn(shape, device=self.device)
            
            # Clear memory
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            with self._gpu_sync():
                mem_before = self.get_memory_stats()["allocated"]
            
            # Run operator
            output = operator(x)
            
            with self._gpu_sync():
                mem_after = self.get_memory_stats()["allocated"]
            
            # Calculate memory used
            memory_used = mem_after - mem_before
            memory_per_batch.append((batch_size, memory_used))
            
            # Cleanup
            del x, output
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        # Analyze scaling
        # Memory should be roughly linear: M = k * batch_size + b
        # If super-linear, there might be a leak
        
        batch_sizes_arr = [b for b, _ in memory_per_batch]
        memories_arr = [m for _, m in memory_per_batch]
        
        # Check linearity using ratio test
        ratios = []
        for i in range(1, len(memory_per_batch)):
            batch_ratio = batch_sizes_arr[i] / batch_sizes_arr[i-1]
            mem_ratio = memories_arr[i] / max(1, memories_arr[i-1])
            ratios.append(mem_ratio / batch_ratio)
        
        avg_ratio = sum(ratios) / len(ratios) if ratios else 1.0
        
        # If avg_ratio > 2.0, memory grows super-linearly
        super_linear_growth = avg_ratio > 2.0
        
        return {
            "super_linear_growth": super_linear_growth,
            "avg_scaling_ratio": avg_ratio,
            "memory_per_batch": [
                (b, m / (1024 ** 2)) for b, m in memory_per_batch
            ],
            "batch_sizes": batch_sizes_arr,
            "memory_mb": [m / (1024 ** 2) for m in memories_arr],
        }
    
    def generate_report(self, results: Dict) -> str:
        """Generate human-readable leak detection report."""
        lines = [
            "=" * 60,
            "GPU Memory Leak Detection Report",
            "=" * 60,
            "",
        ]
        
        if "leak_detected" in results:
            status = "LEAK DETECTED" if results["leak_detected"] else "NO LEAK DETECTED"
            lines.extend([
                f"Status: {status}",
                f"Initial Memory: {results['initial_memory_mb']:.2f} MB",
                f"Final Memory: {results['final_memory_mb']:.2f} MB",
                f"Total Growth: {results['total_growth_mb']:.2f} MB",
                f"Avg Growth/Iter: {results['avg_growth_per_iter_mb']:.4f} MB",
                f"Increase Ratio: {results['increase_ratio']:.2%}",
            ])
        
        if "super_linear_growth" in results:
            lines.extend([
                "",
                "Batch Scaling Analysis:",
                f"Super-linear Growth: {results['super_linear_growth']}",
                f"Avg Scaling Ratio: {results['avg_scaling_ratio']:.2f}",
            ])
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    detector = MemoryLeakDetector()
    
    # Test an operator
    def test_operator(x):
        return torch.nn.functional.relu(x @ x.T)
    
    x = torch.randn(100, 100, device='cuda')
    results = detector.detect_leak_single_run(test_operator, x)
    
    print(detector.generate_report(results))
