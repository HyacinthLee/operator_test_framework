import random
from typing import Any

from .constraint import DeviceConstraint, DTypeConstraint, ShapeConstraint
from .operator_spec import TensorConstraint
from .test_case import TestCase


class TestCaseGenerator:
    """Test case generator class for generating random test inputs based on constraints.

    Attributes:
        seed: Random seed for reproducibility.
    """

    def __init__(self, seed: int | None = None) -> None:
        self.seed: int | None = seed
        self._rng = random.Random(seed)

    def generate_shape(
        self,
        constraint: ShapeConstraint,
        dynamic_dims: dict[int, int] | None = None
    ) -> tuple[int, ...]:
        """Generate a valid shape based on the shape constraint.

        Args:
            constraint: The shape constraint to satisfy.
            dynamic_dims: Optional mapping of dimension index to size for dynamic dims (-1).

        Returns:
            A tuple representing the generated shape.
        """
        shape = []
        for i, dim in enumerate(constraint.expected_shape):
            if dim == -1:
                # Dynamic dimension - use provided value or generate random
                if dynamic_dims and i in dynamic_dims:
                    shape.append(dynamic_dims[i])
                else:
                    shape.append(self._rng.randint(1, 128))
            else:
                shape.append(dim)
        return tuple(shape)

    def generate_dtype(self, constraint: DTypeConstraint) -> str:
        """Generate a valid dtype based on the dtype constraint.

        Args:
            constraint: The dtype constraint to satisfy.

        Returns:
            A string representing the selected dtype.
        """
        return self._rng.choice(constraint.allowed_dtypes)

    def generate_device(self, constraint: DeviceConstraint) -> str:
        """Generate a valid device based on the device constraint.

        Args:
            constraint: The device constraint to satisfy.

        Returns:
            A string representing the selected device.
        """
        device = self._rng.choice(constraint.allowed_devices)
        # Add random device index for cuda
        if device == "cuda":
            device = f"cuda:{self._rng.randint(0, 3)}"
        return device

    def generate_tensor(
        self,
        shape_constraint: ShapeConstraint,
        dtype_constraint: DTypeConstraint,
        device_constraint: DeviceConstraint,
        dynamic_dims: dict[int, int] | None = None
    ) -> dict[str, Any]:
        """Generate a tensor specification based on constraints.

        Args:
            shape_constraint: The shape constraint.
            dtype_constraint: The dtype constraint.
            device_constraint: The device constraint.
            dynamic_dims: Optional mapping for dynamic dimensions.

        Returns:
            A dictionary containing shape, dtype, and device.
        """
        return {
            "shape": self.generate_shape(shape_constraint, dynamic_dims),
            "dtype": self.generate_dtype(dtype_constraint),
            "device": self.generate_device(device_constraint),
        }

    def generate_random_tensor_data(
        self,
        shape: tuple[int, ...],
        dtype: str,
        device: str,
        value_range: tuple[float, float] = (-1.0, 1.0)
    ) -> Any:
        """Generate random tensor data.

        Args:
            shape: The shape of the tensor.
            dtype: The data type of the tensor.
            device: The device for the tensor.
            value_range: The range of random values (min, max).

        Returns:
            A tensor object (numpy array or torch tensor).
        """
        # Try to use PyTorch if available
        try:
            import torch

            dtype_map = {
                "float32": torch.float32,
                "float64": torch.float64,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "int32": torch.int32,
                "int64": torch.int64,
                "int16": torch.int16,
                "int8": torch.int8,
                "uint8": torch.uint8,
                "bool": torch.bool,
                "complex64": torch.complex64,
                "complex128": torch.complex128,
            }

            torch_dtype = dtype_map.get(dtype, torch.float32)

            if dtype in ("float32", "float64", "float16", "bfloat16"):
                data = torch.randn(shape, dtype=torch_dtype, device=device)
                # Scale to value_range
                min_val, max_val = value_range
                data = data * (max_val - min_val) / 2 + (max_val + min_val) / 2
            elif dtype in ("int32", "int64", "int16", "int8"):
                min_val, max_val = int(value_range[0]), int(value_range[1])
                data = torch.randint(min_val, max_val + 1, shape, dtype=torch_dtype, device=device)
            elif dtype == "bool":
                data = torch.randint(0, 2, shape, dtype=torch_dtype, device=device)
            elif dtype in ("complex64", "complex128"):
                real = torch.randn(shape, dtype=torch.float32 if dtype == "complex64" else torch.float64, device=device)
                imag = torch.randn(shape, dtype=torch.float32 if dtype == "complex64" else torch.float64, device=device)
                data = torch.complex(real, imag)
            else:
                data = torch.randn(shape, dtype=torch_dtype, device=device)

            return data

        except ImportError:
            pass

        # Fallback to NumPy
        try:
            import numpy as np

            dtype_map = {
                "float32": np.float32,
                "float64": np.float64,
                "float16": np.float16,
                "int32": np.int32,
                "int64": np.int64,
                "int16": np.int16,
                "int8": np.int8,
                "uint8": np.uint8,
                "bool": bool,
                "complex64": np.complex64,
                "complex128": np.complex128,
            }

            np_dtype = dtype_map.get(dtype, np.float32)

            if dtype in ("float32", "float64", "float16"):
                data = np.random.randn(*shape).astype(np_dtype)
                min_val, max_val = value_range
                data = data * (max_val - min_val) / 2 + (max_val + min_val) / 2
            elif dtype in ("int32", "int64", "int16", "int8", "uint8"):
                min_val, max_val = int(value_range[0]), int(value_range[1])
                data = np.random.randint(min_val, max_val + 1, shape).astype(np_dtype)
            elif dtype == "bool":
                data = np.random.randint(0, 2, shape).astype(bool)
            elif dtype in ("complex64", "complex128"):
                real = np.random.randn(*shape).astype(np.float32 if dtype == "complex64" else np.float64)
                imag = np.random.randn(*shape).astype(np.float32 if dtype == "complex64" else np.float64)
                data = real + 1j * imag
            else:
                data = np.random.randn(*shape).astype(np_dtype)

            return data

        except ImportError:
            raise ImportError("Either PyTorch or NumPy is required for tensor generation")

    def generate_test_case(
        self,
        name: str,
        input_constraints: dict[str, TensorConstraint],
        expected_outputs: dict[str, Any] | None = None,
        dynamic_dims: dict[str, dict[int, int]] | None = None
    ) -> TestCase:
        """Generate a test case based on input constraints.

        Args:
            name: The name of the test case.
            input_constraints: Dictionary mapping input names to TensorConstraint.
            expected_outputs: Optional expected outputs (if None, empty dict used).
            dynamic_dims: Optional mapping of input name to dynamic dimension sizes.

        Returns:
            A TestCase instance with generated inputs.
        """
        expected_outputs = expected_outputs or {}
        dynamic_dims = dynamic_dims or {}

        inputs = {}
        for input_name, constraint in input_constraints.items():
            dims = dynamic_dims.get(input_name, {})
            shape = self.generate_shape(
                ShapeConstraint(constraint.shape),
                dims
            )
            dtype = constraint.dtype
            device = constraint.device

            inputs[input_name] = self.generate_random_tensor_data(shape, dtype, device)

        return TestCase(
            name=name,
            inputs=inputs,
            expected_outputs=expected_outputs
        )

    def generate_batch(
        self,
        base_name: str,
        input_constraints: dict[str, TensorConstraint],
        count: int,
        expected_outputs: dict[str, Any] | None = None
    ) -> list[TestCase]:
        """Generate multiple test cases.

        Args:
            base_name: Base name for test cases (will be suffixed with index).
            input_constraints: Dictionary mapping input names to TensorConstraint.
            count: Number of test cases to generate.
            expected_outputs: Optional expected outputs.

        Returns:
            A list of TestCase instances.
        """
        test_cases = []
        for i in range(count):
            test_case = self.generate_test_case(
                name=f"{base_name}_{i}",
                input_constraints=input_constraints,
                expected_outputs=expected_outputs
            )
            test_cases.append(test_case)
        return test_cases

    def reset_seed(self, seed: int | None = None) -> None:
        """Reset the random seed.

        Args:
            seed: New random seed. If None, uses the original seed.
        """
        if seed is not None:
            self.seed = seed
        self._rng = random.Random(self.seed)
