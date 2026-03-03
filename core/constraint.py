from typing import Any


class ShapeConstraint:
    """Shape constraint class for validating tensor shapes.

    Attributes:
        expected_shape: The expected shape pattern. Supports -1 for dynamic dimensions.
    """

    def __init__(self, expected_shape: tuple[int, ...]) -> None:
        self.expected_shape: tuple[int, ...] = expected_shape

    def validate(self, shape: tuple[int, ...]) -> bool:
        """Validate if the given shape matches the expected shape pattern.

        Args:
            shape: The actual shape to validate.

        Returns:
            True if the shape matches, False otherwise.
        """
        if len(shape) != len(self.expected_shape):
            return False

        for actual, expected in zip(shape, self.expected_shape):
            if expected != -1 and actual != expected:
                return False

        return True


class DTypeConstraint:
    """Data type constraint class for validating tensor data types.

    Attributes:
        allowed_dtypes: List of allowed data type strings.
    """

    def __init__(self, allowed_dtypes: list[str]) -> None:
        self.allowed_dtypes: list[str] = allowed_dtypes

    def validate(self, dtype: str) -> bool:
        """Validate if the given dtype is in the allowed list.

        Args:
            dtype: The data type string to validate.

        Returns:
            True if the dtype is allowed, False otherwise.
        """
        return dtype in self.allowed_dtypes


class DeviceConstraint:
    """Device constraint class for validating tensor device placement.

    Attributes:
        allowed_devices: List of allowed device strings (e.g., 'cpu', 'cuda').
    """

    def __init__(self, allowed_devices: list[str]) -> None:
        self.allowed_devices: list[str] = allowed_devices

    def validate(self, device: str) -> bool:
        """Validate if the given device is in the allowed list.

        Args:
            device: The device string to validate (e.g., 'cpu', 'cuda:0').

        Returns:
            True if the device is allowed, False otherwise.
        """
        # Handle device strings like 'cuda:0', 'cuda:1' by checking prefix
        for allowed in self.allowed_devices:
            if device == allowed or device.startswith(allowed + ":"):
                return True
        return False
