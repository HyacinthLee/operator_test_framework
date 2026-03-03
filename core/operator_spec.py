from typing import Any


class Attribute:
    """Attribute class for defining operator attributes.

    Attributes:
        name: The name of the attribute.
        value_type: The type of the attribute value.
        default_value: The default value of the attribute.
    """

    def __init__(self, name: str, value_type: type, default_value: Any) -> None:
        self.name: str = name
        self.value_type: type = value_type
        self.default_value: Any = default_value


class TensorConstraint:
    """Tensor constraint class for defining tensor specifications.

    Attributes:
        shape: The shape of the tensor.
        dtype: The data type of the tensor.
        device: The device where the tensor is located.
    """

    def __init__(self, shape: tuple[int, ...], dtype: str, device: str) -> None:
        self.shape: tuple[int, ...] = shape
        self.dtype: str = dtype
        self.device: str = device


class OperatorSpec:
    """Operator specification class.

    Attributes:
        name: The name of the operator.
        inputs: The inputs to the operator.
    """

    def __init__(self, name: str, inputs: dict[str, Any]) -> None:
        self.name: str = name
        self.inputs: dict[str, Any] = inputs
