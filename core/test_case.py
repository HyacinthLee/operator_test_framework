from typing import Any


class TestCase:
    """Test case class for defining operator test cases.

    Attributes:
        name: The name of the test case.
        inputs: The input tensors/values for the test case.
        expected_outputs: The expected output tensors/values.
    """

    def __init__(
        self,
        name: str,
        inputs: dict[str, Any],
        expected_outputs: dict[str, Any]
    ) -> None:
        self.name: str = name
        self.inputs: dict[str, Any] = inputs
        self.expected_outputs: dict[str, Any] = expected_outputs
