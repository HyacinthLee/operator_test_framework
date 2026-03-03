from typing import Any

from .test_case import TestCase
from .test_oracle import NumericalOracle


class TestResult:
    """Test result class for storing test execution results.

    Attributes:
        test_name: The name of the test case.
        passed: Whether the test passed.
        message: Optional message describing the result.
        actual_outputs: The actual outputs from the test execution.
        execution_time: Time taken to execute the test in seconds.
    """

    def __init__(
        self,
        test_name: str,
        passed: bool,
        message: str = "",
        actual_outputs: dict[str, Any] | None = None,
        execution_time: float = 0.0
    ) -> None:
        self.test_name: str = test_name
        self.passed: bool = passed
        self.message: str = message
        self.actual_outputs: dict[str, Any] = actual_outputs or {}
        self.execution_time: float = execution_time


class TestRunner:
    """Test runner class for executing test cases.

    Attributes:
        numerical_oracle: NumericalOracle instance for comparing outputs.
    """

    def __init__(self, rtol: float = 1e-5, atol: float = 1e-8) -> None:
        self.numerical_oracle: NumericalOracle = NumericalOracle(rtol=rtol, atol=atol)

    def run(
        self,
        test_case: TestCase,
        operator_func: callable
    ) -> TestResult:
        """Run a single test case against an operator function.

        Args:
            test_case: The test case to execute.
            operator_func: The operator function to test.

        Returns:
            TestResult containing the execution results.
        """
        import time

        start_time = time.time()

        try:
            # Execute the operator
            actual_outputs = operator_func(**test_case.inputs)

            # Normalize outputs to dictionary
            if not isinstance(actual_outputs, dict):
                actual_outputs = {"output": actual_outputs}

            # Compare with expected outputs
            all_passed = True
            messages = []

            for key, expected in test_case.expected_outputs.items():
                if key not in actual_outputs:
                    all_passed = False
                    messages.append(f"Missing output: {key}")
                    continue

                actual = actual_outputs[key]
                if not self.numerical_oracle.compare(actual, expected):
                    all_passed = False
                    messages.append(f"Output '{key}' mismatch: expected {expected}, got {actual}")

            execution_time = time.time() - start_time

            return TestResult(
                test_name=test_case.name,
                passed=all_passed,
                message="; ".join(messages) if messages else "Test passed",
                actual_outputs=actual_outputs,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name=test_case.name,
                passed=False,
                message=f"Exception during execution: {str(e)}",
                actual_outputs={},
                execution_time=execution_time
            )

    def run_batch(
        self,
        test_cases: list[TestCase],
        operator_func: callable
    ) -> list[TestResult]:
        """Run multiple test cases against an operator function.

        Args:
            test_cases: List of test cases to execute.
            operator_func: The operator function to test.

        Returns:
            List of TestResult containing the execution results.
        """
        return [self.run(tc, operator_func) for tc in test_cases]
