from typing import Any


class NumericalOracle:
    """Numerical oracle class for comparing tensor values with tolerance.

    Attributes:
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.
    """

    def __init__(self, rtol: float = 1e-5, atol: float = 1e-8) -> None:
        self.rtol: float = rtol
        self.atol: float = atol

    def compare(self, actual: Any, expected: Any) -> bool:
        """Compare actual and expected values within tolerance.

        Args:
            actual: The actual output value.
            expected: The expected output value.

        Returns:
            True if values are close within tolerance, False otherwise.
        """
        # Handle nested structures (list, tuple)
        if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
            if len(actual) != len(expected):
                return False
            return all(self.compare(a, e) for a, e in zip(actual, expected))

        # Handle dictionaries
        if isinstance(actual, dict) and isinstance(expected, dict):
            if set(actual.keys()) != set(expected.keys()):
                return False
            return all(self.compare(actual[k], expected[k]) for k in actual)

        # Handle numeric types
        try:
            # Try to import numpy for array comparison
            import numpy as np
            if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
                return np.allclose(actual, expected, rtol=self.rtol, atol=self.atol)
        except ImportError:
            pass

        # Try to import torch for tensor comparison
        try:
            import torch
            if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
                return torch.allclose(
                    torch.as_tensor(actual),
                    torch.as_tensor(expected),
                    rtol=self.rtol,
                    atol=self.atol
                )
        except ImportError:
            pass

        # Fallback to simple comparison for scalar values
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            return abs(actual - expected) <= (self.atol + self.rtol * abs(expected))

        # Default equality check
        return actual == expected


class PropertyOracle:
    """Property oracle class for validating operator mathematical properties.

    This class provides methods to verify properties like commutativity,
    associativity, and other algebraic properties of operators.

    Attributes:
        numerical_oracle: NumericalOracle instance for comparing results.
    """

    def __init__(self, rtol: float = 1e-5, atol: float = 1e-8) -> None:
        self.numerical_oracle: NumericalOracle = NumericalOracle(rtol=rtol, atol=atol)

    def check_commutative(
        self,
        op: callable,
        a: Any,
        b: Any
    ) -> bool:
        """Check if an operator is commutative: op(a, b) == op(b, a).

        Args:
            op: The operator function to test.
            a: First operand.
            b: Second operand.

        Returns:
            True if the operator is commutative for the given inputs.
        """
        result_ab = op(a, b)
        result_ba = op(b, a)
        return self.numerical_oracle.compare(result_ab, result_ba)

    def check_associative(
        self,
        op: callable,
        a: Any,
        b: Any,
        c: Any
    ) -> bool:
        """Check if an operator is associative: op(op(a, b), c) == op(a, op(b, c)).

        Args:
            op: The operator function to test.
            a: First operand.
            b: Second operand.
            c: Third operand.

        Returns:
            True if the operator is associative for the given inputs.
        """
        result_ab_c = op(op(a, b), c)
        result_a_bc = op(a, op(b, c))
        return self.numerical_oracle.compare(result_ab_c, result_a_bc)

    def check_identity(
        self,
        op: callable,
        a: Any,
        identity: Any
    ) -> bool:
        """Check if a value acts as identity: op(a, identity) == a.

        Args:
            op: The operator function to test.
            a: The operand to test.
            identity: The proposed identity value.

        Returns:
            True if identity acts as identity element for the operator.
        """
        result = op(a, identity)
        return self.numerical_oracle.compare(result, a)

    def check_distributive(
        self,
        op_mul: callable,
        op_add: callable,
        a: Any,
        b: Any,
        c: Any
    ) -> bool:
        """Check if op_mul distributes over op_add: a * (b + c) == (a * b) + (a * c).

        Args:
            op_mul: The multiplication-like operator.
            op_add: The addition-like operator.
            a: First operand.
            b: Second operand.
            c: Third operand.

        Returns:
            True if op_mul distributes over op_add for the given inputs.
        """
        result_left = op_mul(a, op_add(b, c))
        result_right = op_add(op_mul(a, b), op_mul(a, c))
        return self.numerical_oracle.compare(result_left, result_right)


class GradientOracle:
    """Gradient oracle class for validating gradient correctness.

    This class provides numerical gradient checking to verify that
    analytical gradients computed by autograd are correct.

    Attributes:
        numerical_oracle: NumericalOracle instance for comparing gradients.
        epsilon: Step size for numerical gradient computation.
    """

    def __init__(self, rtol: float = 1e-5, atol: float = 1e-8, epsilon: float = 1e-5) -> None:
        self.numerical_oracle: NumericalOracle = NumericalOracle(rtol=rtol, atol=atol)
        self.epsilon: float = epsilon

    def check_gradient(
        self,
        func: callable,
        inputs: dict[str, Any],
        grad_outputs: dict[str, Any] | None = None,
        input_names: list[str] | None = None
    ) -> dict[str, bool]:
        """Check if analytical gradients match numerical gradients.

        Args:
            func: The function to test. Should return outputs and accept
                  inputs as keyword arguments.
            inputs: Dictionary of input tensors/values.
            grad_outputs: Gradient of the output w.r.t. each output.
                         If None, assumes scalar output with grad=1.
            input_names: List of input names to check gradients for.
                        If None, checks all inputs that are tensors.

        Returns:
            Dictionary mapping input names to True/False for gradient correctness.
        """
        try:
            import torch
        except ImportError:
            raise ImportError("GradientOracle requires PyTorch to be installed")

        results = {}

        # Determine which inputs to check
        if input_names is None:
            input_names = [
                name for name, value in inputs.items()
                if isinstance(value, torch.Tensor) and value.requires_grad
            ]

        # Compute analytical gradients using autograd
        inputs_with_grad = {
            name: (value.clone().detach().requires_grad_(True)
                   if isinstance(value, torch.Tensor) else value)
            for name, value in inputs.items()
        }

        outputs = func(**inputs_with_grad)
        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)

        # Compute gradients
        if grad_outputs is None:
            # Assume scalar output, grad = 1
            for output in outputs:
                if output is not None:
                    output.backward(torch.ones_like(output), retain_graph=True)
        else:
            for output, grad in zip(outputs, grad_outputs.values()):
                if output is not None:
                    output.backward(grad, retain_graph=True)

        analytical_grads = {
            name: inputs_with_grad[name].grad.clone() if inputs_with_grad[name].grad is not None else None
            for name in input_names
        }

        # Compute numerical gradients
        for name in input_names:
            numerical_grad = self._compute_numerical_gradient(
                func, inputs, name, outputs, grad_outputs
            )
            analytical_grad = analytical_grads[name]

            if numerical_grad is None or analytical_grad is None:
                results[name] = numerical_grad is None and analytical_grad is None
            else:
                results[name] = self.numerical_oracle.compare(
                    analytical_grad, numerical_grad
                )

        return results

    def _compute_numerical_gradient(
        self,
        func: callable,
        inputs: dict[str, Any],
        input_name: str,
        outputs: tuple,
        grad_outputs: dict[str, Any] | None = None
    ) -> Any:
        """Compute numerical gradient using finite differences.

        Args:
            func: The function to compute gradient for.
            inputs: Dictionary of input tensors/values.
            input_name: Name of the input to compute gradient for.
            outputs: Expected output structure (for shape reference).
            grad_outputs: Gradient of the output w.r.t. each output.

        Returns:
            Numerical gradient tensor.
        """
        try:
            import torch
        except ImportError:
            raise ImportError("GradientOracle requires PyTorch to be installed")

        input_tensor = inputs[input_name]
        if not isinstance(input_tensor, torch.Tensor):
            return None

        numerical_grad = torch.zeros_like(input_tensor)
        flat_input = input_tensor.reshape(-1)
        flat_grad = numerical_grad.reshape(-1)

        for i in range(flat_input.numel()):
            # Compute f(x + epsilon)
            inputs_plus = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                          for k, v in inputs.items()}
            flat_plus = inputs_plus[input_name].reshape(-1)
            flat_plus[i] = flat_plus[i] + self.epsilon
            outputs_plus = func(**inputs_plus)
            if not isinstance(outputs_plus, (tuple, list)):
                outputs_plus = (outputs_plus,)

            # Compute f(x - epsilon)
            inputs_minus = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                           for k, v in inputs.items()}
            flat_minus = inputs_minus[input_name].reshape(-1)
            flat_minus[i] = flat_minus[i] - self.epsilon
            outputs_minus = func(**inputs_minus)
            if not isinstance(outputs_minus, (tuple, list)):
                outputs_minus = (outputs_minus,)

            # Compute numerical derivative: (f(x+e) - f(x-e)) / (2*e)
            grad_contrib = 0.0
            for out_plus, out_minus in zip(outputs_plus, outputs_minus):
                if out_plus is None or out_minus is None:
                    continue
                diff = (out_plus - out_minus).sum().item()
                grad_contrib += diff / (2 * self.epsilon)

            flat_grad[i] = grad_contrib

        return numerical_grad

    def check_jacobian(
        self,
        func: callable,
        inputs: dict[str, Any],
        input_name: str,
        output_index: int = 0
    ) -> bool:
        """Check if the Jacobian matrix is correct.

        Args:
            func: The function to test.
            inputs: Dictionary of input tensors/values.
            input_name: Name of the input to check Jacobian for.
            output_index: Index of the output to check.

        Returns:
            True if Jacobian is correct.
        """
        try:
            import torch
        except ImportError:
            raise ImportError("GradientOracle requires PyTorch to be installed")

        # Get output shape
        test_inputs = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                      for k, v in inputs.items()}
        output = func(**test_inputs)
        if isinstance(output, (tuple, list)):
            output = output[output_index]

        input_tensor = inputs[input_name]
        if not isinstance(input_tensor, torch.Tensor):
            return True  # Non-tensor inputs have no Jacobian

        output_shape = output.shape
        input_shape = input_tensor.shape

        # Flatten for Jacobian computation
        output_size = output_shape.numel() if len(output_shape) > 0 else 1
        input_size = input_shape.numel() if len(input_shape) > 0 else 1

        # Compute analytical Jacobian
        inputs_with_grad = {
            name: (value.clone().detach().requires_grad_(True)
                   if isinstance(value, torch.Tensor) else value)
            for name, value in inputs.items()
        }

        analytical_jacobian = torch.zeros(output_size, input_size)
        for i in range(output_size):
            output = func(**inputs_with_grad)
            if isinstance(output, (tuple, list)):
                output = output[output_index]

            flat_output = output.reshape(-1)
            if flat_output.numel() > 0:
                grad_output = torch.zeros_like(flat_output)
                grad_output[i] = 1.0
                flat_output.backward(grad_output, retain_graph=True)

                if inputs_with_grad[input_name].grad is not None:
                    analytical_jacobian[i] = inputs_with_grad[input_name].grad.reshape(-1).clone()

                # Clear gradients for next iteration
                inputs_with_grad[input_name].grad.zero_()

        # Compute numerical Jacobian
        numerical_jacobian = torch.zeros(output_size, input_size)
        for j in range(input_size):
            # f(x + epsilon)
            inputs_plus = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                          for k, v in inputs.items()}
            flat_plus = inputs_plus[input_name].reshape(-1)
            flat_plus[j] = flat_plus[j] + self.epsilon
            output_plus = func(**inputs_plus)
            if isinstance(output_plus, (tuple, list)):
                output_plus = output_plus[output_index]
            flat_plus_out = output_plus.reshape(-1)

            # f(x - epsilon)
            inputs_minus = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                           for k, v in inputs.items()}
            flat_minus = inputs_minus[input_name].reshape(-1)
            flat_minus[j] = flat_minus[j] - self.epsilon
            output_minus = func(**inputs_minus)
            if isinstance(output_minus, (tuple, list)):
                output_minus = output_minus[output_index]
            flat_minus_out = output_minus.reshape(-1)

            # Numerical derivative
            numerical_jacobian[:, j] = (flat_plus_out - flat_minus_out) / (2 * self.epsilon)

        return self.numerical_oracle.compare(analytical_jacobian, numerical_jacobian)
