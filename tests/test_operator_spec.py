"""Tests for operator_spec module."""

import pytest

from core.operator_spec import Attribute, OperatorSpec, TensorConstraint


class TestOperatorSpec:
    """Tests for OperatorSpec class."""

    def test_init_with_name_and_inputs(self):
        """Test initialization with name and inputs."""
        inputs = {"x": [1, 2, 3], "y": [4, 5, 6]}
        spec = OperatorSpec(name="add", inputs=inputs)

        assert spec.name == "add"
        assert spec.inputs == inputs

    def test_name_type_annotation(self):
        """Test that name attribute has correct type."""
        spec = OperatorSpec(name="test", inputs={})
        assert isinstance(spec.name, str)

    def test_inputs_type_annotation(self):
        """Test that inputs attribute has correct type."""
        spec = OperatorSpec(name="test", inputs={"a": 1, "b": 2})
        assert isinstance(spec.inputs, dict)


class TestTensorConstraint:
    """Tests for TensorConstraint class."""

    def test_init_with_shape_dtype_device(self):
        """Test initialization with shape, dtype, and device."""
        constraint = TensorConstraint(
            shape=(2, 3, 4),
            dtype="float32",
            device="cuda:0"
        )

        assert constraint.shape == (2, 3, 4)
        assert constraint.dtype == "float32"
        assert constraint.device == "cuda:0"

    def test_shape_with_dynamic_dims(self):
        """Test shape with dynamic dimensions (-1)."""
        constraint = TensorConstraint(
            shape=(-1, 3, -1),
            dtype="float64",
            device="cpu"
        )

        assert constraint.shape == (-1, 3, -1)

    def test_scalar_shape(self):
        """Test shape for scalar tensor (empty tuple)."""
        constraint = TensorConstraint(
            shape=(),
            dtype="int32",
            device="cpu"
        )

        assert constraint.shape == ()

    def test_various_dtypes(self):
        """Test various dtype values."""
        dtypes = ["float32", "float64", "int32", "int64", "bool", "complex64"]

        for dtype in dtypes:
            constraint = TensorConstraint(shape=(1,), dtype=dtype, device="cpu")
            assert constraint.dtype == dtype

    def test_various_devices(self):
        """Test various device values."""
        devices = ["cpu", "cuda:0", "cuda:1", "meta"]

        for device in devices:
            constraint = TensorConstraint(shape=(1,), dtype="float32", device=device)
            assert constraint.device == device


class TestAttribute:
    """Tests for Attribute class."""

    def test_init_with_name_value_type_default_value(self):
        """Test initialization with name, value_type, and default_value."""
        attr = Attribute(
            name="axis",
            value_type=int,
            default_value=0
        )

        assert attr.name == "axis"
        assert attr.value_type == int
        assert attr.default_value == 0

    def test_string_attribute(self):
        """Test string type attribute."""
        attr = Attribute(
            name="padding",
            value_type=str,
            default_value="same"
        )

        assert attr.name == "padding"
        assert attr.value_type == str
        assert attr.default_value == "same"

    def test_float_attribute(self):
        """Test float type attribute."""
        attr = Attribute(
            name="learning_rate",
            value_type=float,
            default_value=0.001
        )

        assert attr.name == "learning_rate"
        assert attr.value_type == float
        assert attr.default_value == 0.001

    def test_bool_attribute(self):
        """Test bool type attribute."""
        attr = Attribute(
            name="training",
            value_type=bool,
            default_value=False
        )

        assert attr.name == "training"
        assert attr.value_type == bool
        assert attr.default_value is False

    def test_list_attribute(self):
        """Test list type attribute."""
        attr = Attribute(
            name="strides",
            value_type=list,
            default_value=[1, 1]
        )

        assert attr.name == "strides"
        assert attr.value_type == list
        assert attr.default_value == [1, 1]

    def test_none_default_value(self):
        """Test attribute with None as default value."""
        attr = Attribute(
            name="bias",
            value_type=object,
            default_value=None
        )

        assert attr.name == "bias"
        assert attr.default_value is None
