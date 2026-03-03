"""Tests for constraint module."""

import pytest

from core.constraint import DeviceConstraint, DTypeConstraint, ShapeConstraint


class TestShapeConstraint:
    """Tests for ShapeConstraint class."""

    def test_init_with_expected_shape(self):
        """Test initialization with expected shape."""
        constraint = ShapeConstraint(expected_shape=(2, 3, 4))
        assert constraint.expected_shape == (2, 3, 4)

    def test_validate_matching_shape(self):
        """Test validation with matching shape."""
        constraint = ShapeConstraint(expected_shape=(2, 3, 4))
        assert constraint.validate((2, 3, 4)) is True

    def test_validate_non_matching_shape(self):
        """Test validation with non-matching shape."""
        constraint = ShapeConstraint(expected_shape=(2, 3, 4))
        assert constraint.validate((2, 3, 5)) is False

    def test_validate_different_rank(self):
        """Test validation with different rank."""
        constraint = ShapeConstraint(expected_shape=(2, 3))
        assert constraint.validate((2, 3, 4)) is False

    def test_validate_with_dynamic_dim(self):
        """Test validation with dynamic dimension (-1)."""
        constraint = ShapeConstraint(expected_shape=(-1, 3, 4))
        assert constraint.validate((2, 3, 4)) is True
        assert constraint.validate((5, 3, 4)) is True
        assert constraint.validate((100, 3, 4)) is True

    def test_validate_with_multiple_dynamic_dims(self):
        """Test validation with multiple dynamic dimensions."""
        constraint = ShapeConstraint(expected_shape=(-1, -1, 4))
        assert constraint.validate((2, 3, 4)) is True
        assert constraint.validate((5, 10, 4)) is True

    def test_validate_dynamic_dim_with_wrong_fixed_dim(self):
        """Test validation fails when fixed dim doesn't match."""
        constraint = ShapeConstraint(expected_shape=(-1, 3, 4))
        assert constraint.validate((2, 5, 4)) is False  # 5 != 3

    def test_validate_scalar_shape(self):
        """Test validation with scalar shape (empty tuple)."""
        constraint = ShapeConstraint(expected_shape=())
        assert constraint.validate(()) is True
        assert constraint.validate((1,)) is False

    def test_validate_1d_shape(self):
        """Test validation with 1D shape."""
        constraint = ShapeConstraint(expected_shape=(10,))
        assert constraint.validate((10,)) is True
        assert constraint.validate((5,)) is False


class TestDTypeConstraint:
    """Tests for DTypeConstraint class."""

    def test_init_with_allowed_dtypes(self):
        """Test initialization with allowed dtypes."""
        constraint = DTypeConstraint(allowed_dtypes=["float32", "float64"])
        assert constraint.allowed_dtypes == ["float32", "float64"]

    def test_validate_allowed_dtype(self):
        """Test validation with allowed dtype."""
        constraint = DTypeConstraint(allowed_dtypes=["float32", "float64", "int32"])
        assert constraint.validate("float32") is True
        assert constraint.validate("float64") is True
        assert constraint.validate("int32") is True

    def test_validate_disallowed_dtype(self):
        """Test validation with disallowed dtype."""
        constraint = DTypeConstraint(allowed_dtypes=["float32", "float64"])
        assert constraint.validate("int32") is False
        assert constraint.validate("bool") is False

    def test_validate_empty_allowed_list(self):
        """Test validation with empty allowed list."""
        constraint = DTypeConstraint(allowed_dtypes=[])
        assert constraint.validate("float32") is False

    def test_validate_single_dtype(self):
        """Test validation with single allowed dtype."""
        constraint = DTypeConstraint(allowed_dtypes=["float16"])
        assert constraint.validate("float16") is True
        assert constraint.validate("float32") is False

    def test_validate_case_sensitive(self):
        """Test that validation is case sensitive."""
        constraint = DTypeConstraint(allowed_dtypes=["float32"])
        assert constraint.validate("Float32") is False
        assert constraint.validate("FLOAT32") is False


class TestDeviceConstraint:
    """Tests for DeviceConstraint class."""

    def test_init_with_allowed_devices(self):
        """Test initialization with allowed devices."""
        constraint = DeviceConstraint(allowed_devices=["cpu", "cuda"])
        assert constraint.allowed_devices == ["cpu", "cuda"]

    def test_validate_cpu(self):
        """Test validation with cpu device."""
        constraint = DeviceConstraint(allowed_devices=["cpu"])
        assert constraint.validate("cpu") is True

    def test_validate_cuda(self):
        """Test validation with cuda device."""
        constraint = DeviceConstraint(allowed_devices=["cuda"])
        assert constraint.validate("cuda") is True
        assert constraint.validate("cuda:0") is True
        assert constraint.validate("cuda:1") is True
        assert constraint.validate("cuda:7") is True

    def test_validate_disallowed_device(self):
        """Test validation with disallowed device."""
        constraint = DeviceConstraint(allowed_devices=["cpu"])
        assert constraint.validate("cuda") is False
        assert constraint.validate("meta") is False

    def test_validate_cpu_not_allowed(self):
        """Test that cpu fails when only cuda is allowed."""
        constraint = DeviceConstraint(allowed_devices=["cuda"])
        assert constraint.validate("cpu") is False

    def test_validate_multiple_devices(self):
        """Test validation with multiple allowed devices."""
        constraint = DeviceConstraint(allowed_devices=["cpu", "cuda", "meta"])
        assert constraint.validate("cpu") is True
        assert constraint.validate("cuda") is True
        assert constraint.validate("cuda:0") is True
        assert constraint.validate("meta") is True

    def test_validate_empty_allowed_list(self):
        """Test validation with empty allowed list."""
        constraint = DeviceConstraint(allowed_devices=[])
        assert constraint.validate("cpu") is False

    def test_validate_with_index(self):
        """Test that device with index is valid."""
        constraint = DeviceConstraint(allowed_devices=["cpu"])
        assert constraint.validate("cpu:0") is True  # cpu:0 is valid with our implementation

    def test_validate_cuda_variations(self):
        """Test various cuda device variations."""
        constraint = DeviceConstraint(allowed_devices=["cuda"])
        assert constraint.validate("cuda:0") is True
        assert constraint.validate("cuda:1") is True
        assert constraint.validate("cuda:10") is True
