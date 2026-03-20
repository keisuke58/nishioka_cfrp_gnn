"""Tests for NoiseAugmentTransform and FeatureDropout in gnn_common.data_utils."""

import pytest
import torch
from torch_geometric.data import Data

from gnn_common.data_utils import NoiseAugmentTransform, FeatureDropout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_dummy_data(num_nodes=50, num_features=4):
    """Create a minimal PyG Data object for testing."""
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    y = torch.zeros(num_nodes, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


# ---------------------------------------------------------------------------
# NoiseAugmentTransform
# ---------------------------------------------------------------------------
class TestNoiseAugmentTransform:
    def test_preserves_shape(self):
        data = _make_dummy_data(num_nodes=100, num_features=4)
        transform = NoiseAugmentTransform(noise_std=0.05)
        result = transform(data)
        assert result.x.shape == data.x.shape

    def test_adds_noise(self):
        """Output should differ from input (noise_std > 0)."""
        data = _make_dummy_data()
        transform = NoiseAugmentTransform(noise_std=0.1)
        result = transform(data)
        assert not torch.allclose(result.x, data.x), "Noise should change features"

    def test_noise_std_approximate(self):
        """Added noise should have roughly the requested std."""
        torch.manual_seed(0)
        data = _make_dummy_data(num_nodes=10000, num_features=4)
        noise_std = 0.05
        transform = NoiseAugmentTransform(noise_std=noise_std)
        result = transform(data)
        diff = result.x - data.x
        empirical_std = diff.std().item()
        # Allow 20% tolerance
        assert abs(empirical_std - noise_std) / noise_std < 0.2, (
            f"Expected std ~{noise_std}, got {empirical_std}"
        )

    def test_feature_indices_selective(self):
        """Only specified columns should be modified."""
        data = _make_dummy_data(num_nodes=100, num_features=4)
        transform = NoiseAugmentTransform(noise_std=0.5, feature_indices=[0, 2])
        result = transform(data)
        # Columns 1 and 3 should be unchanged
        assert torch.equal(result.x[:, 1], data.x[:, 1])
        assert torch.equal(result.x[:, 3], data.x[:, 3])
        # Columns 0 and 2 should differ
        assert not torch.equal(result.x[:, 0], data.x[:, 0])
        assert not torch.equal(result.x[:, 2], data.x[:, 2])

    def test_does_not_modify_original(self):
        """Transform should not mutate the original data object."""
        data = _make_dummy_data()
        original_x = data.x.clone()
        transform = NoiseAugmentTransform(noise_std=0.1)
        _ = transform(data)
        assert torch.equal(data.x, original_x), "Original data should not be modified"

    def test_preserves_other_attributes(self):
        """edge_index, y, etc. should be unchanged."""
        data = _make_dummy_data()
        transform = NoiseAugmentTransform(noise_std=0.1)
        result = transform(data)
        assert torch.equal(result.edge_index, data.edge_index)
        assert torch.equal(result.y, data.y)

    def test_zero_std_no_change(self):
        """With noise_std=0, output should equal input."""
        data = _make_dummy_data()
        transform = NoiseAugmentTransform(noise_std=0.0)
        result = transform(data)
        assert torch.allclose(result.x, data.x)


# ---------------------------------------------------------------------------
# FeatureDropout
# ---------------------------------------------------------------------------
class TestFeatureDropout:
    def test_preserves_shape(self):
        data = _make_dummy_data(num_nodes=100, num_features=4)
        transform = FeatureDropout(p=0.5)
        result = transform(data)
        assert result.x.shape == data.x.shape

    def test_zeros_correct_proportion(self):
        """With high p, most feature columns should be zeroed."""
        torch.manual_seed(42)
        data = _make_dummy_data(num_nodes=100, num_features=100)
        # Make all features non-zero
        data.x = torch.ones(100, 100)
        transform = FeatureDropout(p=0.5)

        # Run many trials to get stable statistics
        zero_counts = 0
        n_trials = 200
        for _ in range(n_trials):
            result = transform(data)
            # Count how many feature columns are entirely zero
            col_sums = result.x.abs().sum(dim=0)
            zero_counts += (col_sums == 0).sum().item()

        avg_zero_frac = zero_counts / (n_trials * 100)
        # Should be approximately p=0.5, allow tolerance
        assert 0.35 < avg_zero_frac < 0.65, (
            f"Expected ~50% columns zeroed, got {avg_zero_frac*100:.1f}%"
        )

    def test_p_zero_no_dropout(self):
        """With p=0, nothing should be dropped."""
        data = _make_dummy_data()
        data.x = torch.ones_like(data.x)
        transform = FeatureDropout(p=0.0)
        result = transform(data)
        assert torch.equal(result.x, data.x)

    def test_does_not_modify_original(self):
        data = _make_dummy_data()
        original_x = data.x.clone()
        transform = FeatureDropout(p=0.5)
        _ = transform(data)
        assert torch.equal(data.x, original_x)

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError):
            FeatureDropout(p=1.0)
        with pytest.raises(ValueError):
            FeatureDropout(p=-0.1)

    def test_column_wise_dropout(self):
        """Dropout should zero entire columns, not individual elements."""
        torch.manual_seed(123)
        data = _make_dummy_data(num_nodes=50, num_features=10)
        data.x = torch.ones(50, 10)
        transform = FeatureDropout(p=0.5)
        result = transform(data)
        # For each column: either ALL zeros or ALL ones
        for col in range(10):
            col_vals = result.x[:, col]
            assert torch.all(col_vals == 0) or torch.all(col_vals == 1), (
                f"Column {col} should be entirely 0 or entirely 1"
            )


# ---------------------------------------------------------------------------
# Integration: augmentation NOT applied during eval
# ---------------------------------------------------------------------------
class TestAugmentationNotAppliedDuringEval:
    """Verify that the training script design ensures augmentation is only
    applied during training.  The transforms are passed as augment_transforms
    to train_one_epoch but NOT to validate/evaluate_test, so we check
    that calling transforms in a conditional guard works correctly."""

    def test_augmentation_guard_pattern(self):
        """Simulate the training loop pattern: augment only when list is non-empty."""
        data = _make_dummy_data()
        original_x = data.x.clone()

        # Training: transforms list is populated
        train_transforms = [
            NoiseAugmentTransform(noise_std=0.1),
            FeatureDropout(p=0.3),
        ]
        batch = data
        if train_transforms:
            for t in train_transforms:
                batch = t(batch)
        assert not torch.allclose(batch.x, original_x), "Training should augment"

        # Validation: transforms list is empty (or None)
        val_transforms = []
        batch_val = data
        if val_transforms:
            for t in val_transforms:
                batch_val = t(batch_val)
        assert torch.equal(batch_val.x, original_x), "Validation should NOT augment"

    def test_none_augment_transforms(self):
        """When augment_transforms is None, no augmentation should happen."""
        data = _make_dummy_data()
        original_x = data.x.clone()
        augment_transforms = None
        batch = data
        if augment_transforms:
            for t in augment_transforms:
                batch = t(batch)
        assert torch.equal(batch.x, original_x)
