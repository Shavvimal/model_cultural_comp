import numpy as np
import pytest

from app.ppca import PPCA


@pytest.fixture(scope="module")
def lowrank_data():
    rng = np.random.default_rng(1)
    latent = rng.standard_normal((500, 2))
    weights = rng.uniform(-1, 1, size=(2, 6))
    return 3 + 2 * (latent @ weights + 0.1 * rng.standard_normal((500, 6)))


def fit(data, **kwargs):
    model = PPCA()
    model.fit(data, d=2, min_obs=1, seed=0, **kwargs)
    return model


class TestFit:
    def test_recovers_planted_two_factor_structure(self, lowrank_data):
        model = fit(lowrank_data)
        assert model.var_exp[-1] > 0.9

    def test_stores_standardization_parameters(self, lowrank_data):
        model = fit(lowrank_data)
        np.testing.assert_allclose(model.means, np.nanmean(lowrank_data, axis=0))
        np.testing.assert_allclose(model.stds, np.nanstd(lowrank_data, axis=0))

    def test_same_seed_same_fit(self, lowrank_data):
        a, b = fit(lowrank_data), fit(lowrank_data)
        np.testing.assert_array_equal(a.C, b.C)

    def test_input_not_mutated(self, lowrank_data):
        before = lowrank_data.copy()
        fit(lowrank_data)
        np.testing.assert_array_equal(lowrank_data, before)

    def test_handles_missing_values(self, lowrank_data):
        rng = np.random.default_rng(2)
        data = lowrank_data.copy()
        data[rng.random(data.shape) < 0.1] = np.nan
        model = fit(data)
        assert np.isfinite(model.C).all()
        assert model.var_exp[-1] > 0.8


class TestTransform:
    def test_standardizes_before_projecting(self, lowrank_data):
        """The 2024 defect: raw data projected without standardization."""
        model = fit(lowrank_data)
        expected = ((lowrank_data - model.means) / model.stds) @ model.C
        np.testing.assert_allclose(model.transform(lowrank_data), expected)

    def test_training_scores_match_external_projection(self, lowrank_data):
        """Complete rows must land exactly where the fit placed them."""
        model = fit(lowrank_data)
        np.testing.assert_allclose(model.transform(lowrank_data), model.transform(), atol=1e-10)

    def test_rejects_missing_values(self, lowrank_data):
        model = fit(lowrank_data)
        bad = lowrank_data.copy()
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="complete"):
            model.transform(bad)

    def test_requires_fit(self):
        with pytest.raises(RuntimeError):
            PPCA().transform(np.zeros((3, 3)))


class TestPersistence:
    def test_save_load_roundtrip(self, lowrank_data, tmp_path):
        model = fit(lowrank_data)
        path = tmp_path / "model.npz"
        model.save(path)

        loaded = PPCA()
        loaded.load(path)
        np.testing.assert_array_equal(loaded.C, model.C)
        np.testing.assert_array_equal(loaded.means, model.means)
        np.testing.assert_array_equal(loaded.stds, model.stds)
        np.testing.assert_allclose(loaded.transform(lowrank_data), model.transform(lowrank_data))
