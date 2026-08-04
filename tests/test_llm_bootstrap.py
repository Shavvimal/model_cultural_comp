import numpy as np
import pandas as pd
import pytest

from app.culture_map import IV_QNS
from app.llm_bootstrap import bootstrap_llm_positions, confidence_ellipses


@pytest.fixture(scope="module")
def responses(fitted_map):
    rng = np.random.default_rng(3)
    rows = []
    for llm, offset in [("model-a", 0.0), ("model-b", 1.5)]:
        for q in IV_QNS:
            for value in 5 + offset + rng.standard_normal(30):
                rows.append({"llm": llm, "question": q, "value": value})
    return pd.DataFrame(rows)


class TestBootstrap:
    def test_replicate_count(self, fitted_map, responses):
        boot = bootstrap_llm_positions(fitted_map, responses, n_boot=50, seed=1)
        assert len(boot) == 2 * 50
        assert set(boot["llm"]) == {"model-a", "model-b"}

    def test_seed_determinism(self, fitted_map, responses):
        a = bootstrap_llm_positions(fitted_map, responses, n_boot=25, seed=1)
        b = bootstrap_llm_positions(fitted_map, responses, n_boot=25, seed=1)
        pd.testing.assert_frame_equal(a, b)

    def test_replicates_center_on_projected_mean(self, fitted_map, responses):
        """Projection is affine, so replicate positions must scatter around
        the projection of the per-question means."""
        boot = bootstrap_llm_positions(fitted_map, responses, n_boot=400, seed=2)
        for llm, group in responses.groupby("llm"):
            means = group.groupby("question")["value"].mean()
            center = fitted_map.project(
                pd.DataFrame([means[list(IV_QNS)].to_numpy()], columns=IV_QNS)
            )
            got = boot[boot["llm"] == llm][["PC1_rescaled", "PC2_rescaled"]].mean()
            np.testing.assert_allclose(
                got.to_numpy(),
                center[["PC1_rescaled", "PC2_rescaled"]].to_numpy()[0],
                atol=0.05,
            )

    def test_missing_question_raises(self, fitted_map, responses):
        incomplete = responses[responses["question"] != "F063"]
        with pytest.raises(ValueError, match="F063"):
            bootstrap_llm_positions(fitted_map, incomplete, n_boot=5, seed=1)


class TestConfidenceEllipses:
    def test_one_row_per_model_with_positive_axes(self, fitted_map, responses):
        boot = bootstrap_llm_positions(fitted_map, responses, n_boot=100, seed=4)
        ellipses = confidence_ellipses(boot)
        assert len(ellipses) == 2
        assert (ellipses["ellipse_width"] > 0).all()
        assert (ellipses["ellipse_height"] > 0).all()
        assert (ellipses["ellipse_width"] >= ellipses["ellipse_height"]).all()
