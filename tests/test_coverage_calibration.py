import pytest

from scripts.coverage_calibration_2026 import analytic_coverage, hotelling_radius


def test_hotelling_uses_corresponding_covariance_normalisation():
    unbiased = hotelling_radius()
    bootstrap = hotelling_radius(divisor_k=True)
    assert unbiased == pytest.approx(10.032682741930149)
    assert bootstrap == pytest.approx(unbiased * 10 / 9)
    assert analytic_coverage(radius=unbiased) == pytest.approx(0.95)
    assert analytic_coverage(divisor_k=True, radius=bootstrap) == pytest.approx(0.95)
    assert analytic_coverage(divisor_k=True, radius=unbiased) == pytest.approx(0.9379068707)


def test_nominal_chi_square_radius_is_only_a_gaussian_benchmark():
    assert analytic_coverage() == pytest.approx(0.8701045946)
    assert analytic_coverage(divisor_k=True) == pytest.approx(0.8470860730)


def test_two_dimensional_hotelling_needs_more_than_two_clusters():
    with pytest.raises(ValueError, match="K > 2"):
        hotelling_radius(k=2)
