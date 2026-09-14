"""The 2026 cohort sets and developer families must describe the same models."""

from app.llm_meta import (
    CHINESE_LLMS_2026,
    COLLECTED_LLMS_2026,
    EXCLUDED_LLMS_2026,
    FAMILY_OF,
    WESTERN_LLMS_2026,
)


def test_families_cover_exactly_the_collected_models():
    assert set(FAMILY_OF) == COLLECTED_LLMS_2026
    assert len(COLLECTED_LLMS_2026 & CHINESE_LLMS_2026) == 10
    assert len(COLLECTED_LLMS_2026 & WESTERN_LLMS_2026) == 7
    assert len(COLLECTED_LLMS_2026) == 17


def test_excluded_models_were_attempted_but_not_collected():
    assert EXCLUDED_LLMS_2026 == frozenset({"kimi-k3"})
    assert EXCLUDED_LLMS_2026 <= CHINESE_LLMS_2026 | WESTERN_LLMS_2026
    assert not EXCLUDED_LLMS_2026 & COLLECTED_LLMS_2026
