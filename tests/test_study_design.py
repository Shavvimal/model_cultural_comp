"""Pin the shared analysis-design constants to the values every stage uses."""

from app.cloud_survey import N_REPEATS, SYSTEM_PROMPTS, SYSTEM_PROMPTS_ZH
from app.culture_map import IV_QNS
from app.study_design import (
    ATTEMPTED_2024,
    ATTEMPTED_2024_ALIASING_CONSERVATIVE,
    COHERENT_2024,
    DESIGN_CALLS,
    MIN_PER_QUESTION,
    PERSONA_PREFIX_FAMILIES,
    PERSONA_PREFIX_FAMILY_OF,
    SENSITIVE_QNS,
    TRIAL_KEY,
)


def test_released_values_are_unchanged():
    assert MIN_PER_QUESTION == 10
    assert DESIGN_CALLS == 50
    assert SENSITIVE_QNS == ("F118", "F120", "F063", "G006", "E025")
    assert TRIAL_KEY == ("llm", "language", "question", "system_prompt_id", "repeat")
    assert (COHERENT_2024, ATTEMPTED_2024, ATTEMPTED_2024_ALIASING_CONSERVATIVE) == (4, 9, 7)
    assert dict(PERSONA_PREFIX_FAMILIES) == {
        "averaging": (0, 1, 3, 4, 6, 7),
        "bare": (2, 5, 8),
        "world_citizen": (9,),
    }
    assert list(PERSONA_PREFIX_FAMILIES) == ["averaging", "bare", "world_citizen"]


def test_design_calls_follow_the_collection_design():
    assert DESIGN_CALLS == len(SYSTEM_PROMPTS) * N_REPEATS


def test_prefix_families_partition_every_persona_prefix_once():
    ids = [i for family in PERSONA_PREFIX_FAMILIES.values() for i in family]
    assert sorted(ids) == list(range(len(SYSTEM_PROMPTS))) == list(range(len(SYSTEM_PROMPTS_ZH)))
    assert sorted(PERSONA_PREFIX_FAMILY_OF) == sorted(ids)
    assert all(
        PERSONA_PREFIX_FAMILY_OF[i] == f for f, m in PERSONA_PREFIX_FAMILIES.items() for i in m
    )


def test_prefix_family_labels_match_the_english_prefix_text():
    for i in PERSONA_PREFIX_FAMILIES["averaging"]:
        assert any(cue in SYSTEM_PROMPTS[i].lower() for cue in ("average", "typical"))
    for i in PERSONA_PREFIX_FAMILIES["bare"] + PERSONA_PREFIX_FAMILIES["world_citizen"]:
        assert not any(cue in SYSTEM_PROMPTS[i].lower() for cue in ("average", "typical"))
    assert "world citizen" in SYSTEM_PROMPTS[9].lower()


def test_sensitive_items_are_instrument_items():
    assert set(SENSITIVE_QNS) <= set(IV_QNS)
