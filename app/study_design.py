"""Analysis-design constants shared by the 2026 collection, QC and analysis stages.

Each value records a decision that is reported in the paper or the analysis
plan. They are defined once here so a script cannot silently apply a different
rule from the one another stage reports. Changing any value changes published
results; the values below are the ones every released artefact was built with.
"""

from __future__ import annotations

from types import MappingProxyType

# Inclusion rule: a model-language cell enters the primary analysis only if
# every instrument item has at least this many parsed answers (analysis plan).
MIN_PER_QUESTION = 10

# Scheduled trials per item within one cell: ten persona prefixes times five
# repeats (app.cloud_survey.SYSTEM_PROMPTS and N_REPEATS).
DESIGN_CALLS = 50

# Refusal-sensitive items (politics, religion, sexuality and petition signing)
# used by the refusal diagnostics and the petition-signing contrast. The order
# is part of the released reference-set label, so keep it.
SENSITIVE_QNS: tuple[str, ...] = ("F118", "F120", "F063", "G006", "E025")

# Persona-prefix sub-families by system_prompt_id: six prefixes carry an
# averaging cue ("average" or "typical"), three are bare ("a human being",
# "a person", "an individual") and one is "a world citizen". Insertion order
# fixes the row order of the prompt-sensitivity outputs.
PERSONA_PREFIX_FAMILIES: MappingProxyType[str, tuple[int, ...]] = MappingProxyType(
    {
        "averaging": (0, 1, 3, 4, 6, 7),
        "bare": (2, 5, 8),
        "world_citizen": (9,),
    }
)
PERSONA_PREFIX_FAMILY_OF: MappingProxyType[int, str] = MappingProxyType(
    {prefix: family for family, ids in PERSONA_PREFIX_FAMILIES.items() for prefix in ids}
)

# Columns that identify one scheduled trial. Deduplication and joins use this
# key; pandas needs a list, so callers pass ``list(TRIAL_KEY)``.
TRIAL_KEY: tuple[str, ...] = ("llm", "language", "question", "system_prompt_id", "repeat")

# 2024 longitudinal baseline: 4 of 9 attempted Chinese-origin models produced
# parseable corpora (recorded model list). Two AquilaChat2-aliased runs may
# share a base artefact, so the conservative denominator is 7.
COHERENT_2024 = 4
ATTEMPTED_2024 = 9
ATTEMPTED_2024_ALIASING_CONSERVATIVE = 7
