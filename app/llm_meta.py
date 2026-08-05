"""Single source of truth for the models surveyed, both cohorts.

The 2024 cohort was served locally through Ollama at Q4 quantisation; the 2026
cohort is cloud-served and crossed with two administration languages. A model
that produced nothing parseable is recorded here rather than silently dropped,
because a failure rate is itself a result.

Previously four divergent copies of the 2024 list lived in culture_map.py,
culture_map_post_hoc.py and notebooks/9-pca-llm.ipynb; they disagreed about
which entries were commented out. Import from here instead.
"""

# Models of Chinese origin or with Chinese-focused fine-tuning.
CHINESE_LLMS = frozenset(
    {
        "wangshenzhi/gemma2-27b-chinese-chat",
        "qwen2:7b",
        "llama2-chinese:13b",
        "wangrongsheng/llama3-70b-chinese-chat",
        "yi:34b",
        "aquilachat2:34b",
        "kingzeus/llama-3-chinese-8b-instruct-v3:q8_0",
        "xuanyuan:70b",
        "glm4:9b",
    }
)

# Uncensored fine-tunes (alignment/bias-filtered training data).
DOLPHIN_LLMS = frozenset(
    {
        "dolphin-llama3:8b",
        "dolphin-mistral:7b",
        "dolphin-mixtral:8x7b",
    }
)

# Models that never produced coherent survey responses in the 2024 run and
# therefore have no rows in data/collection/.
FAILED_LLMS_2024 = frozenset(
    {
        "yi:34b",  # answers "." only
        "aquilachat2:34b",  # answers "。" or echoes the prompt
        "kingzeus/llama-3-chinese-8b-instruct-v3:q8_0",  # fails intermittently
        "xuanyuan:70b",  # unintelligible output
        "glm4:9b",  # answers "." only
    }
)


def is_chinese(llm: str) -> bool:
    return llm in CHINESE_LLMS


# --- 2026 Ollama Cloud generation ---------------------------------------
#
# The collected cohort is 17 models x two administration languages = 34 cells
# of 500 calls. ``kimi-k3`` stays listed as an attempted model but contributes
# no records: every call returned a billing error (the model needs metered
# "extra usage" outside the serving subscription), so it was excluded before
# any data was collected and plays no part in any denominator.

# Chinese-origin frontier models attempted on Ollama Cloud (August 2026).
CHINESE_LLMS_2026 = frozenset(
    {
        "deepseek-v4-flash",
        "deepseek-v4-flash:0731",
        "deepseek-v4-pro",
        "glm-5.1",
        "glm-5.2",
        "kimi-k2.6",
        "kimi-k2.7-code",
        "kimi-k3",  # attempted; billing error on every call, zero records
        "minimax-m2.7",
        "minimax-m3",
        "qwen3.5:397b",
    }
)

# Western-origin cloud models in the same run.
WESTERN_LLMS_2026 = frozenset(
    {
        "gemma4:31b",
        "gpt-oss:120b",
        "gpt-oss:20b",
        "mistral-large-3:675b",
        "nemotron-3-nano:30b",
        "nemotron-3-super",
        "nemotron-3-ultra",
    }
)


def cohort_2026(llm: str) -> str:
    if llm in CHINESE_LLMS_2026:
        return "Chinese"
    if llm in WESTERN_LLMS_2026:
        return "Western"
    raise ValueError(f"unknown 2026 model: {llm}")
