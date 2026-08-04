"""Single source of truth for the models surveyed in the 2024 collection run.

Previously four divergent copies of this list lived in culture_map.py,
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
