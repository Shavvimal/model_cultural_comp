"""The frozen codebook for coding reasoning traces, and its unit builder.

The initial 2026 write-up used vendor-specific coding rules and retained
only cell counts, so inter-annotator agreement could not be computed.
The camera-ready analysis applies the three binary codes to 900 selected
traces using five LLM annotators under this fixed prompt; agreement is
reported by ``scripts/trace_agreement_2026.py``. A separate human worksheet
remains uncoded and supplies no human validation. The appendix reproduces
the prompt with formatting adaptations. Changing CODEBOOK requires new
annotations; changes to this explanatory module docstring do not.

The reasoning-language code is not asked of the annotators: it is computed
mechanically from the script of the trace (``reasoning_language``).
"""

import re

CODEBOOK = """You are coding a reasoning trace produced by a language model that was asked \
to answer one item of a values survey. Read the trace and assign three binary codes. \
Judge only what the trace says, not the final answer's correctness.

CODES

1. modal_targeting (1/0): the trace uses what a typical, average, common, majority or \
"most people" answer would be, or population statistics, survey distributions, or an \
explicit aim at a "moderate" / "middle-of-the-road" answer, AS THE CRITERION for choosing \
the answer. Count it even if the final answer then deviates from that typical answer. \
Do NOT count: merely restating the assigned role ("I am an average person") without using \
typicality to choose; picking the scale midpoint only as a hedge ("I'll say 5, the middle") \
with no claim about what people commonly answer; AI-neutrality on its own.

2. persona_reasoning (1/0): the trace deliberates in the first person about the values, \
beliefs, experiences or circumstances themselves, as the respondent ("I'm fairly happy with \
my life", "God matters a great deal to me", "I have signed petitions before"), and that \
first-person position drives the answer. Do NOT count third-person simulation ("an average \
person would probably say"), or reasoning about how to simulate a person.

3. guideline_citation (1/0): the trace explicitly refers to policies, guidelines, safety, \
disallowed or sensitive content, or runs a harm/risk/compliance check; OR it invokes the \
model's identity as an AI as a limitation or constraint on answering ("as an AI I don't have \
personal beliefs / a nationality / can't take a stance"). Do NOT count a bare "As an AI, I'll \
simulate an average person" that carries no limitation or constraint.

The codes are independent: any combination of 0s and 1s is allowed.

OUTPUT

Reply with exactly one JSON object and nothing else, for example:
{"modal_targeting": 1, "persona_reasoning": 0, "guideline_citation": 1}"""

CJK = re.compile(r"[　-〿㐀-䶿一-鿿豈-﫿＀-￯]")
LATIN = re.compile(r"[A-Za-z]")


def reasoning_language(text: str) -> str:
    """``english`` / ``cjk`` / ``mixed`` by script share of the trace.

    Share = CJK characters / (CJK + Latin letters). Zero CJK is ``english``;
    a CJK majority is ``cjk``; anything in between is ``mixed`` (typically
    English scaffolding quoting the Chinese item text).
    """
    n_cjk = len(CJK.findall(text))
    n_lat = len(LATIN.findall(text))
    if n_cjk == 0:
        return "english"
    if n_cjk / (n_cjk + n_lat) >= 0.5:
        return "cjk"
    return "mixed"


def build_unit(
    *,
    language: str,
    question: str,
    item_prompt: str,
    persona_prefix: str,
    thinking: str,
    final_answer: str,
) -> str:
    """The user turn for one trace: context, then the trace, then the answer."""
    persona = (
        persona_prefix if persona_prefix else "(none: the item was asked with no persona prefix)"
    )
    return (
        f"Survey administration language: {'Chinese' if language == 'zh' else 'English'}\n"
        f"Persona prefix the model was given: {persona}\n"
        f"Item {question} (English wording): {item_prompt}\n\n"
        f"REASONING TRACE (may be truncated at 2,000 characters):\n<<<\n{thinking}\n>>>\n\n"
        f"Model's final answer: {final_answer!r}\n\n"
        "Return the JSON object."
    )


JSON_OBJECT = re.compile(r"\{[^{}]*\}")
CODES = ("modal_targeting", "persona_reasoning", "guideline_citation")


def parse_labels(text: str) -> dict[str, int]:
    """Extract the three 0/1 codes from an annotator's reply, or raise ValueError."""
    import json

    text = re.sub(r"<think>.*?(?:</think>|\Z)", "", text, flags=re.DOTALL)
    for candidate in JSON_OBJECT.findall(text):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not all(k in obj for k in CODES):
            continue
        out = {}
        for k in CODES:
            v = obj[k]
            if v in (0, 1, True, False):
                out[k] = int(v)
            elif isinstance(v, str) and v.strip() in ("0", "1"):
                out[k] = int(v.strip())
            else:
                break
        else:
            return out
    raise ValueError(f"no valid label object in: {text[:200]!r}")
