# Collection protocol and retained records

| Path | Contents |
|---|---|
| `data/collection/*_responses_df.jsonl` | Eleven 2024 JSONL cell files, 500 parsed responses each; no recoverable variant IDs, failed-attempt text or reasoning excerpts |
| `data/collection_2026/<model>.jsonl`, `<model>__zh.jsonl` | 34 persona cells, 17,000 unique recorded trials; 215 terminal parse failures, including but not limited to refusals |
| `data/collection_2026_nosys/<model>.jsonl` | Seventeen English no-persona files: 8,500 scheduled trials, 8,442 unique recorded trials; qwen3.5:397b has 450 and nemotron-3-ultra 492. Resumed-run duplicates are removed by type-normalised trial keys, keeping the last line in file order: the 82 duplicated keys are all nemotron-3-ultra, and keep-last supersedes 15 parsed records with later failures and 2 failures with later parses. The QC gate does not cover this directory; `scripts/qc_2026.py` prints a report-only duplicate and missing-trial count for it. Absent records are not counted as refusals. |
| `data/trace_samples_2026.json` | Frozen sample of 900 excerpts from successful responses: 30 per trace-emitting cell, three per item, seed 42. Four cells from two non-emitting models are absent; 164 excerpts reach the 2,000-character cap. |
| `data/trace_labels_2026__<annotator>.jsonl` | Frozen-code labels from five LLM annotators; gpt-oss:120b and nemotron-3-super each label 60 own-model excerpts |
| `data/trace_coding_human_worksheet.csv` | Regenerated optionally with `scripts/code_traces_2026.py --worksheet`; no completed human reliability estimate is claimed |

Each 2026 JSONL row records a trial, not a single API attempt:

| Field | Meaning |
|---|---|
| `llm`, `question` | Requested model tag and IVS item code; the historical records do not retain a returned model identity or immutable build digest |
| `system_prompt_id` | Legacy name for the user-message persona-prefix variant, 0–9; no-persona uses 0 |
| `repeat` | 0–4 within a persona variant; 0–49 with no persona |
| `language` | en/zh; earliest English records default to en |
| `prompt_variant` | persona/nosys; older persona records default to persona |
| `raw_content` | Most recent returned answer text; earlier rejected drafts are not exhaustively retained |
| `thinking` | Stored reasoning excerpt, capped at 2,000 characters; empty when no text is emitted |
| `parsed` | Parsed integer or null |
| `error` | Terminal error or null. Every retained error is a `parse:` failure; future collection may also write `provider:` rejections. Transport-only failures are never terminal. A failure need not be a refusal |
| `attempts` | Up to `MAX_ATTEMPTS = 3` calls in the task invocation that wrote the record. The collector re-sweeps transport-deferred trials up to `MAX_SWEEPS = 5` times, so one invocation can make up to 15 calls per trial, and re-running the collector has no bound. Re-sweeps restart the count, so this is not a complete API-call or billing ledger |
| `duration_ms`, `ts` | Recorded latency and final-attempt timestamp |

The retained 2026 records predate schema version 2. No record in either
directory has `schema_version`, `request`, `response`, `request_provenance` or
`task_run_id`. The 17,000 persona records have no `prompt_variant` field, and
4,122 of them have no `language` field; both take the defaults above. All 8,524
no-persona lines carry `language` and `prompt_variant`.

Each request sends two messages in this order. The user message joins the
persona prefix (omitted with no persona), the item wording and the format
instruction. It is followed by a trailing system-role message, the primer, which
is the only system message and reads verbatim `Sure thing! Here is my numerical
answer:` in English and `好的！这是我的数字答案：` in Chinese. It is the
2024-protocol refusal mitigation, kept for comparability.

The F120 anchor order differs between arms: the English prompt names 10 first
("10 means always justifiable and 1 means never justifiable"), while the Chinese
F120 prompt and both F118 prompts name 1 first. Translation caveat: Chinese
prefixes 0, 3 and 6 render "average" as 普通 ("ordinary, common"), which the
analyses count as an averaging cue, as they do 典型 ("typical"). 普通 reads
closer to "ordinary" than to a statistical average. Both strings stay verbatim,
because the retained corpus was collected with them; corrected wording would be a
new, versioned prompt set collected into a new directory.

The historical respondent calls supplied the model tag and messages without
explicit temperature, top-p, top-k, seed, token-limit or reasoning-effort settings.
The provider's resolved defaults and model revision were not retained. Annotator
temperature settings do not establish respondent settings. All English primary
cells were collected before the Chinese cells; administration order was not
randomised or interleaved.

For future collection, schema version 2 additionally records the exact request
arguments, returned response fields, SDK version, host and timeout, plus a
`task_run_id` for each invocation. Optional `CloudSurvey.generation_options` and
`CloudSurvey.thinking` settings are preserved when supplied; unspecified provider
defaults remain explicitly unknown. Each attempted call is separately logged in
`attempt_audit/` with its outcome, including transport failures and retries.
Those private prospective logs are excluded from the public release by default.
A process failure between a call and its log write can still leave an unrecorded
in-flight call. Historical records are unchanged and their missing metadata
cannot be recovered by this logging change.

The main no-persona rule requires at least ten parsed answers per item: 13 cells
qualify, all remain in the quadrant and 12 are farther from the survey reference.
Matched median distances are regenerated in
`data/prompt_control_summary_2026.csv`.
The separately labelled one-answer
sensitivity has 15 eligible cells, 14 farther away. This later collection retains
the refusal-mitigation system primer: it is neither a no-system-prompt experiment
nor a causal test of the reasoning mechanism.

Five-annotator majority coding identifies **typicality or moderation targeting**
in 55%, persona reasoning in 5%, and AI-identity/guideline references in 42%;
Fleiss' kappa is .78, .35 and .75, respectively. Labels overlap. The broad first
code does not demonstrate accurate recovery of human modes or faithful reasoning.

## Portable 2024 schema

Each JSONL line has exactly `llm`, `question`, and `response`. Single-choice
responses are integers, Y002 is an ordered two-element array, Y003 is an array
of selected qualities, and missing answers are null. A filename beginning
`c-` identifies Chinese administration; other files are English. These fields
and row order were preserved from the retained pickles. No variant IDs or
failed-attempt text have been inferred. The eleven files contain 5,500 records.

The frozen trace sample contains 900 successful excerpts and 215 terminal
failure records. Missing error fields are JSON null. The five annotation
panels code only the 900 successful excerpts; their texts, keys and labels
are preserved. Regenerating the sample or annotating again is a new operation,
not part of offline replay. The sample and panels belong in the separate
response archive described in [REPRODUCING.md](REPRODUCING.md).

## New collection

Resuming requires schema-version 2 terminal records whose exact request and
host match the current configuration. Changed prompts, generation options,
thinking settings or host require a new output directory. Historical records
without those fields remain valid offline analysis inputs, but cannot establish
compatibility for appending new responses and therefore cannot be resumed.

Each task invocation makes up to `MAX_ATTEMPTS = 3` calls. A parse failure is
retried, and the last one is persisted as a terminal `parse:` error. Once a
trial has produced a parse failure, it is persisted with its last parse failure
even if a later attempt in the same invocation fails in transport. A 4xx response
other than 408 or 429 is persisted at once as a terminal `provider:` error, which
QC accepts and counts separately from parse failures. A trial with only transport
failures stays absent; `scripts/collect_cloud_2026.py` re-sweeps it up to
`MAX_SWEEPS = 5` times, `SWEEP_BACKOFF_S = 120` seconds apart. That is at most 15
calls per trial per invocation, and there is no bound across invocations. These
rules govern future collection only. The retained corpus was collected when a
trial whose final attempt failed in transport was re-swept even after earlier
parse failures, so refusing trials on unreliable endpoints could receive more
chances to answer.

The API key is sent only in the request header and is excluded from the harness
`repr`. A host containing a username, password or query string is rejected, and
records keep only its scheme, host and port. Error text has every exact
occurrence of the configured key replaced with `[REDACTED]` before it is stored.

If a crash leaves a truncated final line, resuming and the QC gate both stop and
name `file:line`. To recover a new collection, keep a copy of the file, delete
only that incomplete last line (it is not valid JSON and has no closing newline),
check that every remaining line parses, then re-run the same collector command.
The trial has no terminal record, so it is asked again; its earlier calls remain
in `attempt_audit/`. Never edit the frozen retained directories this way.

The collector reads `OLLAMA_API_KEY` and optionally `OLLAMA_HOST` from the
environment or a local `.env`. Run explicitly, from a separate checkout or
with the frozen response directories moved safely aside:

```bash
uv run python scripts/collect_cloud_2026.py --language=en MODEL_TAG
uv run python scripts/collect_cloud_2026.py --language=zh MODEL_TAG
uv run python scripts/collect_cloud_2026.py --prompt-variant=nosys MODEL_TAG
```

These commands call hosted endpoints and can incur charges. With no model
argument, the current driver lists and surveys the account's available models;
that list need not equal the frozen study cohort. Requested model tags and
current endpoint availability cannot guarantee identical future responses.
`make progress` and `make watch` show raw terminal-record line counts during
collection; the QC gate checks actual trial-key completeness afterward.

## Frozen study cohort

The model lists live in `app/llm_meta.py` — the single source of truth. Models that
never produced parseable answers in the 2024 run are recorded there in
`FAILED_LLMS_2024` rather than being silently dropped.

### 2024 cohort — eleven model-language cells

Served locally through [Ollama](https://ollama.com) on consumer hardware. Historical
notes describe Q4-quantised GGUF builds, but the retained model responses and
Modelfiles do not verify the quantisation and digest of every actual served build.
Two models were administered *only* in Chinese and one in both
languages (analysed as two cells, marked `[zh]`/`[en]`).

- **Chinese-origin / Chinese fine-tuned:** `qwen2:7b` (both languages),
  `llama2-chinese:13b` `[zh]`, `wangshenzhi/gemma2-27b-chinese-chat` `[zh]`,
  `wangrongsheng/llama3-70b-chinese-chat`
- **Western:** `llama3:70b`, `mistral:7b`, `gemma2:27b`
- **Uncensored (Dolphin):** `dolphin-llama3:8b`, `dolphin-mistral:7b`,
  `dolphin-mixtral:8x7b`
- **Attempted, excluded for producing nothing parseable:** `yi:34b`,
  `aquilachat2:34b`, `glm4:9b`, `xuanyuan:70b`,
  `kingzeus/llama-3-chinese-8b-instruct-v3` (five model names; the historical
  Modelfiles cannot confirm five distinct base artefacts — the `yi` and `glm`
  Modelfiles point at the AquilaChat2 GGUF — which is disclosed wherever the
  2024 usability denominator is used)

### 2026 cohort — 17 models × two administration languages

Cloud-served (Ollama Cloud; serving precision undisclosed by the provider and
carried as a confound), 34 cells of 500 recorded trials each, English and Chinese arms.

- **Chinese-origin (10):** `deepseek-v4-flash`, `deepseek-v4-flash:0731`,
  `deepseek-v4-pro`, `glm-5.1`, `glm-5.2`, `kimi-k2.6`, `kimi-k2.7-code`,
  `minimax-m2.7`, `minimax-m3`, `qwen3.5:397b`
- **Western (7):** `gemma4:31b`, `gpt-oss:20b`, `gpt-oss:120b`,
  `mistral-large-3:675b`, `nemotron-3-nano:30b`, `nemotron-3-super`,
  `nemotron-3-ultra`

An eighteenth model, `kimi-k3`, was excluded before any data was collected (every
call returned a billing error; zero records, no part in any denominator). One cell,
`nemotron-3-ultra [zh]`, is excluded from position estimates (F120 parsed 4/50,
under the inclusion threshold) — its refusals are analysed as data and its
worst-case Manski bound is still reported.

Each model carries its own upstream licence; check it before reuse.


The Chinese 2024 protocol labelled both F118 (homosexuality) scale poles “always justifiable” and
collapsed some persona prefixes into duplicates. The 2026 translations repair
these defects and translate the formatting instructions and primer. Protocol
changes, collection dates and endpoint revisions confound a direct causal
comparison between the two cohorts. Exact 2026 prompts remain in
`app/cloud_survey.py`; historical 2024 code is retained in the author's original
checkout and is not needed to replay the stored responses.
