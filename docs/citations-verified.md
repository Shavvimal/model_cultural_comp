# Citations — verification and enrichment

Verification pass for the ORACLE @ EMNLP 2026 submission draft
(`~/Code/next-mdx-blog/content/blog/cultural-bias-2026.mdx`).
Every citation below was checked against the arXiv abstract page, publisher page, or
archive landing page on 2026-08-04.

**Headline:** all 7 cited arXiv IDs exist. **None** is a hallucinated identifier.
But **5 of 7** were cited under a title that is not the paper's actual title, and
**2 prose claims in §2 misdescribe what the cited paper found** — including one
(`[^kazemi]`) that is completely wrong about the paper's subject, and one
(`[^llmglobe]`) that reverses the paper's headline result.

---

## 1. Corrected footnote block (paste-ready)

Drop-in replacement for lines 159-187 of `cultural-bias-2026.mdx`. Keys unchanged
so no in-text `[^key]` references need editing. New keys at the end are additions
(§2 of this document explains each).

```mdx
[^iw2005]: Inglehart, R., & Welzel, C. (2005). *Modernization, Cultural Change, and Democracy: The Human Development Sequence*. Cambridge University Press. ISBN 9780521846950.

[^tao2024]: Tao, Y., Viberg, O., Baker, R. S., & Kizilcec, R. F. (2024). [Cultural bias and cultural alignment of large language models](https://doi.org/10.1093/pnasnexus/pgae346). *PNAS Nexus*, 3(9), pgae346. Preprint: arXiv:2311.14096.

[^llmglobe]: Karinshak, E., Hu, A., Kong, K., Rao, V., Wang, J., Wang, J., & Zeng, Y. (2024). [LLM-GLOBE: A Benchmark Evaluating the Cultural Values Embedded in LLM Output](https://arxiv.org/abs/2411.06032). arXiv:2411.06032.

[^kazemi]: Kazemi, S., Gerhardt, G., Katz, J., Kuria, C. I., Pan, E., & Prabhakar, U. (2024). [Cultural Fidelity in Large-Language Models: An Evaluation of Online Language Resources as a Driver of Model Performance in Value Representation](https://arxiv.org/abs/2410.10489). arXiv:2410.10489.

[^scenario]: Dang, T. D. A., Kieu, T., & Masud, S. (2026). [Scenario-based Probing and Steering Cultural Values in Large Language Models — Extended Version](https://arxiv.org/abs/2606.11399). arXiv:2606.11399.

[^steering]: Dang, T. D. A., & Masud, S. (2026). [Cultural Value Alignment Via Latent Activation Steering in Large Language Models](https://arxiv.org/abs/2605.26365). arXiv:2605.26365. ACL 2026 Student Research Workshop (non-archival track).

[^personas]: Greco, C. M., La Cava, L., & Tagarelli, A. (2026). [Culturally Grounded Personas in Large Language Models: Characterization and Alignment with Socio-Psychological Value Frameworks](https://arxiv.org/abs/2601.22396). arXiv:2601.22396.

[^wordassoc]: Dai, X., Zhou, L., Wang, B., & Li, H. (2025). [From Word to World: Evaluate and Mitigate Culture Bias in LLMs via Word Association Test](https://arxiv.org/abs/2505.18562). In *Proceedings of EMNLP 2025* (Oral). arXiv:2505.18562.

[^debate]: Tan, Q., Jiang, L., Zeng, Y., Ding, S., & Xu, X. (2026). [Mitigating Cultural Bias in LLMs via Multi-Agent Cultural Debate](https://arxiv.org/abs/2601.12091). arXiv:2601.12091.

[^au1983]: Au, T. K.-F. (1983). [Chinese and English counterfactuals: The Sapir-Whorf hypothesis revisited](https://doi.org/10.1016/0010-0277(83)90038-0). *Cognition*, 15(1-3), 155-187.

[^tipping]: Tipping, M. E., & Bishop, C. M. (1999). [Probabilistic principal component analysis](https://doi.org/10.1111/1467-9868.00196). *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 61(3), 611-622.

[^pcamagic]: Tran, A. [pca-magic](https://github.com/allentran/pca-magic) (Apache-2.0). The projection implementation is derived from this package; see the repository NOTICE file.

[^cameron]: Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). [Bootstrap-based improvements for inference with clustered errors](https://doi.org/10.1162/rest.90.3.414). *The Review of Economics and Statistics*, 90(3), 414-427.

[^evs]: EVS (2022). *EVS Trend File 1981-2017: Integrated Dataset (EVS 1981-2017)*. GESIS Data Archive, Cologne. ZA7503 Data file Version 3.0.0, doi:10.4232/1.14021.

[^wvs]: Haerpfer, C., Inglehart, R., Moreno, A., Welzel, C., Kizilova, K., Diez-Medrano, J., Lagos, M., Norris, P., Ponarin, E., & Puranen, B. (eds.) (2022). *World Values Survey Trend File (1981-2022) Cross-National Data-Set*. Madrid, Spain & Vienna, Austria: JD Systems Institute & WVSA Secretariat. Data File Version 4.0.0, doi:10.14281/18241.27.

[^promptlang]: Bulté, B., & Rigouts Terryn, A. (2025). [LLMs and Cultural Values: The Impact of Prompt Language and Explicit Cultural Framing](https://arxiv.org/abs/2511.03980). arXiv:2511.03980. Accepted (minor revisions) at *Computational Linguistics*.

[^deepseekweird]: Luther, J., & Brown, D. (2025). [DeepSeek's WEIRD Behavior: The cultural alignment of Large Language Models and the effects of prompt language and cultural prompting](https://arxiv.org/abs/2512.09772). arXiv:2512.09772.

[^perturb]: Rupprecht, J., Ahnert, G., & Strohmaier, M. (2025). [Prompt Perturbations Reveal Human-Like Biases in Large Language Model Survey Responses](https://arxiv.org/abs/2507.07188). arXiv:2507.07188.

[^dasman]: Li, D., Li, L., & Qiu, H. S. (2025). [ChatGPT is not A Man but Das Man: Representativeness and Structural Consistency of Silicon Samples Generated by Large Language Models](https://arxiv.org/abs/2507.02919). arXiv:2507.02919.

[^personareliab]: Taday Morocho, E. E., Cima, L., Fagni, T., Avvenuti, M., & Cresci, S. (2026). [Assessing the Reliability of Persona-Conditioned LLMs as Synthetic Survey Respondents](https://arxiv.org/abs/2602.18462). arXiv:2602.18462.

[^silenced]: Himelstein, R., LeVi, A., Youngmann, B., Nemcovsky, Y., & Mendelson, A. (2025). [Silenced Biases: The Dark Side LLMs Learned to Refuse](https://arxiv.org/abs/2511.03369). arXiv:2511.03369. AAAI 2026, AI Alignment track (Oral).

[^promptprog]: Eren, M., Michalak, E., Cook, B., & Seales Jr., J. (2026). [Prompt Programming for Cultural Bias and Alignment of Large Language Models](https://arxiv.org/abs/2603.16827). arXiv:2603.16827.
```

### Per-citation verification log

| Key | Draft status | Finding |
|---|---|---|
| `[^iw2005]` | ✅ correct | Cambridge UP, 2005, title and subtitle exact. Add ISBN if the venue wants it. |
| `[^tao2024]` | ✅ correct | *PNAS Nexus* 3(9) pgae346, doi 10.1093/pnasnexus/pgae346 confirmed. Authors confirmed: Yan Tao, Olga Viberg, Ryan S. Baker, René F. Kizilcec. arXiv:2311.14096 v1 Nov 2023, v2 Jun 2024 — same paper. |
| `[^llmglobe]` | ⚠️ **title OK, author list missing, PROSE CLAIM WRONG** | Title exact. Authors (absent from draft): Karinshak, Hu, Kong, Rao, Wang (Jingren), Wang (Jindong), Zeng. **The §2 sentence "finding smaller East-West differences than expected" is not what the paper reports** — see §4 below. |
| `[^kazemi]` | ❌ **TITLE WRONG + PROSE CLAIM WRONG** | Draft title "Survey-to-behavior gaps and norm conflicts in LLM value probing" is invented. Actual: *Cultural Fidelity in Large-Language Models: An Evaluation of Online Language Resources as a Driver of Model Performance in Value Representation*. Authors: Sharif Kazemi, Gloria Gerhardt, Jonty Katz, Caroline Ida Kuria, Estelle Pan, Umang Prabhakar. Columbia affiliation confirmed (`ms6578@columbia.edu`) — matches the 2024 blog's "a research paper from Columbia University". |
| `[^scenario]` | ⚠️ title approximate | Actual: *Scenario-based Probing and Steering Cultural Values in Large Language Models — Extended Version*. Dang, Kieu & Masud; 9 Jun 2026; 18 pp. |
| `[^steering]` | ⚠️ title approximate | Actual: *Cultural Value Alignment Via Latent Activation Steering in Large Language Models*. Dang & Masud; 25 May 2026; ACL 2026 SRW non-archival. |
| `[^personas]` | ⚠️ title approximate | Actual: *Culturally Grounded Personas in Large Language Models: Characterization and Alignment with Socio-Psychological Value Frameworks*. Greco, La Cava & Tagarelli; 29 Jan 2026, rev. 3 Jun 2026; under review. |
| `[^wordassoc]` | ❌ **TITLE WRONG** | Draft "Word-association tests for measuring cultural bias in LLMs" is invented. Actual: *From Word to World: Evaluate and Mitigate Culture Bias in LLMs via Word Association Test*. Dai, Zhou, Wang & Li. **EMNLP 2025 (Oral)** — cite the venue, not just arXiv, especially for an EMNLP workshop submission. |
| `[^debate]` | ❌ **TITLE WRONG** | Draft "Multi-agent cultural debate for pluralistic alignment" is invented. Actual: *Mitigating Cultural Bias in LLMs via Multi-Agent Cultural Debate*. Tan, Jiang, Zeng, Ding & Xu; 17 Jan 2026. |
| `[^au1983]` | ⚠️ pagination | Title, author, journal, volume, year all correct. Pages 155-187 correct. Issue is **15(1-3)** (a combined issue), not 15(1) — minor; most style guides accept `15`. |
| `[^tipping]` | ✅ correct | JRSS-B 61(3):611-622, 1999, doi 10.1111/1467-9868.00196. |
| `[^cameron]` | ✅ correct | *Review of Economics and Statistics* 90(3):414-427, 2008, doi 10.1162/rest.90.3.414. Journal's formal name is *The Review of Economics and Statistics*. |
| `[^evs]` | ✅ correct | ZA7503, v3.0.0, released 2022-12-14, doi:10.4232/1.14021. GESIS's own recommended string is *"EVS (2022): EVS Trend File 1981-2017: Integrated Dataset (EVS 1981-2017). GESIS Data Archive, Cologne. ZA7503 Data file Version 3.0.0, doi:10.4232/1.14021."* — the draft drops the `: Integrated Dataset (EVS 1981-2017)` subtitle. Add it. |
| `[^wvs]` | ⚠️ **version drift + truncated editor list** | DOI 10.14281/18241.27 is version-agnostic and **now resolves to Data File Version 4.1.0**, not the 4.0.0 the draft cites. Two actions: (a) confirm against the actual downloaded file which version the pipeline consumed and state it; (b) the DOI alone will not disambiguate, so record the download date in the repo. Draft's `Haerpfer, C., et al.` should be expanded — WVSA's recommended citation names all ten editors (see corrected block). |

---

## 2. New papers to engage (verified)

Ordered by how badly the submission needs them. The first two are, in my judgement,
**required** — a reviewer working on elicitation-language effects will know them,
and the paper's central new finding is squarely their territory.

### (b) Elicitation-language effects — the paper's central finding

**1. `[^promptlang]` Bulté & Rigouts Terryn (2025), arXiv:2511.03980 — accepted at *Computational Linguistics*.**
Probes 10 LLMs with 63 Hofstede VSM + WVS items translated into 11 languages, and finds
that both prompt language and explicit cultural framing shift outputs, but that all models
stay anchored to a small set of cultural defaults (NL, DE, US, JP), with explicit cultural
framing shifting values more than prompt language alone.
*Engagement:* This is the closest prior work to our language finding and it must be cited as
such — it independently establishes that prompt language moves expressed values, so our
contribution is not the existence of the effect but its **magnitude relative to sampling
uncertainty** (1.77 map units against per-cell SDs of ~0.1) on the IW instrument, and the
observation that the 2024 literature's Confucian-region claims rest on cells where language
and developer origin are perfectly confounded.

**2. `[^deepseekweird]` Luther & Brown (2025), arXiv:2512.09772.**
Administers Hofstede VSM13 to GPT and DeepSeek models under both prompt-language switching
and cultural prompting, finding DeepSeek aligns with US values regardless of prompting
strategy while GPT models vary by version.
*Engagement:* Directly on our Chinese-origin question by an independent instrument — cite as
converging evidence that Chinese-developed frontier models do not express Chinese-aligned
values, and note that our 2026 factorial design tests the same contrast with both languages
crossed within model rather than compared across prompting conditions.

**3. `[^promptprog]` Eren, Michalak, Cook & Seales (2026), arXiv:2603.16827.**
Extends the Tao et al. cultural-prompting result from proprietary APIs to open-weight LLMs
and shows DSPy prompt optimization beats hand-written cultural prompt engineering.
*Engagement:* **Contests a novelty claim.** The draft's §2 differentiator "we evaluate
open-weight models rather than a single vendor's API models" is now partially occupied —
cite this and re-pitch the differentiator on measurement auditability (frozen pipeline,
uncertainty regions, classifier-free statistics) rather than on open weights alone.

### (c) Survey-response validity / modal and central-tendency responding

**4. `[^perturb]` Rupprecht, Ahnert & Strohmaier (2025), arXiv:2507.07188.**
Runs 167,000+ simulated WVS interviews over nine LLMs under ten prompt perturbations,
testing explicitly for recency, central-tendency and opinion-floating bias, and finds a
consistent recency bias with larger models more robust — and notes that some models
(e.g. Qwen-2.5-7B) are effectively censored on sensitive items, producing high item
nonresponse and invalid interviews.
*Engagement:* Does double duty. In §5 it supplies the external evidence that our
"mid-scale/modal responding registers as secularity" concern is a real measured phenomenon
rather than a speculative alternative reading; in Limitations it independently corroborates
our "conditional on answering" caveat with a named model family.

**5. `[^dasman]` Li, Li & Qiu (2025), arXiv:2507.02919.**
Shows that silicon samples from GPT-4 and Llama on ANES political items suffer severe
homogenization that suppresses minority viewpoints, plus structural inconsistency across
demographic aggregation levels.
*Engagement:* Strengthens the third reading in §5 — if LLMs systematically collapse toward
the modal response, the homogenisation we observe on the IW map is partly a property of
survey-style elicitation itself, which is precisely why our headline statistics are
classifier-free and why we report per-item response entropy and distance from the
all-midpoint vector.

**6. `[^personareliab]` Taday Morocho, Cima, Fagni, Avvenuti & Cresci (2026), arXiv:2602.18462.**
Evaluates 70,000+ respondent-item instances from the WVS and finds persona prompting yields
no clear aggregate improvement in survey alignment and often degrades it, redistributing
error in ways that damage subgroup fidelity.
*Engagement:* Justifies our design choice not to persona-condition — cite alongside
`[^personas]` to frame our system-prompt-variant design as a deliberate alternative to
demographic conditioning, whose validity this paper puts in question.

### (d) Refusal on sensitive items

**7. `[^silenced]` Himelstein, LeVi, Youngmann, Nemcovsky & Mendelson (2025), arXiv:2511.03369 — AAAI 2026 (Oral), AI Alignment track.**
Introduces "silenced biases": unfair preferences that remain in a model's internal
representations while safety training masks them at the output layer, recoverable via
activation steering that suppresses refusal.
*Engagement:* Sharpens our first Limitation from a sampling caveat into a mechanism claim —
refusal on the homosexuality and abortion items is not missing data but *masked* data, so a
model excluded for refusing may carry exactly the survival-pole values the refusal conceals;
this is the strongest available argument for our decision to release refusals rather than
silently drop them.

### Also worth a line (lower priority)

- **Agarwal, Shukla, Sitaram & Vashistha (2025-26), arXiv:2505.21548, *Fluent but Foreign: Even Regional LLMs Lack Cultural Alignment*** — six Indic and six global LLMs; regionally fine-tuned models align no better with Indian norms than global ones, and US respondents proxy Indian values better than any India-focused model. Direct non-Chinese replication of our "developer origin does not predict cultural position" result; one sentence in §2 or §5.
- **Braun (2025), arXiv:2509.08480, *Acquiescence Bias in Large Language Models*, EMNLP 2025 Findings** — 37,975 question variations across English, German and Polish; LLMs bias toward answering "no" regardless of what agreement means, i.e. the opposite of the human pattern, and the effect is language-dependent. A second, language-crossed response-style confound worth acknowledging in §5 alongside modal responding.

---

## 3. arXiv IDs that do not exist or resolve differently

**None of the seven cited arXiv IDs is nonexistent.** All resolve to real papers.
The problem is titles, not identifiers.

| ID | Resolves? | Matches the draft's description? |
|---|---|---|
| 2411.06032 | ✅ | Title yes; **§2 characterization of the finding: no** (see §4) |
| 2410.10489 | ✅ | **No — different paper subject entirely.** Draft title invented; draft's "norm-conflict scenarios" description is wrong on the facts |
| 2606.11399 | ✅ | Substantively yes; title paraphrased, "— Extended Version" suffix dropped |
| 2605.26365 | ✅ | Substantively yes; title paraphrased |
| 2601.22396 | ✅ | Substantively yes; title paraphrased |
| 2505.18562 | ✅ | Substantively yes; **title invented**; missing EMNLP 2025 venue |
| 2601.12091 | ✅ | Substantively yes; **title invented** ("pluralistic alignment" is not in the paper) |
| 2311.14096 | ✅ | Yes — the `[^tao2024]` preprint, correctly the same paper |

**Duplicate-line caveat.** `[^scenario]` (2606.11399) and `[^steering]` (2605.26365) share
authors (Dang & Masud) and are the same research programme — 2606.11399 is described as an
"Extended Version" and both report the same latent-entanglement result. Listing them in §2
as two separate entries in a "growing elicitation toolkit" inflates the apparent breadth of
that literature and a reviewer familiar with either will notice. Recommend citing them as a
single item: `[^scenario] [^steering]` together with "Dang and colleagues".

---

## 4. Prose corrections required in §2 (not just footnotes)

Two sentences in Related Work state findings the cited papers do not report. Both are
reviewer-visible.

**(a) The Kazemi sentence is wrong on the paper's subject.**

> Draft: "Kazemi et al. [^kazemi] probe values through generated norm-conflict scenarios"

The paper does nothing of the kind. It tests 21 country-language pairs with ~100 verified
WVS questions each and shows that **the availability of online language resources predicts
how faithfully a model represents a country's values** — 44% of variance in GPT-4o, rising
to 72% in GPT-4-turbo, with error rates over five times higher for lowest-resource languages.

This is not a minor fix: it is arguably the **single most supportive prior result for this
paper's new central finding**, and the draft currently wastes it. Suggested replacement:

> Kazemi et al. [^kazemi] — the Columbia study that cited our 2024 analysis — show that the
> volume of online resources in a language predicts how faithfully a model reproduces that
> language community's WVS values, explaining 44-72% of variance across 21 country-language
> pairs with error rates over five times higher in the lowest-resource tier. Their result
> supplies a mechanism for ours: if value fidelity tracks the training corpus available in
> the language of administration, then the language of the prompt should move a model on the
> map, which is what we measure.

**(b) The LLM-GLOBE sentence reverses the paper's result.**

> Draft: "LLM-GLOBE [^llmglobe] ... finding smaller East-West differences than expected —
> consistent with our homogenisation result by an independent instrument."

The paper reports **statistically significant differences between Chinese and US models on
7 of 9 GLOBE dimensions** in the open-generation setting; only power distance and
uncertainty avoidance converge. "Smaller than expected" is not supportable.

What the paper *does* support is a different and still-useful point: *both* groups diverge
significantly from their own human ground truths ("for both US and Chinese models,
significant differences existed between the model values and human ground truths"), and
assertiveness was the only dimension where human and model ratings aligned. Suggested
replacement:

> LLM-GLOBE [^llmglobe] compares Chinese and US models on the GLOBE dimensions and finds
> significant East-West differences on seven of nine — but also that *both* groups diverge
> significantly from the human ground truth of their own development context. That second
> finding, not the first, is the one our result echoes by an independent instrument:
> models are displaced from human value profiles regardless of origin.

**(c) Optional but advisable.** §2 currently contains the claim "No prior IW-map evaluation
of LLMs reports uncertainty at all" (in Contributions, §1). Given `[^scenario]` and
`[^steering]` both work on the Inglehart-Welzel axes, and `[^personas]` evaluates personas
against the IW map explicitly, this claim needs to be checked against those three papers'
statistics sections before submission, or softened to "no prior IW-map evaluation of LLMs
reports confidence regions for model positions."

---

## 5. Checklist

- [ ] Replace footnote block (§1 above) — 5 wrong titles, 4 missing author lists
- [ ] Rewrite the Kazemi sentence in §2 — currently describes a different paper
- [ ] Rewrite the LLM-GLOBE sentence in §2 — currently reverses the finding
- [ ] Merge `[^scenario]` and `[^steering]` into one attribution (same authors, same programme)
- [ ] Add EMNLP 2025 venue to `[^wordassoc]`; ACL 2026 SRW to `[^steering]`; AAAI 2026 to `[^silenced]`
- [ ] Confirm WVS trend file version actually used (DOI now resolves to 4.1.0, draft says 4.0.0)
- [ ] Add EVS subtitle ": Integrated Dataset (EVS 1981-2017)"
- [ ] Add `[^promptlang]` and `[^deepseekweird]` to §2 — required for the language finding
- [ ] Add `[^promptprog]` and re-pitch the open-weight novelty claim
- [ ] Add `[^perturb]`, `[^dasman]` to §5 (modal responding); `[^perturb]`, `[^silenced]` to Limitations
- [ ] Verify the "no prior IW-map evaluation reports uncertainty" claim against arXiv:2606.11399, 2605.26365, 2601.22396
