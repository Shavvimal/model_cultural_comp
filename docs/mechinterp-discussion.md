# Mechanistic interpretability in the Discussion — verified passage, footnotes, and guardrails

Support material for the ORACLE @ EMNLP 2026 submission
(`~/Code/next-mdx-blog/content/blog/cultural-bias-2026.mdx`).
Every citation below was checked against the arXiv abstract page, ACL Anthology entry,
PMLR proceedings page, or publisher landing page on **2026-08-04**. Nothing here is
paraphrased from memory; titles are exact.

Intended placement: immediately after the **"Defaults, not destiny — and not ephemera
either"** paragraph in §5 Discussion, before the closing "What the result establishes"
paragraph. It picks up the two objections that paragraph brackets and answers the
second one mechanistically rather than only statistically.

---

## 1. Ready-to-paste MDX passage

```mdx
**Below the prompt: what interpretability would add.** Both objections are, in the end,
claims about mechanism, and mechanism is now partially observable. Sparse-dictionary
decompositions of production-scale models recover interpretable features in bulk —
Anthropic's Claude 3 Sonnet decomposition scales to 34 million features and reports
features for bias, deception and sycophancy among them, several of which change the
model's behaviour when clamped [^scalingmono] — and the cross-layer successor,
attribution graphs, traces which of those features a model actually uses on a given
prompt [^biology]. Value-laden behaviour is beginning to yield the same treatment:
culture-general and culture-specific neurons comprising under 1% of a model's units,
whose ablation costs up to 30% on cultural benchmarks while general language
understanding is largely unaffected [^cultureneurons]; intrinsic and prompt-induced value
expression running through partly shared, partly distinct components that generalise
across languages [^dualvalue]; cultural steering vectors constructed from sparse features
rather than from prompts [^cue]; and a cultural-customization direction conserved across
non-English languages, which surfaces localized knowledge a model already holds but does
not volunteer [^culturalvector]. None of this makes a measured default less real; it makes
it *addressable*. If a default is carried by identifiable internal structure, then "what
values does this deployment express, and can they be adjusted?" becomes a question about
the model rather than only about the prompt that happened to be used — and the automated
pipelines already built for extracting and monitoring trait directions [^personavec] are a
plausible template for what continuous auditing of a cultural default would look like.

We are deliberate about how far this licenses the claim. Identification is demonstrated;
reliable control is not. The closest work to ours reports *latent entanglement* —
intervening on one cultural dimension drags others with it, because the dimensions are
encoded as coupled structures [^steering] — and in a controlled head-to-head,
sparse-autoencoder steering is outperformed by plain prompting and by finetuning
[^axbench]. The decomposition itself has documented failure modes that are not tuning
artefacts: feature absorption, where a parent feature silently stops firing because a more
specific child has absorbed it, and which varying dictionary size or sparsity does not fix
[^absorption]; and reconstruction error of which about half, and over 90% of its norm, is
linearly predictable from the input activation — dense structure the sparse basis cannot
represent [^darkmatter]. The field's own twenty-nine-author stocktake describes mechanistic
interpretability as *promising* assurance over model behaviour rather than yet providing it
[^openproblems]. The defensible position is therefore narrow: interpretability is a route
towards auditing and adjusting cultural defaults below the prompt, not a means of
certifying them. It is also, for our specific finding, a source of testable hypotheses —
multilingual transformers encode output language and conceptual content separably, at
different depths and patchable independently [^tongue] [^langsteer], while production-scale
circuit tracing finds a largely language-independent conceptual core with language-specific
input and output stages [^biology], which makes our 1.77-unit language displacement a
well-posed mechanistic question: whether Chinese administration routes the same value
features through a different output stage, or engages different value features altogether.
Deciding it means running the instrument and the circuit tracer on the same open weights —
the natural next step for this line of work, and one that open tooling now supports.
```

### If space is tight (8-page limit)

Cut the second paragraph's final sentence-pair (from "It is also, for our specific
finding…") into Future Work and drop `[^tongue]`, `[^langsteer]`, `[^cue]`,
`[^personavec]`. The load-bearing minimum is: `[^scalingmono]` + `[^biology]`
(identification), `[^cultureneurons]` or `[^culturalvector]` (value-specific
identification), `[^steering]` + `[^axbench]` (control is demonstrated but unreliable),
`[^openproblems]` (the field's own framing). That is six footnotes and roughly eight
sentences.

---

## 2. New footnote entries (paste at the end of the footnote block)

```mdx
[^scalingmono]: Templeton, A., Conerly, T., Marcus, J., Lindsey, J., Bricken, T., Chen, B., Pearce, A., Citro, C., Ameisen, E., Jones, A., Cunningham, H., Turner, N. L., McDougall, C., MacDiarmid, M., Tamkin, A., Durmus, E., Hume, T., Mosconi, F., Freeman, C. D., Sumers, T. R., Rees, E., Batson, J., Jermyn, A., Carter, S., Olah, C., & Henighan, T. (2024). [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/). *Transformer Circuits Thread*. Archived as arXiv:2605.29358.

[^biology]: Lindsey, J., Gurnee, W., Ameisen, E., Chen, B., Pearce, A., Turner, N. L., Citro, C., Abrahams, D., Carter, S., Hosmer, B., Marcus, J., Sklar, M., Templeton, A., Bricken, T., McDougall, C., Cunningham, H., Henighan, T., Jermyn, A., Jones, A., Persic, A., Qi, Z., Thompson, T. B., Zimmerman, S., Rivoire, K., Conerly, T., Olah, C., & Batson, J. (2025). [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html). *Transformer Circuits Thread*. Companion methods paper: Ameisen, E., Lindsey, J., Pearce, A., Gurnee, W., et al. (2025). [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html).

[^personavec]: Chen, R., Arditi, A., Sleight, H., Evans, O., & Lindsey, J. (2025). [Persona Vectors: Monitoring and Controlling Character Traits in Language Models](https://arxiv.org/abs/2507.21509). arXiv:2507.21509.

[^cultureneurons]: Yamamoto, T., Kumon, R., Bollegala, D., & Yanaka, H. (2026). [Neuron-Level Analysis of Cultural Understanding in Large Language Models](https://arxiv.org/abs/2510.08284). ICLR 2026. arXiv:2510.08284.

[^dualvalue]: Han, J., Lim, J., Kong, I., & Jo, Y. (2026). [Dual Mechanisms of Value Expression: Intrinsic vs. Prompted Values in Large Language Models](https://arxiv.org/abs/2509.24319). ICML 2026. arXiv:2509.24319.

[^cue]: Khanuja, S., Liu, H., Zhang, S., Lambert, J., Chen, M., Mathews, R., & Wang, L. (2026). [Steering LLMs for Culturally Localized Generation](https://arxiv.org/abs/2603.23301). arXiv:2603.23301.

[^culturalvector]: Veselovsky, V., Argın, B., Stroebl, B., Wendler, C., West, R., Evans, J., Griffiths, T. L., & Narayanan, A. (2026). [Localized Cultural Knowledge is Conserved and Controllable in Large Language Models](https://aclanthology.org/2026.findings-acl.2141/). In *Findings of the Association for Computational Linguistics: ACL 2026*, 43152-43178. Preprint: arXiv:2504.10191.

[^axbench]: Wu, Z., Arora, A., Geiger, A., Wang, Z., Huang, J., Jurafsky, D., Manning, C. D., & Potts, C. (2025). [AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders](https://proceedings.mlr.press/v267/wu25a.html). In *Proceedings of the 42nd International Conference on Machine Learning*, PMLR 267:67035-67080. arXiv:2501.17148.

[^absorption]: Chanin, D., Wilken-Smith, J., Dulka, T., Bhatnagar, H., Golechha, S., & Bloom, J. (2025). [A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders](https://arxiv.org/abs/2409.14507). NeurIPS 2025 (Oral). arXiv:2409.14507.

[^darkmatter]: Engels, J., Riggs, L., & Tegmark, M. (2025). [Decomposing The Dark Matter of Sparse Autoencoders](https://arxiv.org/abs/2410.14670). *Transactions on Machine Learning Research*. arXiv:2410.14670.

[^openproblems]: Sharkey, L., Chughtai, B., Batson, J., Lindsey, J., Wu, J., Bushnaq, L., Goldowsky-Dill, N., Heimersheim, S., Ortega, A., Bloom, J., Biderman, S., Garriga-Alonso, A., Conmy, A., Nanda, N., Rumbelow, J., Wattenberg, M., Schoots, N., Miller, J., Michaud, E. J., Casper, S., Tegmark, M., Saunders, W., Bau, D., Todd, E., Geiger, A., Geva, M., Hoogland, J., Murfet, D., & McGrath, T. (2025). [Open Problems in Mechanistic Interpretability](https://arxiv.org/abs/2501.16496). *Transactions on Machine Learning Research*. arXiv:2501.16496.

[^tongue]: Dumas, C., Wendler, C., Veselovsky, V., Monea, G., & West, R. (2025). [Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers](https://arxiv.org/abs/2411.08745). arXiv:2411.08745.

[^langsteer]: Chou, C.-T., Liu, G., Sun, J., Blondin, C., Zhu, K., Sharma, V., & O'Brien, S. (2025). [Causal Language Control in Multilingual Transformers via Sparse Feature Steering](https://arxiv.org/abs/2507.13410). arXiv:2507.13410.
```

### Optional extras (verified, not used in the passage)

Hold these in reserve for reviewer response or an appendix:

```mdx
[^bindingheads]: Floro, A., & Benedetto, L. (2026). [Cultural Binding Heads in Language Models](https://arxiv.org/abs/2605.28543). arXiv:2605.28543.

[^saelang]: Andrylie, L. M., Rahmanisa, I., Ihsani, M. K., Wicaksono, A. F., Wibowo, H. A., & Aji, A. F. (2026). [Sparse Autoencoders Can Capture Language-Specific Concepts Across Diverse Languages](https://arxiv.org/abs/2507.11230). arXiv:2507.11230.
```

`[^bindingheads]` is a preprint (May 2026, revised July 2026) reporting 2-3 causally
implicated mid-layer attention heads per model for binding cultural items to identities
across eight models, with ablation reducing binding strength 9-23% and the mechanism
attributed to pre-training rather than instruction tuning. Attractive but unrefereed;
use only if a reviewer presses for evidence that the routing is attention-mediated.

---

## 3. Verification table

| Key | Claim it supports in the passage | Verified detail | Status |
| --- | --- | --- | --- |
| `[^scalingmono]` | Features identified at production scale, including bias-relevant ones; clamping changes behaviour | Abstract states "up to 34 million features"; explicitly names features for "deception, power-seeking, sycophancy, and bias"; feature clamping is the Golden Gate demonstration | Transformer Circuits Thread 2024; arXiv:2605.29358 (28 May 2026) |
| `[^biology]` | Attribution graphs on a production model; shared multilingual conceptual core | Claude 3.5 Haiku; ten case studies; the multilingual section reports a shared conceptual space, with shared circuitry increasing with scale (Haiku shares >2× the proportion of features between languages vs a smaller model) | Transformer Circuits Thread, Mar 2025 |
| `[^personavec]` | Automated pipeline for extracting/monitoring trait directions | Directions for evil, sycophancy, hallucination; monitoring + steering + finetuning-drift prediction; Anthropic Fellows Program | arXiv:2507.21509 (v3, 5 Sep 2025). **No venue claimed** — none listed |
| `[^cultureneurons]` | Culture-general/specific neurons, <1% of units, ablation costs ≤30% | Confirmed on abstract; shallow-to-middle MLP layers; general NLU largely unaffected | ICLR 2026; arXiv:2510.08284 (9 Oct 2025, rev. 29 Mar 2026) |
| `[^dualvalue]` | Intrinsic vs prompted value mechanisms, partly shared, generalising across languages | Value vectors (residual-stream directions) + value neurons (MLP); shared components generalise across languages; intrinsic pathway promotes diversity, prompted pathway strengthens instruction compliance | ICML 2026; arXiv:2509.24319 (29 Sep 2025) |
| `[^cue]` | SAE features → cultural steering vectors, better than prompting for long-tail concepts | Sparse autoencoders → Cultural Embeddings (CuE) → residual-stream intervention with controllable strength; elicits rarer long-tail cultural concepts than prompting | arXiv:2603.23301 (24 Mar 2026); Google DeepMind + CMU. **Preprint — no venue** |
| `[^culturalvector]` | Non-English cultural knowledge present but not surfaced; one conserved steering direction | "Explicit-implicit localization gap"; a cultural customization vector conserved across all non-English languages; vector steering retains diversity and reduces stereotypes relative to explicit prompting | Findings of ACL 2026, pp. 43152-43178; anthology ID `2026.findings-acl.2141`; preprint arXiv:2504.10191 |
| `[^axbench]` | SAE steering loses to prompting and finetuning head-to-head | Gemma-2-2B and -9B; "prompting outperforms all existing methods, followed by finetuning"; "SAEs are not competitive" for steering | ICML 2025 (spotlight), PMLR 267:67035-67080; arXiv:2501.17148 |
| `[^absorption]` | Absorption is structural, not tunable | Parent features fail to fire when absorbed by child features; caused by sparsity optimisation over hierarchies; varying SAE size/sparsity does not fix it | NeurIPS 2025 Oral; arXiv:2409.14507 (22 Sep 2024) |
| `[^darkmatter]` | Roughly half the SAE error, >90% of its norm, linearly predictable | Confirmed verbatim on the abstract | TMLR; arXiv:2410.14670 (v1 18 Oct 2024; v2 25 Mar 2025) |
| `[^openproblems]` | Field self-describes as unsolved; assurance is a promise | 29 authors (Sharkey, Chughtai, Batson, Lindsey, Wu, …, McGrath); "promises to provide greater assurance over AI system behavior"; three buckets of open problems | TMLR, accepted 20 Sep 2025; arXiv:2501.16496 (27 Jan 2025) |
| `[^tongue]` | Language and concept separable, at different depths | Output language encoded at an earlier layer than the concept; language and concept independently patchable; mean-across-languages concept patching preserves or improves performance | arXiv:2411.08745 (v1 13 Nov 2024; v4 25 Jun 2025). Earlier ICML 2024 mech-interp workshop version titled "How Do Llamas Process Multilingual Text?" |
| `[^langsteer]` | A single language feature causally controls output language | Pretrained SAEs on Gemma-2B/9B residual streams; one feature at one layer gives up to 90% language-shift success (FastText), semantics preserved (LaBSE); en → zh/ja/es/fr; effect peaks in mid-to-late layers | arXiv:2507.13410 (17 Jul 2025, rev. 15 Oct 2025). **Preprint — no venue** |

Already in the paper, reused here unchanged: `[^steering]` (Dang & Masud, arXiv:2605.26365,
ACL 2026 SRW non-archival). Re-verified: the abstract itself reports "a consistent
phenomenon of latent entanglement, where interventions along one cultural dimension induce
shifts along another" and that cultural values are "encoded as coupled structures, limiting
precise alignment" — so the paper's own cited steering work supplies the side-effect caveat.
That is worth saying explicitly to a reviewer: the caveat is not imported from a hostile
source, it is the steering paper's own finding.

---

## 4. Do not claim

Statements the verified literature does **not** support. Keep these out of the paper.

1. **"Mechanistic interpretability can certify / prove a model is culturally unbiased."**
   `[^openproblems]` explicitly frames assurance as a promise. The strongest available
   verb is *audit*, *monitor*, or *provide evidence about* — never *certify*, *guarantee*,
   or *prove*.
2. **"There is a feature (or circuit) for survival/self-expression or traditional/secular
   values."** Nothing in the literature identifies features aligned to the
   Inglehart-Welzel axes, or to any values-survey factor structure. The verified work
   covers *cultural knowledge* neurons `[^cultureneurons]`, *culturally salient concept*
   features `[^cue]`, an *explicit-cultural-customization* direction `[^culturalvector]`,
   and *value-expression* components `[^dualvalue]` — related, but not the same object as
   an IW axis.
3. **"Interpretability gives a reliable control knob for cultural values."** `[^axbench]`
   is a head-to-head in which prompting beats SAE steering; `[^steering]` reports latent
   entanglement across cultural dimensions. Both must accompany any steering claim.
4. **"Steering along one cultural dimension leaves the others fixed."** Directly
   contradicted by `[^steering]`.
5. **"The 1.77-unit language displacement is explained by language-specific circuits."**
   No published work traces values-survey responses under different prompt languages to
   specific internal mechanisms. `[^tongue]`, `[^langsteer]` and `[^biology]` establish
   only that *language identity and conceptual content are separable in general*. Phrase
   as a hypothesis the literature makes well-posed and testable — never as an account.
6. **"Models represent culture language-independently, so the language effect must be an
   output-stage artefact."** `[^biology]` and `[^tongue]` concern conceptual
   representations generally, not value dispositions; `[^culturalvector]` points the other
   way, finding non-English cultural knowledge that English-default behaviour suppresses.
   The routing-vs-content question is open; do not resolve it in either direction.
7. **"Millions of features means we can read a model's values off its features."**
   `[^absorption]` and `[^darkmatter]` bound this: features silently fail to fire when
   absorbed, and roughly half the reconstruction error (over 90% of its norm) is dense
   structure the sparse basis cannot represent. Feature lists are neither complete nor
   guaranteed monosemantic.
8. **"Bigger or sparser dictionaries will fix the reliability problem."** `[^absorption]`
   states the opposite: varying SAE size or sparsity does not fix absorption.
9. **"Attribution-graph and persona-vector results have been replicated on the models we
   measure."** They have not. `[^biology]` and `[^personavec]` are Anthropic first-party
   work on Anthropic models; the open `circuit-tracer` release has been exercised on
   Gemma-2-2B, Llama-3.1 and Qwen3-4B — none of which is in either of our cohorts. If the
   passage implies portability, say "open tooling now supports this on open-weight
   models", not "this has been shown on models like ours".
10. **"Interpretability explains *why* models default to Western values."** No verified
    work adjudicates between the paper's own three readings (training-data gravity,
    alignment convergence, measurement artefact). Interpretability is offered here as a
    method for future adjudication, not as evidence for any of them.
11. **"Golden Gate Claude demonstrates that cultural steering works."** It was a 24-hour
    public demo (23 May 2024) of single-feature clamping, not a controlled result. Cite
    `[^scalingmono]` for the underlying finding; do not cite the demo in a paper.
12. **"Steering is free."** No verified source claims capability neutrality for cultural
    or persona steering. Do not assert that an intervention leaves general capability,
    fluency, or calibration intact.

---

## 5. One cross-check against the existing draft

The current "Defaults, not destiny" paragraph already contains the sentence *"interior
features corresponding to value-laden concepts are identifiable and causally manipulable
in current interpretability work"* — currently uncited. Attach `[^scalingmono]`
`[^cultureneurons]` `[^culturalvector]` to it, or delete the sentence and let the new
passage carry the claim; leaving it unsupported is the kind of thing an ORACLE reviewer
will flag given the paper's own auditability argument.
