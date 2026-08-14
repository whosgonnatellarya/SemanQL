# gql-confidence-thing
a confidence scoring system for llm-generated graphql queries against shopify's admin api schema. (currently, will prob generalize after)

## the problem

shopify's sidekick agent generates graphql queries from natural language merchant requests. the problem is that it fails silently.

shopify's own engineering team documented this during fine-tuning: the model learned reward hacking behaviors, using `customer_tags CONTAINS 'enabled'` instead of the correct `customer_account_status = 'ENABLED'`. both queries are syntactically valid. both run without errors. both return results. only one answers the right question. the merchant has no way to know which one they got.

shopify fixed this at training time using grpo and llm judge consensus, pushing syntax validation accuracy from 93% to 99%. but training fixes aren't runtime fixes. errors still slip through, and a merchant acting on a silently wrong customer segment is making a real business decision on bad data.

## the research question

can you predict, automatically and cheaply, whether a given llm-generated graphql query is likely to be semantically wrong, without re-executing or manually reverifying every single one?

## approach

existing text-to-sql calibration research (2025) shows that llm self-reported confidence is badly miscalibrated, often above 0.9 even on wrong outputs. sub-clause frequency analysis, generating the same query multiple times and measuring consistency across samples, significantly outperforms naive self-reporting (auc ~0.78 vs ~0.2).

this project applies that technique to graphql query generation against shopify's admin api schema. no published calibration research has addressed graphql specifically. that's the gap.

## method

1. baseline: self-probing confidence, ask the model to rate its own output
2. layer 1: sub-clause frequency analysis across multiple sampled generations. catches queries the model is inconsistent about.
3. layer 2: schema validation against a hardcoded shopify admin api field/filter list. catches queries the model is consistently, confidently wrong about, the ones layer 1 misses because every sample agrees on the same wrong field.
4. combined score: `0.6 * subclause_min_confidence + 0.4 * schema_score`
5. evaluation: auc on a labeled dataset of correct vs semantically incorrect queries against shopify's real schema, comparing all four signals (self-probing, layer 1, layer 2, combined)

## results

**dataset:** 75 merchant questions against a hardcoded shopify admin api schema (customers + orders), all labeled: 39 correct (1), 36 wrong (0).

**labeling methodology:** strict binary labels — a query is only marked correct (1) if it's both semantically accurate *and* the ideal way to answer the question (e.g. `customer_account_status` over a `tags`-based workaround, current field names over deprecated ones, documented filter keys with correctly-cased enum values and quoted dates). anything that runs without error but answers the wrong question, or answers the right question the wrong way, is labeled wrong (0). labeling was done with an AI assistant proposing labels against shopify's documented filter-key/field/sortKey reference, spot-checked manually before being accepted.

**generation prompt bugs found during labeling:** going through all 75 entries against shopify's actual documented filter keys surfaced two systemic bugs in `generate_queries.py`'s system prompt, not just one-off bad generations:

1. the prompt told the model `amount_spent` was a valid customer filter key. it isn't — the correct key is `total_spent`. `amount_spent` is a *field* name (for displaying spend), not a *filter* key, and the prompt conflated the two. this alone caused 11 of the labeled-0 entries.
2. the prompt never mentioned that date values in filter strings must be quoted (`created_at:>'2024-01-01'`, not `created_at:>2024-01-01`). every date-range query in the dataset was generated unquoted as a result.

both are fixed in the system prompt now, along with two related corrections: `created_at` isn't a valid customer filter key either (correct: `customer_date`), and `state` values must be uppercase (`ENABLED`, not `enabled`). the 20 entries whose 0 label traced back to either bug were regenerated with the fixed prompt and relabeled. 8 of the 20 flipped to correct once the underlying key/quoting was fixed (customer total_spent range and threshold queries, customer_date range queries, and OR'd multi-country total_spent queries). the other 12 are still labeled 0 — but now for a *different*, not-yet-fixed reason: most of them ask about relative time windows ("last 30 days," "this year," "last 90 days"), and the model has no way to know the real current date, so it hardcodes a plausible-looking one that's disconnected from "now." that's a distinct bug from the one just fixed, still open.

**this is itself a finding worth stating plainly: a bug in the generation prompt doesn't just produce bad queries, it silently biases the evaluation dataset built from those queries.** every eval signal in this project (sub-clause frequency, self-probing, schema validation, combined) was being scored against 20 entries that were mislabeled 0 for a reason that had nothing to do with the merchant's question — the *generator* was wrong, not the underlying task. a confidence-scoring system evaluated on a dataset like that would look worse than it actually is at catching real semantic errors, and better than it actually is at catching prompt-templating bugs, without anyone realizing the difference. dataset quality audits need to check the generation pipeline, not just the labels.

**auc on all 75 labeled entries, with 95% confidence intervals from bootstrap resampling (n=1000):**

```
method                    auc    95% ci
sub_clause_frequency   0.644  [0.53, 0.76]
self_probing_baseline  0.776  [0.67, 0.88]
schema_validation      0.460  [0.34, 0.59]
combined               0.602  [0.47, 0.74]
```

**honest interpretation:** self-probing still leads at n=75, and the gap over sub-clause frequency is now clearer — the confidence intervals barely overlap (`[0.53, 0.76]` vs `[0.67, 0.88]`), unlike at n=25 where they overlapped almost completely. that's a real change in what the data supports, not just noise settling down: more data made the ranking *more* confident, not less, and the ranking still doesn't match the sub-clause frequency paper's headline result. schema validation is now measurably *worse* than random-ish (0.460, essentially no separation) rather than just weak — the hardcoded field/filter allowlist in `schema_validator.py` doesn't capture the actual failure modes in this dataset (most wrong queries here fail on filter *values* — wrong casing, wrong key semantics, hardcoded dates — not on unknown fields, which is mostly what layer 2 checks for). the combined score, still weighted 0.6/0.4 toward sub-clause frequency, is dragged down by both weaker components.

**what these numbers do and don't mean:** they're now a more trustworthy read on this specific pipeline (real schema, real labels, no known generation bugs left in the labeled data) than the n=25 snapshot was — but they still don't generalize past this project's simplified single-schema, hardcoded-allowlist setup. they don't mean sub-clause frequency is a bad technique in general (the paper's result was on text-to-sql at a much larger n); they mean it isn't winning on *this* dataset, and schema validation as implemented here isn't pulling its weight. the combined score's fixed 0.6/0.4 weighting has never been tuned against data — that's the next thing worth revisiting, now that the labels underneath it aren't confounded by prompt bugs.

**limitations:**
- still a small dataset (n=75) for auc — the confidence intervals above are narrower than at n=25 but still wide enough that a handful of different labels would move the ranking
- single schema: shopify admin api customers/orders only, via a hardcoded field/filter allowlist, not the full admin api
- labeling was AI-assisted and manually spot-checked rather than independently double-labeled, so labeler bias/error isn't measured
- self-probing baseline uses a single prompt template and one confidence elicitation per query; it isn't the strongest possible self-reporting baseline
- the "hardcoded relative date" bug (12 entries) is not yet fixed — the model has no grounding on the actual current date, so any question involving "last N days" or "this year" is likely to keep generating plausible-but-wrong absolute dates until the prompt is given a real reference date
- an unrelated, still-open quoting gap: multi-word filter *values* (e.g. `country:United States`) are sometimes left unquoted even now, which is the same category of bug as the date-quoting issue but wasn't in scope for this fix

## research (for now, we still getting there!)

- [calibrating llms for text-to-sql parsing by leveraging sub-clause frequencies](https://arxiv.org/abs/2505.23804) - main method, sub-clause frequency analysis
- [confidence scoring for llm-generated sql in supply chain data extraction](https://arxiv.org/abs/2506.17203) - baseline self-probing approach and why it fails

## sources

- [flow generation through natural language](https://shopify.engineering/fine-tuning-agent-shopify-flow) - reward hacking and the customer_tags example
- [teaching sidekick to say no](https://shopify.engineering/sidekick-curation) - llm judge consensus and refusal training
- [building production-ready agentic systems](https://shopify.engineering/building-production-ready-agentic-systems) - syntax accuracy numbers

## status

pipeline is wired end to end: generation, sub-clause parsing, consistency scoring, schema validation, combined scoring, and an eval harness with bootstrapped confidence intervals. all 75 questions are generated, self-probed, and labeled (see [results](#results)). two systemic bugs in the generation system prompt (wrong customer filter key, unquoted dates) were found and fixed; the still-open hardcoded-relative-date issue is the next thing worth fixing in `generate_queries.py`.
