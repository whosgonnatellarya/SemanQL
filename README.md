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

**dataset:** 75 merchant questions total against a hardcoded shopify admin api schema (customers + orders). 25 are labeled and evaluated so far; the other 50 were added later to broaden coverage (date ranges, multi-filter customer segmentation, sortKey-based sorting, deprecated-vs-current field name traps, ambiguous intent) and are pending manual labeling.

**labeling methodology:** strict binary labels — a query is only marked correct (1) if it's both semantically accurate *and* the ideal way to answer the question (e.g. `customer_account_status` over a `tags`-based workaround, current field names over deprecated ones). anything that runs without error but answers the wrong question, or answers the right question the wrong way, is labeled wrong (0). labeling was done with an AI assistant proposing labels, spot-checked manually against the schema before being accepted.

**auc on the 25 labeled entries, with 95% confidence intervals from bootstrap resampling (n=1000):**

```
method                    auc    95% ci
sub_clause_frequency   0.638  [0.47, 0.81]
self_probing_baseline  0.728  [0.49, 0.92]
schema_validation      0.526  [0.30, 0.73]
combined               0.580  [0.33, 0.81]
```

**honest interpretation:** at n=25, self-probing outperforms sub-clause frequency, which is the opposite of what the sub-clause frequency paper reports at scale. this isn't a refutation of the method — the confidence intervals above all overlap heavily (e.g. sub-clause frequency's `[0.47, 0.81]` and self-probing's `[0.49, 0.92]` overlap almost entirely), meaning the point estimates aren't statistically distinguishable yet. it's consistent with the paper's own finding that sub-clause frequency needs a larger sample to show its advantage over naive self-reported confidence. the 25-entry numbers are a snapshot, not a verdict.

**what these numbers do and don't mean:** they show the relative ranking of four signals on a small, single-schema, single-labeler dataset — useful as a directional check that the pipeline is wired correctly end to end (generation, sub-clause parsing, schema validation, self-probing, combined scoring, evaluation with real error bars). they do *not* yet mean sub-clause frequency is worse than self-reported confidence for this task, or that the combined score's current weighting (0.6 sub-clause / 0.4 schema) is well-tuned. both of those need the full 75-entry evaluation, and ideally more than one labeler, before drawing conclusions.

**limitations:**
- small dataset (n=25 labeled today, n=75 once the rest are labeled) — auc is noisy at this scale, as the wide confidence intervals above show
- single schema: shopify admin api customers/orders only, via a hardcoded field/filter allowlist, not the full admin api
- labeling was AI-assisted and manually spot-checked rather than independently double-labeled, so labeler bias/error isn't measured
- self-probing baseline uses a single prompt template and one confidence elicitation per query; it isn't the strongest possible self-reporting baseline

## research (for now, we still getting there!)

- [calibrating llms for text-to-sql parsing by leveraging sub-clause frequencies](https://arxiv.org/abs/2505.23804) - main method, sub-clause frequency analysis
- [confidence scoring for llm-generated sql in supply chain data extraction](https://arxiv.org/abs/2506.17203) - baseline self-probing approach and why it fails

## sources

- [flow generation through natural language](https://shopify.engineering/fine-tuning-agent-shopify-flow) - reward hacking and the customer_tags example
- [teaching sidekick to say no](https://shopify.engineering/sidekick-curation) - llm judge consensus and refusal training
- [building production-ready agentic systems](https://shopify.engineering/building-production-ready-agentic-systems) - syntax accuracy numbers

## status

pipeline is wired end to end: generation, sub-clause parsing, consistency scoring, schema validation, combined scoring, and an eval harness with bootstrapped confidence intervals. dataset is at 75 questions; 25 are labeled with real auc numbers (see [results](#results)), the other 50 are generated and self-probed but still need manual labeling via `label_dataset.py`.
