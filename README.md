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

## setup

```
git clone <this repo>
cd confidence-thing
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

## usage

run in this order:

1. `python dataset.py` — generates candidate queries for the predefined merchant questions and writes them to `dataset.jsonl` (label `null`). safe to re-run: it only generates questions not already in the file, and leaves existing entries untouched.
2. `python baseline.py` — adds a self-probing confidence score to any entry that doesn't have one yet.
3. `python label_dataset.py` — interactive CLI to hand-label each entry `1` (correct and ideal) or `0` (wrong or not ideal). only shows unlabeled entries by default; pass `--all` to review/relabel everything.
4. `python evaluate.py` — computes auc (with bootstrap confidence intervals) for all four signals on the labeled entries and writes `evaluation_results.json`.

for a single ad-hoc question instead of the batch dataset, use `python main.py` — prompts for one merchant question and prints a confidence report (sub-clause breakdown, schema violations, combined score) without touching `dataset.jsonl`.

## results

**dataset:** 75 merchant questions against a hardcoded shopify admin api schema (customers + orders), all labeled: 39 correct (1), 36 wrong (0).

**labeling methodology:** strict binary labels — a query is only marked correct (1) if it's both semantically accurate *and* the ideal way to answer the question (e.g. the `state` filter over a `tags`-based workaround, current field names over deprecated ones, documented filter keys with correctly-cased enum values and quoted dates). anything that runs without error but answers the wrong question, or answers the right question the wrong way, is labeled wrong (0). labeling was done with an AI assistant proposing labels against shopify's documented filter-key/field/sortKey reference, spot-checked manually before being accepted.

**generation prompt bugs found during labeling:** going through all 75 entries against shopify's actual documented filter keys surfaced two systemic bugs in `generate_queries.py`'s system prompt, not just one-off bad generations:

1. the prompt told the model `amount_spent` was a valid customer filter key. it isn't — the correct key is `total_spent`. `amount_spent` is a *field* name (for displaying spend), not a *filter* key, and the prompt conflated the two. this alone caused 11 of the labeled-0 entries.
2. the prompt never mentioned that date values in filter strings must be quoted (`created_at:>'2024-01-01'`, not `created_at:>2024-01-01`). every date-range query in the dataset was generated unquoted as a result.

both are fixed in the system prompt now, along with two related corrections: `created_at` isn't a valid customer filter key either (correct: `customer_date`), and `state` values must be uppercase (`ENABLED`, not `enabled`). the 20 entries whose 0 label traced back to either bug were regenerated with the fixed prompt and relabeled. 8 of the 20 flipped to correct once the underlying key/quoting was fixed (customer total_spent range and threshold queries, customer_date range queries, and OR'd multi-country total_spent queries). the other 12 are still labeled 0 — but now for a *different*, not-yet-fixed reason: most of them ask about relative time windows ("last 30 days," "this year," "last 90 days"), and the model has no way to know the real current date, so it hardcodes a plausible-looking one that's disconnected from "now." that's a distinct bug from the one just fixed, still open.

**this is itself a finding worth stating plainly: a bug in the generation prompt doesn't just produce bad queries, it silently biases the evaluation dataset built from those queries.** every eval signal in this project (sub-clause frequency, self-probing, schema validation, combined) was being scored against 20 entries that were mislabeled 0 for a reason that had nothing to do with the merchant's question — the *generator* was wrong, not the underlying task. a confidence-scoring system evaluated on a dataset like that would look worse than it actually is at catching real semantic errors, and better than it actually is at catching prompt-templating bugs, without anyone realizing the difference. dataset quality audits need to check the generation pipeline, not just the labels.

**validator bugs found in a full pipeline audit, fixed separately from the labels:** the labeling pass above fixed what the *generator* got wrong. a follow-up audit of every file in the pipeline found that `schema_validator.py` — layer 2, the signal meant to catch exactly this kind of wrongness — had never been updated to match, plus three unrelated bugs:

1. `VALID_FILTER_KEYS` still listed the old, wrong keys (`amount_spent`, `created_at`) and was missing the corrected ones (`total_spent`, `customer_date`) entirely, along with nearly every order-side filter key. this meant schema validation was scoring queries against the *pre-fix* schema — actively rewarding the bug the labeling pass had just fixed, and penalizing the fix. it's also now split into separate customer/order key sets (detected from whether the query root is `customers(` or `orders(`), since a key valid on one isn't valid on the other.
2. the validator only ever checked that a filter *key* was recognized — it never looked at the *value*. `state:enabled` (should be `state:ENABLED`) and `fulfillment_status:on_hold` (not a real value) both passed with no violation. added explicit value checks for `state`, `financial_status`, `fulfillment_status`, `status`, and unquoted dates on any date-typed key.
3. the clause splitter only recognized `AND`/`OR` as separators, so `NOT tag:test` was checked as filter key `"NOT tag"` (a false violation on a valid query), and `total_spent:>1000 NOT country:US` was treated as one clause and only the first key was ever checked (the `NOT`-ed condition was silently never validated at all). fixed to split on `NOT` too and strip `NOT`/`-` negation prefixes before checking the key.
4. separately, in `parse_subclause.py`: the regex that extracts the filter string stopped at the first `"` it saw, including *escaped* quotes — so any filter using a quoted multi-word value (`country:\"United States\"`, the pattern the date-quoting fix now encourages generally) got silently truncated before parsing. confirmed this was corrupting 2 of the 75 entries' filter strings before the fix.

none of these needed the dataset regenerated or relabeled — they were purely in how the *existing* generated queries got scored, not in what got generated. re-running `evaluate.py` with the fixes in place was enough.

**auc on all 75 labeled entries, with 95% confidence intervals from bootstrap resampling (n=1000):**

```
method                    auc    95% ci
sub_clause_frequency   0.644  [0.53, 0.76]
self_probing_baseline  0.776  [0.67, 0.88]
schema_validation      0.661  [0.57, 0.75]
combined               0.719  [0.60, 0.83]
```

**honest interpretation:** schema validation jumped from 0.460 (worse than a coin flip) to 0.661 — not because anything about the queries changed, but because the validator finally checks the schema it was always supposed to check. that 0.460 number in every earlier version of this README was never a real measurement of the technique; it was a measurement of a stale allowlist. combined moved from 0.602 to 0.719 for the same reason, and now has overlapping confidence intervals with self-probing (`[0.60, 0.83]` vs `[0.67, 0.88]`) rather than trailing it clearly — the two are no longer statistically distinguishable at this sample size. sub-clause frequency is unchanged (0.644) since none of this audit touched how sub-clause consistency is computed or stored. self-probing still has the highest point estimate, but the story is no longer "self-probing clearly beats both structural signals" — it's "self-probing, schema validation, and combined are now close enough that n=75 can't cleanly separate them," which is a meaningfully different conclusion than any earlier version of this README supported.

**what these numbers do and don't mean:** they're the first numbers in this project not known to be measuring a bug in the pipeline rather than the technique. that doesn't mean they're final — the combined score's 0.6/0.4 weighting still hasn't been tuned against data (worth revisiting now that both components are trustworthy), and the still-open issues below mean there's likely more measurement error left to find. it does mean the previous headline conclusion ("self-probing wins clearly") should be retired; the current data doesn't support declaring a winner.

**limitations:**
- still a small dataset (n=75) for auc — the confidence intervals above are narrower than at n=25 but still wide enough that a handful of different labels would move the ranking
- single schema: shopify admin api customers/orders only, via a hardcoded field/filter allowlist, not the full admin api
- labeling was AI-assisted and manually spot-checked rather than independently double-labeled, so labeler bias/error isn't measured
- self-probing baseline uses a single prompt template and one confidence elicitation per query; it isn't the strongest possible self-reporting baseline
- the "hardcoded relative date" generation bug (12 entries still labeled 0 for this) is not yet fixed — the model has no grounding on the actual current date, so "last N days"/"this year" questions keep generating plausible-but-wrong absolute dates
- general multi-word filter *values* (e.g. `country:United States`) can still be left unquoted by the generator outside of dates specifically — narrower than before (the parser now handles it once quoted, and dates are enforced), but the generator itself isn't required to quote non-date multi-word values yet

## research (for now, we still getting there!)

- [calibrating llms for text-to-sql parsing by leveraging sub-clause frequencies](https://arxiv.org/abs/2505.23804) - main method, sub-clause frequency analysis
- [confidence scoring for llm-generated sql in supply chain data extraction](https://arxiv.org/abs/2506.17203) - baseline self-probing approach and why it fails

## sources

- [flow generation through natural language](https://shopify.engineering/fine-tuning-agent-shopify-flow) - reward hacking and the customer_tags example
- [teaching sidekick to say no](https://shopify.engineering/sidekick-curation) - llm judge consensus and refusal training
- [building production-ready agentic systems](https://shopify.engineering/building-production-ready-agentic-systems) - syntax accuracy numbers

## status

pipeline is wired end to end: generation, sub-clause parsing, consistency scoring, schema validation, combined scoring, and an eval harness with bootstrapped confidence intervals. all 75 questions are generated, self-probed, and labeled (see [results](#results)). a full pipeline audit found and fixed bugs on both sides — the generation prompt (wrong customer filter key, unquoted dates) and the schema validator (stale/wrong filter keys, no value-level checks, broken `NOT` handling, an escaped-quote parsing bug) — plus decoupled `main.py` from `dataset.py`, corrected `requirements.txt`, and added `.env.example`/setup docs. the still-open hardcoded-relative-date generation issue is the next thing worth fixing in `generate_queries.py`.
