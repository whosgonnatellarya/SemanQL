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
2. main method: sub-clause frequency analysis across multiple sampled generations
3. evaluation: auc on a labeled dataset of correct vs semantically incorrect queries against shopify's real schema

## research (for now, we still getting there!)

- [calibrating llms for text-to-sql parsing by leveraging sub-clause frequencies](https://arxiv.org/abs/2505.23804) — main method, sub-clause frequency analysis
- [confidence scoring for llm-generated sql in supply chain data extraction](https://arxiv.org/abs/2506.17203) — baseline self-probing approach and why it fails

## sources

- [flow generation through natural language](https://shopify.engineering/fine-tuning-agent-shopify-flow) — reward hacking and the customer_tags example
- [teaching sidekick to say no](https://shopify.engineering/sidekick-curation) — llm judge consensus and refusal training
- [building production-ready agentic systems](https://shopify.engineering/building-production-ready-agentic-systems) — syntax accuracy numbers

## status

still working on it lolz
