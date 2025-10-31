Title: Training Our Relevance Reward Model: How RMSearch Learns to Rank

![DPO evaluation accuracy](../../images/dpo.png)
![ADPO evaluation accuracy](../../images/adpo.jpeg)

## Summary (Overview)
Our reward model underpins every ranked list we ship. Classic direct preference optimization (DPO) gives the model one comparison per training example: a chosen sentence versus a rejected sentence. Advanced-batched DPO (ADPO) takes the same batch and evaluates many pairings, keeping one chosen key in place and rotating through five sampled negatives. That wider lens produces more gradients per step and keeps the chosen content from being duplicated throughout the dataset, reducing the risk of over-learning.

## Make DPO Dataset
We start with curated content—think internal how-to guides, support articles, and policy notes—and generate queries that mirror what customers ask. For each query we select two short passages: one that clearly answers the question (for example, “To enable single sign-on, configure the identity provider and sync roles”) and one that drifts away from the need (“Our platform integrates with multiple analytics partners”). These pairs become the building blocks of the DPO dataset, capturing crisp yes-or-no style judgments on relevance without any code-level overhead.

## Make Advanced Batching Dataset
ADPO reuses the same queries but groups information differently. Each record holds the query, one trusted positive passage, and five sampled negatives drawn from the corpus. Picture the positive as “Incident responders should check the runbook in RMSearch to triage outages,” while the negatives mention unrelated licensing policies or marketing metrics. During training the batch yields multiple comparisons between the single positive and each negative, giving us more signal from the same curated content.

## Training
With both datasets prepared, we adapt our base reward model using lightweight LoRA adapters. The standard DPO run consumes one comparison at a time, so improvements depend on covering a large variety of pairs. ADPO squeezes more information out of every batch because the loss is computed across the extra combinations; the optimizer sees several nuanced differences at once and stays anchored on what makes the positive truly relevant.

## Experiment
Evaluation on held-out queries shows that both approaches lift relevance scoring, but ADPO maintains a higher accuracy curve. The attached figures highlight the gap: DPO improves steadily before leveling, while ADPO keeps climbing thanks to the richer comparisons inside each batch. That translates directly to better ranking for customer queries, especially in long-tail knowledge areas where we cannot afford to overfit on a single phrasing.
