# ADAPT-IQ: Context-Injection Creativity Test (CICT)

**A novel benchmark measuring cognitive flexibility and "cognitive inertia" in frontier AI systems.**

Submitted to the [Google DeepMind x Kaggle "Measuring Progress Toward AGI" Hackathon](https://www.kaggle.com/competitions/kaggle-measuring-agi).

## Overview

Current AI evaluations often measure static knowledge retrieval or logical reasoning within a fixed context. However, true fluid intelligence requires the ability to adapt, improvise, and overcome when the environment changes or new, unexpected information is introduced. 

ADAPT-IQ introduces the **Context-Injection Creativity Test (CICT)**. By forcing a mid-task pivot, ADAPT-IQ exposes whether a model is truly reasoning fluidly or merely executing a pre-computed, crystallized pattern.

This benchmark directly targets the **Executive Functions** track, specifically evaluating cognitive flexibility and inhibitory control (the ability to suppress an initial solution path when it becomes invalid).

## The Benchmark Flow

ADAPT-IQ operates in a multi-turn conversational format across three phases:

1. **Phase 1 (Initial Problem):** The model is presented with a complex, open-ended problem across one of six domains and asked to generate a solution.
2. **Phase 2 (Context Injection):** The model is presented with a disruptive piece of new information (e.g., a critical material becomes unavailable, a budget is slashed, the domain shifts).
3. **Phase 3 (Adaptive Resolution):** The model must generate a revised solution that accommodates the new context.

## Evaluation Metrics

The Kaggle Benchmarks SDK implementation automates this multi-turn interaction. The evaluation logic uses deterministic structured parsing and regex assertions to compute a composite score based on three metrics:

*   **Adaptation Score (50%):** Does the final solution successfully incorporate the disruptive context? Evaluated by checking for the presence of specific `success_criteria` keywords.
*   **Non-Perseveration Score (30%):** Does the model successfully abandon its initial assumptions? Evaluated by asserting the absence of `failure_criteria` keywords. This measures **cognitive inertia**.
*   **Context Acknowledgment (20%):** Does the model explicitly acknowledge the new constraints rather than ignoring them?

## Repository Structure

*   `task.py`: The core Kaggle Benchmarks SDK task implementation containing the multi-turn logic and regex-based evaluation scoring.
*   `benchmark.py`: The Kaggle Benchmarks SDK runner definition.
*   `data/adapt_iq_dataset.json`: The 60 hand-crafted, synthetic scenarios across 6 domains (Resource Management, Social Dynamics, Engineering & Design, Scientific Reasoning, Creative Problem Solving, Cross-Domain Adaptation).
*   `adapt_iq_evaluation_notebook.py`: The evaluation script used to run the benchmark against frontier models via API.
*   `data/evaluation_results.json`: The raw output scores from our evaluation runs.

## Results & Insights

We evaluated ADAPT-IQ across frontier models (GPT-4.1-mini, GPT-4.1-nano, Gemini-2.5-Flash) and weaker baselines (GPT-3.5-turbo, GPT-2) to validate discriminatory power.

**Key Finding: Cognitive Inertia Exists.** Weaker models exhibit severe cognitive inertia. When presented with disruptive context, they often acknowledge the new information in their preamble but proceed to output their Phase 1 solution with minor cosmetic changes — failing the non-perseveration metric. The benchmark provides a clear performance gradient from 0.217 to 0.975, making it a strong discriminator across model capability tiers.

## Usage

To run this benchmark locally using the Kaggle Benchmarks SDK:

```bash
pip install kaggle-benchmarks
python benchmark.py
```

## License

CCO
