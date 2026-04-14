"""
ADAPT-IQ: Adaptive Improvisation & Context-Shift Intelligence Quotient
Kaggle Benchmarks SDK Task Implementation

This task evaluates AI cognitive flexibility by presenting models with a
two-phase problem: an initial scenario followed by a disruptive context
injection. The model must demonstrate adaptive reasoning rather than
perseveration on its initial solution.

Track: Executive Functions (primary) + Learning (secondary)

Scoring:
    Adaptation Score      (50%) — success criteria regex matches
    Non-Perseveration     (30%) — absence of failure criteria
    Context Acknowledgment(20%) — explicit reference to new constraints

All scores are continuous floats in [0.0, 1.0]. No LLM-as-judge.
"""

import json
import re
import os
from pathlib import Path
from typing import Any

# ============================================================
# TASK METADATA
# ============================================================
TASK_NAME = "adapt-iq-cognitive-flexibility"
TASK_VERSION = "2.0.0"
TASK_DESCRIPTION = (
    "ADAPT-IQ measures cognitive flexibility and adaptive improvisation in AI systems. "
    "Each task presents a complex scenario, then injects disruptive new information that "
    "invalidates or significantly complicates the initial approach. The model must adapt "
    "its reasoning rather than perseverate on the original solution. This benchmark "
    "isolates the Executive Function of cognitive flexibility and the Learning faculty "
    "of belief updating under new evidence. "
    "Dataset: 100 scenarios across 6 domains (Resource Management, Social Dynamics, "
    "Engineering & Design, Scientific Reasoning, Creative Problem Solving, "
    "Cross-Domain Adaptation) at 3 difficulty levels (easy, medium, hard)."
)

# ============================================================
# PROMPT TEMPLATES
# ============================================================

SYSTEM_PROMPT = (
    "You are a highly capable problem-solving assistant. You will be presented with "
    "complex, real-world scenarios that require careful analysis and creative problem-solving.\n\n"
    "Your responses should be:\n"
    "1. Comprehensive and actionable\n"
    "2. Directly responsive to ALL information provided\n"
    "3. Adaptive when new constraints or context are introduced\n"
    "4. Specific rather than generic\n\n"
    "When new information is provided that changes the situation, you MUST update your "
    "approach accordingly. Ignoring or minimizing new constraints is not acceptable."
)

PHASE_1_TEMPLATE = (
    "## Scenario\n\n"
    "{initial_prompt}\n\n"
    "Please provide a detailed, actionable plan or response to this scenario. "
    "Be specific and comprehensive."
)

PHASE_2_TEMPLATE = (
    "## IMPORTANT UPDATE\n\n"
    "{disruptive_context}\n\n"
    "Given this critical new information, you must revise your previous response. "
    "Your updated plan must:\n"
    "1. Directly address the new information provided above\n"
    "2. Explain specifically what changes from your previous approach\n"
    "3. Provide a complete, revised plan that works within all new constraints\n\n"
    "Please provide your revised, comprehensive response now."
)


def build_conversation(scenario: dict) -> list[dict]:
    """
    Build the Phase 1 conversation messages for a scenario.

    Returns a list of message dicts (system + user) ready to send
    to any OpenAI-compatible chat API. The model's Phase 1 response
    must be appended before calling build_phase2_message().
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": PHASE_1_TEMPLATE.format(
            initial_prompt=scenario["initial_prompt"]
        )},
    ]


def build_phase2_message(scenario: dict) -> dict:
    """
    Build the Phase 2 disruptive context message.

    This message is appended to the conversation after the model's
    Phase 1 response, triggering the adaptive reasoning challenge.
    """
    return {
        "role": "user",
        "content": PHASE_2_TEMPLATE.format(
            disruptive_context=scenario["disruptive_context"]
        ),
    }


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def check_adaptation_score(phase2_response: str, scenario: dict) -> dict:
    """
    Evaluate whether the model's Phase 2 response successfully adapts
    to the injected context.

    Scoring logic:
    - success_criteria: pipe-separated regex alternatives; at least 2 of 3
      must match for adaptation_passed = True.
    - failure_criteria: simple substring checks; any match means the model
      perseverated on its original solution.

    Returns a dict with:
        adaptation_score       float  — proportion of success criteria met (0.0–1.0)
        non_perseveration_score float — 1.0 if no failure criteria matched, else 0.0
        adaptation_passed      bool
        non_perseveration_passed bool
        criteria_matched       list[str]
        failure_criteria_matched list[str]
        success_criteria_count int
        total_success_criteria int
    """
    response_lower = phase2_response.lower()

    # ── Success criteria ────────────────────────────────────────────────────
    # Each criterion may be a pipe-separated list of alternatives (OR logic).
    success_matches = []
    for criterion in scenario.get("success_criteria", []):
        alternatives = [a.strip() for a in criterion.split("|")]
        if any(alt.lower() in response_lower for alt in alternatives):
            success_matches.append(criterion)

    total_success = len(scenario.get("success_criteria", []))
    required_matches = max(2, (total_success + 1) // 2)  # majority, min 2
    adaptation_passed = len(success_matches) >= required_matches

    # Continuous score: fraction of success criteria met (capped at 1.0)
    adaptation_score = min(1.0, len(success_matches) / required_matches) if required_matches > 0 else 0.0

    # ── Failure criteria ────────────────────────────────────────────────────
    # Any match indicates the model perseverated on its Phase 1 answer.
    failure_matches = [
        c for c in scenario.get("failure_criteria", [])
        if c.lower() in response_lower
    ]
    non_perseveration_passed = len(failure_matches) == 0
    non_perseveration_score = 1.0 if non_perseveration_passed else 0.0

    return {
        "adaptation_score":           round(adaptation_score, 4),
        "non_perseveration_score":    non_perseveration_score,
        "adaptation_passed":          adaptation_passed,
        "non_perseveration_passed":   non_perseveration_passed,
        "criteria_matched":           success_matches,
        "failure_criteria_matched":   failure_matches,
        "success_criteria_count":     len(success_matches),
        "total_success_criteria":     total_success,
    }


def check_context_acknowledgment(phase2_response: str, scenario: dict) -> float:
    """
    Evaluate whether the model explicitly referenced the disruptive context
    rather than issuing a generic "I will update my plan" response.

    Strategy:
    1. Extract all numbers/percentages from the disruptive context.
    2. Extract key domain nouns (capitalised words, length ≥ 4).
    3. Require at least half of these tokens to appear in the Phase 2 response.

    Returns a float: 1.0 (acknowledged) or 0.0 (ignored).
    """
    disruptive_text = scenario.get("disruptive_context", "")
    response_lower  = phase2_response.lower()

    # Numbers and percentages (e.g. "60%", "500", "3.5")
    numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', disruptive_text)

    # Capitalised domain nouns (likely key constraint terms)
    key_nouns = re.findall(r'\b[A-Z][a-z]{3,}\b', disruptive_text)[:5]

    # Long lowercase content words (filter common stop-words)
    STOP = {
        "information", "reveals", "however", "because", "which", "their",
        "there", "about", "these", "those", "where", "while", "additional",
        "shows", "indicates", "update", "important", "critical", "please",
    }
    content_words = [
        w for w in re.findall(r'\b[a-z]{6,}\b', disruptive_text.lower())
        if w not in STOP
    ][:5]

    tokens = numbers + [n.lower() for n in key_nouns] + content_words
    if not tokens:
        return 1.0  # No tokens to check — give benefit of the doubt

    found = sum(1 for t in tokens if t.lower() in response_lower)
    threshold = max(1, len(tokens) // 2)
    return 1.0 if found >= threshold else 0.0


def compute_composite_score(
    adaptation_result: dict,
    context_score: float,
) -> float:
    """
    Compute the ADAPT-IQ composite score (0.0 – 1.0).

    Weights:
        Adaptation (success criteria met):  50%
        Non-perseveration (no failure):     30%
        Context acknowledgment:             20%

    Note: The length penalty has been removed in v2.0.0. Short responses
    that fully satisfy the criteria should not be penalised — concise,
    accurate adaptation is the goal.
    """
    composite = (
        0.50 * adaptation_result["adaptation_score"] +
        0.30 * adaptation_result["non_perseveration_score"] +
        0.20 * context_score
    )
    return round(composite, 4)


# ============================================================
# MAIN EVALUATION FUNCTION
# ============================================================

def evaluate_response(
    scenario: dict,
    phase1_response: str,
    phase2_response: str,
) -> dict:
    """
    Full evaluation of a model's responses to a single ADAPT-IQ scenario.

    Args:
        scenario        : The scenario dict from adapt_iq_dataset.json
        phase1_response : The model's response to the initial prompt
        phase2_response : The model's response after the disruptive context injection

    Returns:
        A result dict containing the composite score, all sub-scores, and
        diagnostic metadata for analysis.
    """
    adaptation_result = check_adaptation_score(phase2_response, scenario)
    context_score     = check_context_acknowledgment(phase2_response, scenario)
    composite_score   = compute_composite_score(adaptation_result, context_score)

    return {
        # Identity
        "scenario_id":                scenario["scenario_id"],
        "domain":                     scenario["domain"],
        "difficulty":                 scenario["difficulty"],
        # Scores
        "composite_score":            composite_score,
        "adaptation_score":           adaptation_result["adaptation_score"],
        "non_perseveration_score":    adaptation_result["non_perseveration_score"],
        "context_acknowledgment":     context_score,
        # Boolean pass/fail for quick filtering
        "adaptation_passed":          adaptation_result["adaptation_passed"],
        "non_perseveration_passed":   adaptation_result["non_perseveration_passed"],
        "context_acknowledged":       context_score == 1.0,
        # Diagnostic detail
        "success_criteria_matched":   adaptation_result["success_criteria_count"],
        "total_success_criteria":     adaptation_result["total_success_criteria"],
        "failure_criteria_triggered": len(adaptation_result["failure_criteria_matched"]),
        "failure_criteria_list":      adaptation_result["failure_criteria_matched"],
        "phase2_response_length":     len(phase2_response),
    }


# ============================================================
# BENCHMARK RUNNER (for local testing and CLI use)
# ============================================================

def run_benchmark_on_model(
    model_fn,
    dataset_path: str,
    output_path: str = None,
) -> dict:
    """
    Run the full ADAPT-IQ benchmark on a model.

    Args:
        model_fn      : Callable that takes a list of message dicts and returns a str
        dataset_path  : Path to adapt_iq_dataset.json
        output_path   : Optional path to save results JSON

    Returns:
        Aggregated benchmark results dict with per-domain and per-difficulty breakdowns.
    """
    with open(dataset_path) as f:
        scenarios = json.load(f)

    results = []

    for i, scenario in enumerate(scenarios):
        print(f"  [{i+1:3d}/{len(scenarios)}] {scenario['scenario_id']} "
              f"({scenario['domain']}, {scenario['difficulty']})", end="", flush=True)

        # ── Phase 1 ──────────────────────────────────────────────────────────
        messages = build_conversation(scenario)
        phase1_response = model_fn(messages)

        # ── Phase 2 ──────────────────────────────────────────────────────────
        messages.append({"role": "assistant", "content": phase1_response})
        messages.append(build_phase2_message(scenario))
        phase2_response = model_fn(messages)

        # ── Score ─────────────────────────────────────────────────────────────
        result = evaluate_response(scenario, phase1_response, phase2_response)
        results.append(result)
        print(f"  score={result['composite_score']:.3f}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n = len(results)
    avg_composite         = round(sum(r["composite_score"] for r in results) / n, 4)
    adaptation_rate       = round(sum(1 for r in results if r["adaptation_passed"]) / n, 4)
    non_perseveration_rate= round(sum(1 for r in results if r["non_perseveration_passed"]) / n, 4)
    context_ack_rate      = round(sum(1 for r in results if r["context_acknowledged"]) / n, 4)

    # Domain breakdown
    domain_scores = {}
    for r in results:
        domain_scores.setdefault(r["domain"], []).append(r["composite_score"])
    domain_averages = {d: round(sum(s)/len(s), 4) for d, s in domain_scores.items()}

    # Difficulty breakdown
    difficulty_scores = {}
    for r in results:
        difficulty_scores.setdefault(r["difficulty"], []).append(r["composite_score"])
    difficulty_averages = {d: round(sum(s)/len(s), 4) for d, s in difficulty_scores.items()}

    aggregated = {
        "benchmark":              TASK_NAME,
        "version":                TASK_VERSION,
        "total_scenarios":        n,
        "average_composite_score": avg_composite,
        "adaptation_rate":        adaptation_rate,
        "non_perseveration_rate": non_perseveration_rate,
        "context_acknowledgment_rate": context_ack_rate,
        "domain_averages":        domain_averages,
        "difficulty_averages":    difficulty_averages,
        "individual_results":     results,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(aggregated, f, indent=2)
        print(f"\nResults saved to {output_path}")

    print(f"\n{'='*50}")
    print(f"ADAPT-IQ Results Summary")
    print(f"  Average Composite Score: {avg_composite:.4f}")
    print(f"  Adaptation Rate:         {adaptation_rate:.4f}")
    print(f"  Non-Perseveration Rate:  {non_perseveration_rate:.4f}")
    print(f"  Context Ack Rate:        {context_ack_rate:.4f}")
    print(f"\nDomain Averages:")
    for domain, avg in sorted(domain_averages.items()):
        print(f"  {domain:<35} {avg:.4f}")
    print(f"\nDifficulty Averages:")
    for diff, avg in sorted(difficulty_averages.items()):
        print(f"  {diff:<10} {avg:.4f}")
    print(f"{'='*50}")

    return aggregated
