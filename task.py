"""
ADAPT-IQ: Adaptive Improvisation & Context-Shift Intelligence Quotient
Kaggle Benchmarks SDK Task Implementation

This task evaluates AI cognitive flexibility by presenting models with a
two-phase problem: an initial scenario followed by a disruptive context
injection. The model must demonstrate adaptive reasoning rather than
perseveration on its initial solution.

Track: Executive Functions (primary) + Learning (secondary)
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
TASK_DESCRIPTION = (
    "ADAPT-IQ measures cognitive flexibility and adaptive improvisation in AI systems. "
    "Each task presents a complex scenario, then injects disruptive new information that "
    "invalidates or significantly complicates the initial approach. The model must adapt "
    "its reasoning rather than perseverate on the original solution. This benchmark "
    "isolates the Executive Function of cognitive flexibility and the Learning faculty "
    "of belief updating under new evidence."
)

# ============================================================
# PROMPT TEMPLATES
# ============================================================

SYSTEM_PROMPT = """You are a highly capable problem-solving assistant. You will be presented with complex, real-world scenarios that require careful analysis and creative problem-solving.

Your responses should be:
1. Comprehensive and actionable
2. Directly responsive to ALL information provided
3. Adaptive when new constraints or context are introduced
4. Specific rather than generic

When new information is provided that changes the situation, you MUST update your approach accordingly. Ignoring or minimizing new constraints is not acceptable."""

PHASE_1_TEMPLATE = """## Scenario

{initial_prompt}

Please provide a detailed, actionable plan or response to this scenario. Be specific and comprehensive."""

PHASE_2_TEMPLATE = """## IMPORTANT UPDATE

{disruptive_context}

Given this critical new information, you must revise your previous response. Your updated plan must:
1. Directly address the new information provided above
2. Explain specifically what changes from your previous approach
3. Provide a complete, revised plan that works within all new constraints

Please provide your revised, comprehensive response now."""


def build_conversation(scenario: dict) -> list[dict]:
    """Build the multi-turn conversation for a scenario."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PHASE_1_TEMPLATE.format(
            initial_prompt=scenario["initial_prompt"]
        )},
        # Note: The model's Phase 1 response will be inserted here by the evaluator
        # before Phase 2 is sent. This is handled by the evaluation loop.
    ]


def build_phase2_message(scenario: dict) -> dict:
    """Build the Phase 2 disruptive context message."""
    return {
        "role": "user",
        "content": PHASE_2_TEMPLATE.format(
            disruptive_context=scenario["disruptive_context"]
        )
    }


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def check_adaptation_score(response: str, scenario: dict) -> dict:
    """
    Check if the model's Phase 2 response successfully adapts to the new context.
    
    Returns a dict with:
    - adaptation_passed: bool
    - non_perseveration_passed: bool
    - criteria_matched: list of matched success criteria
    - failure_criteria_matched: list of matched failure criteria
    """
    response_lower = response.lower()
    
    # Check success criteria (at least 2 of 3 must match)
    success_matches = []
    for criterion in scenario["success_criteria"]:
        # Each criterion can be a pipe-separated list of alternatives
        alternatives = criterion.split("|")
        for alt in alternatives:
            if alt.lower() in response_lower:
                success_matches.append(criterion)
                break
    
    # Check failure criteria (none should match)
    failure_matches = []
    for criterion in scenario["failure_criteria"]:
        if criterion.lower() in response_lower:
            failure_matches.append(criterion)
    
    # Adaptation passed if at least 2/3 success criteria met
    adaptation_passed = len(success_matches) >= 2
    
    # Non-perseveration passed if no failure criteria matched
    non_perseveration_passed = len(failure_matches) == 0
    
    return {
        "adaptation_passed": adaptation_passed,
        "non_perseveration_passed": non_perseveration_passed,
        "criteria_matched": success_matches,
        "failure_criteria_matched": failure_matches,
        "success_criteria_count": len(success_matches),
        "total_success_criteria": len(scenario["success_criteria"])
    }


def check_context_acknowledgment(phase2_response: str, scenario: dict) -> bool:
    """
    Check if the model explicitly acknowledges the disruptive context
    rather than ignoring it.
    """
    # Extract key terms from the disruptive context
    disruptive_text = scenario["disruptive_context"].lower()
    
    # Look for numbers/percentages from the disruptive context
    numbers = re.findall(r'\d+(?:\.\d+)?%?', disruptive_text)
    
    # Look for key nouns (simplified: words > 5 chars not in common words)
    common_words = {"information", "reveals", "shows", "additional", "however", "because", 
                    "which", "their", "there", "about", "these", "those", "where", "while"}
    key_words = [w for w in re.findall(r'\b[a-z]{5,}\b', disruptive_text) 
                 if w not in common_words][:10]
    
    response_lower = phase2_response.lower()
    
    # Count how many key terms appear in the response
    matches = sum(1 for n in numbers if n in response_lower)
    matches += sum(1 for w in key_words if w in response_lower)
    
    # Threshold: at least 3 key terms from the disruptive context appear in response
    return matches >= 3


def compute_composite_score(
    adaptation_result: dict,
    context_acknowledged: bool,
    response_length: int
) -> float:
    """
    Compute a composite score from 0.0 to 1.0.
    
    Scoring breakdown:
    - Adaptation (success criteria met): 50%
    - Non-perseveration (no failure criteria): 30%
    - Context acknowledgment: 20%
    """
    # Adaptation score: proportion of success criteria met
    if adaptation_result["total_success_criteria"] > 0:
        adaptation_score = min(1.0, adaptation_result["success_criteria_count"] / 
                               max(2, adaptation_result["total_success_criteria"]))
    else:
        adaptation_score = 0.0
    
    non_perseveration_score = 1.0 if adaptation_result["non_perseveration_passed"] else 0.0
    context_score = 1.0 if context_acknowledged else 0.0
    
    # Penalize very short responses (likely incomplete)
    length_penalty = 1.0 if response_length >= 200 else (response_length / 200)
    
    composite = (
        0.50 * adaptation_score +
        0.30 * non_perseveration_score +
        0.20 * context_score
    ) * length_penalty
    
    return round(composite, 4)


# ============================================================
# MAIN EVALUATION LOOP (for local testing)
# ============================================================

def evaluate_response(scenario: dict, phase1_response: str, phase2_response: str) -> dict:
    """
    Full evaluation of a model's responses to a single scenario.
    
    Args:
        scenario: The scenario dict from the dataset
        phase1_response: The model's response to the initial prompt
        phase2_response: The model's response after the disruptive context injection
    
    Returns:
        Evaluation result dict
    """
    adaptation_result = check_adaptation_score(phase2_response, scenario)
    context_acknowledged = check_context_acknowledgment(phase2_response, scenario)
    
    composite_score = compute_composite_score(
        adaptation_result,
        context_acknowledged,
        len(phase2_response)
    )
    
    return {
        "scenario_id": scenario["scenario_id"],
        "domain": scenario["domain"],
        "difficulty": scenario["difficulty"],
        "composite_score": composite_score,
        "adaptation_passed": adaptation_result["adaptation_passed"],
        "non_perseveration_passed": adaptation_result["non_perseveration_passed"],
        "context_acknowledged": context_acknowledged,
        "success_criteria_matched": adaptation_result["success_criteria_count"],
        "total_success_criteria": adaptation_result["total_success_criteria"],
        "failure_criteria_triggered": len(adaptation_result["failure_criteria_matched"]),
        "phase2_response_length": len(phase2_response),
        "details": adaptation_result
    }


def run_benchmark_on_model(model_fn, dataset_path: str, output_path: str = None):
    """
    Run the full ADAPT-IQ benchmark on a model.
    
    Args:
        model_fn: A callable that takes a list of messages and returns a string response
        dataset_path: Path to the adapt_iq_dataset.json file
        output_path: Optional path to save results JSON
    
    Returns:
        Aggregated benchmark results
    """
    with open(dataset_path) as f:
        scenarios = json.load(f)
    
    results = []
    
    for i, scenario in enumerate(scenarios):
        print(f"Evaluating scenario {i+1}/{len(scenarios)}: {scenario['scenario_id']}")
        
        # Phase 1: Initial problem
        messages = build_conversation(scenario)
        phase1_response = model_fn(messages)
        
        # Phase 2: Disruptive context injection
        messages.append({"role": "assistant", "content": phase1_response})
        messages.append(build_phase2_message(scenario))
        phase2_response = model_fn(messages)
        
        # Evaluate
        result = evaluate_response(scenario, phase1_response, phase2_response)
        results.append(result)
    
    # Aggregate results
    total_scenarios = len(results)
    avg_composite = sum(r["composite_score"] for r in results) / total_scenarios
    adaptation_rate = sum(1 for r in results if r["adaptation_passed"]) / total_scenarios
    non_perseveration_rate = sum(1 for r in results if r["non_perseveration_passed"]) / total_scenarios
    context_ack_rate = sum(1 for r in results if r["context_acknowledged"]) / total_scenarios
    
    # Domain breakdown
    domain_scores = {}
    for r in results:
        d = r["domain"]
        if d not in domain_scores:
            domain_scores[d] = []
        domain_scores[d].append(r["composite_score"])
    
    domain_averages = {d: round(sum(scores)/len(scores), 4) 
                       for d, scores in domain_scores.items()}
    
    # Difficulty breakdown
    difficulty_scores = {}
    for r in results:
        diff = r["difficulty"]
        if diff not in difficulty_scores:
            difficulty_scores[diff] = []
        difficulty_scores[diff].append(r["composite_score"])
    
    difficulty_averages = {d: round(sum(scores)/len(scores), 4) 
                           for d, scores in difficulty_scores.items()}
    
    aggregated = {
        "benchmark": "ADAPT-IQ",
        "total_scenarios": total_scenarios,
        "overall_composite_score": round(avg_composite, 4),
        "adaptation_rate": round(adaptation_rate, 4),
        "non_perseveration_rate": round(non_perseveration_rate, 4),
        "context_acknowledgment_rate": round(context_ack_rate, 4),
        "domain_scores": domain_averages,
        "difficulty_scores": difficulty_averages,
        "per_scenario_results": results
    }
    
    if output_path:
        with open(output_path, "w") as f:
            json.dump(aggregated, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    print(f"\n{'='*50}")
    print(f"ADAPT-IQ BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"Overall Composite Score: {avg_composite:.4f}")
    print(f"Adaptation Rate: {adaptation_rate:.2%}")
    print(f"Non-Perseveration Rate: {non_perseveration_rate:.2%}")
    print(f"Context Acknowledgment Rate: {context_ack_rate:.2%}")
    print(f"\nDomain Scores:")
    for domain, score in domain_averages.items():
        print(f"  {domain}: {score:.4f}")
    print(f"\nDifficulty Scores:")
    for diff, score in difficulty_averages.items():
        print(f"  {diff}: {score:.4f}")
    
    return aggregated


if __name__ == "__main__":
    # Demo: Test the evaluation functions with a sample response
    import sys
    sys.path.insert(0, "/home/ubuntu/adapt_iq")
    
    with open("/home/ubuntu/adapt_iq/data/adapt_iq_dataset.json") as f:
        scenarios = json.load(f)
    
    # Test with a mock "good" response for RM-001
    test_scenario = scenarios[0]
    
    mock_phase2_good = """
    Given the chemical contamination affecting 40% of the reservoir, I must completely revise my plan.
    
    The effective usable capacity is now only 36% (60% × 0.6 = 36%), which represents a critical shortage.
    
    Emergency Actions:
    1. Immediately halt all agricultural water use from the contaminated supply
    2. Activate emergency water trucking from neighboring municipalities
    3. Contact state emergency management for desalination unit deployment
    4. Issue a boil-water advisory for all residents
    5. Prioritize drinking water for the 500,000 residents above all other uses
    
    Alternative Sources:
    - Emergency water trucking: 2 million liters/day capacity
    - Neighboring municipality transfer agreement: activate immediately
    - Mobile desalination units: request 3 units from state emergency reserves
    
    The contaminated 40% must be isolated and remediation begun immediately.
    """
    
    mock_phase2_bad = """
    My original water conservation plan remains appropriate. The reservoir is at 60% capacity,
    which gives us sufficient buffer for the 3-month drought period. I recommend the original
    conservation measures: reducing residential use by 15%, limiting agricultural irrigation
    to off-peak hours, and public awareness campaigns.
    """
    
    print("Testing GOOD response:")
    result_good = evaluate_response(test_scenario, "initial response", mock_phase2_good)
    print(f"  Composite Score: {result_good['composite_score']}")
    print(f"  Adaptation Passed: {result_good['adaptation_passed']}")
    print(f"  Non-Perseveration Passed: {result_good['non_perseveration_passed']}")
    
    print("\nTesting BAD response (perseveration):")
    result_bad = evaluate_response(test_scenario, "initial response", mock_phase2_bad)
    print(f"  Composite Score: {result_bad['composite_score']}")
    print(f"  Adaptation Passed: {result_bad['adaptation_passed']}")
    print(f"  Non-Perseveration Passed: {result_bad['non_perseveration_passed']}")
