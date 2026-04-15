"""
ADAPT-IQ Full Evaluation Runner
Evaluates all 100 scenarios against 3 live models:
  - gpt-4.1-nano
  - gpt-4.1-mini
  - gemini-2.5-flash

Uses the same 3-phase CICT paradigm and composite scoring as task.py
"""

import json
import re
import os
import time
from openai import OpenAI

client = OpenAI()

MODELS = [
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "gemini-2.5-flash",
]

SYSTEM_PROMPT = (
    "You are a highly capable AI assistant. "
    "Respond thoroughly and adapt your reasoning when new information is provided."
)

def call_model(model: str, messages: list, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=1500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"    [Attempt {attempt+1}] Error calling {model}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return ""

def evaluate_scenario(scenario: dict, model: str) -> dict:
    """Run the 3-phase CICT evaluation for one scenario."""
    
    # Phase 1: Initial problem
    phase1_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": scenario["initial_prompt"]},
    ]
    phase1_response = call_model(model, phase1_messages)
    
    if not phase1_response:
        return None
    
    # Phase 2: Context injection (disruptive information)
    phase2_messages = phase1_messages + [
        {"role": "assistant", "content": phase1_response},
        {"role": "user", "content": scenario["disruptive_context"]},
    ]
    phase2_response = call_model(model, phase2_messages)
    
    if not phase2_response:
        return None
    
    # Scoring
    response_lower = phase2_response.lower()
    
    # 1. Adaptation Rate (50%): Check success criteria
    success_criteria = scenario.get("success_criteria", [])
    criteria_matched = []
    for criterion in success_criteria:
        pattern = criterion.lower()
        if re.search(pattern, response_lower):
            criteria_matched.append(criterion)
    
    adaptation_rate = len(criteria_matched) / len(success_criteria) if success_criteria else 0.0
    adaptation_passed = adaptation_rate >= 0.67  # at least 2/3 criteria
    
    # 2. Non-Perseveration (30%): Check failure criteria NOT triggered
    failure_criteria = scenario.get("failure_criteria", [])
    failure_triggered = []
    for fc in failure_criteria:
        if fc.lower() in response_lower:
            failure_triggered.append(fc)
    
    non_perseveration_passed = len(failure_triggered) == 0
    non_perseveration_score = 1.0 if non_perseveration_passed else 0.0
    
    # 3. Context Acknowledgment (20%): Does response acknowledge the new info?
    ack_patterns = [
        r"new information",
        r"update[d]?",
        r"however",
        r"given (that|this|the)",
        r"in light of",
        r"now that",
        r"with this",
        r"revised",
        r"changed",
        r"adjust",
        r"reconsider",
        r"taking into account",
        r"based on (this|the new)",
        r"important(ly)?",
        r"critical(ly)?",
        r"significant(ly)?",
    ]
    context_acknowledged = any(re.search(p, response_lower) for p in ack_patterns)
    context_score = 1.0 if context_acknowledged else 0.0
    
    # Composite score
    composite_score = (
        0.50 * adaptation_rate +
        0.30 * non_perseveration_score +
        0.20 * context_score
    )
    
    return {
        "scenario_id": scenario["scenario_id"],
        "domain": scenario["domain"],
        "difficulty": scenario["difficulty"],
        "model": model,
        "composite_score": round(composite_score, 4),
        "adaptation_rate": round(adaptation_rate, 4),
        "adaptation_passed": adaptation_passed,
        "non_perseveration_passed": non_perseveration_passed,
        "context_acknowledged": context_acknowledged,
        "success_criteria_matched": len(criteria_matched),
        "total_success_criteria": len(success_criteria),
        "failure_criteria_triggered": len(failure_triggered),
        "phase2_response_length": len(phase2_response),
        "details": {
            "adaptation_passed": adaptation_passed,
            "non_perseveration_passed": non_perseveration_passed,
            "criteria_matched": criteria_matched,
            "failure_criteria_matched": failure_triggered,
            "success_criteria_count": len(criteria_matched),
            "total_success_criteria": len(success_criteria),
        },
        "phase1_response": phase1_response[:500],  # truncate for storage
        "phase2_response": phase2_response[:500],  # truncate for storage
    }

def main():
    # Load dataset
    with open("data/adapt_iq_dataset.json") as f:
        scenarios = json.load(f)
    
    print(f"Loaded {len(scenarios)} scenarios")
    print(f"Models to evaluate: {MODELS}")
    print(f"Total evaluations: {len(scenarios) * len(MODELS)}\n")
    
    all_results = []
    
    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Evaluating model: {model}")
        print(f"{'='*60}")
        
        model_results = []
        model_scores = []
        
        for i, scenario in enumerate(scenarios):
            sid = scenario["scenario_id"]
            domain = scenario["domain"]
            difficulty = scenario["difficulty"]
            
            print(f"  [{i+1:3d}/100] {sid} ({domain}, {difficulty})...", end=" ", flush=True)
            
            result = evaluate_scenario(scenario, model)
            
            if result:
                model_results.append(result)
                model_scores.append(result["composite_score"])
                print(f"score={result['composite_score']:.3f}")
            else:
                print("FAILED")
            
            # Small delay to avoid rate limits
            time.sleep(0.3)
        
        avg_score = sum(model_scores) / len(model_scores) if model_scores else 0
        print(f"\n  {model} average: {avg_score:.4f} (n={len(model_results)})")
        
        all_results.extend(model_results)
        
        # Save intermediate results
        with open("data/evaluation_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Intermediate save: {len(all_results)} total results")
    
    # Final save
    with open("data/evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    
    from collections import defaultdict
    model_summary = defaultdict(list)
    for r in all_results:
        model_summary[r["model"]].append(r["composite_score"])
    
    for model, scores in sorted(model_summary.items()):
        avg = sum(scores) / len(scores)
        print(f"{model}: n={len(scores)}, avg={avg:.4f}, min={min(scores):.3f}, max={max(scores):.3f}")
    
    print(f"\nTotal results saved: {len(all_results)}")
    print("Done!")

if __name__ == "__main__":
    main()
