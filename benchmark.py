"""
ADAPT-IQ Kaggle Benchmark Definition
=====================================
This file defines the benchmark structure using the Kaggle Benchmarks SDK pattern.
It wraps the ADAPT-IQ task into a format compatible with Kaggle's evaluation infrastructure.

Usage:
    python benchmark.py --model gpt-4o --output results.json
    python benchmark.py --model gemini-2.0-flash --output results.json
"""

import json
import os
import argparse
from pathlib import Path
from openai import OpenAI
from task import run_benchmark_on_model, build_conversation, build_phase2_message

# ============================================================
# SUPPORTED MODELS
# ============================================================
SUPPORTED_MODELS = {
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4.1": "gpt-4.1",
    "gpt-4.1-mini": "gpt-4.1-mini",
    "gemini-2.0-flash": "gemini-2.0-flash-exp",
    "gemini-1.5-pro": "gemini-1.5-pro",
    "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
}


def create_openai_model_fn(model_name: str, client: OpenAI):
    """Create a model function for OpenAI-compatible models."""
    def model_fn(messages: list) -> str:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=1500,
        )
        return response.choices[0].message.content
    return model_fn


def run_adapt_iq_benchmark(model_name: str, dataset_path: str, output_path: str):
    """
    Run the ADAPT-IQ benchmark on a specified model.
    
    This is the main entry point for the Kaggle Benchmarks evaluation.
    """
    print(f"\n{'='*60}")
    print(f"ADAPT-IQ: Cognitive Flexibility Benchmark")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"{'='*60}\n")
    
    # Initialize OpenAI client (works with OpenAI-compatible APIs)
    client = OpenAI()
    
    # Resolve model name
    resolved_model = SUPPORTED_MODELS.get(model_name, model_name)
    
    # Create model function
    model_fn = create_openai_model_fn(resolved_model, client)
    
    # Run benchmark
    results = run_benchmark_on_model(model_fn, dataset_path, output_path)
    
    return results


# ============================================================
# KAGGLE BENCHMARKS SDK INTERFACE
# ============================================================

class AdaptIQTask:
    """
    ADAPT-IQ Task class following Kaggle Benchmarks SDK conventions.
    
    Each task instance represents a single scenario from the ADAPT-IQ dataset.
    The task is evaluated in two phases:
    1. Phase 1: Model responds to the initial problem
    2. Phase 2: Model adapts its response after a disruptive context injection
    """
    
    def __init__(self, scenario: dict):
        self.scenario = scenario
        self.scenario_id = scenario["scenario_id"]
        self.domain = scenario["domain"]
        self.difficulty = scenario["difficulty"]
    
    def get_prompt(self) -> list[dict]:
        """Return the initial conversation prompt (Phase 1)."""
        return build_conversation(self.scenario)
    
    def get_followup(self, phase1_response: str) -> list[dict]:
        """Return the full conversation with Phase 2 injection."""
        messages = build_conversation(self.scenario)
        messages.append({"role": "assistant", "content": phase1_response})
        messages.append(build_phase2_message(self.scenario))
        return messages
    
    def evaluate(self, phase1_response: str, phase2_response: str) -> dict:
        """Evaluate the model's responses and return a score."""
        from task import evaluate_response
        return evaluate_response(self.scenario, phase1_response, phase2_response)
    
    @property
    def metadata(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "required_adaptation": self.scenario["required_adaptation"],
            "failure_mode_anchor": self.scenario["failure_mode_anchor"]
        }


class AdaptIQBenchmark:
    """
    ADAPT-IQ Benchmark collection following Kaggle Benchmarks SDK conventions.
    
    Benchmark: adapt-iq-cognitive-flexibility
    Track: Executive Functions
    """
    
    BENCHMARK_NAME = "adapt-iq-cognitive-flexibility"
    BENCHMARK_VERSION = "1.0.0"
    TRACK = "executive_functions"
    SECONDARY_TRACK = "learning"
    
    def __init__(self, dataset_path: str):
        with open(dataset_path) as f:
            self.scenarios = json.load(f)
        self.tasks = [AdaptIQTask(s) for s in self.scenarios]
    
    def __len__(self):
        return len(self.tasks)
    
    def __iter__(self):
        return iter(self.tasks)
    
    def get_task(self, scenario_id: str) -> AdaptIQTask:
        """Get a specific task by scenario ID."""
        for task in self.tasks:
            if task.scenario_id == scenario_id:
                return task
        raise ValueError(f"Scenario {scenario_id} not found")
    
    def get_tasks_by_domain(self, domain: str) -> list:
        """Get all tasks for a specific domain."""
        return [t for t in self.tasks if t.domain == domain]
    
    def get_tasks_by_difficulty(self, difficulty: str) -> list:
        """Get all tasks for a specific difficulty level."""
        return [t for t in self.tasks if t.difficulty == difficulty]
    
    def run(self, model_fn, output_path: str = None) -> dict:
        """Run the full benchmark on a model function."""
        dataset_path = "/home/ubuntu/adapt_iq/data/adapt_iq_dataset.json"
        return run_benchmark_on_model(model_fn, dataset_path, output_path)
    
    @property
    def metadata(self) -> dict:
        return {
            "name": self.BENCHMARK_NAME,
            "version": self.BENCHMARK_VERSION,
            "track": self.TRACK,
            "secondary_track": self.SECONDARY_TRACK,
            "total_tasks": len(self.tasks),
            "domains": list(set(t.domain for t in self.tasks)),
            "difficulty_levels": list(set(t.difficulty for t in self.tasks)),
            "description": (
                "ADAPT-IQ evaluates cognitive flexibility in AI systems through "
                "Context-Injection Creativity Tests (CICT). Each scenario presents "
                "an initial problem followed by disruptive new information, measuring "
                "whether models can adapt their reasoning or perseverate on initial solutions."
            )
        }


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ADAPT-IQ Benchmark")
    parser.add_argument("--model", default="gpt-4.1-mini", 
                        help="Model to evaluate (default: gpt-4.1-mini)")
    parser.add_argument("--dataset", 
                        default="/home/ubuntu/adapt_iq/data/adapt_iq_dataset.json",
                        help="Path to dataset JSON")
    parser.add_argument("--output", default="results.json",
                        help="Output path for results JSON")
    parser.add_argument("--subset", type=int, default=None,
                        help="Run on a subset of N scenarios (for testing)")
    
    args = parser.parse_args()
    
    if args.subset:
        # Load and subset the dataset for testing
        with open(args.dataset) as f:
            full_data = json.load(f)
        subset_path = f"/tmp/adapt_iq_subset_{args.subset}.json"
        with open(subset_path, "w") as f:
            json.dump(full_data[:args.subset], f)
        dataset_path = subset_path
    else:
        dataset_path = args.dataset
    
    results = run_adapt_iq_benchmark(args.model, dataset_path, args.output)
