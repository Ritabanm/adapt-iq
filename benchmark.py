"""
ADAPT-IQ Kaggle Benchmark Definition
=====================================
This file defines the benchmark structure using the Kaggle Benchmarks SDK pattern.
It wraps the ADAPT-IQ task into a format compatible with Kaggle's evaluation
infrastructure.

Benchmark:  adapt-iq-cognitive-flexibility
Version:    2.0.0
Track:      Executive Functions (primary) · Learning (secondary)
Dataset:    100 scenarios × 6 domains × 3 difficulty levels

Usage (CLI):
    python benchmark.py --model gpt-4.1-mini --output results.json
    python benchmark.py --model gemini-2.5-flash --output results.json
    python benchmark.py --model gpt-4.1-nano --subset 10  # quick smoke test
"""

import json
import os
import argparse
from pathlib import Path
from openai import OpenAI

from task import (
    run_benchmark_on_model,
    build_conversation,
    build_phase2_message,
    evaluate_response,
    TASK_NAME,
    TASK_VERSION,
    TASK_DESCRIPTION,
)

# ============================================================
# DEFAULT DATASET PATH
# ============================================================
DEFAULT_DATASET = str(Path(__file__).parent / "data" / "adapt_iq_dataset.json")


# ============================================================
# SUPPORTED MODELS
# ============================================================
# Keys are the short names used on the CLI; values are the API identifiers.
# Add new models here — no other changes needed.
SUPPORTED_MODELS = {
    # OpenAI GPT-4.1 family
    "gpt-4.1":           "gpt-4.1",
    "gpt-4.1-mini":      "gpt-4.1-mini",
    "gpt-4.1-nano":      "gpt-4.1-nano",
    # OpenAI GPT-4o family
    "gpt-4o":            "gpt-4o",
    "gpt-4o-mini":       "gpt-4o-mini",
    # Google Gemini (via OpenAI-compatible endpoint)
    "gemini-2.5-flash":  "gemini-2.5-flash",
    "gemini-2.0-flash":  "gemini-2.0-flash",
    "gemini-1.5-pro":    "gemini-1.5-pro",
    # Anthropic Claude (via OpenAI-compatible endpoint)
    "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet": "claude-3-7-sonnet-20250219",
}


# ============================================================
# MODEL FUNCTION FACTORY
# ============================================================

def create_openai_model_fn(model_name: str, client: OpenAI, temperature: float = 0.3):
    """
    Create a model callable for any OpenAI-compatible API.

    Args:
        model_name  : The API model identifier string.
        client      : An initialised OpenAI client.
        temperature : Sampling temperature (lower = more deterministic).
                      0.3 is recommended for reproducible benchmark results.

    Returns:
        A callable: messages (list[dict]) → response_text (str)
    """
    def model_fn(messages: list) -> str:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=1500,
        )
        return response.choices[0].message.content.strip()
    return model_fn


# ============================================================
# KAGGLE BENCHMARKS SDK INTERFACE
# ============================================================

class AdaptIQTask:
    """
    A single ADAPT-IQ evaluation task (one scenario).

    Follows the Kaggle Benchmarks SDK task interface:
    - get_prompt()     → Phase 1 messages
    - get_followup()   → Phase 2 messages (after Phase 1 response)
    - evaluate()       → score dict
    - metadata         → task metadata dict
    """

    def __init__(self, scenario: dict):
        self.scenario    = scenario
        self.scenario_id = scenario["scenario_id"]
        self.domain      = scenario["domain"]
        self.difficulty  = scenario["difficulty"]

    def get_prompt(self) -> list[dict]:
        """
        Return the Phase 1 conversation messages.
        Send these to the model to get the initial solution (Phase 1 response).
        """
        return build_conversation(self.scenario)

    def get_followup(self, phase1_response: str) -> list[dict]:
        """
        Return the full conversation including the disruptive context injection.
        Append the model's Phase 1 response, then send these to get Phase 2.

        Args:
            phase1_response : The model's Phase 1 answer (string).

        Returns:
            Full message list ready for the Phase 2 API call.
        """
        messages = build_conversation(self.scenario)
        messages.append({"role": "assistant", "content": phase1_response})
        messages.append(build_phase2_message(self.scenario))
        return messages

    def evaluate(self, phase1_response: str, phase2_response: str) -> dict:
        """
        Score the model's two-phase responses.

        Args:
            phase1_response : Model's answer to the initial prompt.
            phase2_response : Model's revised answer after context injection.

        Returns:
            Result dict with composite_score and all sub-scores.
        """
        return evaluate_response(self.scenario, phase1_response, phase2_response)

    @property
    def metadata(self) -> dict:
        """Task metadata for logging and analysis."""
        return {
            "scenario_id":        self.scenario_id,
            "domain":             self.domain,
            "difficulty":         self.difficulty,
            "required_adaptation": self.scenario.get("required_adaptation", ""),
            "failure_mode_anchor": self.scenario.get("failure_mode_anchor", ""),
        }


class AdaptIQBenchmark:
    """
    The full ADAPT-IQ benchmark collection.

    Follows the Kaggle Benchmarks SDK benchmark interface:
    - Iterable over AdaptIQTask instances
    - run(model_fn) → aggregated results dict
    - metadata       → benchmark metadata dict

    Example usage:
        benchmark = AdaptIQBenchmark("data/adapt_iq_dataset.json")
        client = OpenAI()
        model_fn = create_openai_model_fn("gpt-4.1-mini", client)
        results = benchmark.run(model_fn, output_path="results.json")
        print(f"Average score: {results['average_composite_score']:.4f}")
    """

    BENCHMARK_NAME    = "adapt-iq-cognitive-flexibility"
    BENCHMARK_VERSION = TASK_VERSION
    TRACK             = "executive_functions"
    SECONDARY_TRACK   = "learning"

    def __init__(self, dataset_path: str = DEFAULT_DATASET):
        with open(dataset_path) as f:
            self.scenarios = json.load(f)
        self.tasks = [AdaptIQTask(s) for s in self.scenarios]
        self._dataset_path = dataset_path

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)

    def __repr__(self) -> str:
        return (
            f"AdaptIQBenchmark(tasks={len(self.tasks)}, "
            f"domains={len(set(t.domain for t in self.tasks))}, "
            f"version={self.BENCHMARK_VERSION})"
        )

    def get_task(self, scenario_id: str) -> AdaptIQTask:
        """Retrieve a specific task by its scenario ID (e.g. 'RM-001')."""
        for task in self.tasks:
            if task.scenario_id == scenario_id:
                return task
        raise ValueError(f"Scenario '{scenario_id}' not found in dataset.")

    def get_tasks_by_domain(self, domain: str) -> list:
        """Return all tasks for a given domain."""
        return [t for t in self.tasks if t.domain == domain]

    def get_tasks_by_difficulty(self, difficulty: str) -> list:
        """Return all tasks for a given difficulty level (easy/medium/hard)."""
        return [t for t in self.tasks if t.difficulty == difficulty]

    def run(self, model_fn, output_path: str = None) -> dict:
        """
        Run the full benchmark against a model function.

        Args:
            model_fn    : Callable (messages: list[dict]) → str
            output_path : Optional JSON file path to save results.

        Returns:
            Aggregated results dict.
        """
        return run_benchmark_on_model(model_fn, self._dataset_path, output_path)

    @property
    def metadata(self) -> dict:
        """Benchmark metadata for the Kaggle Benchmarks SDK."""
        domains     = sorted(set(t.domain     for t in self.tasks))
        difficulties= sorted(set(t.difficulty for t in self.tasks))
        return {
            "name":              self.BENCHMARK_NAME,
            "version":           self.BENCHMARK_VERSION,
            "track":             self.TRACK,
            "secondary_track":   self.SECONDARY_TRACK,
            "total_tasks":       len(self.tasks),
            "domains":           domains,
            "difficulty_levels": difficulties,
            "description":       TASK_DESCRIPTION,
        }


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def run_adapt_iq_benchmark(
    model_name: str,
    dataset_path: str = DEFAULT_DATASET,
    output_path: str = "results.json",
    temperature: float = 0.3,
) -> dict:
    """
    High-level convenience function to run ADAPT-IQ on a named model.

    Args:
        model_name   : Short name (see SUPPORTED_MODELS) or full API identifier.
        dataset_path : Path to adapt_iq_dataset.json.
        output_path  : Where to save the results JSON.
        temperature  : Sampling temperature (default 0.3 for reproducibility).

    Returns:
        Aggregated benchmark results dict.
    """
    print(f"\n{'='*60}")
    print(f"ADAPT-IQ v{TASK_VERSION}: Cognitive Flexibility Benchmark")
    print(f"Model:   {model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"{'='*60}\n")

    client = OpenAI()
    resolved_model = SUPPORTED_MODELS.get(model_name, model_name)
    model_fn = create_openai_model_fn(resolved_model, client, temperature=temperature)

    benchmark = AdaptIQBenchmark(dataset_path)
    return benchmark.run(model_fn, output_path)


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ADAPT-IQ Cognitive Flexibility Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Supported model shortcuts:\n  "
            + "\n  ".join(SUPPORTED_MODELS.keys())
            + "\n\nAny other string is passed directly to the API as the model name."
        ),
    )
    parser.add_argument(
        "--model", default="gpt-4.1-mini",
        help="Model to evaluate (default: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET,
        help="Path to adapt_iq_dataset.json",
    )
    parser.add_argument(
        "--output", default="results.json",
        help="Output path for results JSON (default: results.json)",
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Evaluate only the first N scenarios (useful for smoke testing)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="Sampling temperature (default: 0.3)",
    )

    args = parser.parse_args()

    # Handle subset mode
    if args.subset:
        with open(args.dataset) as f:
            full_data = json.load(f)
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="adapt_iq_subset_"
        )
        json.dump(full_data[:args.subset], tmp)
        tmp.close()
        dataset_path = tmp.name
        print(f"Subset mode: evaluating first {args.subset} scenarios.")
    else:
        dataset_path = args.dataset

    results = run_adapt_iq_benchmark(
        args.model,
        dataset_path=dataset_path,
        output_path=args.output,
        temperature=args.temperature,
    )
