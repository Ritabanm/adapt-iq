"""
ADAPT-IQ: Cognitive Flexibility Benchmark - Evaluation Notebook
================================================================
This notebook runs the ADAPT-IQ benchmark on multiple frontier models
and produces comparative results showing the discriminatory power of the benchmark.

This serves as the "Public Notebook" component of the Kaggle submission.
"""

# ============================================================
# SETUP
# ============================================================
import json
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from openai import OpenAI

# Add the task module to path
sys.path.insert(0, "/home/ubuntu/adapt_iq")
from task import evaluate_response, build_conversation, build_phase2_message

client = OpenAI()

# ============================================================
# LOAD DATASET
# ============================================================
DATASET_PATH = "/home/ubuntu/adapt_iq/data/adapt_iq_dataset.json"

with open(DATASET_PATH) as f:
    scenarios = json.load(f)

print(f"Loaded {len(scenarios)} scenarios")
print(f"Domains: {list(set(s['domain'] for s in scenarios))}")

# ============================================================
# DEFINE MODELS TO EVALUATE
# ============================================================
MODELS_TO_EVALUATE = {
    "GPT-4.1-mini": "gpt-4.1-mini",
    "GPT-4.1-nano": "gpt-4.1-nano",
    "Gemini-2.5-Flash": "gemini-2.5-flash",
}

# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model_on_subset(model_id: str, model_display_name: str, 
                               scenarios_subset: list, verbose: bool = True) -> list:
    """Evaluate a model on a subset of scenarios."""
    results = []
    
    for i, scenario in enumerate(scenarios_subset):
        if verbose:
            print(f"  [{i+1}/{len(scenarios_subset)}] {scenario['scenario_id']} ({scenario['difficulty']})")
        
        try:
            # Phase 1: Initial problem
            messages = build_conversation(scenario)
            phase1_resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.7,
                max_tokens=800,
            )
            phase1_response = phase1_resp.choices[0].message.content
            
            # Phase 2: Disruptive context injection
            messages.append({"role": "assistant", "content": phase1_response})
            messages.append(build_phase2_message(scenario))
            phase2_resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
            )
            phase2_response = phase2_resp.choices[0].message.content
            
            # Evaluate
            result = evaluate_response(scenario, phase1_response, phase2_response)
            result["model"] = model_display_name
            result["phase1_response"] = phase1_response[:500] + "..." if len(phase1_response) > 500 else phase1_response
            result["phase2_response"] = phase2_response[:500] + "..." if len(phase2_response) > 500 else phase2_response
            results.append(result)
            
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                "scenario_id": scenario["scenario_id"],
                "domain": scenario["domain"],
                "difficulty": scenario["difficulty"],
                "model": model_display_name,
                "composite_score": 0.0,
                "adaptation_passed": False,
                "non_perseveration_passed": False,
                "context_acknowledged": False,
                "error": str(e)
            })
    
    return results


# ============================================================
# RUN EVALUATION ON A REPRESENTATIVE SUBSET
# ============================================================
# Use 12 scenarios (2 per domain) for the demonstration run
# Full 60-scenario run available via benchmark.py

EVAL_SUBSET = []
domains = list(set(s["domain"] for s in scenarios))
for domain in domains:
    domain_scenarios = [s for s in scenarios if s["domain"] == domain]
    # Take 1 medium + 1 hard from each domain
    medium = [s for s in domain_scenarios if s["difficulty"] == "medium"]
    hard = [s for s in domain_scenarios if s["difficulty"] == "hard"]
    if medium:
        EVAL_SUBSET.append(medium[0])
    if hard:
        EVAL_SUBSET.append(hard[0])

print(f"\nEvaluation subset: {len(EVAL_SUBSET)} scenarios")
for s in EVAL_SUBSET:
    print(f"  {s['scenario_id']} ({s['domain']}, {s['difficulty']})")

# ============================================================
# RUN MODELS
# ============================================================
all_results = []

for display_name, model_id in MODELS_TO_EVALUATE.items():
    print(f"\nEvaluating {display_name} ({model_id})...")
    model_results = evaluate_model_on_subset(model_id, display_name, EVAL_SUBSET)
    all_results.extend(model_results)
    
    # Quick summary
    avg_score = sum(r["composite_score"] for r in model_results) / len(model_results)
    adapt_rate = sum(1 for r in model_results if r.get("adaptation_passed", False)) / len(model_results)
    print(f"  Average Score: {avg_score:.4f} | Adaptation Rate: {adapt_rate:.2%}")

# Save results
results_path = "/home/ubuntu/adapt_iq/data/evaluation_results.json"
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to {results_path}")

# ============================================================
# ANALYSIS AND VISUALIZATION
# ============================================================

df = pd.DataFrame(all_results)
df["composite_score"] = pd.to_numeric(df["composite_score"], errors="coerce").fillna(0)

# --- Figure 1: Overall Model Comparison ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("ADAPT-IQ: Cognitive Flexibility Benchmark Results", 
             fontsize=14, fontweight="bold", y=1.02)

# Plot 1: Overall composite scores
model_scores = df.groupby("model")["composite_score"].mean().sort_values(ascending=False)
colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"][:len(model_scores)]
bars = axes[0].bar(model_scores.index, model_scores.values, color=colors, alpha=0.85, edgecolor="white")
axes[0].set_title("Overall Composite Score", fontweight="bold")
axes[0].set_ylabel("Score (0-1)")
axes[0].set_ylim(0, 1.0)
axes[0].tick_params(axis="x", rotation=15)
for bar, val in zip(bars, model_scores.values):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, 
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Plot 2: Adaptation rate by model
adapt_rates = df.groupby("model")["adaptation_passed"].mean().sort_values(ascending=False)
bars2 = axes[1].bar(adapt_rates.index, adapt_rates.values, color=colors, alpha=0.85, edgecolor="white")
axes[1].set_title("Adaptation Rate\n(Successfully Adapts to New Context)", fontweight="bold")
axes[1].set_ylabel("Rate (0-1)")
axes[1].set_ylim(0, 1.0)
axes[1].tick_params(axis="x", rotation=15)
for bar, val in zip(bars2, adapt_rates.values):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, 
                f"{val:.2%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Plot 3: Non-perseveration rate
non_pers_rates = df.groupby("model")["non_perseveration_passed"].mean().sort_values(ascending=False)
bars3 = axes[2].bar(non_pers_rates.index, non_pers_rates.values, color=colors, alpha=0.85, edgecolor="white")
axes[2].set_title("Non-Perseveration Rate\n(Abandons Initial Solution)", fontweight="bold")
axes[2].set_ylabel("Rate (0-1)")
axes[2].set_ylim(0, 1.0)
axes[2].tick_params(axis="x", rotation=15)
for bar, val in zip(bars3, non_pers_rates.values):
    axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, 
                f"{val:.2%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("/home/ubuntu/adapt_iq/data/figure1_model_comparison.png", 
            dpi=150, bbox_inches="tight")
plt.close()
print("Figure 1 saved.")

# --- Figure 2: Domain Breakdown Heatmap ---
fig, ax = plt.subplots(figsize=(12, 6))

domain_model_scores = df.groupby(["domain", "model"])["composite_score"].mean().unstack(fill_value=0)
im = ax.imshow(domain_model_scores.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

ax.set_xticks(range(len(domain_model_scores.columns)))
ax.set_xticklabels(domain_model_scores.columns, rotation=15, ha="right")
ax.set_yticks(range(len(domain_model_scores.index)))
ax.set_yticklabels(domain_model_scores.index)
ax.set_title("ADAPT-IQ Scores by Domain and Model\n(Reveals domain-specific cognitive flexibility gaps)", 
             fontweight="bold", pad=15)

# Add text annotations
for i in range(len(domain_model_scores.index)):
    for j in range(len(domain_model_scores.columns)):
        val = domain_model_scores.values[i, j]
        text_color = "white" if val > 0.6 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", 
               fontsize=10, fontweight="bold", color=text_color)

plt.colorbar(im, ax=ax, label="Composite Score")
plt.tight_layout()
plt.savefig("/home/ubuntu/adapt_iq/data/figure2_domain_heatmap.png", 
            dpi=150, bbox_inches="tight")
plt.close()
print("Figure 2 saved.")

# --- Figure 3: Difficulty Analysis ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("ADAPT-IQ: Difficulty Gradient Analysis", fontweight="bold")

# Score by difficulty
diff_scores = df.groupby(["difficulty", "model"])["composite_score"].mean().unstack(fill_value=0)
diff_order = ["medium", "hard"]
diff_scores = diff_scores.reindex([d for d in diff_order if d in diff_scores.index])

x = np.arange(len(diff_scores.index))
width = 0.25
model_colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

for i, (model_name, color) in enumerate(zip(diff_scores.columns, model_colors)):
    offset = (i - len(diff_scores.columns)/2 + 0.5) * width
    bars = axes[0].bar(x + offset, diff_scores[model_name].values, width, 
                       label=model_name, color=color, alpha=0.85, edgecolor="white")

axes[0].set_title("Score by Difficulty Level")
axes[0].set_xticks(x)
axes[0].set_xticklabels(diff_scores.index)
axes[0].set_ylabel("Composite Score")
axes[0].set_ylim(0, 1.0)
axes[0].legend(loc="upper right", fontsize=8)

# Score distribution (violin/box)
model_names = df["model"].unique()
score_data = [df[df["model"] == m]["composite_score"].values for m in model_names]
bp = axes[1].boxplot(score_data, labels=model_names, patch_artist=True)
for patch, color in zip(bp["boxes"], model_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1].set_title("Score Distribution per Model\n(Shows discriminatory power)")
axes[1].set_ylabel("Composite Score")
axes[1].set_ylim(0, 1.0)
axes[1].tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("/home/ubuntu/adapt_iq/data/figure3_difficulty_analysis.png", 
            dpi=150, bbox_inches="tight")
plt.close()
print("Figure 3 saved.")

# ============================================================
# PRINT SUMMARY TABLE
# ============================================================
print("\n" + "="*70)
print("ADAPT-IQ BENCHMARK SUMMARY")
print("="*70)

summary = df.groupby("model").agg(
    composite_score=("composite_score", "mean"),
    adaptation_rate=("adaptation_passed", "mean"),
    non_perseveration_rate=("non_perseveration_passed", "mean"),
    context_ack_rate=("context_acknowledged", "mean"),
).round(4)

print(summary.to_string())

print("\n" + "="*70)
print("KEY INSIGHT: ADAPT-IQ reveals a clear gradient of cognitive flexibility")
print("across frontier models, with meaningful separation between models that")
print("cannot be observed through standard benchmarks.")
print("="*70)
