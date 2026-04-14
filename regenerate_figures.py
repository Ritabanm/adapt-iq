"""
Regenerate all ADAPT-IQ figures with updated 100-scenario, 3-model results.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict

# Load results
with open("data/evaluation_results.json") as f:
    results = json.load(f)

with open("data/adapt_iq_dataset.json") as f:
    dataset = json.load(f)

print(f"Loaded {len(results)} results, {len(dataset)} scenarios")

# Organize data
model_results = defaultdict(list)
for r in results:
    model_results[r["model"]].append(r)

models = ["gpt-4.1-nano", "gpt-4.1-mini", "gemini-2.5-flash"]
model_labels = ["GPT-4.1-nano", "GPT-4.1-mini", "Gemini-2.5-Flash"]
model_colors = ["#2196F3", "#4CAF50", "#FF9800"]

# ============================================================
# FIGURE 1: Model Comparison Bar Chart
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

model_avgs = []
model_stds = []
for model in models:
    scores = [r["composite_score"] for r in model_results[model]]
    model_avgs.append(np.mean(scores))
    model_stds.append(np.std(scores))

bars = ax.bar(model_labels, model_avgs, color=model_colors, 
              yerr=model_stds, capsize=8, edgecolor='black', linewidth=0.8,
              error_kw=dict(elinewidth=1.5, ecolor='black'))

for bar, avg, std in zip(bars, model_avgs, model_stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
            f'{avg:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylim(0, 1.1)
ax.set_ylabel("ADAPT-IQ Composite Score", fontsize=13)
ax.set_title("ADAPT-IQ Model Performance\n(100 Scenarios, 3 Domains × 6 Categories)", fontsize=14, fontweight='bold')
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect Score')
ax.set_xlabel("Model", fontsize=13)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("data/figure1_model_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1 saved")

# ============================================================
# FIGURE 2: Domain Performance Heatmap
# ============================================================
domains = [
    "Resource Management",
    "Social Dynamics",
    "Engineering & Design",
    "Scientific Reasoning",
    "Creative Problem Solving",
    "Cross-Domain Adaptation",
]

domain_scores = {}
for model in models:
    domain_scores[model] = {}
    for domain in domains:
        scores = [r["composite_score"] for r in model_results[model] if r["domain"] == domain]
        domain_scores[model][domain] = np.mean(scores) if scores else 0.0

# Build matrix: rows=models, cols=domains
matrix = np.array([[domain_scores[m][d] for d in domains] for m in models])

fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')

ax.set_xticks(range(len(domains)))
ax.set_xticklabels([d.replace(" ", "\n") for d in domains], fontsize=10)
ax.set_yticks(range(len(models)))
ax.set_yticklabels(model_labels, fontsize=11)

for i in range(len(models)):
    for j in range(len(domains)):
        val = matrix[i, j]
        color = 'white' if val < 0.65 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', 
                fontsize=10, fontweight='bold', color=color)

plt.colorbar(im, ax=ax, label='Composite Score')
ax.set_title("ADAPT-IQ Domain Performance Heatmap\n(100 Scenarios × 3 Models)", 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("data/figure2_domain_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2 saved")

# ============================================================
# FIGURE 3: Discriminatory Power — Score Distribution
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for idx, (model, label, color) in enumerate(zip(models, model_labels, model_colors)):
    scores = [r["composite_score"] for r in model_results[model]]
    ax = axes[idx]
    
    bins = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.01]
    ax.hist(scores, bins=bins, color=color, edgecolor='black', alpha=0.85)
    ax.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(scores):.3f}')
    ax.set_title(f"{label}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Composite Score", fontsize=11)
    if idx == 0:
        ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(0, 1.05)

fig.suptitle("ADAPT-IQ Score Distribution by Model\n(100 Scenarios Each)", 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("data/figure3_discriminatory_power.png", dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 saved")

# ============================================================
# FIGURE 4: Difficulty Analysis
# ============================================================
difficulties = ["easy", "medium", "hard"]
diff_colors = ["#66BB6A", "#FFA726", "#EF5350"]

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(models))
width = 0.25

for i, (diff, color) in enumerate(zip(difficulties, diff_colors)):
    diff_avgs = []
    for model in models:
        scores = [r["composite_score"] for r in model_results[model] if r["difficulty"] == diff]
        diff_avgs.append(np.mean(scores) if scores else 0)
    
    bars = ax.bar(x + i * width, diff_avgs, width, label=diff.capitalize(), 
                  color=color, edgecolor='black', linewidth=0.7, alpha=0.9)
    for bar, val in zip(bars, diff_avgs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x + width)
ax.set_xticklabels(model_labels, fontsize=11)
ax.set_ylabel("Average Composite Score", fontsize=12)
ax.set_title("ADAPT-IQ Performance by Difficulty Level\n(Easy / Medium / Hard)", 
             fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.15)
ax.legend(title="Difficulty", fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("data/figure4_difficulty_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4 saved")

# ============================================================
# FIGURE 5: Benchmark Design Diagram (keep existing or update)
# ============================================================
# Keep the existing figure4_benchmark_design.png as-is (it's the design diagram)
# Just rename the new figure 4 to figure4_difficulty_analysis.png

# Print summary stats
print("\n=== FINAL SUMMARY ===")
for model, label in zip(models, model_labels):
    scores = [r["composite_score"] for r in model_results[model]]
    print(f"{label}: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}, "
          f"min={min(scores):.3f}, max={max(scores):.3f}, n={len(scores)}")

# By difficulty
print("\n=== BY DIFFICULTY ===")
for diff in difficulties:
    scores_all = [r["composite_score"] for r in results if r["difficulty"] == diff]
    print(f"{diff}: mean={np.mean(scores_all):.4f}, n={len(scores_all)}")

print("\nAll figures regenerated successfully!")
