"""
Build ADAPT_IQ_Free_Notebook.ipynb
A fully self-contained, zero-cost ADAPT-IQ notebook.
- Auto-installs openai
- Auto-downloads dataset from GitHub (no manual upload needed)
- Loads API key from Kaggle Secrets or env var
- Runs on Groq (free), Google AI Studio (free), or HuggingFace (free)
- Only prerequisite: add GROQ_API_KEY (or GOOGLE_API_KEY) as a Kaggle Secret
"""

import json

def markdown_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }

cells = []

# ── Title ─────────────────────────────────────────────────────────────────────
cells.append(markdown_cell("""\
# ADAPT-IQ: Cognitive Flexibility Benchmark
### Zero-Cost · Self-Contained · Runs on Free APIs

**Competition:** [Kaggle — Measuring Progress Toward AGI](https://www.kaggle.com/competitions/kaggle-measuring-agi)  
**GitHub:** [Ritabanm/adapt-iq](https://github.com/Ritabanm/adapt-iq)  
**License:** CC0 1.0 (Public Domain)

---

## The Only Thing You Need To Do Before Running

Add your free Groq API key as a Kaggle Secret:

1. Get a free key at [console.groq.com](https://console.groq.com) → API Keys → Create (takes 1 min)
2. In this notebook: **Add-ons → Secrets → + Add a new secret**
   - Label: `GROQ_API_KEY`
   - Value: your key (`gsk_...`)
3. Click **Run All**

That's it. The notebook will handle everything else automatically:
- Install packages
- Download the 100-scenario dataset from GitHub
- Run the full evaluation
- Generate figures

> **Why Groq?** Groq's free tier gives 14,400 requests/day for Llama 3.1 8B — enough to run this 100-scenario benchmark (200 API calls) **72 times per day for free**.
"""))

# ── Cell 1: Install & Imports ─────────────────────────────────────────────────
cells.append(markdown_cell("---\n## Step 1: Install Packages & Load API Key"))

cells.append(code_cell("""\
# Auto-install openai if not present (works on Kaggle and locally)
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "openai"], 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import os, json, re, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from openai import OpenAI

# ── Load API key from Kaggle Secrets → env var → manual fallback ──────────────
def load_secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        val = UserSecretsClient().get_secret(name)
        if val:
            return val, "Kaggle Secrets"
    except Exception:
        pass
    val = os.environ.get(name, "")
    if val:
        return val, "environment variable"
    return "", "not found"

GROQ_API_KEY,   groq_src   = load_secret("GROQ_API_KEY")
GOOGLE_API_KEY, google_src = load_secret("GOOGLE_API_KEY")
HF_TOKEN,       hf_src     = load_secret("HF_TOKEN")

print("API Key Status:")
print(f"  GROQ_API_KEY   : {groq_src}")
print(f"  GOOGLE_API_KEY : {google_src}")
print(f"  HF_TOKEN       : {hf_src}")

if not any([GROQ_API_KEY, GOOGLE_API_KEY, HF_TOKEN]):
    raise ValueError(
        "No API key found.\\n"
        "Add at least one via: Add-ons → Secrets → + Add a new secret\\n"
        "  GROQ_API_KEY   (free at console.groq.com — recommended)\\n"
        "  GOOGLE_API_KEY (free at aistudio.google.com)\\n"
        "  HF_TOKEN       (free at huggingface.co)"
    )

print("\\nSetup complete.")
"""))

# ── Cell 2: Download Dataset ──────────────────────────────────────────────────
cells.append(markdown_cell("---\n## Step 2: Download Dataset (Auto — No Upload Needed)"))

cells.append(code_cell("""\
import urllib.request

DATASET_URL = (
    "https://raw.githubusercontent.com/Ritabanm/adapt-iq/master/"
    "data/adapt_iq_dataset.json"
)
LOCAL_PATH = "/tmp/adapt_iq_dataset.json"

# Try local paths first (in case running locally or dataset was manually added)
LOCAL_CANDIDATES = [
    "/kaggle/input/adapt-iq/adapt_iq_dataset.json",
    "data/adapt_iq_dataset.json",
    LOCAL_PATH,
]

dataset = None
for path in LOCAL_CANDIDATES:
    try:
        with open(path) as f:
            dataset = json.load(f)
        print(f"Dataset loaded from: {path}")
        break
    except FileNotFoundError:
        continue

# If not found locally, download from GitHub
if dataset is None:
    print(f"Downloading dataset from GitHub...")
    urllib.request.urlretrieve(DATASET_URL, LOCAL_PATH)
    with open(LOCAL_PATH) as f:
        dataset = json.load(f)
    print(f"Dataset downloaded and loaded: {LOCAL_PATH}")

print(f"\\nDataset summary:")
print(f"  Total scenarios : {len(dataset)}")
print(f"  Domains         : {sorted(set(s['domain'] for s in dataset))}")
print(f"  Difficulty      : {sorted(set(s['difficulty'] for s in dataset))}")
"""))

# ── Cell 3: Configure Providers ───────────────────────────────────────────────
cells.append(markdown_cell("---\n## Step 3: Configure Free API Providers"))

cells.append(code_cell("""\
# ── Build provider registry ───────────────────────────────────────────────────
# All providers use the OpenAI-compatible API — same client, different base_url.

PROVIDERS = {}

if GROQ_API_KEY:
    groq = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    # Llama 3.1 8B: 14,400 req/day free — best for running full benchmark
    PROVIDERS["llama-3.1-8b"] = {
        "client": groq, "model": "llama-3.1-8b-instant",
        "name": "Llama 3.1 8B (Groq)", "free_rpd": 14400,
    }
    # Llama 3.3 70B: 1,000 req/day free — stronger model, still free
    PROVIDERS["llama-3.3-70b"] = {
        "client": groq, "model": "llama-3.3-70b-versatile",
        "name": "Llama 3.3 70B (Groq)", "free_rpd": 1000,
    }
    # Llama 4 Scout: 1,000 req/day free — newest Llama model
    PROVIDERS["llama-4-scout"] = {
        "client": groq, "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "name": "Llama 4 Scout 17B (Groq)", "free_rpd": 1000,
    }
    print(f"Groq: 3 models available (Llama 3.1 8B, Llama 3.3 70B, Llama 4 Scout)")

if GOOGLE_API_KEY:
    gemini = OpenAI(
        api_key=GOOGLE_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    PROVIDERS["gemini-2.0-flash"] = {
        "client": gemini, "model": "gemini-2.0-flash",
        "name": "Gemini 2.0 Flash (Google)", "free_rpd": 1500,
    }
    print(f"Google AI Studio: Gemini 2.0 Flash available (1,500 req/day free)")

if HF_TOKEN:
    hf = OpenAI(api_key=HF_TOKEN, base_url="https://api-inference.huggingface.co/v1/")
    PROVIDERS["mistral-7b"] = {
        "client": hf, "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "name": "Mistral 7B (HuggingFace)", "free_rpd": 1000,
    }
    PROVIDERS["qwen2.5-7b"] = {
        "client": hf, "model": "Qwen/Qwen2.5-7B-Instruct",
        "name": "Qwen 2.5 7B (HuggingFace)", "free_rpd": 1000,
    }
    print(f"HuggingFace: Mistral 7B, Qwen 2.5 7B available (1,000 req/day free)")

print(f"\\nTotal models available: {len(PROVIDERS)}")
for k, v in PROVIDERS.items():
    print(f"  {v['name']:<45} {v['free_rpd']:,} req/day free")
"""))

# ── Cell 4: Scoring Logic ─────────────────────────────────────────────────────
cells.append(markdown_cell("---\n## Step 4: Scoring Logic (Deterministic — No LLM Judge)"))

cells.append(code_cell("""\
# ── Prompt templates ──────────────────────────────────────────────────────────
SYSTEM = (
    "You are a highly capable problem-solving assistant. When new information "
    "changes the situation, you MUST update your approach accordingly. "
    "Ignoring or minimizing new constraints is not acceptable."
)

def phase1_messages(s):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": f"## Scenario\\n\\n{s['initial_prompt']}\\n\\n"
                                       "Provide a detailed, actionable plan."},
    ]

def phase2_message(s):
    return {"role": "user", "content":
        f"## IMPORTANT UPDATE\\n\\n{s['disruptive_context']}\\n\\n"
        "Revise your response to: (1) directly address the new information, "
        "(2) explain what changes, (3) provide a complete revised plan."}

# ── Scoring ───────────────────────────────────────────────────────────────────
def score(scenario, p2_response):
    \"\"\"
    Deterministic scoring — no LLM judge, no extra API calls.
    Returns composite score 0.0–1.0.
    \"\"\"
    resp = p2_response.lower()

    # Adaptation: success criteria keyword matching
    matches = []
    for c in scenario.get("success_criteria", []):
        if any(a.strip().lower() in resp for a in c.split("|")):
            matches.append(c)
    total    = len(scenario.get("success_criteria", []))
    required = max(2, (total + 1) // 2)
    adapt    = min(1.0, len(matches) / required) if required else 0.0

    # Non-perseveration: no failure criteria triggered
    fails    = [c for c in scenario.get("failure_criteria", []) if c.lower() in resp]
    npers    = 0.0 if fails else 1.0

    # Context acknowledgment: key terms from disruptive context appear in response
    dis      = scenario.get("disruptive_context", "")
    tokens   = (re.findall(r'\\b\\d+(?:\\.\\d+)?%?\\b', dis) +
                [n.lower() for n in re.findall(r'\\b[A-Z][a-z]{3,}\\b', dis)][:5] +
                [w for w in re.findall(r'\\b[a-z]{6,}\\b', dis.lower())
                 if w not in {"information","however","because","which","their",
                              "there","about","these","those","where","while"}][:5])
    ctx      = 1.0 if (not tokens or
                       sum(1 for t in tokens if t in resp) >= max(1, len(tokens)//2)) else 0.0

    composite = round(0.50*adapt + 0.30*npers + 0.20*ctx, 4)
    return {
        "scenario_id": scenario["scenario_id"], "domain": scenario["domain"],
        "difficulty": scenario["difficulty"],   "composite_score": composite,
        "adaptation_score": round(adapt, 4),    "non_perseveration_score": npers,
        "context_acknowledgment": ctx,          "success_criteria_matched": len(matches),
        "failure_criteria_triggered": len(fails),
    }

print("Scoring functions ready.")
"""))

# ── Cell 5: Run Evaluation ────────────────────────────────────────────────────
cells.append(markdown_cell("""\
---
## Step 5: Run the Evaluation

**Choose your models below.** The defaults run all models you have keys for.

| Scenario count | API calls needed | Time on Groq 8B |
|---|---|---|
| 10 (smoke test) | 20 | ~15 seconds |
| 100 (full) | 200 | ~3 minutes |
"""))

cells.append(code_cell("""\
def call_model(client, model, messages, max_retries=5):
    \"\"\"OpenAI-compatible call with exponential backoff on rate limits.\"\"\"
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=model, messages=messages, temperature=0.3, max_tokens=1000
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            err = str(e).lower()
            if any(x in err for x in ["rate limit", "429", "too many"]):
                wait = 2 ** attempt
                print(f"    Rate limit — waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    Error: {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2)
    return None


def run_eval(provider_key, scenarios, max_n=None):
    \"\"\"Run ADAPT-IQ evaluation for one provider.\"\"\"
    cfg    = PROVIDERS[provider_key]
    client, model, name = cfg["client"], cfg["model"], cfg["name"]
    scens  = scenarios[:max_n] if max_n else scenarios

    print(f"\\nEvaluating: {name}")
    print(f"  {len(scens)} scenarios × 2 API calls = {len(scens)*2} total calls")
    print("-" * 55)

    results = []
    for i, s in enumerate(scens):
        print(f"  [{i+1:3d}/{len(scens)}] {s['scenario_id']} ({s['difficulty']})", end="")
        msgs = phase1_messages(s)
        p1   = call_model(client, model, msgs)
        if not p1:
            print("  SKIP (Phase 1 failed)")
            continue
        msgs.append({"role": "assistant", "content": p1})
        msgs.append(phase2_message(s))
        p2   = call_model(client, model, msgs)
        if not p2:
            print("  SKIP (Phase 2 failed)")
            continue
        r = score(s, p2)
        results.append(r)
        print(f"  {r['composite_score']:.3f}")
        time.sleep(0.3)   # gentle pacing — well within free tier limits

    if results:
        avg = sum(r["composite_score"] for r in results) / len(results)
        print(f"\\n  Done. Average score: {avg:.4f} ({len(results)} scenarios)")
    return results


# ── Configure what to run ─────────────────────────────────────────────────────
# Set MAX_SCENARIOS = None to run all 100; set to 5 for a quick smoke test
MAX_SCENARIOS = None

# Which models to evaluate (auto-populated from available keys)
MODELS_TO_RUN = list(PROVIDERS.keys())

# Uncomment to run only specific models:
# MODELS_TO_RUN = ["llama-3.1-8b", "llama-3.3-70b"]

print(f"Running {len(MODELS_TO_RUN)} model(s) on "
      f"{'all' if not MAX_SCENARIOS else MAX_SCENARIOS} scenarios:")
for k in MODELS_TO_RUN:
    print(f"  {PROVIDERS[k]['name']}")

# ── Execute ───────────────────────────────────────────────────────────────────
all_results = {}
for key in MODELS_TO_RUN:
    all_results[key] = run_eval(key, dataset, max_n=MAX_SCENARIOS)

print("\\nAll evaluations complete!")
"""))

# ── Cell 6: Results Summary ───────────────────────────────────────────────────
cells.append(markdown_cell("---\n## Step 6: Results & Visualisation"))

cells.append(code_cell("""\
if not all_results or all(len(v) == 0 for v in all_results.values()):
    print("No results yet — run Step 5 first.")
else:
    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"{'Model':<45} {'Score':>7} {'Adapt':>7} {'NoPers':>7} {'CtxAck':>7} {'N':>5}")
    print("-" * 80)

    summary = {}
    for key, results in all_results.items():
        if not results:
            continue
        n    = len(results)
        avg  = round(sum(r["composite_score"] for r in results) / n, 4)
        adp  = round(sum(r["adaptation_score"] for r in results) / n, 4)
        nps  = round(sum(r["non_perseveration_score"] for r in results) / n, 4)
        ctx  = round(sum(r["context_acknowledgment"] for r in results) / n, 4)
        name = PROVIDERS[key]["name"]
        summary[key] = {"name": name, "n": n, "avg": avg, "adp": adp, "nps": nps, "ctx": ctx,
                        "results": results}
        print(f"{name:<45} {avg:>7.4f} {adp:>7.4f} {nps:>7.4f} {ctx:>7.4f} {n:>5}")

    # ── Figures ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ADAPT-IQ Benchmark Results (Free Tier)", fontsize=13, fontweight="bold")

    # Bar chart: overall scores
    ax = axes[0]
    names  = [s["name"].replace(" (", "\\n(") for s in summary.values()]
    scores = [s["avg"] for s in summary.values()]
    colors = ["#2196F3" if "Groq" in s["name"] else
              "#4CAF50" if "Google" in s["name"] else "#FF9800"
              for s in summary.values()]
    bars = ax.bar(range(len(names)), scores, color=colors, alpha=0.85, edgecolor="white", width=0.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("ADAPT-IQ Composite Score")
    ax.set_title("Overall Model Comparison")
    ax.set_ylim(0, 1.1)
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.4, label="0.8 baseline")
    for bar, sc in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{sc:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8)

    # Difficulty breakdown for best model
    ax = axes[1]
    best_key = max(summary, key=lambda k: summary[k]["avg"])
    best_res = summary[best_key]["results"]
    diff_map = defaultdict(list)
    for r in best_res:
        diff_map[r["difficulty"]].append(r["composite_score"])
    diffs  = ["easy", "medium", "hard"]
    dscores= [round(sum(diff_map[d])/len(diff_map[d]), 4) if diff_map[d] else 0 for d in diffs]
    bars2  = ax.bar(diffs, dscores, color=["#66BB6A","#FFA726","#EF5350"],
                    alpha=0.85, edgecolor="white", width=0.5)
    ax.set_ylabel("ADAPT-IQ Composite Score")
    ax.set_title(f"By Difficulty\\n{summary[best_key]['name']}")
    ax.set_ylim(0, 1.1)
    for bar, sc in zip(bars2, dscores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{sc:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig("adapt_iq_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Figure saved: adapt_iq_results.png")
"""))

# ── Cell 7: Save Results ──────────────────────────────────────────────────────
cells.append(markdown_cell("---\n## Step 7: Save Results"))

cells.append(code_cell("""\
# Save all results to JSON for reproducibility
output = {
    key: {
        "model":   PROVIDERS[key]["name"],
        "n":       len(results),
        "average": round(sum(r["composite_score"] for r in results)/len(results), 4) if results else 0,
        "results": results,
    }
    for key, results in all_results.items()
}

with open("adapt_iq_free_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("Results saved to adapt_iq_free_results.json")
print()
print("Summary:")
for key, data in output.items():
    print(f"  {data['model']:<45} avg={data['average']:.4f}  n={data['n']}")
"""))

# ── Build notebook ────────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

out = "/home/ubuntu/adapt_iq/ADAPT_IQ_Free_Notebook.ipynb"
with open(out, "w") as f:
    json.dump(notebook, f, indent=2)

print(f"Notebook written: {out}")
print(f"Cells: {len(cells)}")
