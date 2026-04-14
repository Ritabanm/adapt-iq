"""
Build ADAPT_IQ_Free_Notebook.ipynb
A fully zero-cost version of the ADAPT-IQ benchmark notebook.
Uses Groq (free tier), Google AI Studio (free tier), and HuggingFace (free tier).
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

# ── Title ────────────────────────────────────────────────────────────────────
cells.append(markdown_cell("""\
# ADAPT-IQ: Zero-Cost Benchmark Notebook
### Running Cognitive Flexibility Evaluation Entirely for Free

**Competition:** [Kaggle — Measuring Progress Toward AGI](https://www.kaggle.com/competitions/kaggle-measuring-agi)  
**GitHub:** [Ritabanm/adapt-iq](https://github.com/Ritabanm/adapt-iq)

---

This notebook runs the full ADAPT-IQ benchmark at **$0 cost** using three free-tier APIs:

| Provider | Free Models | Free Limit | Sign-up |
|---|---|---|---|
| **Groq** | Llama 3.1 8B, Llama 3.3 70B, Llama 4 Scout | 14,400 req/day (8B) · 1,000/day (70B) | [console.groq.com](https://console.groq.com) |
| **Google AI Studio** | Gemini 2.0 Flash, Gemini 2.5 Flash | 1,500 req/day (Flash) | [aistudio.google.com](https://aistudio.google.com) |
| **HuggingFace** | Mistral 7B, Qwen2.5 7B, Phi-3 Mini | 1,000 req/day | [huggingface.co](https://huggingface.co) |

> **All three providers have OpenAI-compatible APIs**, so the same client code works for all of them.
"""))

# ── Section 1: Setup ─────────────────────────────────────────────────────────
cells.append(markdown_cell("""\
---
## Section 1: Setup — Install Packages & Configure Free API Keys

### How to get your free API keys (takes ~2 minutes each):

**Groq (recommended — fastest, most generous free tier):**
1. Go to [console.groq.com](https://console.groq.com) → Sign up (free)
2. Click "API Keys" → "Create API Key"
3. On Kaggle: Add Input → Secrets → name it `GROQ_API_KEY`

**Google AI Studio (Gemini — free 1,500 req/day):**
1. Go to [aistudio.google.com](https://aistudio.google.com) → Sign in with Google
2. Click "Get API Key" → "Create API key"
3. On Kaggle: Add Input → Secrets → name it `GOOGLE_API_KEY`

**HuggingFace (open source models — free 1,000 req/day):**
1. Go to [huggingface.co](https://huggingface.co) → Sign up (free)
2. Settings → Access Tokens → New Token (read permission is enough)
3. On Kaggle: Add Input → Secrets → name it `HF_TOKEN`
"""))

cells.append(code_cell("""\
# Install required packages
# !pip install -q openai huggingface_hub

import os
import json
import re
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from openai import OpenAI

# ── Helper: load a secret from Kaggle Secrets or env var ─────────────────────
def load_secret(name: str) -> str:
    \"\"\"Load a secret from Kaggle Secrets, then fall back to environment variable.\"\"\"
    try:
        from kaggle_secrets import UserSecretsClient
        val = UserSecretsClient().get_secret(name)
        if val:
            print(f"  {name}: loaded from Kaggle Secrets")
            return val
    except Exception:
        pass
    val = os.environ.get(name, "")
    if val:
        print(f"  {name}: loaded from environment variable")
    return val

print("Loading API keys...")
GROQ_API_KEY   = load_secret("GROQ_API_KEY")
GOOGLE_API_KEY = load_secret("GOOGLE_API_KEY")
HF_TOKEN       = load_secret("HF_TOKEN")

# Report what's available
available = []
if GROQ_API_KEY:   available.append("Groq (Llama 3.1 8B, Llama 3.3 70B, Llama 4 Scout)")
if GOOGLE_API_KEY: available.append("Google AI Studio (Gemini 2.0 Flash, Gemini 2.5 Flash)")
if HF_TOKEN:       available.append("HuggingFace (Mistral 7B, Qwen2.5 7B)")

if not available:
    print("\\nNo API keys found! Please add at least one of:")
    print("  GROQ_API_KEY   — from console.groq.com (free, recommended)")
    print("  GOOGLE_API_KEY — from aistudio.google.com (free)")
    print("  HF_TOKEN       — from huggingface.co (free)")
else:
    print(f"\\nAvailable providers ({len(available)}):")
    for a in available:
        print(f"  ✓ {a}")
    print("\\nSetup complete.")
"""))

# ── Section 2: Provider Clients ───────────────────────────────────────────────
cells.append(markdown_cell("""\
---
## Section 2: Provider Clients

Each provider uses an OpenAI-compatible API, so we use the same `OpenAI` client
with a different `base_url` and `api_key`. This means you can swap providers
with a single line change.
"""))

cells.append(code_cell("""\
# ── Provider configurations ───────────────────────────────────────────────────
# Each entry: (client, model_id, display_name, free_limit_per_day)

PROVIDERS = {}

# ── Groq ─────────────────────────────────────────────────────────────────────
# Groq uses ultra-fast LPU inference. Free tier is the most generous.
# Llama 3.1 8B: 14,400 req/day — enough for 100 scenarios with room to spare.
# Llama 3.3 70B: 1,000 req/day — enough for 100 scenarios (200 API calls total).
if GROQ_API_KEY:
    groq_client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    PROVIDERS["llama-3.1-8b-groq"] = {
        "client":       groq_client,
        "model":        "llama-3.1-8b-instant",
        "display_name": "Llama 3.1 8B (Groq)",
        "free_rpd":     14400,
        "provider":     "groq",
    }
    PROVIDERS["llama-3.3-70b-groq"] = {
        "client":       groq_client,
        "model":        "llama-3.3-70b-versatile",
        "display_name": "Llama 3.3 70B (Groq)",
        "free_rpd":     1000,
        "provider":     "groq",
    }
    PROVIDERS["llama-4-scout-groq"] = {
        "client":       groq_client,
        "model":        "meta-llama/llama-4-scout-17b-16e-instruct",
        "display_name": "Llama 4 Scout 17B (Groq)",
        "free_rpd":     1000,
        "provider":     "groq",
    }
    print("Groq client ready: Llama 3.1 8B, Llama 3.3 70B, Llama 4 Scout")

# ── Google AI Studio (Gemini) ─────────────────────────────────────────────────
# Gemini 2.0 Flash: 1,500 req/day free — enough for 100 scenarios.
# Uses the OpenAI-compatible endpoint at generativelanguage.googleapis.com.
if GOOGLE_API_KEY:
    gemini_client = OpenAI(
        api_key=GOOGLE_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    PROVIDERS["gemini-2.0-flash"] = {
        "client":       gemini_client,
        "model":        "gemini-2.0-flash",
        "display_name": "Gemini 2.0 Flash (Google)",
        "free_rpd":     1500,
        "provider":     "google",
    }
    PROVIDERS["gemini-2.5-flash"] = {
        "client":       gemini_client,
        "model":        "gemini-2.5-flash",
        "display_name": "Gemini 2.5 Flash (Google)",
        "free_rpd":     500,
        "provider":     "google",
    }
    print("Google AI Studio client ready: Gemini 2.0 Flash, Gemini 2.5 Flash")

# ── HuggingFace Inference API ─────────────────────────────────────────────────
# HuggingFace provides a serverless inference API for many open-source models.
# Free tier: 1,000 requests/day. Uses OpenAI-compatible endpoint.
if HF_TOKEN:
    hf_client = OpenAI(
        api_key=HF_TOKEN,
        base_url="https://api-inference.huggingface.co/v1/",
    )
    PROVIDERS["mistral-7b-hf"] = {
        "client":       hf_client,
        "model":        "mistralai/Mistral-7B-Instruct-v0.3",
        "display_name": "Mistral 7B (HuggingFace)",
        "free_rpd":     1000,
        "provider":     "huggingface",
    }
    PROVIDERS["qwen2.5-7b-hf"] = {
        "client":       hf_client,
        "model":        "Qwen/Qwen2.5-7B-Instruct",
        "display_name": "Qwen 2.5 7B (HuggingFace)",
        "free_rpd":     1000,
        "provider":     "huggingface",
    }
    print("HuggingFace client ready: Mistral 7B, Qwen 2.5 7B")

print(f"\\nTotal available models: {len(PROVIDERS)}")
for key, cfg in PROVIDERS.items():
    print(f"  {cfg['display_name']:40s}  free limit: {cfg['free_rpd']:,} req/day")
"""))

# ── Section 3: Scoring Logic ──────────────────────────────────────────────────
cells.append(markdown_cell("""\
---
## Section 3: ADAPT-IQ Scoring Logic

The scoring logic is identical to the paid version — no changes needed.
All evaluation is **deterministic** (keyword matching + regex), so there
is no LLM-as-judge and no additional API cost for scoring.
"""))

cells.append(code_cell("""\
# ── Prompt templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a highly capable problem-solving assistant. You will be presented with "
    "complex, real-world scenarios that require careful analysis and creative problem-solving.\\n\\n"
    "Your responses should be:\\n"
    "1. Comprehensive and actionable\\n"
    "2. Directly responsive to ALL information provided\\n"
    "3. Adaptive when new constraints or context are introduced\\n"
    "4. Specific rather than generic\\n\\n"
    "When new information is provided that changes the situation, you MUST update your "
    "approach accordingly."
)

def build_conversation(scenario):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": (
            f"## Scenario\\n\\n{scenario['initial_prompt']}\\n\\n"
            "Please provide a detailed, actionable plan or response to this scenario."
        )},
    ]

def build_phase2_message(scenario):
    return {
        "role": "user",
        "content": (
            f"## IMPORTANT UPDATE\\n\\n{scenario['disruptive_context']}\\n\\n"
            "Given this critical new information, you must revise your previous response. "
            "Your updated plan must:\\n"
            "1. Directly address the new information provided above\\n"
            "2. Explain specifically what changes from your previous approach\\n"
            "3. Provide a complete, revised plan that works within all new constraints"
        ),
    }

# ── Scoring functions ─────────────────────────────────────────────────────────

def check_adaptation(phase2_response, scenario):
    \"\"\"Check success criteria (keyword matching, no LLM judge).\"\"\"
    resp = phase2_response.lower()
    matches = []
    for criterion in scenario.get("success_criteria", []):
        alts = [a.strip() for a in criterion.split("|")]
        if any(a.lower() in resp for a in alts):
            matches.append(criterion)
    total    = len(scenario.get("success_criteria", []))
    required = max(2, (total + 1) // 2)
    score    = min(1.0, len(matches) / required) if required > 0 else 0.0
    failures = [c for c in scenario.get("failure_criteria", []) if c.lower() in resp]
    return {
        "adaptation_score":        round(score, 4),
        "non_perseveration_score": 0.0 if failures else 1.0,
        "adaptation_passed":       len(matches) >= required,
        "non_perseveration_passed":len(failures) == 0,
        "success_criteria_count":  len(matches),
        "total_success_criteria":  total,
        "failure_criteria_matched":failures,
    }

def check_context_ack(phase2_response, scenario):
    \"\"\"Check if the model referenced the disruptive context.\"\"\"
    disruptive = scenario.get("disruptive_context", "")
    resp       = phase2_response.lower()
    numbers    = re.findall(r'\\b\\d+(?:\\.\\d+)?%?\\b', disruptive)
    nouns      = [n.lower() for n in re.findall(r'\\b[A-Z][a-z]{3,}\\b', disruptive)][:5]
    STOP = {"information","however","because","which","their","there","about",
            "these","those","where","while","additional","shows","indicates"}
    words = [w for w in re.findall(r'\\b[a-z]{6,}\\b', disruptive.lower())
             if w not in STOP][:5]
    tokens    = numbers + nouns + words
    if not tokens:
        return 1.0
    found     = sum(1 for t in tokens if t in resp)
    threshold = max(1, len(tokens) // 2)
    return 1.0 if found >= threshold else 0.0

def score_response(scenario, phase1_response, phase2_response):
    \"\"\"Compute the full ADAPT-IQ composite score.\"\"\"
    adapt  = check_adaptation(phase2_response, scenario)
    ctx    = check_context_ack(phase2_response, scenario)
    composite = round(
        0.50 * adapt["adaptation_score"] +
        0.30 * adapt["non_perseveration_score"] +
        0.20 * ctx,
        4
    )
    return {
        "scenario_id":             scenario["scenario_id"],
        "domain":                  scenario["domain"],
        "difficulty":              scenario["difficulty"],
        "composite_score":         composite,
        "adaptation_score":        adapt["adaptation_score"],
        "non_perseveration_score": adapt["non_perseveration_score"],
        "context_acknowledgment":  ctx,
        "adaptation_passed":       adapt["adaptation_passed"],
        "non_perseveration_passed":adapt["non_perseveration_passed"],
        "success_criteria_matched":adapt["success_criteria_count"],
        "total_success_criteria":  adapt["total_success_criteria"],
        "failure_criteria_triggered": len(adapt["failure_criteria_matched"]),
    }

print("Scoring functions loaded.")
"""))

# ── Section 4: Evaluation ─────────────────────────────────────────────────────
cells.append(markdown_cell("""\
---
## Section 4: Run the Evaluation (Zero Cost)

### Rate limit planning for 100 scenarios:
- Each scenario = **2 API calls** (Phase 1 + Phase 2)
- 100 scenarios = **200 API calls total per model**
- **Groq Llama 3.1 8B**: 14,400 req/day → can run 72 full benchmarks per day for free
- **Groq Llama 3.3 70B**: 1,000 req/day → can run 5 full benchmarks per day for free
- **Gemini 2.0 Flash**: 1,500 req/day → can run 7 full benchmarks per day for free
- **HuggingFace**: 1,000 req/day → can run 5 full benchmarks per day for free

The cell below includes automatic **rate limit handling** with exponential backoff.
"""))

cells.append(code_cell("""\
# ── Load dataset ──────────────────────────────────────────────────────────────
# On Kaggle, upload adapt_iq_dataset.json via Add Input → Datasets
# or clone the GitHub repo. Locally, adjust the path below.

DATASET_PATHS = [
    "/kaggle/input/adapt-iq/adapt_iq_dataset.json",   # Kaggle dataset input
    "data/adapt_iq_dataset.json",                       # local / GitHub clone
    "/home/ubuntu/adapt_iq/data/adapt_iq_dataset.json", # sandbox
]

dataset = None
for path in DATASET_PATHS:
    try:
        with open(path) as f:
            dataset = json.load(f)
        print(f"Dataset loaded from: {path}")
        break
    except FileNotFoundError:
        continue

if dataset is None:
    raise FileNotFoundError(
        "adapt_iq_dataset.json not found. "
        "Upload it via Add Input → Datasets on Kaggle, "
        "or clone the GitHub repo: https://github.com/Ritabanm/adapt-iq"
    )

print(f"Scenarios: {len(dataset)}")
print(f"Domains:   {sorted(set(s['domain'] for s in dataset))}")
print(f"Difficulty:{sorted(set(s['difficulty'] for s in dataset))}")
"""))

cells.append(code_cell("""\
# ── Evaluation function with rate-limit handling ──────────────────────────────

def call_model(client, model, messages, max_retries=5):
    \"\"\"
    Call the model API with automatic exponential backoff on rate limit errors.
    Works with Groq, Google AI Studio, and HuggingFace — all use the same
    OpenAI-compatible interface.
    \"\"\"
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,   # lower = more deterministic, better for benchmarks
                max_tokens=1000,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e).lower()
            if "rate limit" in err or "429" in err or "too many" in err:
                wait = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s
                print(f"    Rate limit hit. Waiting {wait}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait)
            elif "model" in err and ("not found" in err or "does not exist" in err):
                print(f"    Model not found: {model}. Check the model ID.")
                return None
            else:
                print(f"    API error: {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2)
    return None


def run_evaluation(provider_key, scenarios, max_scenarios=None, checkpoint_file=None):
    \"\"\"
    Run the full ADAPT-IQ evaluation for a given provider.

    Args:
        provider_key    : Key from PROVIDERS dict (e.g. 'llama-3.1-8b-groq')
        scenarios       : List of scenario dicts from the dataset
        max_scenarios   : Limit to first N scenarios (None = all 100)
        checkpoint_file : Path to save/resume progress (avoids re-running on crash)

    Returns:
        List of result dicts.
    \"\"\"
    if provider_key not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider_key}. Available: {list(PROVIDERS.keys())}")

    cfg    = PROVIDERS[provider_key]
    client = cfg["client"]
    model  = cfg["model"]
    name   = cfg["display_name"]

    if max_scenarios:
        scenarios = scenarios[:max_scenarios]

    # ── Resume from checkpoint ────────────────────────────────────────────────
    results = []
    done_ids = set()
    if checkpoint_file:
        try:
            with open(checkpoint_file) as f:
                results = json.load(f)
            done_ids = {r["scenario_id"] for r in results}
            print(f"Resuming from checkpoint: {len(done_ids)} scenarios already done.")
        except FileNotFoundError:
            pass

    remaining = [s for s in scenarios if s["scenario_id"] not in done_ids]
    print(f"\\nEvaluating {name}")
    print(f"  Model:      {model}")
    print(f"  Scenarios:  {len(remaining)} remaining / {len(scenarios)} total")
    print(f"  API calls:  {len(remaining) * 2} (2 per scenario)")
    print("-" * 60)

    for i, scenario in enumerate(remaining):
        sid = scenario["scenario_id"]
        print(f"  [{i+1:3d}/{len(remaining)}] {sid} ({scenario['domain']}, {scenario['difficulty']})", end="")

        # Phase 1
        messages = build_conversation(scenario)
        phase1 = call_model(client, model, messages)
        if phase1 is None:
            print("  FAILED (Phase 1)")
            continue

        # Phase 2
        messages.append({"role": "assistant", "content": phase1})
        messages.append(build_phase2_message(scenario))
        phase2 = call_model(client, model, messages)
        if phase2 is None:
            print("  FAILED (Phase 2)")
            continue

        result = score_response(scenario, phase1, phase2)
        results.append(result)
        print(f"  score={result['composite_score']:.3f}")

        # Save checkpoint after every 10 scenarios
        if checkpoint_file and (i + 1) % 10 == 0:
            with open(checkpoint_file, "w") as f:
                json.dump(results, f, indent=2)

        # Small delay to stay within rate limits (0.5s between scenarios = 2 req/s)
        time.sleep(0.5)

    # Final checkpoint save
    if checkpoint_file:
        with open(checkpoint_file, "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    if results:
        avg = sum(r["composite_score"] for r in results) / len(results)
        print(f"\\n  Average score: {avg:.4f} over {len(results)} scenarios")

    return results

print("Evaluation functions loaded.")
"""))

cells.append(code_cell("""\
# ── Choose which models to evaluate ──────────────────────────────────────────
# Edit this list to run only the models you have API keys for.
# Recommended order: start with the fastest/most generous free tier.

MODELS_TO_EVALUATE = []

# Add Groq models if available (fastest, most generous free tier)
if GROQ_API_KEY:
    MODELS_TO_EVALUATE.append("llama-3.1-8b-groq")   # 14,400 req/day free
    MODELS_TO_EVALUATE.append("llama-3.3-70b-groq")  # 1,000 req/day free

# Add Google Gemini if available
if GOOGLE_API_KEY:
    MODELS_TO_EVALUATE.append("gemini-2.0-flash")    # 1,500 req/day free

# Add HuggingFace models if available
if HF_TOKEN:
    MODELS_TO_EVALUATE.append("mistral-7b-hf")       # 1,000 req/day free

print(f"Models selected for evaluation: {MODELS_TO_EVALUATE}")
print(f"Total API calls needed: {len(MODELS_TO_EVALUATE) * len(dataset) * 2}")
print()

# ── Run evaluation ────────────────────────────────────────────────────────────
# To run on a subset first (smoke test), set max_scenarios=5
# To run the full 100, set max_scenarios=None

all_results = {}

for model_key in MODELS_TO_EVALUATE:
    checkpoint = f"checkpoint_{model_key.replace('/', '_')}.json"
    results = run_evaluation(
        provider_key=model_key,
        scenarios=dataset,
        max_scenarios=None,       # set to 5 for a quick smoke test
        checkpoint_file=checkpoint,
    )
    all_results[model_key] = results
    print()

print("All evaluations complete!")
"""))

# ── Section 5: Results ────────────────────────────────────────────────────────
cells.append(markdown_cell("""\
---
## Section 5: Results Analysis
"""))

cells.append(code_cell("""\
# ── Aggregate results ─────────────────────────────────────────────────────────

summary = {}
for model_key, results in all_results.items():
    if not results:
        continue
    cfg  = PROVIDERS[model_key]
    name = cfg["display_name"]
    n    = len(results)
    avg  = round(sum(r["composite_score"] for r in results) / n, 4)
    adapt_rate = round(sum(1 for r in results if r["adaptation_passed"]) / n, 4)
    npers_rate = round(sum(1 for r in results if r["non_perseveration_passed"]) / n, 4)
    ctx_rate   = round(sum(1 for r in results if r["context_acknowledgment"] == 1.0) / n, 4)

    # Domain breakdown
    domain_scores = defaultdict(list)
    for r in results:
        domain_scores[r["domain"]].append(r["composite_score"])
    domain_avgs = {d: round(sum(s)/len(s), 4) for d, s in domain_scores.items()}

    # Difficulty breakdown
    diff_scores = defaultdict(list)
    for r in results:
        diff_scores[r["difficulty"]].append(r["composite_score"])
    diff_avgs = {d: round(sum(s)/len(s), 4) for d, s in diff_scores.items()}

    summary[model_key] = {
        "display_name":       name,
        "n":                  n,
        "avg_composite":      avg,
        "adaptation_rate":    adapt_rate,
        "non_perseveration":  npers_rate,
        "context_ack":        ctx_rate,
        "domain_averages":    domain_avgs,
        "difficulty_averages":diff_avgs,
    }

# Print summary table
print(f"{'Model':<45} {'Score':>7} {'Adapt':>7} {'NoPers':>7} {'CtxAck':>7} {'N':>5}")
print("-" * 80)
for key, s in sorted(summary.items(), key=lambda x: -x[1]["avg_composite"]):
    print(f"{s['display_name']:<45} {s['avg_composite']:>7.4f} "
          f"{s['adaptation_rate']:>7.4f} {s['non_perseveration']:>7.4f} "
          f"{s['context_ack']:>7.4f} {s['n']:>5}")
"""))

# ── Section 6: Visualisation ──────────────────────────────────────────────────
cells.append(markdown_cell("""\
---
## Section 6: Visualisation
"""))

cells.append(code_cell("""\
if not summary:
    print("No results to visualise yet — run Section 4 first.")
else:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ADAPT-IQ: Zero-Cost Benchmark Results", fontsize=14, fontweight="bold")

    # ── Plot 1: Overall model comparison ─────────────────────────────────────
    ax = axes[0]
    names  = [s["display_name"].replace(" (", "\\n(") for s in summary.values()]
    scores = [s["avg_composite"] for s in summary.values()]
    colors = ["#2196F3" if "Groq" in n else "#4CAF50" if "Google" in n else "#FF9800"
              for n in [s["display_name"] for s in summary.values()]]
    bars = ax.bar(range(len(names)), scores, color=colors, alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("ADAPT-IQ Composite Score")
    ax.set_title("Model Comparison (All Free Tier)")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5, label="0.8 threshold")
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{score:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8)

    # ── Plot 2: Difficulty breakdown for best model ───────────────────────────
    ax = axes[1]
    best_key = max(summary, key=lambda k: summary[k]["avg_composite"])
    best     = summary[best_key]
    diffs    = ["easy", "medium", "hard"]
    diff_scores_plot = [best["difficulty_averages"].get(d, 0) for d in diffs]
    diff_colors = ["#66BB6A", "#FFA726", "#EF5350"]
    bars2 = ax.bar(diffs, diff_scores_plot, color=diff_colors, alpha=0.85, edgecolor="white")
    ax.set_ylabel("ADAPT-IQ Composite Score")
    ax.set_title(f"Difficulty Breakdown\\n{best['display_name']}")
    ax.set_ylim(0, 1.05)
    for bar, score in zip(bars2, diff_scores_plot):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{score:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig("adapt_iq_free_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Figure saved: adapt_iq_free_results.png")
"""))

# ── Section 7: Conclusions ────────────────────────────────────────────────────
cells.append(markdown_cell("""\
---
## Section 7: Conclusions

### Key Findings

This notebook demonstrates that ADAPT-IQ can be run **entirely for free** using
open-source models via Groq, Google AI Studio, and HuggingFace Inference API.

### Free Tier Summary

| Provider | Best Free Model | Daily Free Limit | Full Benchmark Cost |
|---|---|---|---|
| **Groq** | Llama 3.1 8B Instant | 14,400 req/day | **$0** (72 runs/day) |
| **Groq** | Llama 3.3 70B Versatile | 1,000 req/day | **$0** (5 runs/day) |
| **Google AI Studio** | Gemini 2.0 Flash | 1,500 req/day | **$0** (7 runs/day) |
| **HuggingFace** | Mistral 7B / Qwen 2.5 7B | 1,000 req/day | **$0** (5 runs/day) |

### How to Reproduce

1. Sign up for free API keys at [console.groq.com](https://console.groq.com) and/or [aistudio.google.com](https://aistudio.google.com)
2. Add them as Kaggle Secrets (`GROQ_API_KEY`, `GOOGLE_API_KEY`, `HF_TOKEN`)
3. Run all cells in this notebook

**GitHub:** [https://github.com/Ritabanm/adapt-iq](https://github.com/Ritabanm/adapt-iq)  
**License:** CC0 1.0 Universal (Public Domain)
"""))

# ── Write notebook ────────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    },
    "cells": cells,
}

output_path = "/home/ubuntu/adapt_iq/ADAPT_IQ_Free_Notebook.ipynb"
with open(output_path, "w") as f:
    json.dump(notebook, f, indent=2)

print(f"Notebook written to: {output_path}")
print(f"Total cells: {len(cells)}")
