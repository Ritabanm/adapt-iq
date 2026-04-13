# ADAPT-IQ: Complete Kaggle Hackathon Submission Guide

## What Has Been Built

This package contains everything needed to submit a competitive entry to the **Google DeepMind x Kaggle "Measuring Progress Toward AGI - Cognitive Abilities"** hackathon (deadline: **April 16, 2026**).

---

## Deliverables Summary

| File | Purpose | Status |
|---|---|---|
| `data/adapt_iq_dataset.json` | 60-scenario benchmark dataset | Complete |
| `data/adapt_iq_dataset.csv` | CSV version for easy inspection | Complete |
| `task.py` | Core evaluation logic (Kaggle SDK) | Complete |
| `benchmark.py` | Benchmark runner + CLI | Complete |
| `adapt_iq_evaluation_notebook.py` | Public notebook for Kaggle | Complete |
| `Kaggle_Writeup.md` | Competition writeup (≤1,500 words) | Complete |
| `data/figure1_model_comparison.png` | Results visualization | Complete |
| `data/figure2_domain_heatmap.png` | Domain breakdown heatmap | Complete |
| `data/figure3_discriminatory_power.png` | Discriminatory power chart | Complete |
| `data/figure4_benchmark_design.png` | Benchmark design infographic | Complete |

---

## Step-by-Step Submission Instructions

### Step 1: Create the Kaggle Benchmark

1. Go to [https://www.kaggle.com/benchmarks](https://www.kaggle.com/benchmarks)
2. Click **"New Benchmark"**
3. Name it: `adapt-iq-cognitive-flexibility`
4. Upload `data/adapt_iq_dataset.json` as the dataset
5. Upload `task.py` and `benchmark.py` as the task code
6. Set the benchmark to **Private** (it will be made public after the deadline)
7. Save the benchmark URL: `https://www.kaggle.com/benchmarks/<your-username>/adapt-iq-cognitive-flexibility`

### Step 2: Create the Kaggle Writeup

1. Go to the competition page: [https://www.kaggle.com/competitions/kaggle-measuring-agi](https://www.kaggle.com/competitions/kaggle-measuring-agi)
2. Click **"New Writeup"**
3. **Title:** `ADAPT-IQ: Context-Injection Creativity Test for Measuring Cognitive Flexibility`
4. **Track:** Select **"Executive Functions"** (primary)
5. Copy and paste the content from `Kaggle_Writeup.md` into the writeup editor
6. **Attachments > Project Links > Benchmark:** Add the URL from Step 1
7. **Media Gallery:** Upload `figure4_benchmark_design.png` as the cover image, then add the other figures
8. **Project Files:** Upload `adapt_iq_evaluation_notebook.py` as a public notebook
9. Click **"Submit"** (not just "Save") before the deadline

### Step 3: Verify Submission

- Confirm the writeup shows a "Submitted" status (not "Draft")
- Confirm the Kaggle Benchmark link is attached under "Project Links"
- Confirm the cover image is set in the Media Gallery

---

## What Makes This Submission Competitive

### Novelty Argument (30% of score)
ADAPT-IQ introduces a fundamentally new evaluation paradigm: the **Context-Injection Creativity Test (CICT)**. Unlike all existing benchmarks that evaluate models on static inputs, CICT forces a mid-task pivot, revealing "cognitive inertia" — a failure mode that is invisible to static benchmarks but critical for real-world AI deployment.

**Prior work and the gap:**
- **ARC (Abstraction and Reasoning Corpus):** Tests pattern recognition on novel visual puzzles. Does not test adaptation to mid-task context changes.
- **BIG-Bench:** Large collection of static tasks. No multi-turn context injection paradigm.
- **MMLU:** Pure knowledge retrieval. No adaptive reasoning required.
- **HELM:** Comprehensive static evaluation. No cognitive flexibility measurement.
- **HellaSwag / WinoGrande:** Tests commonsense reasoning. No disruptive context injection.
- **CogEval (2023):** Tests cognitive biases. Does not test adaptive improvisation.
- **GAIA (2023):** Tests real-world task completion. Does not isolate cognitive flexibility as a faculty.

**ADAPT-IQ is the first benchmark to specifically isolate the Executive Function of cognitive flexibility through controlled context injection across diverse real-world domains.**

### Dataset Quality (50% of score)
- 60 hand-crafted scenarios with verifiable, unambiguous answers
- Explicit `success_criteria` and `failure_criteria` for deterministic evaluation
- Covers 6 diverse domains to prevent domain-specific gaming
- Two difficulty levels with clear gradient
- Fully synthetic data (no contamination risk)

### Writeup Quality (20% of score)
- Follows the official template exactly
- Cites the DeepMind cognitive framework paper
- Explains the problem statement, construction, dataset, and results clearly
- Under 1,500 words

---

## Existing Research Landscape (Deep Research Summary)

The following is a summary of the deep research conducted to confirm the novelty of ADAPT-IQ:

**What already exists:**
- Benchmarks for static reasoning (MMLU, BIG-Bench, HELM)
- Benchmarks for in-context learning (few-shot prompting evaluations)
- Benchmarks for cognitive biases (CogEval)
- Benchmarks for planning (PlanBench)
- Benchmarks for instruction following (IFEval)

**What does NOT exist (the gap ADAPT-IQ fills):**
- No benchmark specifically tests **mid-task context injection** as a measure of cognitive flexibility
- No benchmark uses the **failure mode anchor** concept to detect perseveration
- No benchmark evaluates **cross-domain adaptive improvisation** (e.g., pivoting from a military strategy to a humanitarian one)
- The DeepMind cognitive framework explicitly calls out cognitive flexibility as under-measured

**Conclusion:** Even if individual components of ADAPT-IQ have been explored in isolation, the specific paradigm of Context-Injection Creativity Testing as a measure of Executive Function cognitive flexibility is novel and directly addresses the competition's stated goal.

---

## Track Alignment

| Competition Track | ADAPT-IQ Alignment |
|---|---|
| **Executive Functions (primary)** | Directly tests cognitive flexibility (switching between task rules when context changes), inhibitory control (abandoning initial solution), and working memory (holding both old and new constraints simultaneously) |
| **Learning (secondary)** | Tests belief updating — can the model update its internal state when given corrective/new information? |

---

## Scoring Breakdown

| Metric | Weight | How ADAPT-IQ Scores |
|---|---|---|
| Dataset quality & task construction | 50% | Verifiable answers, 60 scenarios, clean code, robust regex evaluation |
| Writeup quality | 20% | Follows template, cites framework paper, explains construction clearly |
| Novelty, insights, discriminatory power | 30% | First CICT benchmark, clear performance gradient (0.22 to 0.975), domain-specific insights |

---

## Contact
For any questions about this submission, refer to the Kaggle competition discussion forum or the Kaggle Benchmarks documentation.
