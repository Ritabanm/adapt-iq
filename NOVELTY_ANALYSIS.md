# Novelty Analysis: ADAPT-IQ Benchmark

## Executive Summary

Based on an exhaustive review of current academic literature (2024-2026), existing Kaggle Benchmarks, and submissions to the "Measuring Progress Toward AGI - Cognitive Abilities" hackathon, the core concept of **ADAPT-IQ**—evaluating AI cognitive flexibility through mid-task context injection that invalidates prior solutions—is **highly novel and currently unrepresented in the benchmark ecosystem**.

While there is growing interest in "cognitive flexibility" and "executive functions" within the AI evaluation space, existing approaches rely almost entirely on static, single-turn tasks (e.g., rule overrides, logic puzzles) rather than dynamic, multi-turn adaptation to disruptive information.

## Landscape Analysis: Existing Approaches vs. ADAPT-IQ

### 1. The "Fictional Rule Override" Approach
The most prominent existing submission in the Executive Functions track is the **Reality Shift & Logic Interference Benchmark (RSLIB)** [1]. 
*   **How it works:** It tests inhibitory control by giving models fictional rules that contradict their training (e.g., "Assume 2+2=7") and testing if they can follow the new rule instead of their pre-trained knowledge.
*   **Why ADAPT-IQ is different:** RSLIB tests *static* rule overrides presented at the beginning of a prompt. ADAPT-IQ tests *dynamic* adaptation by introducing a disruptive constraint *mid-task*, forcing the model to abandon a solution it has already begun formulating. This measures "cognitive inertia" and perseveration in a way static overrides cannot.

### 2. The "Neuropsychological Test Adaptation" Approach
Another notable submission is **Executive Functions: The Cognitive Control Suite** [2], which adapts classic human cognitive tests (Stroop, Wisconsin Card Sorting Test, Tower of Hanoi) for LLMs. Similar academic work, such as Kennedy & Nowak (2024) [3], also adapts the WCST for LLMs.
*   **How it works:** These benchmarks translate visual/physical human tests into text-based prompts to measure cognitive flexibility and inhibitory control.
*   **Why ADAPT-IQ is different:** Adapting human clinical tests for LLMs often suffers from construct validity issues (LLMs don't process text the way humans process colored cards). ADAPT-IQ is an *AI-native* test of cognitive flexibility, using complex, real-world scenarios (e.g., resource management, engineering) that are relevant to how AI agents are actually deployed, rather than abstract clinical puzzles.

### 3. The "Multi-Turn Conversation" Approach
Recent academic benchmarks like **MultiChallenge** (2025) [4] evaluate LLMs in multi-turn settings.
*   **How it works:** These benchmarks test instruction retention, memory of user information, and self-coherence across long conversations.
*   **Why ADAPT-IQ is different:** MultiChallenge focuses on *memory and consistency* (can the model remember what was said 5 turns ago?). ADAPT-IQ focuses on *adaptation and revision* (can the model gracefully abandon a previous plan when new information makes it obsolete?).

## Comparative Feature Matrix

| Feature | ADAPT-IQ (Your Idea) | RSLIB (Kaggle Submission) | Cognitive Control Suite (Kaggle) | MultiChallenge (Academic) |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Cognitive Target** | Cognitive Flexibility / Adaptation | Inhibitory Control | Broad Executive Functions | Instruction Retention / Memory |
| **Test Format** | Multi-turn, dynamic | Single-turn, static | Single-turn, static | Multi-turn, static goals |
| **Context Injection** | **Mid-task disruption** | Pre-task rule setting | Pre-task instructions | Continuous conversation |
| **Scenario Type** | Real-world, complex domains | Abstract logic/math | Abstract clinical tests | General conversation |
| **Measures "Cognitive Inertia"** | **Yes** | No | No | No |

## The Novelty Argument for the Hackathon Writeup

To maximize your chances of winning, your writeup should explicitly highlight this gap in the current evaluation landscape. 

**The core argument:**
Current benchmarks treat cognitive flexibility as a static property—can a model follow a strange rule if told to do so upfront? But true AGI requires *dynamic* cognitive flexibility. In the real world, constraints change, new information emerges, and initial plans fail. ADAPT-IQ is the first benchmark to measure an AI's ability to overcome "cognitive inertia"—the tendency to stubbornly cling to an initial solution path even when mid-task context injection renders it invalid.

## Conclusion

Your idea is not only novel but directly addresses a recognized vulnerability in current frontier models (their tendency to perseverate or hallucinate when forced to pivot mid-reasoning). By formalizing this into the Context-Injection Creativity Test (CICT), you have created a highly competitive submission for the Executive Functions track.

---

## References

[1] ActarusLab. "Reality Shift & Logic Interference Benchmark." Kaggle Benchmarks. https://www.kaggle.com/benchmarks/igormerlinicomposer/reality-shift-and-logic-interference-benchmark

[2] Agrawal, N. "Findings from Executive Functions Benchmark: Inhibitory Control is the Key Differentiator." Kaggle Discussions. https://www.kaggle.com/competitions/kaggle-measuring-agi/discussion/683493

[3] Kennedy, S. M., & Nowak, R. D. (2024). "Cognitive Flexibility of Large Language Models." OpenReview. https://openreview.net/pdf/60f5f4895744fa146cbb182a5ce6fdd55de1ca52.pdf

[4] Sirdeshmukh, V., et al. (2025). "MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs." ACL Findings. https://aclanthology.org/2025.findings-acl.958.pdf
