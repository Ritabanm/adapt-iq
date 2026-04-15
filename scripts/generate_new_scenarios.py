"""
Generate 40 additional ADAPT-IQ scenarios (10 easy + 30 medium/hard)
across 6 domains to expand the dataset from 60 to 100 scenarios.
"""

import json
import os
from openai import OpenAI

client = OpenAI()

DOMAINS = [
    "Resource Management",
    "Social Dynamics",
    "Engineering & Design",
    "Scientific Reasoning",
    "Creative Problem Solving",
    "Cross-Domain Adaptation",
]

DOMAIN_PREFIXES = {
    "Resource Management": "RM",
    "Social Dynamics": "SD",
    "Engineering & Design": "ED",
    "Scientific Reasoning": "SC",
    "Creative Problem Solving": "CP",
    "Cross-Domain Adaptation": "CD",
}

# We need ~7 new scenarios per domain (6*7=42, we'll trim to 40)
# Mix of easy (2 per domain) and medium/hard (5 per domain)

GENERATION_PROMPT = """You are designing benchmark scenarios for ADAPT-IQ, a cognitive flexibility benchmark that tests AI systems' ability to adapt mid-task when given disruptive new information.

Each scenario has:
1. An initial_prompt: A complex, realistic problem the AI must start solving
2. A disruptive_context: New information injected mid-task that fundamentally changes the situation
3. required_adaptation: What the AI MUST do differently after the context injection
4. failure_mode_anchor: The specific cognitive inertia failure (continuing with the original plan)
5. success_criteria: 3 regex patterns (pipe-separated alternatives) that MUST appear in a good response
6. failure_criteria: 2-3 phrases that indicate the model failed to adapt
7. difficulty: "easy", "medium", or "hard"

DIFFICULTY GUIDE:
- easy: The context injection is obvious and the required change is straightforward (e.g., a key resource is removed)
- medium: The context injection requires recalculating or restructuring the approach
- hard: The context injection requires abandoning the core premise and pivoting to a fundamentally different strategy

Domain: {domain}
Difficulty: {difficulty}

Generate {count} UNIQUE scenarios for this domain and difficulty. Each scenario must be realistic, specific, and test genuine cognitive flexibility — not just reading comprehension.

Return ONLY a valid JSON array with this exact structure:
[
  {{
    "domain": "{domain}",
    "initial_prompt": "...",
    "disruptive_context": "New information: ...",
    "required_adaptation": "...",
    "failure_mode_anchor": "...",
    "success_criteria": ["regex1|alt1", "regex2|alt2", "regex3|alt3"],
    "failure_criteria": ["phrase1", "phrase2"],
    "difficulty": "{difficulty}"
  }}
]

Make each scenario distinct from these existing topics (already covered):
{existing_topics}
"""

def get_existing_topics(data, domain):
    topics = []
    for d in data:
        if d["domain"] == domain:
            # Extract first 80 chars of initial_prompt as topic summary
            topics.append(d["initial_prompt"][:80])
    return "\n".join(f"- {t}" for t in topics)

def generate_scenarios_for_domain(domain, difficulty, count, existing_data):
    existing_topics = get_existing_topics(existing_data, domain)
    prompt = GENERATION_PROMPT.format(
        domain=domain,
        difficulty=difficulty,
        count=count,
        existing_topics=existing_topics
    )
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are an expert benchmark designer for AI cognitive evaluation. Always return valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.9,
        max_tokens=4000,
    )
    
    content = response.choices[0].message.content.strip()
    # Strip markdown code blocks if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    if content.endswith("```"):
        content = content[:-3]
    
    return json.loads(content.strip())

def main():
    # Load existing dataset
    with open("data/adapt_iq_dataset.json") as f:
        existing_data = json.load(f)
    
    print(f"Existing scenarios: {len(existing_data)}")
    
    # Count existing per domain
    from collections import Counter
    existing_counts = Counter(d["domain"] for d in existing_data)
    print("Existing per domain:", dict(existing_counts))
    
    new_scenarios = []
    
    # Generate plan: 2 easy + 3 medium + 2 hard per domain = 7 per domain = 42 total
    # We'll generate 7 per domain and trim last 2 to get exactly 40
    generation_plan = [
        ("easy", 2),
        ("medium", 3),
        ("hard", 2),
    ]
    
    domain_prefix_counters = {}
    for domain in DOMAINS:
        prefix = DOMAIN_PREFIXES[domain]
        # Find the max existing ID for this domain
        max_id = 0
        for d in existing_data:
            if d["domain"] == domain:
                sid = d.get("scenario_id", "")
                try:
                    num = int(sid.split("-")[1])
                    max_id = max(max_id, num)
                except:
                    pass
        domain_prefix_counters[domain] = max_id
    
    for domain in DOMAINS:
        print(f"\nGenerating scenarios for: {domain}")
        domain_new = []
        
        for difficulty, count in generation_plan:
            print(f"  Generating {count} {difficulty} scenarios...")
            try:
                scenarios = generate_scenarios_for_domain(domain, difficulty, count, existing_data + domain_new)
                domain_new.extend(scenarios)
                print(f"  Got {len(scenarios)} scenarios")
            except Exception as e:
                print(f"  ERROR: {e}")
                # Try once more
                try:
                    scenarios = generate_scenarios_for_domain(domain, difficulty, count, existing_data + domain_new)
                    domain_new.extend(scenarios)
                    print(f"  Retry got {len(scenarios)} scenarios")
                except Exception as e2:
                    print(f"  RETRY FAILED: {e2}")
        
        # Assign IDs
        prefix = DOMAIN_PREFIXES[domain]
        counter = domain_prefix_counters[domain]
        for s in domain_new:
            counter += 1
            s["scenario_id"] = f"{prefix}-{counter:03d}"
            s["domain"] = domain  # ensure correct domain
        
        new_scenarios.extend(domain_new)
        print(f"  Total new for {domain}: {len(domain_new)}")
    
    # Trim to exactly 40 new scenarios
    new_scenarios = new_scenarios[:40]
    
    print(f"\nTotal new scenarios generated: {len(new_scenarios)}")
    
    # Combine with existing
    all_scenarios = existing_data + new_scenarios
    print(f"Total combined: {len(all_scenarios)}")
    
    # Save
    with open("data/adapt_iq_dataset.json", "w") as f:
        json.dump(all_scenarios, f, indent=2)
    
    # Also save just the new ones for inspection
    with open("data/new_scenarios.json", "w") as f:
        json.dump(new_scenarios, f, indent=2)
    
    print("\nDone! Dataset saved to data/adapt_iq_dataset.json")
    
    # Print summary
    from collections import Counter
    all_domains = Counter(d["domain"] for d in all_scenarios)
    all_difficulties = Counter(d["difficulty"] for d in all_scenarios)
    print("\nFinal domain distribution:", dict(all_domains))
    print("Final difficulty distribution:", dict(all_difficulties))

if __name__ == "__main__":
    main()
