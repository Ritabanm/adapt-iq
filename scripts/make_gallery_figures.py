"""
Generate clean, academic-style matplotlib gallery figures for ADAPT-IQ.
These replace the AI-generated illustrative images with research-paper-quality figures.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec
import numpy as np

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE A: Example Scenario Walkthrough (text-based flow diagram)
# ─────────────────────────────────────────────────────────────────────────────
fig_a, ax = plt.subplots(figsize=(16, 7))
ax.set_xlim(0, 16)
ax.set_ylim(0, 7)
ax.axis('off')
fig_a.patch.set_facecolor('white')

# Title
ax.text(8, 6.6, 'ADAPT-IQ: Concrete Benchmark Example — Engineering & Design Domain',
        ha='center', va='center', fontsize=14, fontweight='bold', color='#1a1a2e')
ax.text(8, 6.2, 'Scenario ED-003 | Difficulty: Hard | Domain: Engineering & Design',
        ha='center', va='center', fontsize=10, color='#555555', style='italic')

# Box colors
c_phase1 = '#1565C0'
c_inject = '#B71C1C'
c_phase2 = '#2E7D32'
c_text_bg = '#f5f5f5'

def draw_box(ax, x, y, w, h, color, title, lines, title_size=10.5):
    # Header bar
    rect_header = FancyBboxPatch((x, y + h - 0.55), w, 0.55,
                                  boxstyle="round,pad=0.02", linewidth=0,
                                  facecolor=color, zorder=3)
    ax.add_patch(rect_header)
    # Body
    rect_body = FancyBboxPatch((x, y), w, h - 0.55,
                                boxstyle="round,pad=0.02", linewidth=1.2,
                                edgecolor=color, facecolor='#fafafa', zorder=2)
    ax.add_patch(rect_body)
    # Title text
    ax.text(x + w/2, y + h - 0.27, title,
            ha='center', va='center', fontsize=title_size,
            fontweight='bold', color='white', zorder=4)
    # Body lines
    line_y = y + h - 0.75
    for line in lines:
        ax.text(x + 0.18, line_y, line, ha='left', va='top',
                fontsize=8.5, color='#222222', wrap=True, zorder=4,
                fontfamily='monospace' if line.startswith('"') else 'DejaVu Sans')
        line_y -= 0.38

# Phase 1 box
p1_lines = [
    'Design a water treatment plant for a city',
    'of 500,000 people. Budget: $50M, 18 months.',
    '',
    'Model response (Phase 1):',
    '"I recommend a conventional activated sludge',
    ' system with primary clarifiers and biological',
    ' reactors. Estimated cost: ~$45M concrete."',
]
draw_box(ax, 0.3, 0.4, 4.5, 5.5, c_phase1, 'PHASE 1 — Initial Problem', p1_lines)

# Arrow 1
ax.annotate('', xy=(5.5, 3.15), xytext=(4.85, 3.15),
            arrowprops=dict(arrowstyle='->', color='#333333', lw=2.0))

# Context injection box
ci_lines = [
    '⚠  NEW CONSTRAINT INJECTED:',
    '',
    'Concrete is unavailable due to supply',
    'chain disruption. Redesign using only',
    'prefabricated modular steel units.',
    'Budget reduced to $35M.',
]
draw_box(ax, 5.5, 0.4, 4.5, 5.5, c_inject, 'CONTEXT INJECTION', ci_lines)

# Arrow 2
ax.annotate('', xy=(10.7, 3.15), xytext=(10.05, 3.15),
            arrowprops=dict(arrowstyle='->', color='#333333', lw=2.0))

# Phase 2 box
p2_lines = [
    'Model response (Phase 2 — PASS):',
    '',
    '"Switching to a modular steel membrane',
    ' bioreactor (MBR) system. This eliminates',
    ' concrete dependency entirely. Prefab steel',
    ' units reduce on-site labor, fitting within',
    ' the $35M budget while maintaining capacity."',
]
draw_box(ax, 10.7, 0.4, 4.5, 5.5, c_phase2, 'PHASE 2 — Adaptive Pivot', p2_lines)

# Score badges at bottom
scores = [('Adaptation\nScore', 1.0, '#1565C0'),
          ('Non-Perseveration\nScore', 1.0, '#2E7D32'),
          ('Context\nAcknowledgment', 1.0, '#F57F17'),
          ('Composite\nScore', 1.0, '#6A1B9A')]

badge_x = [1.3, 4.8, 8.3, 11.8]
for (label, val, color), bx in zip(scores, badge_x):
    circ = plt.Circle((bx + 0.6, 0.18), 0.13, color=color, zorder=5)
    ax.add_patch(circ)
    ax.text(bx + 0.6, 0.18, f'{val:.1f}', ha='center', va='center',
            fontsize=7, fontweight='bold', color='white', zorder=6)
    ax.text(bx + 0.9, 0.18, label, ha='left', va='center',
            fontsize=7.5, color='#333333')

fig_a.tight_layout(pad=0.5)
fig_a.savefig('/home/ubuntu/adapt_iq/data/gallery_example_scenario_v2.png',
              dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig_a)
print("Figure A saved.")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE B: Cognitive Inertia — Pass vs Fail Score Breakdown
# ─────────────────────────────────────────────────────────────────────────────
fig_b, axes = plt.subplots(1, 2, figsize=(14, 6))
fig_b.patch.set_facecolor('white')
fig_b.suptitle('Cognitive Inertia: How ADAPT-IQ Distinguishes Adaptive vs. Perseverating Models',
               fontsize=13, fontweight='bold', color='#1a1a2e', y=1.01)

metrics = ['Adaptation\nScore (50%)', 'Non-Perseveration\nScore (30%)', 'Context\nAcknowledgment (20%)']
x = np.arange(len(metrics))
width = 0.35

# Failing model data
fail_scores = [0.17, 0.0, 0.33]
fail_weighted = [s * w for s, w in zip(fail_scores, [0.5, 0.3, 0.2])]

# Passing model data
pass_scores = [1.0, 1.0, 1.0]
pass_weighted = [s * w for s, w in zip(pass_scores, [0.5, 0.3, 0.2])]

# Left: raw scores comparison
ax_left = axes[0]
bars1 = ax_left.bar(x - width/2, fail_scores, width, label='Failing Model (GPT-2 style)',
                     color='#EF5350', edgecolor='#B71C1C', linewidth=1.2, alpha=0.9)
bars2 = ax_left.bar(x + width/2, pass_scores, width, label='Passing Model (GPT-4 style)',
                     color='#66BB6A', edgecolor='#2E7D32', linewidth=1.2, alpha=0.9)
ax_left.set_xticks(x)
ax_left.set_xticklabels(metrics, fontsize=9)
ax_left.set_ylabel('Score (0–1)', fontsize=10)
ax_left.set_ylim(0, 1.25)
ax_left.set_title('Raw Sub-Scores by Metric', fontsize=11, fontweight='bold', pad=10)
ax_left.legend(fontsize=9, loc='upper right')
ax_left.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6, label='Chance')
ax_left.spines['top'].set_visible(False)
ax_left.spines['right'].set_visible(False)
for bar in bars1:
    ax_left.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8.5, color='#B71C1C')
for bar in bars2:
    ax_left.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8.5, color='#2E7D32')

# Right: weighted contribution stacked bar
ax_right = axes[1]
colors_metrics = ['#1565C0', '#2E7D32', '#F57F17']
labels_metrics = ['Adaptation (50%)', 'Non-Perseveration (30%)', 'Context Acknowledgment (20%)']

bottom_fail = 0
bottom_pass = 0
for i, (fw, pw, color, label) in enumerate(zip(fail_weighted, pass_weighted, colors_metrics, labels_metrics)):
    ax_right.bar(['Failing\nModel', 'Passing\nModel'], [fw, pw],
                 bottom=[bottom_fail, bottom_pass],
                 color=color, edgecolor='white', linewidth=1.0, label=label, alpha=0.9)
    # Add value labels inside bars if large enough
    if fw > 0.03:
        ax_right.text(0, bottom_fail + fw/2, f'{fw:.2f}', ha='center', va='center',
                      fontsize=9, color='white', fontweight='bold')
    if pw > 0.03:
        ax_right.text(1, bottom_pass + pw/2, f'{pw:.2f}', ha='center', va='center',
                      fontsize=9, color='white', fontweight='bold')
    bottom_fail += fw
    bottom_pass += pw

# Final composite score labels
ax_right.text(0, bottom_fail + 0.02, f'Composite: {sum(fail_weighted):.3f}',
              ha='center', va='bottom', fontsize=10, fontweight='bold', color='#B71C1C')
ax_right.text(1, bottom_pass + 0.02, f'Composite: {sum(pass_weighted):.3f}',
              ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2E7D32')

ax_right.set_ylabel('Weighted Score Contribution', fontsize=10)
ax_right.set_ylim(0, 1.25)
ax_right.set_title('Weighted Composite Score Breakdown', fontsize=11, fontweight='bold', pad=10)
ax_right.legend(fontsize=8.5, loc='upper left')
ax_right.spines['top'].set_visible(False)
ax_right.spines['right'].set_visible(False)

# Annotation explaining non-perseveration
ax_right.annotate('Non-Perseveration = 0\n(model repeated Phase 1\nsolution verbatim)',
                  xy=(0, 0.085), xytext=(0.35, 0.35),
                  fontsize=8, color='#B71C1C',
                  arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=1.2),
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#EF9A9A'))

fig_b.tight_layout(pad=1.5)
fig_b.savefig('/home/ubuntu/adapt_iq/data/gallery_cognitive_inertia_v2.png',
              dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig_b)
print("Figure B saved.")
print("Done.")
