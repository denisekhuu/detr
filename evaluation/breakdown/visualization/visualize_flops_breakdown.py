import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Load results from JSON
json_path = Path(__file__).parent / 'flops_breakdown_results.json'
with open(json_path, 'r') as f:
    data = json.load(f)

entries = data['detr_resnet50']

# Extract mean values (index 0 = mean from fmt_res: mean, std, min, max)
MEAN = 0
heads              = [e['heads']                                              for e in entries]
backbone_flops     = [e['backbone_flops'][MEAN]                               for e in entries]
transformer_flops  = [e['transformer_flops'][MEAN]                            for e in entries]
detr_flops         = [e['detr_flops'][MEAN]                                   for e in entries]
other_flops   = [d - b for d, b in
                      zip(detr_flops, backbone_flops)]

# ── Style ─────────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-poster')
sns.set_palette("husl")
colors = ['#1B4F72', '#2874A6', '#E67E22', '#3498DB', '#F39C12']


plt.figure(figsize=(8, 6))
plt.plot(heads, other_flops, color=colors[2], marker='^', linewidth=3, markersize=10,
         label='Total (excluding Backbone)', markerfacecolor='white', markeredgewidth=2)

plt.xlabel('Number of Attention Heads', fontsize=14, fontweight='bold')
plt.ylabel('GMACs', fontsize=14, fontweight='bold')
plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=12)
plt.grid(True, alpha=0.4, linestyle='--')
plt.xticks(heads)

y_min = min(min(transformer_flops), min(other_flops))
y_max = max(max(transformer_flops), max(other_flops))
y_margin = (y_max - y_min) * 0.1
plt.ylim(y_min - y_margin, y_max + y_margin)

max_val = other_flops[-1]
plt.axhline(y=max_val / 2, color='red',    linestyle=':', alpha=0.6, linewidth=2, label='Half Max')
plt.axhline(y=max_val / 4, color='darkred', linestyle=':', alpha=0.6, linewidth=2, label='Quarter Max')

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# ── Plot 2: Stacked area chart ─────────────────────────────────────────────────
total_flops     = [b + p for b, p in zip(backbone_flops, other_flops)]
backbone_plus_t = backbone_flops

plt.figure(figsize=(8, 6))
plt.fill_between(heads, 0, backbone_flops,                              color=colors[0], alpha=0.7, label='Backbone')
plt.fill_between(heads, backbone_flops, total_flops,                color=colors[1], alpha=0.7, label='Total')
plt.xlabel('Number of Attention Heads', fontsize=12, fontweight='bold')
plt.ylabel('GMACs', fontsize=12, fontweight='bold')
plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
plt.grid(True, alpha=0.4, linestyle='--')
plt.xticks(heads)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()
