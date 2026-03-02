import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Load data ─────────────────────────────────────────────────────────────────
flops_path     = Path(__file__).parent.parent / 'breakdown' / 'visualization' / 'flops_breakdown_results.json'
precision_path = Path(__file__).parent.parent / 'precision' / 'sliced_test_stats.json'

with open(flops_path, 'r') as f:
    flops_data = json.load(f)

with open(precision_path, 'r') as f:
    prec_data = json.load(f)

# ── Parse FLOPs (deduplicate heads, take first occurrence) ────────────────────
seen_heads = set()
entries = []
for e in flops_data['detr_resnet50']:
    h = e['heads']
    if h not in seen_heads:
        seen_heads.add(h)
        entries.append(e)
entries.sort(key=lambda x: x['heads'])

MEAN = 0
heads      = [e['heads']            for e in entries]
detr_flops = [e['detr_flops'][MEAN] for e in entries]

# ── Parse precision ───────────────────────────────────────────────────────────
def get_coco(h):
    return prec_data[f'heads_{h}']['coco_eval_bbox']

ap_all = [get_coco(h)[0] for h in heads]
ap50   = [get_coco(h)[1] for h in heads]
ap75   = [get_coco(h)[2] for h in heads]
ar100  = [get_coco(h)[8] for h in heads]

# ── Style ─────────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-poster')
COLOR_GMAC = '#1B4F72'
COLOR_AP   = '#E67E22'
COLOR_AP50 = '#2ECC71'
COLOR_AP75 = '#9B59B6'
COLOR_AR   = '#E74C3C'

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

# GMACs (left axis)
l_gmac, = ax1.plot(heads, detr_flops, color=COLOR_GMAC, marker='o', linewidth=2.5,
                   markersize=9, label='Total GMACs',
                   markerfacecolor='white', markeredgewidth=2)

# AP / AR (right axis)
l_ap,   = ax2.plot(heads, ap_all, color=COLOR_AP,   marker='s', linewidth=2.5,
                   markersize=9, label='AP@[.50:.95]',
                   markerfacecolor='white', markeredgewidth=2, linestyle='--')
l_ap50, = ax2.plot(heads, ap50,   color=COLOR_AP50, marker='^', linewidth=2.5,
                   markersize=9, label='AP@0.50',
                   markerfacecolor='white', markeredgewidth=2, linestyle='--')
l_ap75, = ax2.plot(heads, ap75,   color=COLOR_AP75, marker='D', linewidth=2.5,
                   markersize=9, label='AP@0.75',
                   markerfacecolor='white', markeredgewidth=2, linestyle='--')
l_ar,   = ax2.plot(heads, ar100,  color=COLOR_AR,   marker='P', linewidth=2.5,
                   markersize=9, label='AR@100',
                   markerfacecolor='white', markeredgewidth=2, linestyle='--')

# Axes labels
ax1.set_xlabel('Number of Attention Heads', fontsize=13, fontweight='bold')
ax1.set_ylabel('Total GMACs',               fontsize=13, fontweight='bold', color=COLOR_GMAC)
ax2.set_ylabel('Score',                     fontsize=13, fontweight='bold', color=COLOR_AP)
ax1.tick_params(axis='y', labelcolor=COLOR_GMAC)
ax2.tick_params(axis='y', labelcolor='black')

ax1.set_xticks(heads)
ax1.grid(True, alpha=0.4, linestyle='--')
ax1.spines['top'].set_visible(False)

plt.title('DETR-ResNet50: Total GMACs vs Detection Performance',
          fontsize=14, fontweight='bold', pad=15)

lines = [l_gmac, l_ap, l_ap50, l_ap75, l_ar]
ax1.legend(lines, [l.get_label() for l in lines],
           fontsize=11, frameon=True, fancybox=True, shadow=True, loc='upper left')

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'scalability_gmac_precision.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'Heads':>6} {'GMACs':>10} {'AP@.50:.95':>12} {'AP@0.50':>10} {'AP@0.75':>10} {'AR@100':>10}")
print("-" * 62)
for i, h in enumerate(heads):
    print(f"{h:>6} {detr_flops[i]:>10.2f} {ap_all[i]:>12.4f} "
          f"{ap50[i]:>10.4f} {ap75[i]:>10.4f} {ar100[i]:>10.4f}")