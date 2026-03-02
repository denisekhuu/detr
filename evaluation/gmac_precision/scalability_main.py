import json
import matplotlib.pyplot as plt
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

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-poster')
COLOR = '#1B4F72'

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(detr_flops, ap_all, color=COLOR, linewidth=2.5,
        marker='o', markersize=9, markerfacecolor='white', markeredgewidth=2)

# Annotate each point with its head count
for gmac, ap, h in zip(detr_flops, ap_all, heads):
    ax.annotate(f'{h}h', xy=(gmac, ap),
                xytext=(6, 4), textcoords='offset points',
                fontsize=10, color=COLOR, fontweight='bold')

ax.set_xlabel('Total GMACs', fontsize=13, fontweight='bold')
ax.set_ylabel('AP@[.50:.95]', fontsize=13, fontweight='bold')
ax.set_title('DETR-ResNet50: GMACs vs AP@[.50:.95]', fontsize=14, fontweight='bold', pad=15)

ax.grid(True, alpha=0.4, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'scalability_gmac_precision.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'Heads':>6} {'GMACs':>10} {'AP@.50:.95':>12}")
print("-" * 30)
for i, h in enumerate(heads):
    print(f"{h:>6} {detr_flops[i]:>10.2f} {ap_all[i]:>12.4f}")