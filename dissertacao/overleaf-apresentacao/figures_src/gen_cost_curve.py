"""Training cost curve illustrating O(L^2) scaling of attention."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

L = np.array([128, 256, 512, 1024, 2048])
cost = (L / 128.0) ** 2  # normalized so L=128 => cost=1

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(L, cost, 'o-', color='#1565C0', linewidth=2, markersize=7)

# Annotate each point
for l, c in zip(L, cost):
    ax.annotate(f'{c:.0f}x', xy=(l, c), xytext=(8, 4),
                textcoords='offset points', fontsize=9, color='#1565C0')

ax.set_xlabel('Sequence length $L$', fontsize=12)
ax.set_ylabel('Relative training cost', fontsize=12)
ax.set_title('Attention cost grows as $O(L^2)$', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=10)
ax.set_xticks(L)
ax.set_xticklabels([str(l) for l in L])

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), '..', 'figures', 'cost_curve.pdf')
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved {out_path}")
