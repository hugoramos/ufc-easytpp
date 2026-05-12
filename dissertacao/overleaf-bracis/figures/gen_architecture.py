"""Generate Figure: HoTHP architecture diagram.
Shows the pipeline and highlights the difference from THP/RoTHP."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3.5)
ax.axis('off')

# Colors
c_input = '#E3F2FD'
c_norm = '#FFF3E0'
c_attn = '#FFEBEE'
c_ff = '#E8F5E9'
c_output = '#F3E5F5'
c_border = '#555555'

# Box positions: x, y, width, height
boxes = [
    (0.3, 1.2, 1.4, 1.0, c_input, 'Event\nembedding\n$e(k_i)$'),
    (2.1, 1.2, 1.4, 1.0, c_norm, 'Temporal\nnormalization\n$\\tilde{t}_i$'),
    (3.9, 1.2, 1.8, 1.0, c_attn, 'Hyperbolic\nattention\n$\\mathbf{q}^T \\mathbf{B}(\\Delta t) \\mathbf{k}$'),
    (6.1, 1.2, 1.3, 1.0, c_ff, 'Feed-\nforward'),
    (7.8, 1.2, 1.8, 1.0, c_output, 'Intensity\n$\\lambda_k(t)$\n(softplus)'),
]

for x, y, w, h, color, text in boxes:
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1',
                                    facecolor=color, edgecolor=c_border, linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=8.5,
            fontfamily='sans-serif')

# Arrows between boxes
arrow_props = dict(arrowstyle='->', color=c_border, lw=1.5)
connections = [(1.7, 2.1), (3.5, 3.9), (5.7, 6.1), (7.4, 7.8)]
for x_start, x_end in connections:
    ax.annotate('', xy=(x_end, 1.7), xytext=(x_start, 1.7), arrowprops=arrow_props)

# Bracket for encoder layers
ax.annotate('', xy=(5.85, 2.45), xytext=(3.85, 2.45),
            arrowprops=dict(arrowstyle='-', color='gray', lw=1))
ax.text(4.85, 2.65, '$\\times N$ layers', ha='center', va='bottom',
        fontsize=9, color='gray', style='italic')

# Annotation: what's different
ax.annotate('HoTHP: hyperbolic kernel\n(replaces RoPE rotation)',
            xy=(4.8, 1.15), xytext=(4.8, 0.2),
            fontsize=8, ha='center', color='#B71C1C',
            arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=1),
            bbox=dict(boxstyle='round,pad=0.3', fc='#FFEBEE', ec='#B71C1C', alpha=0.8))

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'architecture.pdf')
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved {out_path}")
