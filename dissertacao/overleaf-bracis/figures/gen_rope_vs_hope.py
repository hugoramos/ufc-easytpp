"""Generate Figure: RoPE vs HoPE block comparison.
Shows the 2x2 trigonometric block (RoPE/RoTHP) vs hyperbolic block (HoPE/HoTHP)
and how each affects the attention score as delta_t grows."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

dt = np.linspace(0, 30, 300)
theta = 0.3  # single frequency for illustration
theta_prime = theta + 0.05  # damping coefficient

# RoPE/RoTHP: cos(theta * dt) -- the diagonal element of the rotation block
rope_score = np.cos(theta * dt)

# HoPE/HoTHP: damped cosh
exp_minus = np.exp(-np.abs(dt) * (theta_prime - theta))
exp_plus = np.exp(-np.abs(dt) * (theta_prime + theta))
hope_score = (exp_minus + exp_plus) / 2.0

# Normalize to start at 1
rope_score_n = rope_score / rope_score[0]
hope_score_n = hope_score / hope_score[0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# Panel (a): Single frequency behavior
ax1.plot(dt, rope_score_n, color='#2196F3', linewidth=1.5, label='RoTHP (cos)')
ax1.plot(dt, hope_score_n, color='#F44336', linewidth=1.5, label='HoTHP (damped cosh)')
ax1.axhline(0, color='gray', lw=0.5, alpha=0.3)
ax1.set_xlabel('Time', fontsize=11)
ax1.set_ylabel('Kernel value', fontsize=11)
ax1.set_title(r'(a) Single frequency $\theta$', fontsize=12)
ax1.legend(fontsize=10, frameon=False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(labelsize=9)

# Panel (b): Matrix visualization
# Show the 2x2 blocks side by side at a specific dt value
dt_val = 5.0
cos_v = np.cos(theta * dt_val)
sin_v = np.sin(theta * dt_val)

exp_m = np.exp(-np.abs(dt_val) * (theta_prime - theta))
exp_p = np.exp(-np.abs(dt_val) * (theta_prime + theta))
cosh_v = (exp_m + exp_p) / 2.0
sinh_v = (exp_m - exp_p) / 2.0

ax2.axis('off')
ax2.set_xlim(-0.5, 10)
ax2.set_ylim(-1, 4)

# RoTHP matrix
ax2.text(1.5, 3.5, r'RoTHP: $\mathbf{R}(\Delta t)$', fontsize=11,
         ha='center', fontweight='bold', color='#2196F3')
matrix_text = (f'[{cos_v:+.2f}  {sin_v:+.2f}]\n'
               f'[{-sin_v:+.2f}  {cos_v:+.2f}]')
ax2.text(1.5, 2.0, matrix_text, fontsize=12, ha='center', va='center',
         fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.4', fc='#E3F2FD', ec='#2196F3', lw=1.5))

# HoTHP matrix
ax2.text(6.5, 3.5, r'HoTHP: $e^{-\theta^\prime \Delta t} \cdot \mathbf{B}(\Delta t)$',
         fontsize=11, ha='center', fontweight='bold', color='#F44336')
matrix_text2 = (f'[{cosh_v:+.4f}  {sinh_v:+.4f}]\n'
                f'[{sinh_v:+.4f}  {cosh_v:+.4f}]')
ax2.text(6.5, 2.0, matrix_text2, fontsize=12, ha='center', va='center',
         fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.4', fc='#FFEBEE', ec='#F44336', lw=1.5))

# Labels
ax2.text(1.5, 0.2, 'Oscillates\n(values can increase)', fontsize=9,
         ha='center', color='#2196F3', style='italic')
ax2.text(6.5, 0.2, 'Decays monotonically\n(values always decrease)', fontsize=9,
         ha='center', color='#F44336', style='italic')

ax2.text(4.0, 2.0, 'vs', fontsize=14, ha='center', va='center',
         fontweight='bold', color='gray')

ax2.set_title(f'(b) Block values at $\\Delta t = {dt_val:.0f}$', fontsize=12)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'rope_vs_hope.pdf')
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved {out_path}")
