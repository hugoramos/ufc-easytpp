"""Generate Figure: Attention kernel profiles vs true Hawkes kernel.
Uses ALL 32 frequencies (honest representation)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

d = 64
half_d = d // 2

j = np.arange(1, half_d + 1)
theta_j = 10000.0 ** (-2.0 * (j - 1) / d)  # all 32 frequencies
theta_prime = np.max(theta_j) + 0.05

dt = np.linspace(0, 50, 500)

# True Hawkes kernel: exp(-beta * dt), beta=0.4
true_kernel = np.exp(-0.4 * dt)

# RoTHP: mean of cos(dt * theta_j) over ALL frequencies
rothp_profile = np.mean([np.cos(dt * th) for th in theta_j], axis=0)

# HoTHP: mean of damped cosh over ALL frequencies
hothp_profiles = []
for th in theta_j:
    exp_minus = np.exp(-np.abs(dt) * (theta_prime - th))
    exp_plus = np.exp(-np.abs(dt) * (theta_prime + th))
    hothp_profiles.append((exp_minus + exp_plus) / 2.0)
hothp_profile = np.mean(hothp_profiles, axis=0)

# Normalize all to [0, 1] for comparison
def norm01(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-10)

rothp_norm = norm01(rothp_profile)
hothp_norm = norm01(hothp_profile)

# Plot
fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(dt, true_kernel, 'k--', linewidth=1.5, label='Hawkes kernel')
ax.plot(dt, rothp_norm, color='#2196F3', linewidth=1.2, label='RoTHP')
ax.plot(dt, hothp_norm, color='#F44336', linewidth=1.2, label='HoTHP')

ax.set_xlabel('Time', fontsize=11)
ax.set_ylabel('Attention score', fontsize=11)
ax.legend(fontsize=10, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=9)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'kernel_alignment.pdf')
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved {out_path}")
