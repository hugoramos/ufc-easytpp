"""Generate Figure 2: NLL vs extrapolation factor (slow and fast decay panels)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Extrapolation factors
x_labels = ['1x', '2x', '5x', '10x']
x_pos = np.array([1, 2, 5, 10])

# Slow decay (beta_norm ~ 0.025)
slow_rothp_mean = np.array([0.937, 1.005, 1.389, 1.540])
slow_rothp_std  = np.array([0.007, 0.009, 0.011, 0.017])
slow_hothp_mean = np.array([0.950, 1.031, 1.403, 1.545])
slow_hothp_std  = np.array([0.008, 0.021, 0.005, 0.018])

# Fast decay (beta_norm ~ 0.40)
fast_rothp_mean = np.array([0.965, 1.583, 2.036, 2.362])
fast_rothp_std  = np.array([0.015, 0.281, 0.347, 0.442])
fast_hothp_mean = np.array([0.958, 1.170, 1.484, 1.593])
fast_hothp_std  = np.array([0.004, 0.006, 0.009, 0.015])

color_rothp = '#2196F3'
color_hothp = '#F44336'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Panel (a): Slow decay
ax1.errorbar(x_pos, slow_rothp_mean, yerr=slow_rothp_std, fmt='o-',
             color=color_rothp, capsize=4, linewidth=1.2, markersize=5, label='RoTHP')
ax1.errorbar(x_pos, slow_hothp_mean, yerr=slow_hothp_std, fmt='s-',
             color=color_hothp, capsize=4, linewidth=1.2, markersize=5, label='HoTHP')
ax1.set_xlabel('Extrapolation factor', fontsize=11)
ax1.set_ylabel('NLL', fontsize=11)
ax1.set_title(r'(a) Slow decay ($\beta \approx 0.025$)', fontsize=12)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(x_labels)
ax1.legend(fontsize=9, frameon=False)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(labelsize=9)

# Panel (b): Fast decay
ax2.errorbar(x_pos, fast_rothp_mean, yerr=fast_rothp_std, fmt='o-',
             color=color_rothp, capsize=4, linewidth=1.2, markersize=5, label='RoTHP')
ax2.errorbar(x_pos, fast_hothp_mean, yerr=fast_hothp_std, fmt='s-',
             color=color_hothp, capsize=4, linewidth=1.2, markersize=5, label='HoTHP')
ax2.set_xlabel('Extrapolation factor', fontsize=11)
ax2.set_ylabel('NLL', fontsize=11)
ax2.set_title(r'(b) Fast decay ($\beta \approx 0.40$)', fontsize=12)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(x_labels)
ax2.legend(fontsize=9, frameon=False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(labelsize=9)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'main_result.pdf')
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved {out_path}")
