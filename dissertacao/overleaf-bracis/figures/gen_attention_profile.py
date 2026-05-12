"""Generate Figure: Theoretical attention score profiles for RoTHP, HoTHP, ALiBi.
Single panel, all three normalized."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

d = 64
half_d = d // 2

j = np.arange(1, half_d + 1)
theta_j = 10000.0 ** (-2.0 * (j - 1) / d)
theta_prime = np.max(theta_j) + 0.05

# Aligned q = k
np.random.seed(42)
q = np.random.randn(d)
q /= np.linalg.norm(q)
k = q.copy()

q1, q2 = q[0::2], q[1::2]
k1, k2 = k[0::2], k[1::2]

dt = np.linspace(0, 50, 500)

scores_rothp = np.zeros_like(dt)
scores_hothp = np.zeros_like(dt)

for i, t in enumerate(dt):
    cos_t = np.cos(t * theta_j)
    sin_t = np.sin(t * theta_j)
    scores_rothp[i] = np.sum((q1*k1 + q2*k2) * cos_t +
                              (-q1*k2 + q2*k1) * sin_t) / np.sqrt(d)

    abs_t = np.abs(t)
    sign_t = np.sign(t) if t != 0 else 0.0
    exp_minus = np.exp(-abs_t * (theta_prime - theta_j))
    exp_plus = np.exp(-abs_t * (theta_prime + theta_j))
    decay_cosh = (exp_minus + exp_plus) / 2.0
    decay_sinh = sign_t * (exp_minus - exp_plus) / 2.0
    scores_hothp[i] = np.sum((q1*k1 + q2*k2) * decay_cosh +
                              (q1*k2 + q2*k1) * decay_sinh) / np.sqrt(d)

scores_alibi = -dt / 8.0

def norm01(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-10)

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(dt, norm01(scores_rothp), color='#2196F3', linewidth=1.2, label='RoTHP')
ax.plot(dt, norm01(scores_hothp), color='#F44336', linewidth=1.2, label='HoTHP')
ax.plot(dt, norm01(scores_alibi), color='#4CAF50', linewidth=1.2,
        linestyle='--', label='ALiBi')

ax.set_xlabel('Time', fontsize=11)
ax.set_ylabel('Attention score', fontsize=11)
ax.legend(fontsize=10, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=9)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'attention_profile.pdf')
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved {out_path}")
