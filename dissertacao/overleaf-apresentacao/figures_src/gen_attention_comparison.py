# overleaf-apresentacao/figures_src/gen_attention_comparison.py
"""Side-by-side attention weight matrices for RoTHP and HoTHP.
Shows that RoTHP has no recency bias; HoTHP decays from the diagonal."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

np.random.seed(0)
L = 40
d = 64
half_d = d // 2

j = np.arange(1, half_d + 1)
theta_j = 10000.0 ** (-2.0 * (j - 1) / d)
theta_prime = np.max(theta_j) + 0.05

# Synthetic timestamps: irregular gaps drawn from exponential
gaps = np.random.exponential(scale=1.0, size=L)
times = np.cumsum(gaps)  # shape [L]

# Random unit queries and keys (same for both models)
Q = np.random.randn(L, d)
K = np.random.randn(L, d)
Q /= np.linalg.norm(Q, axis=-1, keepdims=True)
K /= np.linalg.norm(K, axis=-1, keepdims=True)

Q1, Q2 = Q[:, 0::2], Q[:, 1::2]  # [L, half_d]
K1, K2 = K[:, 0::2], K[:, 1::2]

# RoTHP scores: q_rot dot k_rot using cos/sin kernel
scores_rothp = np.zeros((L, L))
for i in range(L):
    cos_i = np.cos(times[i] * theta_j)
    sin_i = np.sin(times[i] * theta_j)
    q1r = Q1[i] * cos_i - Q2[i] * sin_i
    q2r = Q1[i] * sin_i + Q2[i] * cos_i
    for jj in range(L):
        cos_j = np.cos(times[jj] * theta_j)
        sin_j = np.sin(times[jj] * theta_j)
        k1r = K1[jj] * cos_j - K2[jj] * sin_j
        k2r = K1[jj] * sin_j + K2[jj] * cos_j
        scores_rothp[i, jj] = (np.sum(q1r * k1r) + np.sum(q2r * k2r)) / np.sqrt(d)

# HoTHP scores: hyperbolic kernel per pair
scores_hothp = np.zeros((L, L))
for i in range(L):
    for jj in range(L):
        delta = times[i] - times[jj]
        abs_d = abs(delta)
        sign_d = np.sign(delta) if delta != 0 else 0.0
        exp_minus = np.exp(-abs_d * (theta_prime - theta_j))
        exp_plus  = np.exp(-abs_d * (theta_prime + theta_j))
        dc = (exp_minus + exp_plus) / 2.0
        ds = sign_d * (exp_minus - exp_plus) / 2.0
        scores_hothp[i, jj] = (
            np.sum((Q1[i]*K1[jj] + Q2[i]*K2[jj]) * dc +
                   (Q1[i]*K2[jj] + Q2[i]*K1[jj]) * ds)
        ) / np.sqrt(d)

# Apply causal mask and softmax
def causal_softmax(scores):
    mask = np.triu(np.ones_like(scores), k=1) * -1e9
    s = scores + mask
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)

w_rothp = causal_softmax(scores_rothp)
w_hothp = causal_softmax(scores_hothp)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

for ax, w, title in zip(axes,
                         [w_rothp, w_hothp],
                         ['RoTHP', 'HoTHP']):
    im = ax.imshow(w, aspect='auto', cmap='Blues', vmin=0)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Key event index', fontsize=10)
    ax.set_ylabel('Query event index', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), '..', 'figures', 'attention_comparison.pdf')
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved {out_path}")
