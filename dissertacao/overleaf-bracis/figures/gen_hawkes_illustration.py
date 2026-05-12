"""Generate Figure: Multivariate Hawkes process illustration.
Two event types, two intensity lines, showing self- and cross-excitation."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Parameters
mu = np.array([0.3, 0.25])
alpha = np.array([
    [0.6, 0.2],   # alpha[0,0]=self-excitation of type 0, alpha[0,1]=type 1 excites type 0
    [0.3, 0.5],   # alpha[1,0]=type 0 excites type 1, alpha[1,1]=self-excitation of type 1
])
beta = 0.5

# Hand-picked events: (time, type)
events = [
    (1.5, 0), (3.5, 0), (4.5, 1),
    (8.0, 1), (9.5, 0), (10.5, 1),
    (15.0, 0), (16.0, 1),
]

t = np.linspace(0, 20, 2000)
type_colors = ['#2196F3', '#F44336']
type_labels = ['Type 1', 'Type 2']

# Compute intensity for each type separately
n_types = 2
intensities = [np.full_like(t, mu[k]) for k in range(n_types)]

for ev_time, ev_type in events:
    for k in range(n_types):
        mask = t > ev_time
        intensities[k][mask] += alpha[k, ev_type] * np.exp(-beta * (t[mask] - ev_time))

fig, (ax_ev, ax0, ax1) = plt.subplots(3, 1, figsize=(9, 7),
                                        height_ratios=[0.6, 1.5, 1.5],
                                        sharex=True)
fig.subplots_adjust(hspace=0.08)

# Top panel: events
for ev_time, ev_type in events:
    ax_ev.plot(ev_time, 0.5, 'o', color=type_colors[ev_type], markersize=8, zorder=5)
    ax_ev.vlines(ev_time, 0.15, 0.85, color=type_colors[ev_type], lw=1.5, alpha=0.6)

for k in range(n_types):
    ax_ev.plot([], [], 'o', color=type_colors[k], markersize=7, label=type_labels[k])
ax_ev.legend(fontsize=9, frameon=False, loc='center right', ncol=1)
ax_ev.set_yticks([])
ax_ev.set_ylabel('Events', fontsize=11)
ax_ev.set_xlim(0, 20)
ax_ev.set_ylim(0, 1)
for spine in ax_ev.spines.values():
    spine.set_visible(False)

# Middle panel: intensity of type 0
ax0.plot(t, intensities[0], color=type_colors[0], linewidth=1.5)
ax0.axhline(mu[0], color='gray', ls=':', lw=1, alpha=0.5)

# Dots at post-jump for type 0
for ev_time, ev_type in events:
    pre = mu[0]
    for pt, pk in events:
        if pt < ev_time:
            pre += alpha[0, pk] * np.exp(-beta * (ev_time - pt))
    post = pre + alpha[0, ev_type]
    ax0.plot(ev_time, post, 'o', color=type_colors[ev_type], markersize=4, zorder=5)

ax0.set_ylabel('$\lambda_1(t)$', fontsize=12, color=type_colors[0])
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.set_ylim(0, None)
ax0.tick_params(labelsize=9)

# Bottom panel: intensity of type 1
ax1.plot(t, intensities[1], color=type_colors[1], linewidth=1.5)
ax1.axhline(mu[1], color='gray', ls=':', lw=1, alpha=0.5)

# Dots at post-jump for type 1
for ev_time, ev_type in events:
    pre = mu[1]
    for pt, pk in events:
        if pt < ev_time:
            pre += alpha[1, pk] * np.exp(-beta * (ev_time - pt))
    post = pre + alpha[1, ev_type]
    ax1.plot(ev_time, post, 'o', color=type_colors[ev_type], markersize=4, zorder=5)

ax1.set_ylabel('$\lambda_2(t)$', fontsize=12, color=type_colors[1])
ax1.set_xlabel('Time', fontsize=11)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylim(0, None)
ax1.tick_params(labelsize=9)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'hawkes_illustration.pdf')
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved {out_path}")
