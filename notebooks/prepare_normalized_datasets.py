"""
Prepara versões normalizadas dos datasets para teste de ablação.

Ablação 1: retweet_norm — retweet com timestamps normalizados (÷ mean gap por seq)
Ablação 2: taxi com max_len maior (não requer novo dataset, só config)

O dataset normalizado é salvo no formato EasyTPP JSON local.

Execução:
    python3.11 notebooks/prepare_normalized_datasets.py
"""

import json, time
import numpy as np
from pathlib import Path

t0 = time.time()

# ─────────────────────────────────────────────────────────────────────────────

def normalize_dataset(hf_name, output_dir, max_seqs_per_split=None):
    """
    Carrega dataset do HuggingFace, normaliza timestamps por sequência
    (divide por mean gap), e salva como JSON local no formato EasyTPP.
    """
    from datasets import load_dataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_all = []

    for split in ['train', 'validation', 'test']:
        print(f"  [{split}] ", end="", flush=True)
        try:
            ds = load_dataset(hf_name, split=split)
        except Exception:
            print(f"split '{split}' não encontrado, tentando 'dev'...", flush=True)
            try:
                ds = load_dataset(hf_name, split='dev')
            except Exception:
                print("SKIP", flush=True)
                continue

        sequences = []
        for i, row in enumerate(ds):
            if max_seqs_per_split and i >= max_seqs_per_split:
                break

            t = np.array(row['time_since_start'], dtype=np.float64)
            types = list(row['type_event'])
            dim = row.get('dim_process', max(types) + 1 if types else 1)

            if len(t) < 3:
                continue

            # Normalização: divide todos os timestamps pela média dos gaps
            gaps = np.diff(t)
            mean_gap = gaps[gaps > 0].mean() if np.any(gaps > 0) else 1.0
            if mean_gap <= 0:
                mean_gap = 1.0

            t_norm = (t - t[0]) / mean_gap

            # Recomputa time_since_last_event
            dt_norm = np.diff(t_norm)
            dt_norm = np.insert(dt_norm, 0, 0.0)  # primeiro evento: dt=0

            seq = {
                'time_since_start': t_norm.tolist(),
                'time_since_last_event': dt_norm.tolist(),
                'type_event': types,
                'dim_process': dim,
            }
            sequences.append(seq)

            # Stats para validação
            if split == 'train':
                stats_all.append({
                    'mean_gap_orig': float(mean_gap),
                    'mean_dt_norm': float(dt_norm[1:].mean()) if len(dt_norm) > 1 else 0,
                    'max_dt_norm': float(dt_norm.max()),
                    'len': len(t),
                })

        out_file = output_dir / f'{split}.json'
        with open(out_file, 'w') as f:
            json.dump(sequences, f)
        print(f"{len(sequences)} seqs → {out_file}", flush=True)

    # Print stats
    if stats_all:
        mean_gaps = [s['mean_gap_orig'] for s in stats_all]
        mean_dts = [s['mean_dt_norm'] for s in stats_all]
        max_dts = [s['max_dt_norm'] for s in stats_all]
        print(f"  Stats (train):", flush=True)
        print(f"    mean_gap original: median={np.median(mean_gaps):.4f} "
              f"mean={np.mean(mean_gaps):.4f}", flush=True)
        print(f"    mean_dt normalizado: median={np.median(mean_dts):.4f} "
              f"(esperado ≈1.0)", flush=True)
        print(f"    max_dt normalizado: median={np.median(max_dts):.2f} "
              f"max={np.max(max_dts):.2f}", flush=True)

    # Save metadata
    meta = {
        'source': hf_name,
        'normalization': 'per-sequence mean gap',
        'description': 'Timestamps divididos pela média dos gaps de cada sequência',
    }
    if stats_all:
        meta['dim_process'] = stats_all[0].get('dim_process', None)

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Criar retweet normalizado
# ─────────────────────────────────────────────────────────────────────────────

print(f"[{time.time()-t0:.0f}s] Preparando retweet_norm...", flush=True)
normalize_dataset(
    hf_name='easytpp/retweet',
    output_dir='datasets/retweet_norm',
)

print(f"\n[{time.time()-t0:.0f}s] Preparando stackoverflow_norm...", flush=True)
normalize_dataset(
    hf_name='easytpp/stackoverflow',
    output_dir='datasets/stackoverflow_norm',
)

print(f"\n[Total: {time.time()-t0:.0f}s]")
print("\nDatasets salvos em:")
print("  datasets/retweet_norm/")
print("  datasets/stackoverflow_norm/")
print()
print("Para usar no benchmark, adicione ao DATASETS:")
print("""
'retweet_norm': {
    'hf_name': None,
    'local_dir': './datasets/retweet_norm',
    'num_event_types': 3,
    'train_max_len': 50,
    'eval_1x': 50,
    'eval_5x': 250,
},
""")
