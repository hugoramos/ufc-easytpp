# scripts/peek_pkl.py
import os, pickle, math, numpy as np

BASE = './data/taxi'
MAP = {'train.pkl':'train','dev.pkl':'dev','test.pkl':'test'}

def short(v, n=3):
    if isinstance(v, (list, tuple)):
        return f'{type(v).__name__}(len={len(v)}): ' + ', '.join(short(x,1) for x in v[:n])
    if isinstance(v, dict):
        ks = list(v.keys())[:n]
        return f'dict(keys={ks} ...)'
    if isinstance(v, np.ndarray):
        return f'ndarray(shape={v.shape}, dtype={v.dtype})'
    return f'{type(v).__name__}'

for fn, key in MAP.items():
    p = os.path.join(BASE, fn)
    d = pickle.load(open(p,'rb'))
    print(f'\n=== {fn} ===')
    print('top-level keys:', list(d.keys()))
    print('dim_process:', d.get('dim_process'))
    inner = d.get(key)
    if inner is None:
        print(f'[{fn}] não tem chave "{key}" — talvez já esteja “plano”.')
        print('Plano? tem time_seqs?', 'time_seqs' in d, '| type_seqs?', 'type_seqs' in d)
        continue
    print(f'[{key}] type:', type(inner).__name__)
    if isinstance(inner, list):
        print(f'[{key}] len:', len(inner))
        if inner:
            print(f'[{key}][0] type:', type(inner[0]).__name__, '| preview:', short(inner[0]))
            if isinstance(inner[0], (list, tuple)) and inner and len(inner[0])>=1:
                print(f'[{key}][0][0] type:', type(inner[0][0]).__name__)
    elif isinstance(inner, dict):
        print(f'[{key}] keys:', list(inner.keys()))
