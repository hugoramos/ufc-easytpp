# scripts/unwrap_splits.py
import os, pickle

base = './data/taxi'
mapping = {
    'train.pkl': 'train',
    'dev.pkl': 'dev',
    'test.pkl': 'test',
}

for fn, key in mapping.items():
    path = os.path.join(base, fn)
    with open(path, 'rb') as f:
        d = pickle.load(f)

    # Se já estiver no formato plano, não faz nada
    if any(k in d for k in ['time_seqs', 'time_delta_seqs', 'type_seqs']):
        print(f'{fn} já parece estar no formato plano; ignorando.')
        continue

    if key not in d:
        raise ValueError(f'{fn} não contém a chave esperada "{key}". Chaves: {list(d.keys())}')

    inner = d[key]
    dim = d.get('dim_process')

    # Construir formato esperado pelo EasyTPP:
    out = dict(inner)  # copia time_seqs/time_delta_seqs/type_seqs/...
    if dim is not None:
        out['dim_process'] = dim
        out['type_count'] = dim  # ajuda manter consistência em alguns dumps

    with open(path, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'✔ Regravado {fn} no formato plano com dim_process={dim}')
