# scripts/unwrap_splits_v2.py
import os, pickle, sys

BASE = './data/taxi'
MAPPING = {
    'train.pkl': 'train',
    'dev.pkl':   'dev',
    'test.pkl':  'test',
}

def to_plain(inner):
    """
    Converte 'inner' para um dict com chaves:
      - time_seqs
      - time_delta_seqs
      - type_seqs
    Retorna (out_dict, note_str)
    """
    # Caso 1: já é dict no formato certo
    if isinstance(inner, dict) and any(k in inner for k in ('time_seqs','type_seqs','time_delta_seqs')):
        return dict(inner), 'já era dict com sequências'

    # Caso 2: inner é tupla/lista de 3 elementos
    if isinstance(inner, (list, tuple)) and len(inner) == 3:
        time_seqs, time_delta_seqs, type_seqs = inner
        return {
            'time_seqs': time_seqs,
            'time_delta_seqs': time_delta_seqs,
            'type_seqs': type_seqs,
        }, 'convertido de lista/tupla (time, delta, type)'

    # Caso 3: inner é dict mas com outra chave envelopando
    if isinstance(inner, dict):
        # tenta achar listas plausíveis
        cand_time = None
        cand_delta = None
        cand_type = None
        for k in inner.keys():
            lk = k.lower()
            if cand_time is None and ('time_seqs' in lk or lk in ('times','time')):
                cand_time = inner[k]
            if cand_delta is None and ('delta' in lk or 'time_delta' in lk):
                cand_delta = inner[k]
            if cand_type is None and ('type_seqs' in lk or 'event_types' in lk or lk in ('types','type')):
                cand_type = inner[k]
        if cand_time is not None and cand_type is not None:
            return {
                'time_seqs': cand_time,
                'time_delta_seqs': cand_delta,
                'type_seqs': cand_type,
            }, 'dict com chaves não padronizadas'
        # às vezes vem algo como {'sequences': {...}}
        if 'sequences' in inner and isinstance(inner['sequences'], dict):
            seqs = inner['sequences']
            if all(k in seqs for k in ('time_seqs','time_delta_seqs','type_seqs')):
                return dict(seqs), 'dict dentro de sequences'
    # Se chegou aqui, formato desconhecido
    raise ValueError(f'Formato inesperado: type(inner)={type(inner)}, keys={list(inner.keys()) if isinstance(inner, dict) else "n/a"}')

def process_file(fn, split_key):
    path = os.path.join(BASE, fn)
    with open(path, 'rb') as f:
        d = pickle.load(f)

    dim = d.get('dim_process')
    # Se já tiver no formato plano, só garante que tem dim_process
    if any(k in d for k in ('time_seqs','time_delta_seqs','type_seqs')):
        out = dict(d)
        if dim is not None:
            out['dim_process'] = dim
            out['type_count'] = dim
        with open(path, 'wb') as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'✔ {fn}: já estava plano; dim_process={dim}')
        return

    if split_key not in d:
        raise ValueError(f'{fn} não contém a chave "{split_key}". Chaves: {list(d.keys())}')

    inner = d[split_key]
    out, note = to_plain(inner)
    if dim is not None:
        out['dim_process'] = dim
        out['type_count'] = dim

    with open(path, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'✔ {fn}: {note}; dim_process={dim}')

if __name__ == '__main__':
    for fn, key in MAPPING.items():
        process_file(fn, key)
    print('OK: arquivos regravados no formato plano.')
