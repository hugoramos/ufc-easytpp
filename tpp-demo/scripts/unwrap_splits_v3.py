# scripts/unwrap_splits_v3.py
import os, pickle, math

BASE = './data/taxi'
MAPPING = {
    'train.pkl': 'train',
    'dev.pkl':   'dev',
    'test.pkl':  'test',
}

def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))

def _seq_is_numeric_list(seq):
    return isinstance(seq, list) and (len(seq)==0 or all(_is_number(v) for v in seq))

def _derive_deltas_from_times(times):
    if not times: return []
    deltas = [times[0]]
    for i in range(1, len(times)):
        deltas.append(times[i]-times[i-1])
    return deltas

def _derive_times_from_deltas(deltas):
    times = []
    acc = 0.0
    for d in deltas:
        acc += d
        times.append(acc)
    return times

def _extract_seq_from_dict_per_sequence(d):
    """
    d: dict que representa UMA sequência
    Retorna (times, deltas, types) ou None se não reconhecer.
    """
    # tenta tempo absoluto
    times = None
    for k in ('time_seqs','times','time_since_start','time_since_begin','timestamps'):
        if k in d and _seq_is_numeric_list(d[k]):
            times = list(d[k]); break

    # tenta deltas
    deltas = None
    for k in ('time_delta_seqs','deltas','time_since_last_event'):
        if k in d and _seq_is_numeric_list(d[k]):
            deltas = list(d[k]); break

    # tenta tipos
    types = None
    for k in ('type_seqs','types','event_types','type_event','labels'):
        if k in d and isinstance(d[k], list):
            types = list(d[k]); break

    # se não achar tipos, alguns dumps usam ints/np.int em outra chave
    if types is None and 'y' in d and isinstance(d['y'], list):
        types = list(d['y'])

    # reconcilia faltantes
    if times is None and deltas is not None:
        times = _derive_times_from_deltas(deltas)
    if deltas is None and times is not None:
        deltas = _derive_deltas_from_times(times)

    # validação básica
    if times is None or deltas is None or types is None:
        return None
    if not (len(times)==len(deltas)==len(types)):
        m = min(len(times), len(deltas), len(types))
        times, deltas, types = times[:m], deltas[:m], types[:m]
    return times, deltas, types

def to_plain(inner):
    """
    Converte 'inner' para um dict com chaves:
      - time_seqs: List[List[float]]
      - time_delta_seqs: List[List[float]]
      - type_seqs: List[List[int]]
    """
    # Caso 1: já está plano
    if isinstance(inner, dict) and any(k in inner for k in ('time_seqs','time_delta_seqs','type_seqs')):
        out = {}
        out['time_seqs'] = list(inner.get('time_seqs', []))
        out['time_delta_seqs'] = list(inner.get('time_delta_seqs', []))
        out['type_seqs'] = list(inner.get('type_seqs', []))
        # sanity fix se faltarem alguns
        if not out['time_delta_seqs'] and out['time_seqs']:
            out['time_delta_seqs'] = [_derive_deltas_from_times(ts) for ts in out['time_seqs']]
        if not out['time_seqs'] and out['time_delta_seqs']:
            out['time_seqs'] = [_derive_times_from_deltas(ds) for ds in out['time_delta_seqs']]
        return out, 'já plano'

    # Caso 2: tupla/lista de 3
    if isinstance(inner, (list, tuple)) and len(inner) == 3:
        time_seqs, time_delta_seqs, type_seqs = inner
        return {
            'time_seqs': list(time_seqs),
            'time_delta_seqs': list(time_delta_seqs),
            'type_seqs': list(type_seqs),
        }, 'tupla/lista (time, delta, type)'

    # Caso 3: lista de sequências (cada item é um dict por sequência)
    if isinstance(inner, list) and inner and isinstance(inner[0], dict):
        T, D, Y = [], [], []
        for i, seqd in enumerate(inner):
            r = _extract_seq_from_dict_per_sequence(seqd)
            if r is None:
                raise ValueError(f'Sequência {i} não reconhecida; chaves={list(seqd.keys())}')
            t, d, y = r
            T.append(t); D.append(d); Y.append(y)
        return {'time_seqs': T, 'time_delta_seqs': D, 'type_seqs': Y}, 'lista de dicts (1 dict por sequência)'

    # Caso 4: lista de listas brutas (pouco comum). Tentamos heurística:
    if isinstance(inner, list) and inner and isinstance(inner[0], (list, tuple)):
        # Heurística: se cada item tem 3 listas, interpretamos como (t, d, y) por sequência
        if all(isinstance(el, (list, tuple)) and len(el)==3 for el in inner):
            T = [list(el[0]) for el in inner]
            D = [list(el[1]) for el in inner]
            Y = [list(el[2]) for el in inner]
            return {'time_seqs': T, 'time_delta_seqs': D, 'type_seqs': Y}, 'lista de (t,d,y) por sequência'
        # Se cada item for lista de números → pode ser “apenas tempos” de várias seqs (sem tipos)
        if all(_seq_is_numeric_list(el) for el in inner):
            T = [list(el) for el in inner]
            D = [_derive_deltas_from_times(el) for el in T]
            # tipos desconhecidos → não dá para inferir
            raise ValueError('Lista de listas numéricas sem tipos; não consigo inferir type_seqs.')
    raise ValueError(f'Formato inesperado: type(inner)={type(inner)}')

def process_file(fn, split_key):
    path = os.path.join(BASE, fn)
    with open(path, 'rb') as f:
        d = pickle.load(f)

    dim = d.get('dim_process')

    # já plano?
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

    # garante consistência
    if dim is not None:
        out['dim_process'] = int(dim)
        out['type_count'] = int(dim)

    with open(path, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'✔ {fn}: {note}; dim_process={dim}')

def main():
    for fn, key in MAPPING.items():
        process_file(fn, key)
    # Harmoniza dim_process global (maior tipo + 1)
    files = list(MAPPING.keys())
    max_type = -1
    for fn in files:
        d = pickle.load(open(os.path.join(BASE, fn),'rb'))
        for seq in d.get('type_seqs', []):
            if seq:
                mt = max(seq)
                if mt > max_type: max_type = mt
    if max_type >= 0:
        global_dim = int(max_type) + 1
        for fn in files:
            p = os.path.join(BASE, fn)
            d = pickle.load(open(p,'rb'))
            d['dim_process'] = global_dim
            d['type_count'] = global_dim
            pickle.dump(d, open(p,'wb'), protocol=pickle.HIGHEST_PROTOCOL)
            print(f'✔ {fn}: dim_process normalizado para {global_dim}')
    print('OK: arquivos prontos.')

if __name__ == '__main__':
    main()
