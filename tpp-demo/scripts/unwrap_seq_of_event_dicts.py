# scripts/unwrap_seq_of_event_dicts.py
import os, pickle, math

BASE = './data/taxi'
MAP = {'train.pkl': 'train', 'dev.pkl': 'dev', 'test.pkl': 'test'}

TYPE_KEYS  = ['type_event','event_type','type','label','y','k']
TIME_KEYS  = ['time_since_start','timestamp','time','t','times']
DELTA_KEYS = ['time_since_last_event','dtime','delta','dt','time_delta','time_since_prev']

def _is_num(x):
    try:
        return isinstance(x, (int, float)) and math.isfinite(float(x))
    except Exception:
        return False

def _to_list(x):
    # aceita int/float/np e listas/tuplas
    if isinstance(x, (list, tuple)):
        return list(x)
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass
    return [x] if _is_num(x) else None

def _derive_deltas(times):
    if not times: return []
    out = [times[0]]
    for i in range(1, len(times)):
        out.append(times[i] - times[i-1])
    return out

def _derive_times(deltas):
    out, acc = [], 0.0
    for d in deltas:
        acc += d
        out.append(acc)
    return out

def _pick_first(d, keys):
    for k in keys:
        if k in d:
            return k
    return None

def convert_file(fn, split_key):
    path = os.path.join(BASE, fn)
    with open(path, 'rb') as f:
        data = pickle.load(f)

    # já plano?
    if any(k in data for k in ('time_seqs','time_delta_seqs','type_seqs')):
        print(f'{fn}: já está no formato plano; pulando.')
        return

    if split_key not in data:
        raise ValueError(f'{fn} não tem a chave "{split_key}". Chaves: {list(data.keys())}')

    sequences = data[split_key]
    if not isinstance(sequences, list) or not sequences or not isinstance(sequences[0], list):
        raise ValueError(f'{fn}: formato inesperado; esperado list[ list[ dict ] ].')

    T_all, D_all, Y_all = [], [], []
    min_type, max_type = None, None

    for si, seq in enumerate(sequences):
        if not seq:  # sequência vazia
            T_all.append([]); D_all.append([]); Y_all.append([]); continue

        # cada item deve ser um dict de evento
        if not isinstance(seq[0], dict):
            raise ValueError(f'{fn}: sequência {si} não é lista de dicts.')

        times, deltas, types = [], [], []

        for ev in seq:
            # tipo
            tk = _pick_first(ev, TYPE_KEYS)
            if tk is None:
                raise ValueError(f'{fn}: evento sem chave de tipo. Chaves: {list(ev.keys())}')
            y = ev[tk]
            if isinstance(y, (list, tuple)):
                # raro: alguns dumps guardam 1-hot; pega argmax
                if len(y)==0:
                    continue
                y = int(max(range(len(y)), key=lambda i: y[i]))
            else:
                y = int(y)
            types.append(y)
            min_type = y if min_type is None else min(min_type, y)
            max_type = y if max_type is None else max(max_type, y)

            # tempo
            ak = _pick_first(ev, TIME_KEYS)
            dk = _pick_first(ev, DELTA_KEYS)
            a = ev[ak] if ak in ev else None
            d = ev[dk] if dk in ev else None

            # normaliza pra lista
            a_list = _to_list(a) if a is not None else None
            d_list = _to_list(d) if d is not None else None

            # a maioria dos dumps guarda UM valor por evento (não lista)
            if a_list is not None and len(a_list)==1: a_list = a_list[0]
            if d_list is not None and len(d_list)==1: d_list = d_list[0]

            times.append(float(a_list) if a_list is not None else None)
            deltas.append(float(d_list) if d_list is not None else None)

        # reconciliar
        # se só tempos absolutos
        if all(t is not None for t in times) and any(d is None for d in deltas):
            deltas = _derive_deltas(times)
        # se só deltas
        if all(d is not None for d in deltas) and any(t is None for t in times):
            times = _derive_times(deltas)

        # sanidade: cortar no menor comprimento válido
        m = min(len(times), len(deltas), len(types))
        times, deltas, types = times[:m], deltas[:m], types[:m]

        T_all.append(times)
        D_all.append(deltas)
        Y_all.append(types)

    # normaliza para 0-based se necessário
    if min_type is not None and min_type >= 1 and (min_type == 1):
        # shift -1 se não houver 0 em nenhum lugar
        has_zero = False
        for seq in Y_all:
            if any(t == 0 for t in seq):
                has_zero = True; break
        if not has_zero:
            for i in range(len(Y_all)):
                Y_all[i] = [t-1 for t in Y_all[i]]
            max_type -= 1
            print(f'{fn}: tipos pareciam 1-based; convertidos para 0-based.')

    dim = data.get('dim_process')
    if dim is None and max_type is not None:
        dim = int(max_type) + 1
    if dim is None:
        raise ValueError(f'{fn}: não foi possível definir dim_process.')

    out = {
        'time_seqs': T_all,
        'time_delta_seqs': D_all,
        'type_seqs': Y_all,
        'dim_process': int(dim),
        'type_count': int(dim),
    }

    with open(path, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'{fn}: convertido → formato plano; dim_process={dim} | seqs={len(T_all)}')

def harmonize_dim():
    files = list(MAP.keys())
    mx = -1
    for fn in files:
        d = pickle.load(open(os.path.join(BASE, fn),'rb'))
        for seq in d.get('type_seqs', []):
            if seq:
                m = max(seq)
                if m > mx: mx = m
    if mx >= 0:
        dim = int(mx) + 1
        for fn in files:
            p = os.path.join(BASE, fn)
            d = pickle.load(open(p,'rb'))
            d['dim_process'] = dim
            d['type_count']  = dim
            pickle.dump(d, open(p,'wb'), protocol=pickle.HIGHEST_PROTOCOL)
            print(f'{fn}: dim_process normalizado para {dim}')

if __name__ == '__main__':
    for fn, key in MAP.items():
        convert_file(fn, key)
    harmonize_dim()
    print('OK: splits regravados no formato aceito pelo EasyTPP.')
