# scripts/flatten_to_event_dicts.py
import os, pickle

BASE = './data/taxi'
SPLITS = [('train.pkl','train'), ('dev.pkl','dev'), ('test.pkl','test')]

def make_event_dicts(times, deltas, types):
    events = []
    n = min(len(times), len(deltas), len(types))
    for i in range(n):
        events.append({
            'time_since_start': float(times[i]),
            'time_since_last_event': float(deltas[i]),
            'type_event': int(types[i]),
            # campos auxiliares (opcionais, mas úteis p/ debug):
            'idx_event': i
        })
    return events

def process_file(fn, split_key):
    p = os.path.join(BASE, fn)
    data = pickle.load(open(p, 'rb'))

    # Aceita dois formatos de entrada:
    # A) data[split_key] já é dict com time_seqs/time_delta_seqs/type_seqs
    # B) data tem essas chaves na raiz (caso tenha sobrado algo antigo)
    if split_key in data and isinstance(data[split_key], dict):
        split_dict = data[split_key]
    elif all(k in data for k in ('time_seqs','time_delta_seqs','type_seqs')):
        split_dict = {k: data[k] for k in ('time_seqs','time_delta_seqs','type_seqs')}
    else:
        raise ValueError(f'{fn}: não achei dict com time_seqs/time_delta_seqs/type_seqs para o split "{split_key}". Chaves: {list(data.keys())}')

    T = split_dict['time_seqs']
    D = split_dict['time_delta_seqs']
    Y = split_dict['type_seqs']

    assert isinstance(T, list) and isinstance(D, list) and isinstance(Y, list), 'Esperado listas de sequências'
    assert len(T) == len(D) == len(Y), 'time_seqs, time_delta_seqs e type_seqs devem ter o mesmo número de sequências'

    seqs_as_event_dicts = []
    for times, deltas, types in zip(T, D, Y):
        seqs_as_event_dicts.append(make_event_dicts(times, deltas, types))

    out = { split_key: seqs_as_event_dicts }
    # preserve dim_process
    dim = data.get('dim_process') or data.get('type_count')
    if dim is not None:
        out['dim_process'] = int(dim)
        out['type_count'] = int(dim)

    pickle.dump(out, open(p, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    print(f'{fn}: convertido para List[List[Dict]] com chave "{split_key}", dim_process={out.get("dim_process")} | seqs={len(seqs_as_event_dicts)}')

if __name__ == '__main__':
    for fn, key in SPLITS:
        process_file(fn, key)
    print('OK: splits no formato esperado pelo loader (lista de dicionários de eventos).')
