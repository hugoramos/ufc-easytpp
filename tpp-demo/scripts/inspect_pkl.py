# scripts/inspect_pkl.py
import pickle, os, sys

base = './data/taxi'
files = ['train.pkl','dev.pkl','test.pkl']
for fn in files:
    path = os.path.join(base, fn)
    with open(path,'rb') as f:
        d = pickle.load(f)
    # muitos dumps no formato EasyTPP têm essas chaves:
    time_keys = [k for k in ['time_seqs','times'] if k in d]
    type_keys = [k for k in ['type_seqs','types','event_types'] if k in d]
    dim = d.get('dim_process')
    all_types = []
    if type_keys:
        for seq in d[type_keys[0]]:
            all_types.extend(seq)
    print('===', fn, '===')
    print('dim_process no arquivo:', dim)
    if all_types:
        mx = max(all_types)
        print('maior id de tipo encontrado:', mx, '=> num_event_types sugerido:', mx+1)
    else:
        print('não achei sequências de tipos; chaves presentes:', list(d.keys()))
