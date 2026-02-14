# scripts/rewrap_for_easytpp.py
import os, pickle

BASE = './data/taxi'
SPLITS = [('train.pkl','train'), ('dev.pkl','dev'), ('test.pkl','test')]

def rewrap(fn, split_key):
    p = os.path.join(BASE, fn)
    d = pickle.load(open(p,'rb'))

    # Se já está no formato esperado, não mexe
    if split_key in d and isinstance(d[split_key], dict):
        print(f'{fn}: já contém "{split_key}". OK.')
        return

    # Precisa ter time_seqs/… na raiz para reembrulhar
    keys_needed = {'time_seqs','time_delta_seqs','type_seqs'}
    if not keys_needed.issubset(d.keys()):
        raise ValueError(f'{fn}: não achei {keys_needed} na raiz. Chaves: {list(d.keys())}')

    dim = d.get('dim_process')
    split_dict = {
        'time_seqs': d['time_seqs'],
        'time_delta_seqs': d['time_delta_seqs'],
        'type_seqs': d['type_seqs'],
    }

    out = {split_key: split_dict}
    if dim is not None:
        out['dim_process'] = int(dim)
        out['type_count'] = int(dim)

    pickle.dump(out, open(p,'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    print(f'{fn}: reembrulhado -> possui chave "{split_key}" e dim_process={out.get("dim_process")}')

def main():
    for fn, key in SPLITS:
        rewrap(fn, key)
    print('OK: arquivos reembrulhados para o formato esperado pelo loader.')

if __name__ == '__main__':
    main()
