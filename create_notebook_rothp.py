import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cell 1: Markdown Intro
cell1 = nbf.v4.new_markdown_cell("""
# Comparação RoTHP vs THP no Dataset Retweet

Neste notebook, testaremos a hipótese do paper RoTHP em um dataset real (**Retweet**): 
**"O RoTHP é mais robusto em dados reais e ruidosos devido à sua invariância a translação."**

Compararemos:
1. **RoTHP Original**: Apenas RoPE na atenção.
2. **RoTHP Hybrid**: RoPE na atenção + Embedding Absoluto na entrada.
3. **THP Tradicional**: Apenas Embedding Absoluto.

Usaremos o dataset oficial `easytpp/retweet` hospedado no HuggingFace.
""")

# Cell 2: Imports
cell2 = nbf.v4.new_code_cell("""
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

project_root = '/Users/hugoramossoares/Sites/EasyTemporalPointProcess'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import Models
from easy_tpp.model.torch_model.torch_rothp import RoTHP
from easy_tpp.model.torch_model.torch_thp import THP

print("Bibliotecas carregadas.")
""")

# Cell 3: Load Dataset
cell3 = nbf.v4.new_code_cell("""
# =============================================================================
# CARREGAR DATASET (Retweet via HuggingFace)
# =============================================================================
print("Carregando dataset 'easytpp/retweet'...")
dataset = load_dataset("easytpp/retweet")

train_data = dataset['train']
dev_data = dataset['validation']
test_data = dataset['test']

print(f"\\nDataset Retweet carregado!")
print(f"Treino: {len(train_data)} sequências")
print(f"Dev:    {len(dev_data)} sequências")
print(f"Teste:  {len(test_data)} sequências")

# Verificar metadados do primeiro exemplo
sample = train_data[0]
print("\\nExemplo de sequência:")
print(f"Event Types: {sample['type_event'][:10]}...")
print(f"Time deltas: {sample['time_since_last_event'][:10]}...")
print(f"Dim Process (Num Event Types): {sample['dim_process']}")
""")

# Cell 4: Collate Function
cell4 = nbf.v4.new_code_cell("""
# =============================================================================
# PREPARAÇÃO DE BATCH (Adaptado para formato HuggingFace)
# =============================================================================
def collate_fn(batch_list):
    \"\"\"Converte lista de dicts (HF) para tensores de batch do EasyTPP.\"\"\"
    time_seqs = []
    time_delta_seqs = []
    type_seqs = []
    
    max_len = 0
    for item in batch_list:
        # Converter listas python para tensores, se necessário
        ts = item['time_since_start']
        td = item['time_since_last_event']
        ev = item['type_event']
        
        if len(ts) > max_len:
            max_len = len(ts)
            
        time_seqs.append(torch.tensor(ts, dtype=torch.float32))
        time_delta_seqs.append(torch.tensor(td, dtype=torch.float32))
        type_seqs.append(torch.tensor(ev, dtype=torch.long))

    batch_size = len(batch_list)
    
    # Padding
    pad_time = torch.zeros(batch_size, max_len)
    pad_delta = torch.zeros(batch_size, max_len)
    pad_type = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, max_len) # mask de atenção pad
    batch_non_pad_mask = torch.zeros(batch_size, max_len)
    
    for i in range(batch_size):
        l = len(time_seqs[i])
        pad_time[i, :l] = time_seqs[i]
        pad_delta[i, :l] = time_delta_seqs[i]
        pad_type[i, :l] = type_seqs[i]
        batch_non_pad_mask[i, :l] = 1
        
        # Causal Mask (Triangular Superior)
        causal_mask = torch.tril(torch.ones(l, l))
        attention_mask[i, :l, :l] = causal_mask

    return (pad_time, pad_delta, pad_type, batch_non_pad_mask, attention_mask)

# Teste rápido do Collate
sample_batch = [train_data[0], train_data[1]]
batch_tensors = collate_fn(sample_batch)
print("Batch Tensors Shapes:", [t.shape for t in batch_tensors])
""")

# Cell 5: Config and Loop
cell5 = nbf.v4.new_code_cell("""
# =============================================================================
# CONFIGURAÇÃO DO MODELO
# =============================================================================

class ModelConfig:
    def __init__(self, hidden_size=64, num_heads=4, num_layers=2):
        self.hidden_size = hidden_size
        self.time_emb_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = 0.1
        self.use_ln = True
        
        # Retweet Specs
        self.num_event_types = 3      # 3 tipos: small, medium, large
        self.num_event_types_pad = 4  # +1 para pad
        self.pad_token_id = 3
        
        self.loss_integral_num_sample_per_step = 20
        self.use_mc_samples = False
        self.gpu = -1 # CPU por enquanto
        self.thinning = None

def train_eval_loop(model_class, name, config, train_ds, test_ds, epochs=5, batch_size=32, lr=1e-3):
    print(f"\\n>>> Iniciando Treinamento: {name}")
    torch.manual_seed(42)
    model = model_class(config)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    history = []
    
    for epoch in range(1, epochs+1):
        # Treino
        model.train()
        total_loss = 0
        total_events = 0
        
        indices = np.random.permutation(len(train_ds))
        max_batches = 200 
        
        for i in range(0, len(indices), batch_size):
            if i // batch_size > max_batches: break
            
            batch_idx = indices[i:i+batch_size]
            batch_list = [train_ds[int(k)] for k in batch_idx]
            batch_data = collate_fn(batch_list)
            
            optimizer.zero_grad()
            loss, num_events = model.loglike_loss(batch_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_events += num_events
            
        nll_train = total_loss / (total_events + 1e-9)
        
        # Avaliação
        model.eval()
        with torch.no_grad():
            test_subset = [test_ds[i] for i in range(min(500, len(test_ds)))]
            batch_test = collate_fn(test_subset)
            loss_test, num_test = model.loglike_loss(batch_test)
            nll_test = (loss_test / num_test).item()
        
        print(f"  Ep {epoch} | Train NLL: {nll_train:.4f} | Test NLL: {nll_test:.4f}")
        history.append(nll_test)
        
    return history
""")

# Cell 6: Run Comparison
cell6 = nbf.v4.new_code_cell("""
# =============================================================================
# EXECUÇÃO COMPARATIVA
# =============================================================================

# Configuração Moderada (2 Layers, 64 Hidden)
config = ModelConfig(hidden_size=64, num_heads=4, num_layers=2)

# 1. RoTHP Original
hist_rothp = train_eval_loop(RoTHP, "RoTHP Original", config, train_data, test_data, epochs=5)

# 2. THP Tradicional
hist_thp = train_eval_loop(THP, "THP Tradicional", config, train_data, test_data, epochs=5)
""")

# Cell 7: Plot
cell7 = nbf.v4.new_code_cell("""
plt.figure(figsize=(10, 6))
epochs = range(1, 6)

plt.plot(epochs, hist_rothp, 'r-o', label='RoTHP Original')
plt.plot(epochs, hist_thp, 'g-^', label='THP Tradicional')

plt.xlabel('Época')
plt.ylabel('NLL (Menor é melhor)')
plt.title('Retweet Real Data: Comparação de Modelos (2 Layers)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
""")

nb.cells = [cell1, cell2, cell3, cell4, cell5, cell6, cell7]

with open('notebooks/rothp_retweet_comparison_v2.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook gerado com sucesso: notebooks/rothp_retweet_comparison_v2.ipynb")
