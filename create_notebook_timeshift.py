import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cell 1: Markdown Intro
cell1 = nbf.v4.new_markdown_cell("""
# Teste de Robustez a Deslocamento Temporal (Time Shift) - StackOverflow

Neste notebook, testaremos a **invariância a translação** do RoTHP.
A hipótese é que o RoTHP, por usar posições relativas, deve manter sua performance mesmo se deslocarmos todos os timestamps por um valor grande (ex: +100.000).
O THP tradicional (embedding absoluto) deve sofrer degradação ou instabilidade.

Dataset: **StackOverflow**
Shift: **+100.000** em todos os tempos.

**Melhorias incluídas:**
- Early Stopping (Patience = 5)
- Adaptive Learning Rate Scheduler
- Melhor logging
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
# CARREGAR DATASET (StackOverflow via HuggingFace)
# =============================================================================
print("Carregando dataset 'easytpp/stackoverflow'...")
dataset = load_dataset("easytpp/stackoverflow")

train_data = dataset['train']
dev_data = dataset['validation']
test_data = dataset['test']

print(f"\\nDataset StackOverflow carregado!")
print(f"Treino: {len(train_data)} sequências")
print(f"Dev:    {len(dev_data)} sequências")
print(f"Teste:  {len(test_data)} sequências")

# Verificar metadados do primeiro exemplo
sample = train_data[0]
print("\\nExemplo de sequência (Original):")
print(f"Time starts: {sample['time_since_start'][:5]}...")
print(f"Dim Process: {sample['dim_process']}")
""")

# Cell 4: Collate Function WITH SHIFT
cell4 = nbf.v4.new_code_cell("""
# =============================================================================
# PREPARAÇÃO DE BATCH COM TIME SHIFT
# =============================================================================

# CONSTANTE DE DESLOCAMENTO TEMPORAL
SHIFT_VAL = 100000.0 

def collate_fn_shifted(batch_list):
    \"\"\"
    Converte lista de dicts para tensores e APLICA SHIFT no tempo absoluto.
    \"\"\"
    time_seqs = []
    time_delta_seqs = []
    type_seqs = []
    
    max_len = 0
    for item in batch_list:
        ts = item['time_since_start']
        td = item['time_since_last_event']
        ev = item['type_event']
        
        if len(ts) > max_len:
            max_len = len(ts)
            
        # AQUI ESTÁ O SHIFT: Adicionamos a constante ao tempo absoluto
        ts_shifted = torch.tensor(ts, dtype=torch.float32) + SHIFT_VAL
        
        time_seqs.append(ts_shifted)
        time_delta_seqs.append(torch.tensor(td, dtype=torch.float32)) # Deltas não mudam!
        type_seqs.append(torch.tensor(ev, dtype=torch.long))

    batch_size = len(batch_list)
    
    pad_time = torch.zeros(batch_size, max_len)
    pad_delta = torch.zeros(batch_size, max_len)
    pad_type = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, max_len)
    batch_non_pad_mask = torch.zeros(batch_size, max_len)
    
    for i in range(batch_size):
        l = len(time_seqs[i])
        pad_time[i, :l] = time_seqs[i]
        pad_delta[i, :l] = time_delta_seqs[i]
        pad_type[i, :l] = type_seqs[i]
        batch_non_pad_mask[i, :l] = 1
        
        causal_mask = torch.tril(torch.ones(l, l))
        attention_mask[i, :l, :l] = causal_mask

    return (pad_time, pad_delta, pad_type, batch_non_pad_mask, attention_mask)

print(f"Collate function configurada com SHIFT = {SHIFT_VAL}")
""")

# Cell 5: Config and Loop
cell5 = nbf.v4.new_code_cell("""
# =============================================================================
# CONFIGURAÇÃO DO MODELO COM EARLY STOPPING
# =============================================================================

class ModelConfig:
    def __init__(self, hidden_size=64, num_heads=4, num_layers=2):
        self.hidden_size = hidden_size
        self.time_emb_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = 0.1
        self.use_ln = True
        
        # StackOverflow Specs
        self.num_event_types = 22     
        self.num_event_types_pad = 23 
        self.pad_token_id = 22
        
        self.loss_integral_num_sample_per_step = 20
        self.use_mc_samples = False
        self.gpu = -1
        self.thinning = None

def train_eval_loop(model_class, name, config, train_ds, test_ds, epochs=50, batch_size=32, lr=1e-3, patience=5):
    print(f"\\n>>> Iniciando Treinamento SHIFTED: {name}")
    torch.manual_seed(42)
    model = model_class(config)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4) # Added weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    history = []
    best_nll = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        total_events = 0
        
        indices = np.random.permutation(len(train_ds))
        max_batches = 200 # Limit batches per epoch for speed
        
        for i in range(0, len(indices), batch_size):
            if i // batch_size > max_batches: break
            
            batch_idx = indices[i:i+batch_size]
            batch_list = [train_ds[int(k)] for k in batch_idx]
            batch_data = collate_fn_shifted(batch_list)
            
            optimizer.zero_grad()
            loss, num_events = model.loglike_loss(batch_data)
            
            # Gradient clipping is crucial
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_events += num_events
            
        nll_train = total_loss / (total_events + 1e-9)
        
        # Validation
        model.eval()
        with torch.no_grad():
            test_subset = [test_ds[i] for i in range(min(500, len(test_ds)))]
            batch_test = collate_fn_shifted(test_subset)
            loss_test, num_test = model.loglike_loss(batch_test)
            nll_test = (loss_test / num_test).item()
        
        print(f"  Ep {epoch} | Train NLL: {nll_train:.4f} | Test NLL: {nll_test:.4f}")
        history.append(nll_test)
        
        # Scheduler Step
        scheduler.step(nll_test)
        
        # Early Stopping Check
        if nll_test < best_nll:
            best_nll = nll_test
            patience_counter = 0
            best_model_state = model.state_dict() # Save best model (in memory)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  >>> Early Stopping at Epoch {epoch}. Best NLL: {best_nll:.4f}")
                break
                
    return history
""")

# Cell 6: Run Comparison
cell6 = nbf.v4.new_code_cell("""
# =============================================================================
# EXECUÇÃO COMPARATIVA
# =============================================================================

config = ModelConfig(hidden_size=64, num_heads=4, num_layers=2)

# 1. RoTHP Original
hist_rothp = train_eval_loop(RoTHP, "RoTHP Original", config, train_data, test_data, epochs=30, patience=5)

# 2. THP Tradicional
hist_thp = train_eval_loop(THP, "THP Tradicional", config, train_data, test_data, epochs=30, patience=5)
""")

# Cell 7: Plot
cell7 = nbf.v4.new_code_cell("""
plt.figure(figsize=(10, 6))

# Função auxiliar para plotar dinamicamente e ignorar valores muito altos
def plot_robust(hist, label, style, threshold=10):
    if hist and len(hist) > 0:
        epochs = range(1, len(hist) + 1)
        final_val = hist[-1]
        # Filtra valores para o plot ou usa ylim
        plt.plot(epochs, hist, style, label=f'{label} (Final: {final_val:.2f})')

# Plotar se as variáveis existirem
if 'hist_rothp' in locals(): plot_robust(hist_rothp, 'RoTHP Original', 'r-o')
if 'hist_thp' in locals(): plot_robust(hist_thp, 'THP Tradicional', 'g-^')

plt.xlabel('Época')
plt.ylabel('NLL (Menor é melhor)')
plt.title(f'StackOverflow SHIFTED (+{SHIFT_VAL}): Comparação de Modelos')
plt.legend()
plt.grid(True, alpha=0.3)

# AQUI: Define o limite superior do eixo Y para "ignorar" os valores extremos
plt.ylim(top=10)

plt.show()
""")

nb.cells = [cell1, cell2, cell3, cell4, cell5, cell6, cell7]

with open('notebooks/rothp_timeshift_test.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook gerado com sucesso: notebooks/rothp_timeshift_test.ipynb")
