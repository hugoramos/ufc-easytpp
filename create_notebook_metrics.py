import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cell 1: Markdown Intro
cell1 = nbf.v4.new_markdown_cell("""
# Comparação de Modelos: NLL, Acurácia e RMSE - Dataset Sintético (Long Range)

Neste notebook, compararemos **NHP, THP e THP-ExpDecay** em um **dataset sintético de Hawkes Process** projetado especificamente para testar dependências de longo alcance.

**Características do Dataset:**
- Sequências longas: 300 a 600 eventos (vs ~100 no Retweet)
- Processo gerador: Hawkes Multivariado com decaimento lento (memória longa)
- 5 tipos de eventos

O objetivo é verificar se, em um cenário onde o histórico distante importa e as sequências são longas, a arquitetura Transformer consegue superar a RNN (NHP), que tipicamente sofre com vanishing gradients em sequências longas.

**Configurações:**
- Model Capacity: Hidden Size 128, 4 Layers, 8 Heads
- Normalização temporal aplicada
- Máscara de atenção correta (causal + padding)
""")

# Cell 2: Imports
cell2 = nbf.v4.new_code_cell("""
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import DataLoader

project_root = '/Users/hugoramossoares/Sites/EasyTemporalPointProcess'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import Models
from easy_tpp.model.torch_model.torch_rothp import RoTHP
from easy_tpp.model.torch_model.torch_thp import THP
from easy_tpp.model.torch_model.torch_nhp import NHP
from easy_tpp.model.torch_model.torch_thp_expdecay import THPExpDecay
from easy_tpp.model.torch_model.torch_attnhp import AttNHP

print("Bibliotecas carregadas.")
""")

# Cell 3: Load Dataset
cell3 = nbf.v4.new_code_cell("""
# =============================================================================
# CARREGAR DATASET SINTÉTICO (Long Range)
# =============================================================================
import pickle

dataset_path = os.path.join(project_root, 'data/synthetic_long')
print(f"Carregando dataset sintético de '{dataset_path}'...")

if not os.path.exists(dataset_path):
    # Tentar caminho relativo se rodar localmente fora do root
    dataset_path = '../data/synthetic_long'
    print(f"Tentando caminho relativo: '{dataset_path}'...")

with open(os.path.join(dataset_path, 'train.pkl'), 'rb') as f:
    train_data = pickle.load(f)

with open(os.path.join(dataset_path, 'test.pkl'), 'rb') as f:
    test_data = pickle.load(f)

print(f"\\nDataset carregado!")
print(f"Treino: {len(train_data)} sequências")
print(f"Teste:  {len(test_data)} sequências")

# =============================================================================
# EXTRAÇÃO AUTOMÁTICA DE METADADOS
# =============================================================================
def get_dataset_specs(data):
    max_type = 0
    for item in data:
        types = item['type_event']
        if len(types) > 0:
            current_max = max(types)
            if current_max > max_type:
                max_type = current_max
    
    # Assumindo IDs sequenciais 0, 1, ..., max_type
    num_types = max_type + 1
    return num_types

# Calcular specs baseados no treino
NUM_EVENT_TYPES = get_dataset_specs(train_data)
PAD_TOKEN_ID = NUM_EVENT_TYPES
NUM_EVENT_TYPES_PAD = NUM_EVENT_TYPES + 1

print(f"\\nMetadados detectados automaticamente:")
print(f"Num Event Types: {NUM_EVENT_TYPES}")
print(f"Pad Token ID:    {PAD_TOKEN_ID}")
print(f"Total Vocab:     {NUM_EVENT_TYPES_PAD}")

# Estatísticas de comprimento de sequência
seq_lengths = [len(item['time_since_start']) for item in train_data]
print(f"\\nEstatísticas de Comprimento de Sequência (Treino):")
print(f"  Média:   {np.mean(seq_lengths):.1f}")
print(f"  Mediana: {np.median(seq_lengths):.1f}")
print(f"  Min:     {np.min(seq_lengths)}")
print(f"  Max:     {np.max(seq_lengths)}")
print(f"  Std:     {np.std(seq_lengths):.1f}")

# Calcular escala temporal para normalização
all_deltas = []
for item in train_data:
    td = item['time_since_last_event']
    all_deltas.extend([d for d in td if d > 0])

TIME_SCALE = np.mean(all_deltas)
print(f"\\nEscala Temporal (média dos deltas no treino):")
print(f"  Mean delta:   {TIME_SCALE:.4f}")
print(f"  Median delta: {np.median(all_deltas):.4f}")
print(f"  Max delta:    {np.max(all_deltas):.4f}")
print(f"  Min delta:    {np.min(all_deltas):.4f}")
print(f"  >> Todos os tempos serão divididos por {TIME_SCALE:.4f} para normalização")
""")

# Cell 4: Collate Function WITH SHIFT
cell4 = nbf.v4.new_code_cell("""
# =============================================================================
# PREPARAÇÃO DE BATCH COM NORMALIZAÇÃO TEMPORAL
# =============================================================================

SHIFT_VAL = 0.0 

def collate_fn_shifted(batch_list):
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
            
        # NORMALIZAÇÃO TEMPORAL
        # 1. Subtrair t[0] para começar em 0
        ts_raw = torch.tensor(ts, dtype=torch.float64) + SHIFT_VAL
        ts_normalized = ts_raw - ts_raw[0]
        
        # 2. Dividir pela escala temporal (mean delta) para trazer valores para ~O(1)
        #    Isso é CRUCIAL para o THP, que usa decay LINEAR: factor * delta_t
        #    Sem normalização, delta_t pode ser ~1000+, explodindo a intensidade
        ts_final = (ts_normalized / TIME_SCALE).to(torch.float32)
        td_final = (torch.tensor(td, dtype=torch.float64) / TIME_SCALE).to(torch.float32)
        
        time_seqs.append(ts_final)
        time_delta_seqs.append(td_final) 
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
        
        # MÁSCARA DE ATENÇÃO: Combina causal + padding (convenção EasyTPP: 1 = BLOQUEAR)
        # 1. Causal: triu(k=1) bloqueia posições futuras
        causal_mask = torch.triu(torch.ones(max_len, max_len), diagonal=1)
        # 2. Padding: bloqueia COLUNAS de padding (key positions que são padding)
        causal_mask[:, l:] = 1
        # 3. Padding rows: queries de padding bloqueiam tudo
        causal_mask[l:, :] = 1
        attention_mask[i] = causal_mask

    return (pad_time, pad_delta, pad_type, batch_non_pad_mask, attention_mask)

print(f"Collate function configurada com SHIFT = {SHIFT_VAL}")
""")

# Cell 5: Config
cell5 = nbf.v4.new_code_cell("""
class ThinningConfig:
    def __init__(self, dtime_max=5.0, num_sample=200, num_exp=500):
        self.num_sample = num_sample     # Amostras para calcular a integral/predição
        self.num_exp = num_exp           # Amostras para o algoritmo de thinning
        self.over_sample_rate = 10.0
        self.patience_counter = 5
        self.num_samples_boundary = 20
        self.dtime_max = dtime_max       # Horizonte máximo de tempo para predição (em unidades normalizadas)

class ModelConfig:
    def __init__(self, num_types, pad_id, num_types_pad, hidden_size=64, num_heads=4, num_layers=2):
        self.hidden_size = hidden_size
        self.time_emb_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = 0.1
        self.use_ln = True
        
        # Specs Automáticos
        self.num_event_types = num_types
        self.num_event_types_pad = num_types_pad
        self.pad_token_id = pad_id
        
        self.loss_integral_num_sample_per_step = 20
        self.use_mc_samples = False
        self.gpu = -1
        
        # IMPORTANTE: Configuração de Thinning para habilitar .predict()
        self.thinning = ThinningConfig()
        
        # NHP Specifics
        self.model_specs = {'beta': 1.0, 'bias': True}

    def __str__(self):
        return str(self.__dict__)

print("Configuração definida com Thinning ativado.")
""")

# Cell 6: Train and Evaluate Function
cell6 = nbf.v4.new_code_cell("""
def compute_metrics(model, test_ds):
    model.eval()
    
    total_acc = 0
    total_rmse = 0
    total_events = 0
    
    # Avaliar em subset do teste para ser rápido (ex: 100 sequências)
    subset_size = min(100, len(test_ds))
    subset = [test_ds[i] for i in range(subset_size)]
    
    # Processar um por um ou em batches pequenos para evitar estourar memória na amostragem
    batch_size_eval = 10 
    
    with torch.no_grad():
        for i in range(0, subset_size, batch_size_eval):
            batch_list = subset[i:i+batch_size_eval]
            batch = collate_fn_shifted(batch_list)
            
            # Unpack batch para pegar targets
            # batch = (time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask)
            _, time_delta_target, type_target, mask_target, _ = batch
            
            # Predict
            # dtimes_pred: [B, L-1] (deltas previstos)
            # types_pred:  [B, L-1] (tipos previstos)
            dtimes_pred, types_pred = model.predict_one_step_at_every_event(batch)
            
            # Ajustar targets (removemos o primeiro evento pois não prevemos o passado, 
            # e o predict já remove o último evento da entrada para prever o próximo)
            # O output do predict alinha com o target do índice 1 ao fim.
            
            target_types = type_target[:, 1:]
            target_deltas = time_delta_target[:, 1:]
            target_mask = mask_target[:, 1:]
            
            # Calcular Acurácia (Type)
            correct = (types_pred == target_types) * target_mask
            total_acc += correct.sum().item()
            
            # Calcular RMSE (Time)
            se = ((dtimes_pred - target_deltas) ** 2) * target_mask
            total_rmse += se.sum().item()
            
            total_events += target_mask.sum().item()
            
    avg_acc = total_acc / (total_events + 1e-9)
    avg_rmse = np.sqrt(total_rmse / (total_events + 1e-9))
    
    return avg_acc, avg_rmse

def train_eval_loop(model_class, name, config, train_ds, test_ds, epochs=50, batch_size=64, lr=5e-4, patience=15):
    import time as time_module
    print(f"\\n>>> Treinando: {name}")
    torch.manual_seed(42)
    model = model_class(config)
    
    # Contar parâmetros
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parâmetros treináveis: {num_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    history_nll = []
    best_nll = float('inf')
    patience_counter = 0
    clip_val = 1.0 
    
    t_train_start = time_module.time()
    
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        total_events = 0
        
        indices = np.random.permutation(len(train_ds))
        max_batches = 100
        
        for i in range(0, len(indices), batch_size):
            if i // batch_size > max_batches: break
            batch_idx = indices[i:i+batch_size]
            batch_list = [train_ds[int(k)] for k in batch_idx]
            
            try:
                batch_data = collate_fn_shifted(batch_list)
                optimizer.zero_grad()
                loss, num_events = model.loglike_loss(batch_data)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                optimizer.step()
                
                total_loss += loss.item()
                total_events += num_events
            except Exception as e:
                continue
            
        nll_train = total_loss / (total_events + 1e-9)
        
        # Validation NLL (batches menores para evitar padding excessivo)
        model.eval()
        val_loss_total = 0
        val_events_total = 0
        with torch.no_grad():
            test_subset = [test_ds[i] for i in range(min(200, len(test_ds)))]
            val_batch_size = 32
            for vi in range(0, len(test_subset), val_batch_size):
                val_batch = collate_fn_shifted(test_subset[vi:vi+val_batch_size])
                vl, vn = model.loglike_loss(val_batch)
                if not (torch.isnan(vl) or torch.isinf(vl)):
                    val_loss_total += vl.item()
                    val_events_total += vn
        nll_test = val_loss_total / (val_events_total + 1e-9)
        
        print(f"  Ep {epoch} | Train NLL: {nll_train:.4f} | Val NLL: {nll_test:.4f}")
        history_nll.append(nll_test)
        
        scheduler.step(nll_test)
        
        if nll_test < best_nll:
            best_nll = nll_test
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  >>> Early Stopping.")
                break
    
    t_train_end = time_module.time()
    train_time = t_train_end - t_train_start
    num_epochs_run = len(history_nll)
    
    # Final Metrics Calculation
    print(f"  Calculando métricas finais (Acc, RMSE)...")
    t_eval_start = time_module.time()
    final_acc, final_rmse = compute_metrics(model, test_ds)
    t_eval_end = time_module.time()
    eval_time = t_eval_end - t_eval_start
    
    total_time = train_time + eval_time
    print(f"  >>> Resultado Final {name}: Acc={final_acc:.4f}, RMSE={final_rmse:.4f}")
    print(f"  >>> Tempo: Treino={train_time:.1f}s ({num_epochs_run} épocas, {train_time/num_epochs_run:.1f}s/ep) | Eval={eval_time:.1f}s | Total={total_time:.1f}s")
    print(f"  >>> Parâmetros: {num_params:,}")
    
    return history_nll, final_acc, final_rmse, train_time, eval_time, num_params, model
""")

# Cell 7: Run Comparison
cell7 = nbf.v4.new_code_cell("""
config = ModelConfig(
    num_types=NUM_EVENT_TYPES,
    pad_id=PAD_TOKEN_ID,
    num_types_pad=NUM_EVENT_TYPES_PAD,
    hidden_size=64, 
    num_heads=2, 
    num_layers=2
)

results = {}

# Executar cada modelo individualmente (comente/descomente conforme necessário)

# --- NHP ---
hist, acc, rmse, t_train, t_eval, n_params, trained_model = train_eval_loop(NHP, "NHP (RNN)", config, train_data, test_data)
results["NHP (RNN)"] = {'hist': hist, 'acc': acc, 'rmse': rmse, 'train_time': t_train, 'eval_time': t_eval, 'total_time': t_train + t_eval, 'num_params': n_params, 'model': trained_model}

# --- THP ---
hist, acc, rmse, t_train, t_eval, n_params, trained_model = train_eval_loop(THP, "THP Tradicional", config, train_data, test_data)
results["THP Tradicional"] = {'hist': hist, 'acc': acc, 'rmse': rmse, 'train_time': t_train, 'eval_time': t_eval, 'total_time': t_train + t_eval, 'num_params': n_params, 'model': trained_model}

# --- THP-ExpDecay (PROPOSTO: Transformer + Exponential Decay) ---
hist, acc, rmse, t_train, t_eval, n_params, trained_model = train_eval_loop(THPExpDecay, "THP-ExpDecay (Proposto)", config, train_data, test_data)
results["THP-ExpDecay (Proposto)"] = {'hist': hist, 'acc': acc, 'rmse': rmse, 'train_time': t_train, 'eval_time': t_eval, 'total_time': t_train + t_eval, 'num_params': n_params, 'model': trained_model}

# --- RoTHP ---
#hist, acc, rmse, t_train, t_eval, n_params, trained_model = train_eval_loop(RoTHP, "RoTHP (Transformer)", config, train_data, test_data)
#results["RoTHP (Transformer)"] = {'hist': hist, 'acc': acc, 'rmse': rmse, 'train_time': t_train, 'eval_time': t_eval, 'total_time': t_train + t_eval, 'num_params': n_params, 'model': trained_model}
""")

# Cell 8: Plotting
cell8 = nbf.v4.new_code_cell("""
# Plot 1: NLL Curves
plt.figure(figsize=(10, 5))
for name, res in results.items():
    if 'hist' in res and len(res['hist']) > 0:
        plt.plot(res['hist'], label=f"{name}")
plt.xlabel('Época')
plt.ylabel('NLL')
plt.title('Curvas de Convergência (NLL) - Dataset Sintético')
# plt.ylim(top=5) # Zoom removido para ver escala completa se necessário
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Plot 2: Bar Chart Metrics
names = list(results.keys())
accs = [results[n]['acc'] for n in names]
rmses = [results[n]['rmse'] for n in names]

x = np.arange(len(names))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))

# Accuracy bars (Axis 1)
rects1 = ax1.bar(x - width/2, accs, width, label='Acurácia', color='skyblue')
ax1.set_ylabel('Acurácia (Maior é melhor)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(0, 1.0)

# RMSE bars (Axis 2)
ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, rmses, width, label='RMSE Time', color='salmon')
ax2.set_ylabel('RMSE Tempo (Menor é melhor)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=15, ha='right')
ax1.set_title(f'Comparação de Métricas de Predição (Dados Originais - Retweet)')

fig.tight_layout()
plt.show()

# Tabela resumo
import pandas as pd
df_res = pd.DataFrame({
    'Modelo': names,
    'Acurácia': accs,
    'RMSE Tempo': rmses,
    'Final NLL': [results[n]['hist'][-1] for n in names],
    'Treino (s)': [results[n]['train_time'] for n in names],
    'Eval (s)': [results[n]['eval_time'] for n in names],
    'Total (s)': [results[n]['total_time'] for n in names],
    'Parâmetros': [results[n]['num_params'] for n in names],
})
print(df_res.to_string(index=False))

# Plot 3: Tempo de Execução
fig, ax = plt.subplots(figsize=(10, 5))
train_times = [results[n]['train_time'] for n in names]
eval_times = [results[n]['eval_time'] for n in names]
ax.bar(x - width/2, train_times, width, label='Treino', color='steelblue')
ax.bar(x + width/2, eval_times, width, label='Avaliação', color='coral')
ax.set_ylabel('Tempo (segundos)')
ax.set_title('Tempo de Execução por Modelo')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
plt.show()
""")

# Cell 9: Intensity Visualization
cell9 = nbf.v4.new_code_cell("""
# =============================================================================
# ANÁLISE DE SENSIBILIDADE DO THINNING ALGORITHM
# =============================================================================
# Testa diferentes configs de thinning SEM retreinar os modelos.
# Isso mostra como os hiperparâmetros de predição afetam Acc/RMSE.

import time as time_module

thinning_configs = [
    {'dtime_max': 1.0,  'num_sample': 200, 'num_exp': 500, 'label': 'dmax=1.0'},
    {'dtime_max': 2.0,  'num_sample': 200, 'num_exp': 500, 'label': 'dmax=2.0'},
    {'dtime_max': 3.0,  'num_sample': 200, 'num_exp': 500, 'label': 'dmax=3.0'},
    {'dtime_max': 5.0,  'num_sample': 200, 'num_exp': 500, 'label': 'dmax=5.0 (default)'},
    {'dtime_max': 10.0, 'num_sample': 200, 'num_exp': 500, 'label': 'dmax=10.0'},
]

print("=" * 80)
print("ANÁLISE DE SENSIBILIDADE: Thinning Config vs Métricas de Predição")
print("=" * 80)

sensitivity_results = []

for tc in thinning_configs:
    print(f"\\n--- Config: {tc['label']} ---")
    
    for name, res in results.items():
        model = res['model']
        
        # Recriar o EventSampler com novos parâmetros de thinning
        from easy_tpp.model.torch_model.torch_thinning import EventSampler
        model.event_sampler = EventSampler(
            num_sample=tc['num_sample'],
            num_exp=tc['num_exp'],
            over_sample_rate=10.0,
            num_samples_boundary=20,
            dtime_max=tc['dtime_max'],
            patience_counter=5,
            device=model.device
        )
        
        t0 = time_module.time()
        acc, rmse = compute_metrics(model, test_data)
        t1 = time_module.time()
        
        print(f"  {name:25s} | Acc={acc:.4f} | RMSE={rmse:.4f} | Time={t1-t0:.1f}s")
        sensitivity_results.append({
            'config': tc['label'],
            'model': name,
            'acc': acc,
            'rmse': rmse,
        })

# Tabela consolidada
df_sens = pd.DataFrame(sensitivity_results)
print("\\n" + "=" * 80)
print("TABELA CONSOLIDADA")
print("=" * 80)
pivot_acc = df_sens.pivot(index='config', columns='model', values='acc')
pivot_rmse = df_sens.pivot(index='config', columns='model', values='rmse')
print("\\nACURÁCIA:")
print(pivot_acc.to_string())
print("\\nRMSE:")
print(pivot_rmse.to_string())

# Plot comparativo
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name in results.keys():
    model_data = df_sens[df_sens['model'] == name]
    axes[0].plot(range(len(model_data)), model_data['acc'].values, 'o-', label=name)
    axes[1].plot(range(len(model_data)), model_data['rmse'].values, 'o-', label=name)

tick_labels = [tc['label'] for tc in thinning_configs]
axes[0].set_xticks(range(len(thinning_configs)))
axes[0].set_xticklabels(tick_labels, rotation=30, ha='right')
axes[0].set_ylabel('Acurácia')
axes[0].set_title('Acurácia vs dtime_max')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_xticks(range(len(thinning_configs)))
axes[1].set_xticklabels(tick_labels, rotation=30, ha='right')
axes[1].set_ylabel('RMSE')
axes[1].set_title('RMSE vs dtime_max')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.suptitle('Sensibilidade ao Horizonte de Predição (dtime_max)', fontsize=14)
fig.tight_layout()
plt.show()
""")

# Cell 10: Intensity Visualization
cell10 = nbf.v4.new_code_cell("""
# =============================================================================
# VISUALIZAÇÃO DA FUNÇÃO DE INTENSIDADE: NHP vs THP vs THP-ExpDecay
# =============================================================================
# Para uma sequência de teste, plotamos:
# 1. A função de intensidade λ(t) de cada modelo entre eventos consecutivos
# 2. Linhas verticais nos eventos reais
# 3. Predições de cada modelo

import matplotlib.patches as mpatches

# --- Configuração ---
SEQ_IDX = 5            # Qual sequência do teste usar
EVENT_RANGE = (5, 15)  # Faixa de eventos para visualizar
NUM_POINTS = 200       # Pontos para plotar a curva entre eventos

# Cores por modelo
model_colors = {
    'NHP (RNN)': 'tab:blue',
    'THP Tradicional': 'tab:green',
    'THP-ExpDecay (Proposto)': 'tab:red',
}
event_type_names = {0: 'Small', 1: 'Medium', 2: 'Large'}  # Retweet types

# --- Preparar a sequência ---
seq = test_data[SEQ_IDX]
ts_raw = torch.tensor(seq['time_since_start'], dtype=torch.float64)
ts_norm = ((ts_raw - ts_raw[0]) / TIME_SCALE).to(torch.float32)
td_raw = torch.tensor(seq['time_since_last_event'], dtype=torch.float64)
td_norm = (td_raw / TIME_SCALE).to(torch.float32)
types = torch.tensor(seq['type_event'], dtype=torch.long)

start_ev, end_ev = EVENT_RANGE
ts_slice = ts_norm[start_ev:end_ev+1]
types_slice = types[start_ev:end_ev+1]

print(f"Sequência {SEQ_IDX}: {len(ts_norm)} eventos totais")
print(f"Visualizando eventos {start_ev} a {end_ev}")
print(f"Tempos (normalizados): {ts_slice[:5].tolist()}...")
print(f"Tipos: {types_slice[:5].tolist()}...")

# --- Computar intensidade para cada modelo ---
def compute_intensity_curve(model, ts_full, td_full, types_full, start_ev, end_ev, num_points=200):
    model.eval()
    intervals = []
    
    with torch.no_grad():
        for i in range(start_ev, end_ev):
            t_start = ts_full[i].item()
            t_end = ts_full[i+1].item()
            type_next = types_full[i+1].item()
            
            if t_end <= t_start:
                continue
            
            # Grade de tempos entre t_i e t_{i+1}
            dt_grid = torch.linspace(0.001, t_end - t_start, num_points)
            
            # Preparar batch: sequência até o evento i (inclusive)
            seq_len = i + 1
            time_seq = ts_full[:seq_len].unsqueeze(0)
            td_seq = td_full[:seq_len].unsqueeze(0)
            type_seq = types_full[:seq_len].unsqueeze(0)
            sample_dtimes = dt_grid.unsqueeze(0).unsqueeze(0)
            
            try:
                intensities = model.compute_intensities_at_sample_times(
                    time_seq, td_seq, type_seq, 
                    sample_dtimes.expand(1, seq_len, num_points),
                    compute_last_step_only=True
                )
                
                # [num_points, num_types]
                intensity_values = intensities[0, 0, :, :].cpu().numpy()
                
                intervals.append({
                    't_grid': (dt_grid + t_start).numpy(),
                    'intensities': intensity_values,
                    't_start': t_start,
                    't_end': t_end,
                    'type_next': type_next,
                })
            except Exception as e:
                print(f"  Erro no intervalo {i}->{i+1}: {e}")
                continue
    
    return intervals

# --- Computar predições de cada modelo ---
def get_predictions_for_seq(model, ts_full, td_full, types_full, start_ev, end_ev):
    model.eval()
    
    seq_len = end_ev + 1
    batch = (
        ts_full[:seq_len].unsqueeze(0),
        td_full[:seq_len].unsqueeze(0),
        types_full[:seq_len].unsqueeze(0),
        torch.ones(1, seq_len),
        torch.triu(torch.ones(seq_len, seq_len), diagonal=1).unsqueeze(0)
    )
    
    with torch.no_grad():
        dtimes_pred, types_pred = model.predict_one_step_at_every_event(batch)
    
    preds = []
    for i in range(start_ev, end_ev):
        t_pred = ts_full[i].item() + dtimes_pred[0, i].item()
        type_pred = types_pred[0, i].item()
        preds.append({'t_pred': t_pred, 'type_pred': type_pred})
    
    return preds

# --- Computar tudo ---
print("\\nComputando intensidades e predições...")
model_data = {}
for name, res in results.items():
    model = res['model']
    color = model_colors.get(name, 'gray')
    
    print(f"  {name}...")
    intervals = compute_intensity_curve(model, ts_norm, td_norm, types, start_ev, end_ev, NUM_POINTS)
    preds = get_predictions_for_seq(model, ts_norm, td_norm, types, start_ev, end_ev)
    model_data[name] = {'intervals': intervals, 'preds': preds, 'color': color}

# =============================================================================
# PLOT 1: Intensidade Total λ(t) + Erro de Predição
# =============================================================================
fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
ax_main = axes[0]
ax_err = axes[1]

# Curvas de intensidade
for name, md in model_data.items():
    color = md['color']
    for j, interval in enumerate(md['intervals']):
        total_intensity = interval['intensities'].sum(axis=1)
        label = name if j == 0 else None
        ax_main.plot(interval['t_grid'], total_intensity, color=color, alpha=0.8, linewidth=1.5, label=label)

# Eventos reais (linhas verticais)
for i in range(start_ev, end_ev + 1):
    t = ts_norm[i].item()
    etype = types[i].item()
    ax_main.axvline(x=t, color='black', linestyle='--', alpha=0.4, linewidth=0.8)
    ylim = ax_main.get_ylim()
    ypos = ylim[1] * 0.95 if ylim[1] > 0 else 1.0
    ax_main.text(t, ypos, f'{event_type_names.get(etype, str(etype))}', 
                 rotation=90, va='top', ha='right', fontsize=7, color='black', alpha=0.7)

ax_main.set_ylabel('Intensidade Total λ(t)', fontsize=12)
ax_main.set_title(f'Função de Intensidade: Sequência {SEQ_IDX}, Eventos {start_ev}-{end_ev}', fontsize=14)
ax_main.legend(loc='upper right', fontsize=10)
ax_main.grid(True, alpha=0.2)
ax_main.set_xlim(ts_norm[start_ev].item() - 0.1, ts_norm[end_ev].item() + 0.1)

# Erro de predição temporal
bar_width = (ts_norm[end_ev].item() - ts_norm[start_ev].item()) / (end_ev - start_ev) * 0.2
for mi, (name, md) in enumerate(model_data.items()):
    errors = []
    positions = []
    for j, pred in enumerate(md['preds']):
        t_real = ts_norm[start_ev + j + 1].item()
        errors.append(abs(pred['t_pred'] - t_real))
        positions.append(t_real + (mi - 1) * bar_width)
    ax_err.bar(positions, errors, width=bar_width * 0.9, color=md['color'], alpha=0.7, label=name)

ax_err.set_xlabel('Tempo (normalizado)', fontsize=12)
ax_err.set_ylabel('|t_pred - t_real|', fontsize=12)
ax_err.set_title('Erro Absoluto de Predição Temporal por Evento', fontsize=11)
ax_err.legend(fontsize=9)
ax_err.grid(True, alpha=0.2)
ax_err.set_xlim(ts_norm[start_ev].item() - 0.1, ts_norm[end_ev].item() + 0.1)

fig.tight_layout()
plt.show()

# =============================================================================
# PLOT 2: Intensidade por Tipo (detalhe de 1 intervalo)
# =============================================================================
DETAIL_INTERVAL = 3

fig, axes = plt.subplots(1, len(model_data), figsize=(5 * len(model_data), 4), sharey=True)
if len(model_data) == 1:
    axes = [axes]

for idx, (name, md) in enumerate(model_data.items()):
    ax = axes[idx]
    
    if DETAIL_INTERVAL < len(md['intervals']):
        interval = md['intervals'][DETAIL_INTERVAL]
        type_colors = ['tab:cyan', 'tab:orange', 'tab:purple']
        
        for k in range(interval['intensities'].shape[1]):
            type_label = event_type_names.get(k, f'Type {k}')
            ax.plot(interval['t_grid'], interval['intensities'][:, k], 
                    color=type_colors[k % len(type_colors)], linewidth=2, label=type_label)
        
        ax.axvline(x=interval['t_end'], color='black', linestyle='--', linewidth=2, 
                   label=f'Real: {event_type_names.get(interval["type_next"], "?")}')
        
        pred = md['preds'][DETAIL_INTERVAL]
        ax.axvline(x=pred['t_pred'], color=md['color'], linestyle=':', linewidth=2,
                   label=f'Pred: {event_type_names.get(pred["type_pred"], "?")}')
    
    ax.set_title(name, fontsize=11)
    ax.set_xlabel('Tempo')
    if idx == 0:
        ax.set_ylabel('λ_k(t)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

fig.suptitle(f'Intensidade por Tipo: Intervalo {start_ev+DETAIL_INTERVAL} → {start_ev+DETAIL_INTERVAL+1}', fontsize=13)
fig.tight_layout()
plt.show()

# =============================================================================
# Tabela resumo de predições por evento
# =============================================================================
print("\\n" + "=" * 80)
print("RESUMO: Predição por Evento")
print("=" * 80)
print(f"{'Ev':<5} {'Real (tipo, t)':<22}", end='')
for name in model_data.keys():
    short = name[:18]
    print(f" {short:<25}", end='')
print()
print("-" * 80)

for j in range(min(end_ev - start_ev, 10)):
    real_t = ts_norm[start_ev + j + 1].item()
    real_type = types[start_ev + j + 1].item()
    real_label = event_type_names.get(real_type, str(real_type))
    
    print(f"{start_ev+j+1:<5} {real_label+', t='+f'{real_t:.3f}':<22}", end='')
    
    for name, md in model_data.items():
        pred = md['preds'][j]
        pred_label = event_type_names.get(pred['type_pred'], str(pred['type_pred']))
        t_err = abs(pred['t_pred'] - real_t)
        type_ok = 'OK' if pred['type_pred'] == real_type else 'X '
        print(f" {pred_label} {type_ok} dt={t_err:.3f}", end='    ')
    print()
""")

nb.cells = [cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8, cell9]

with open('notebooks/rothp_metrics_comparison.ipynb', 'w') as f:
    nbf.write(nb, f)
