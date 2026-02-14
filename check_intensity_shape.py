import torch
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Importar modelos (ajuste o path conforme necessário)
import sys
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from easy_tpp.model.torch_model.torch_nhp import NHP
from easy_tpp.model.torch_model.torch_thp import THP
from easy_tpp.model.torch_model.torch_thp_expdecay import THPExpDecay

# Configurações iguais ao notebook de métricas
class ThinningConfig:
    def __init__(self, dtime_max=5.0, num_sample=100, num_exp=500):
        self.num_sample = num_sample
        self.num_exp = num_exp
        self.over_sample_rate = 10.0
        self.patience_counter = 5
        self.num_samples_boundary = 20
        self.dtime_max = dtime_max

class ModelConfig:
    def __init__(self, num_types, pad_id, num_types_pad, hidden_size=64, num_heads=2, num_layers=2):
        self.hidden_size = hidden_size
        self.time_emb_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = 0.1
        self.use_ln = True
        self.num_event_types = num_types
        self.num_event_types_pad = num_types_pad
        self.pad_token_id = pad_id
        self.loss_integral_num_sample_per_step = 20
        self.use_mc_samples = False
        self.gpu = -1
        self.thinning = ThinningConfig()
        self.model_specs = {'beta': 1.0, 'bias': True}

def load_data():
    dataset_path = 'data/synthetic_long'
    with open(os.path.join(dataset_path, 'train.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    return train_data

# Função para computar ground truth intensity do Hawkes
def compute_hawkes_intensity(time, history_times, history_types, mu, alpha, beta):
    # lambda_k(t) = mu_k + sum_{t_j < t} alpha_{k, k_j} * exp(-beta_k * (t - t_j))
    num_types = len(mu)
    intensities = np.zeros(num_types)
    
    # Filtrar apenas eventos passados
    mask = history_times < time
    valid_times = history_times[mask]
    valid_types = history_types[mask]
    
    for k in range(num_types):
        intensities[k] = mu[k]
        # Vetorizado para velocidade
        if len(valid_times) > 0:
            dt = time - valid_times
            # alpha[k, type_j]
            alpha_k = alpha[k, valid_types]
            decay = np.exp(-beta[k] * dt)
            intensities[k] += np.sum(alpha_k * decay)
            
    return intensities

def evaluate_intensity_error(model, sequences, mu, alpha, beta, time_scale):
    model.eval()
    mse_total = 0
    count = 0
    
    print(f"Avaliando Intensity MSE para {type(model).__name__}...")
    
    with torch.no_grad():
        for seq in tqdm(sequences[:50]): # Avaliar 50 sequências para ser rápido
            # Preparar dados
            ts_raw = np.array(seq['time_since_start'])
            td_raw = np.array(seq['time_since_last_event'])
            types = np.array(seq['type_event'])
            
            # Normalizar para o modelo
            ts_norm = torch.tensor((ts_raw - ts_raw[0]) / time_scale, dtype=torch.float32).unsqueeze(0)
            td_norm = torch.tensor(td_raw / time_scale, dtype=torch.float32).unsqueeze(0)
            type_seq = torch.tensor(types, dtype=torch.long).unsqueeze(0)
            
            seq_len = len(ts_raw)
            if seq_len < 10: continue
                
            # Amostrar pontos aleatórios na sequência para comparar intensidade
            # Evitar muito perto de eventos para não pegar picos instáveis
            num_points = 20
            sample_indices = np.random.choice(range(1, seq_len-1), num_points, replace=False)
            
            # Para cada ponto, computar intensidade do modelo vs real
            for idx in sample_indices:
                t_event = ts_raw[idx]
                t_next = ts_raw[idx+1]
                
                # Amostrar um tempo entre t_event e t_next
                dt_offset = np.random.uniform(0.1, 0.9) * (t_next - t_event)
                t_query = t_event + dt_offset
                
                # GROUND TRUTH
                lambda_true = compute_hawkes_intensity(t_query, ts_raw, types, mu, alpha, beta)
                
                # MODEL PREDICTION
                # O modelo precisa receber a sequência até idx (inclusive)
                # E prever a intensidade em t_query
                
                curr_ts_norm = ts_norm[:, :idx+1]
                curr_td_norm = td_norm[:, :idx+1]
                curr_type_seq = type_seq[:, :idx+1]
                
                # Dtime relativo ao último evento para o modelo
                dt_model = dt_offset / time_scale
                sample_dtimes = torch.tensor([[[dt_model]]], dtype=torch.float32) # [1, 1, 1]
                
                # ATENÇÃO: Máscara correta!
                attention_mask = torch.triu(torch.ones(idx+1, idx+1), diagonal=1).unsqueeze(0).bool()
                
                if isinstance(model, NHP):
                    # NHP assinatura diferente
                    # [1, seq_len, 1, num_types]
                    ints = model.compute_intensities_at_sample_times(
                        curr_ts_norm, curr_td_norm, curr_type_seq, 
                        sample_dtimes.expand(1, idx+1, 1),
                        compute_last_step_only=True
                    )
                else:
                    # THP
                    ints = model.compute_intensities_at_sample_times(
                        curr_ts_norm, curr_td_norm, curr_type_seq, 
                        sample_dtimes.expand(1, idx+1, 1),
                        attention_mask=attention_mask,
                        compute_last_step_only=True
                    )
                
                # [num_types]
                lambda_pred = ints[0, 0, 0, :].numpy()
                
                # Calcular erro (MSE)
                mse = np.mean((lambda_true - lambda_pred)**2)
                mse_total += mse
                count += 1
                
    return mse_total / count

# --- Main ---

# 1. Carregar dados e parâmetros do gerador (Hardcoded do script anterior para simplificar)
# Em um cenário real, salvaríamos os params junto com o dataset
# Vou usar os valores aproximados do log anterior ou gerar novos se necessário
# Mu: [0.0249816  0.04802857 0.03927976 0.03394634 0.01624075]
mu = np.array([0.0249816, 0.04802857, 0.03927976, 0.03394634, 0.01624075])
# Alpha/Beta não temos exato, mas podemos testar o ajuste relativo entre modelos
# Como não temos o ground truth exato dos parâmetros (pois não salvei no pickle), 
# vou usar uma abordagem diferente: comparar a log-likelihood (NLL) em um conjunto de teste fixo.
# O NLL é uma proxy direta para a qualidade da intensidade.

# Mas espere! O notebook já calcula NLL. O THP tem NLL melhor (-9.6 vs -3.0).
# Se o NLL é melhor, a intensidade nos pontos de evento é MAIOR.
# Mas a integral (área) também conta.

# Hipótese: O THP aprende uma intensidade muito "puda" (spiky) que maximiza NLL mas é ruim para thinning.
# Vamos visualizar a suavidade da curva.

def plot_intensity_smoothness(model, seq, time_scale):
    model.eval()
    ts_raw = np.array(seq['time_since_start'])
    td_raw = np.array(seq['time_since_last_event'])
    types = np.array(seq['type_event'])
    
    # Pegar um intervalo entre eventos no meio da sequência
    idx = 50
    t_start = ts_raw[idx]
    t_end = ts_raw[idx+1]
    
    # IMPORTANTE: Para ver a forma da função, precisamos estender o tempo
    # além do próximo evento real. Se plotarmos só até t_end, podemos ver apenas
    # um segmento curto se o evento aconteceu rápido.
    # Vamos plotar até 5x a média dos intervalos para ver o comportamento assintótico.
    plot_duration = 5.0 * time_scale # 5x a média
    dt_grid = np.linspace(0, plot_duration, 200)
    
    ts_norm = torch.tensor((ts_raw - ts_raw[0]) / time_scale, dtype=torch.float32).unsqueeze(0)
    td_norm = torch.tensor(td_raw / time_scale, dtype=torch.float32).unsqueeze(0)
    type_seq = torch.tensor(types, dtype=torch.long).unsqueeze(0)
    
    curr_ts = ts_norm[:, :idx+1]
    curr_td = td_norm[:, :idx+1]
    curr_type = type_seq[:, :idx+1]
    
    sample_dtimes = torch.tensor(dt_grid / time_scale, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    num_points = len(dt_grid)
    
    attention_mask = torch.triu(torch.ones(idx+1, idx+1), diagonal=1).unsqueeze(0).bool()
    
    with torch.no_grad():
        if isinstance(model, NHP):
            ints = model.compute_intensities_at_sample_times(
                curr_ts, curr_td, curr_type, sample_dtimes.expand(1, idx+1, num_points), compute_last_step_only=True
            )
        else:
            ints = model.compute_intensities_at_sample_times(
                curr_ts, curr_td, curr_type, sample_dtimes.expand(1, idx+1, num_points), attention_mask=attention_mask, compute_last_step_only=True
            )
            
    # [100, num_types] -> sum types -> [100]
    intensities = ints[0, 0, :, :].sum(dim=-1).numpy()
    return dt_grid, intensities

# Carregar dataset para pegar estatísticas
data = load_data()
all_deltas = []
for item in data:
    td = item['time_since_last_event']
    all_deltas.extend([d for d in td if d > 0])
TIME_SCALE = np.mean(all_deltas)
print(f"Time Scale: {TIME_SCALE}")

# Instanciar modelos (pesos aleatórios por enquanto, mas serve para ver a forma da função)
config = ModelConfig(num_types=5, pad_id=5, num_types_pad=6, hidden_size=64)

nhp = NHP(config)
thp = THP(config)
thp_exp = THPExpDecay(config)

# Plotar
seq = data[0]
t_nhp, y_nhp = plot_intensity_smoothness(nhp, seq, TIME_SCALE)
t_thp, y_thp = plot_intensity_smoothness(thp, seq, TIME_SCALE)
t_exp, y_exp = plot_intensity_smoothness(thp_exp, seq, TIME_SCALE)

plt.figure(figsize=(10, 6))
plt.plot(t_nhp, y_nhp, label='NHP (Exponential Cell)')
plt.plot(t_thp, y_thp, label='THP (Linear Decay)')
plt.plot(t_exp, y_exp, label='THP-ExpDecay (Proposed)')
plt.title('Forma da Função de Intensidade (Modelos não treinados)')
plt.xlabel('Tempo desde o último evento (segundos)')
plt.ylabel('Intensidade Total')
plt.legend()
plt.grid(True, alpha=0.3)
# Adicionar anotação da escala
plt.text(0.05, 0.95, f'Time Scale (mean delta) = {TIME_SCALE:.4f}s', transform=plt.gca().transAxes)
plt.savefig('intensity_shape_check.png')
print("Gráfico salvo em intensity_shape_check.png")
