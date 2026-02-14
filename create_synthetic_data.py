import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def simulate_hawkes_process_optimized(mu, alpha, beta, max_events=None):
    """
    Simula um Processo de Hawkes multivariado O(N) usando a propriedade recursiva do kernel exponencial.
    lambda_k(t) = mu_k + R_k(t)
    Onde R_k(t) decai exponencialmente e salta quando ocorre um evento.
    
    Args:
        mu: Base intensity [num_types]
        alpha: Infectivity matrix [num_types, num_types] (influência de j em k)
        beta: Decay rate [num_types] (taxa de decaimento de k)
        max_events: Número máximo de eventos
    
    Returns:
        seq: Lista de tuplas (t, k)
    """
    num_types = len(mu)
    events = [] # [(t, k)]
    current_time = 0.0
    
    # R[k] armazena o valor da componente variável da intensidade de k no tempo current_time
    # Logo após um evento u em t_last, R_k(t_last) aumenta em alpha[k, u]
    R = np.zeros(num_types)
    
    n_events = 0
    while n_events < max_events:
        # 1. Calcular intensidade total atual (lambda_sum)
        # lambda_k(current_time) = mu_k + R_k
        current_intensities = mu + R
        lambda_sum = np.sum(current_intensities)
        
        # 2. Gerar tempo candidato (algoritmo de Ogata adaptado para decay)
        # Como a intensidade é DECRESCENTE entre eventos, lambda_sum é um upper bound
        # para todo t > current_time (até o próximo evento)
        
        # Passo A: Gerar tempo até o próximo evento do processo homogêneo majorante
        # w ~ Exp(lambda_sum)
        w = np.random.exponential(1.0 / lambda_sum)
        candidate_time = current_time + w
        
        # Passo B: Rejeição
        # Atualizar R para o tempo candidato (decay)
        # R_k(t+w) = R_k(t) * exp(-beta_k * w)
        R_candidate = R * np.exp(-beta * w)
        lambdas_candidate = mu + R_candidate
        lambda_sum_candidate = np.sum(lambdas_candidate)
        
        # Teste de aceitação: Aceita com prob lambda(t+w) / lambda(t)
        # Nota: Ogata usa lambda(t+w) / lambda_upper_bound. Aqui o upper bound era lambda(current_time).
        u = np.random.uniform(0, lambda_sum)
        
        if u < lambda_sum_candidate:
            # Aceito! Determinar tipo do evento
            k = 0
            cumulative = 0
            for i in range(num_types):
                cumulative += lambdas_candidate[i]
                if u < cumulative:
                    k = i
                    break
            
            events.append((candidate_time, k))
            n_events += 1
            
            # Atualizar estado
            current_time = candidate_time
            R = R_candidate
            # Adicionar excitação do evento k em todos os tipos
            # O evento k excita o tipo i com alpha[i, k]
            R += alpha[:, k]
            
        else:
            # Rejeitado (falso evento), apenas avança o tempo e atualiza o decay
            current_time = candidate_time
            R = R_candidate
            # Atualiza lambda_sum para o novo (menor) valor para o próximo passo ser mais eficiente
            
    return events

def generate_synthetic_dataset(num_seqs, num_types=5, min_len=200, max_len=500):
    print(f"Gerando {num_seqs} sequências sintéticas (Otimizado)...")
    
    np.random.seed(42)
    
    # Parâmetros
    mu = np.random.uniform(0.01, 0.05, num_types)
    
    # Alpha: alpha[k, j] é influência de j sobre k
    # Vamos fazer uma matriz densa para garantir correlações
    alpha = np.random.uniform(0.1, 0.4, (num_types, num_types))
    
    # Estabilidade
    eig_val = np.abs(np.linalg.eigvals(alpha))
    if np.max(eig_val) >= 1.0:
        alpha = alpha / (np.max(eig_val) + 0.1)
        
    beta = np.random.uniform(0.5, 2.0, num_types) # Decay mais rápido para estabilidade
    
    print(f"Parâmetros gerados.")
    print(f"Mu: {mu}")
    print(f"Max Alpha Eigenvalue: {np.max(np.abs(np.linalg.eigvals(alpha))):.4f}")
    
    dataset = []
    
    # Gera um pouco mais para garantir tamanho após corte
    target_size_factor = 1.1 
    
    for i in tqdm(range(num_seqs)):
        target_len = np.random.randint(min_len, max_len)
        events = simulate_hawkes_process_optimized(mu, alpha, beta, max_events=target_len)
        
        # Formatar para EasyTPP
        time_seq = [e[0] for e in events]
        time_delta_seq = [time_seq[0]] + [time_seq[k] - time_seq[k-1] for k in range(1, len(time_seq))]
        type_seq = [e[1] for e in events]
        
        dataset.append({
            'time_since_start': time_seq,
            'time_since_last_event': time_delta_seq,
            'type_event': type_seq
        })
            
    return dataset

# --- Execução Principal ---

# Configuração
NUM_TYPES = 5
MIN_LEN = 300
MAX_LEN = 600
TRAIN_SIZE = 1000  # Reduzido para teste rápido, mas suficiente
TEST_SIZE = 200

data = generate_synthetic_dataset(TRAIN_SIZE + TEST_SIZE, NUM_TYPES, MIN_LEN, MAX_LEN)

train_data = data[:TRAIN_SIZE]
test_data = data[TRAIN_SIZE:]

output_dir = 'data/synthetic_long'
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, 'train.pkl'), 'wb') as f:
    pickle.dump(train_data, f)
    
with open(os.path.join(output_dir, 'test.pkl'), 'wb') as f:
    pickle.dump(test_data, f)

print(f"\nDataset salvo em {output_dir}")
print(f"Treino: {len(train_data)}, Teste: {len(test_data)}")
print(f"Avg Len: {np.mean([len(x['time_since_start']) for x in train_data]):.1f}")

# Plot
seq = train_data[0]
plt.figure(figsize=(12, 2))
plt.eventplot(seq['time_since_start'], color='black', alpha=0.5)
plt.title(f"Exemplo de Sequência Sintética (Len: {len(seq['time_since_start'])})")
plt.xlabel("Tempo")
plt.yticks([])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'sample_sequence.png'))
print(f"Plot de exemplo salvo.")
