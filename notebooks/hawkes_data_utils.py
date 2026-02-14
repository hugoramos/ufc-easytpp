"""
Utilitários para geração de dados sintéticos Hawkes Process.

Este módulo contém funções para:
- Simular sequências de eventos Hawkes
- Preparar batches para modelos TPP

Pode ser reutilizado por diferentes modelos (Hawkes, RoTHP, THP, NHP, etc.)
"""

import numpy as np
import torch


def simulate_hawkes(mu, alpha, beta, num_seqs, max_time=100.0, seed=42):
    """
    Simula sequências de eventos usando o processo Hawkes univariado.
    
    O processo Hawkes é um processo pontual auto-excitável onde a intensidade
    condicional é dada por:
        λ(t) = μ + α * Σ exp(-β * (t - t_i))
    
    Args:
        mu (float): Taxa base (baseline intensity)
        alpha (float): Parâmetro de excitação (excitation parameter)
        beta (float): Taxa de decaimento (decay rate)
        num_seqs (int): Número de sequências a gerar
        max_time (float): Tempo máximo para cada sequência
        seed (int): Seed para reprodutibilidade
    
    Returns:
        list: Lista de listas, onde cada sublista contém os timestamps dos eventos
    """
    np.random.seed(seed)
    seqs = []
    
    for _ in range(num_seqs):
        timestamps = []
        t = 0.0
        history = []
        
        while t < max_time:
            if not history:
                lambda_barra = mu
            else:
                # colocando alpha máximo(1)
                soma_exponencial_beta = 1 * sum([np.exp(-beta * (t - ti)) for ti in history])
                lambda_barra = mu + alpha * soma_exponencial_beta
            
            taxa_processo_auxiliar = lambda_barra
            
            # sorteio do tempo de espera segundo uma exponencial (poisson homogêneo)
            # P(t) = e^(-lambda t)... E[T]=1/lambda
            tempo_ate_proximo_candidato = np.random.exponential(1 / taxa_processo_auxiliar)
            
            # avança o tempo
            t += tempo_ate_proximo_candidato
            
            if t >= max_time:
                break

            soma_exponencial_beta = sum([np.exp(-beta * (t - ti)) for ti in history])
            lambda_atual = mu + alpha * soma_exponencial_beta
            
            if np.random.uniform(0, 1) * lambda_barra <= lambda_atual:
                timestamps.append(t)
                history.append(t)
        
        seqs.append(timestamps)
    
    return seqs


def prepare_batch(raw_seqs, include_attention_mask=True):
    """
    Prepara um batch de sequências para modelos TPP.
    
    Converte sequências brutas de timestamps em tensores PyTorch com:
    - time_seqs: timestamps absolutos
    - time_deltas: intervalos entre eventos
    - type_seqs: tipos de eventos (0 para univariado)
    - mask: máscara de padding
    - attention_mask: máscara de atenção (triangular superior) ou None
    
    Args:
        raw_seqs (list): Lista de listas de timestamps
        include_attention_mask (bool): Se True, inclui attention_mask para modelos Transformer.
                                       Se False, retorna None (para modelos como Hawkes).
    
    Returns:
        tuple: (time_seqs, time_deltas, type_seqs, mask, attention_mask)
            - time_seqs: [num_seqs, max_len] timestamps
            - time_deltas: [num_seqs, max_len] intervalos
            - type_seqs: [num_seqs, max_len] tipos (0 para univariado)
            - mask: [num_seqs, max_len] máscara de eventos válidos
            - attention_mask: [num_seqs, max_len, max_len] ou None
    """
    num_seqs = len(raw_seqs)
    max_len = max(len(s) for s in raw_seqs) + 1  # +1 para token inicial
    
    time_seqs = torch.zeros(num_seqs, max_len)
    time_deltas = torch.zeros(num_seqs, max_len)
    type_seqs = torch.zeros(num_seqs, max_len).long()
    mask = torch.zeros(num_seqs, max_len)
    
    for i, seq in enumerate(raw_seqs):
        full_seq = [0.0] + seq  # Token inicial em t=0
        l = len(full_seq)
        t_tensor = torch.tensor(full_seq, dtype=torch.float32)
        
        time_seqs[i, :l] = t_tensor
        mask[i, :l] = 1.0
        
        if l > 1:
            time_deltas[i, 1:l] = t_tensor[1:] - t_tensor[:-1]
        
        # repetir último valor válido (pras qualquer conta de delta que eventualmente ocorra fique 0) (máscara será 0)
        if l < max_len:
            time_seqs[i, l:] = full_seq[-1]
    
    # Criar máscara de atenção (triangular superior) apenas se solicitado
    if include_attention_mask:
        # [max_len, max_len] -> [num_seqs, max_len, max_len]
        attention_mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).unsqueeze(0)
        attention_mask = attention_mask.expand(num_seqs, -1, -1).bool()
    else:
        attention_mask = None
    
    return (time_seqs, time_deltas, type_seqs, mask, attention_mask)


def get_default_hawkes_params():
    """
    Retorna parâmetros padrão para geração de dados Hawkes.
    
    Returns:
        dict: Dicionário com parâmetros padrão
    """
    return {
        'mu': 0.3,
        'alpha': 0.6,
        'beta': 1.3,
        'num_train_seqs': 200,
        'num_test_seqs': 50,
        'max_time': 100.0,
        'train_seed': 42,
        'test_seed': 123
    }


def generate_train_test_data(mu=0.3, alpha=0.6, beta=1.3, 
                              num_train_seqs=200, num_test_seqs=50, 
                              max_time=100.0, train_seed=42, test_seed=123):
    """
    Gera dados de treino e teste prontos para uso.
    
    Args:
        mu, alpha, beta: Parâmetros do processo Hawkes
        num_train_seqs: Número de sequências de treino
        num_test_seqs: Número de sequências de teste
        max_time: Tempo máximo por sequência
        train_seed, test_seed: Seeds para reprodutibilidade
    
    Returns:
        tuple: (batch_train, batch_test, train_raw, test_raw)
    """
    train_raw = simulate_hawkes(mu, alpha, beta, num_train_seqs, max_time, seed=train_seed)
    test_raw = simulate_hawkes(mu, alpha, beta, num_test_seqs, max_time, seed=test_seed)
    
    batch_train = prepare_batch(train_raw)
    batch_test = prepare_batch(test_raw)
    
    return batch_train, batch_test, train_raw, test_raw
