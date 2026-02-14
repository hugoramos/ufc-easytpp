import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Adiciona raiz do projeto ao path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from easy_tpp.model.torch_model.torch_rothp import RoTHP
from hawkes_data_utils import simulate_hawkes, prepare_batch

print("RoTHP e utilitários importados com sucesso")

# =============================================================================
# PARÂMETROS PARA GERAÇÃO DE DADOS SINTÉTICOS
# =============================================================================
HAWKES_MU = 0.3
HAWKES_ALPHA = 0.6
HAWKES_BETA = 1.3

NUM_TRAIN_SEQS = 2000
NUM_TEST_SEQS = 500
T_MAX = 100.0

# =============================================================================
# HIPERPARÂMETROS DO MODELO RoTHP
# =============================================================================
HIDDEN_SIZE = 16
NUM_LAYERS = 1
NUM_HEADS = 2
DROPOUT = 0.2

# Hiperparâmetros de treinamento
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 64

# Seed para reprodutibilidade
torch.manual_seed(42)

print("Configuração do RoTHP:")
print(f"  hidden_size = {HIDDEN_SIZE}")
print(f"  num_layers = {NUM_LAYERS}")
print(f"  num_heads = {NUM_HEADS}")
print(f"  dropout = {DROPOUT}")

print("Gerando dados de treino...")
train_raw = simulate_hawkes(HAWKES_MU, HAWKES_ALPHA, HAWKES_BETA, NUM_TRAIN_SEQS, T_MAX, seed=42)

print("Gerando dados de teste...")
test_raw = simulate_hawkes(HAWKES_MU, HAWKES_ALPHA, HAWKES_BETA, NUM_TEST_SEQS, T_MAX, seed=123)

# Preparar batches
batch_train = prepare_batch(train_raw)
batch_test = prepare_batch(test_raw)

print(f"\nDados sintéticos:")
print(f"Treino: {NUM_TRAIN_SEQS} sequências")
print(f"Teste:  {NUM_TEST_SEQS} sequências")
print(f"Shape: {batch_train[0].shape}")

class RoTHPConfig:
    """Configuração simplificada para o modelo RoTHP."""
    def __init__(self, hidden_size=32, num_layers=2, num_heads=4, dropout=0.1):
        # Parâmetros do modelo
        self.hidden_size = hidden_size
        self.time_emb_size = hidden_size  # mesmo tamanho do hidden
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.use_ln = True  # usar layer normalization
        
        # Parâmetros de evento (univariado)
        self.num_event_types = 1
        self.num_event_types_pad = 2  # +1 para padding
        self.pad_token_id = 1  # índice do token de padding
        
        # Parâmetros de perda
        self.loss_integral_num_sample_per_step = 20
        self.use_mc_samples = False  # usar regra do trapézio
        
        # Parâmetros de dispositivo
        self.gpu = -1  # CPU
        
        # Thinning (não usado neste exemplo simples)
        self.thinning = None

# Reset seed antes de criar modelo
torch.manual_seed(42)

config = RoTHPConfig(
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    dropout=DROPOUT
)
model = RoTHP(config)

# Contar parâmetros
num_params = sum(p.numel() for p in model.parameters())
num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Calcular razão dados/parâmetros
total_events_train = sum(len(s) for s in train_raw)
ratio = total_events_train / num_params

print("Modelo RoTHP criado!")
print(f"\nArquitetura:")
print(f"  Hidden size: {HIDDEN_SIZE}")
print(f"  Num layers: {NUM_LAYERS}")
print(f"  Num heads: {NUM_HEADS}")
print(f"  Total parâmetros: {num_params:,}")
print(f"  Parâmetros treináveis: {num_trainable:,}")
print(f"\nDados vs Modelo:")
print(f"  Eventos de treino: {total_events_train:,}")
print(f"  Razão eventos/parâmetros: {ratio:.2f}")

# Optimizer com weight decay para regularização
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Learning rate scheduler para reduzir lr quando estagna
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)

losses_train = []
losses_test = []
best_test_loss = float('inf')
best_epoch = 0

print(f"Iniciando treinamento com Batch Size = {BATCH_SIZE}...\n")

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    
    # Embaralhar dados de treino
    indices = np.random.permutation(len(train_raw))
    
    total_loss_train = 0
    total_events_train = 0
    
    # Mini-batch loop
    for i in range(0, len(train_raw), BATCH_SIZE):
        batch_idx = indices[i:i+BATCH_SIZE]
        batch_seqs = [train_raw[k] for k in batch_idx]
        batch_data = prepare_batch(batch_seqs)
        
        optimizer.zero_grad()
        loss, num_events = model.loglike_loss(batch_data)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss_train += loss.item()
        total_events_train += num_events  # Corrigido: sem .item()
        
    # Média do NLL no treino
    nll_train = total_loss_train / total_events_train
    losses_train.append(nll_train)
    
    # Avaliação no teste
    model.eval()
    with torch.no_grad():
        loss_test, num_test = model.loglike_loss(batch_test)
        nll_test = loss_test / num_test
        losses_test.append(nll_test.item())
    
    # Atualizar scheduler
    scheduler.step(nll_test)
    
    # Track best model
    if nll_test.item() < best_test_loss:
        best_test_loss = nll_test.item()
        best_epoch = epoch
    
    if epoch % 10 == 0 or epoch == 1:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | Train: {nll_train:.4f} | Test: {nll_test.item():.4f} | LR: {current_lr:.6f}")

print(f"\nTreinamento concluído!")
print(f"Melhor teste: {best_test_loss:.4f} (epoch {best_epoch})")

print("Resultado Final")
print("=" * 50)

best_test_epoch = np.argmin(losses_test) + 1

print(f"\nNLL Final (Treino): {losses_train[-1]:.4f}")
print(f"NLL Final (Teste):  {losses_test[-1]:.4f}")

print(f"\nMelhor NLL Teste: {min(losses_test):.4f} (epoch {best_test_epoch})")

# Verificar convergência
gap = losses_test[-1] - losses_train[-1]
print(f"\nGap (Teste - Treino): {gap:.4f}")

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 1. Curvas de Aprendizado
axes[0].plot(losses_train, label='Treino', color='#2E86AB', linewidth=2)
axes[0].plot(losses_test, label='Teste', color='#A23B72', linewidth=2, linestyle='--')
axes[0].axvline(best_test_epoch, color='green', linestyle=':', alpha=0.7, label=f'Melhor: {best_test_epoch}')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('NLL (por evento)')
axes[0].set_title('Curvas de Aprendizado - RoTHP')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Gap Treino-Teste (overfitting monitor)
gap = [t - tr for t, tr in zip(losses_test, losses_train)]
axes[1].plot(gap, color='#E63946', linewidth=2)
axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[1].fill_between(range(len(gap)), gap, alpha=0.3, color='#E63946')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Gap (Teste - Treino)')
axes[1].set_title('Overfitting Monitor')
axes[1].grid(True, alpha=0.3)

# 3. Exemplo de Sequência
example_seq = train_raw[0][:50]
axes[2].eventplot([example_seq], colors='#2E86AB', lineoffsets=0, linelengths=0.5)
axes[2].set_xlabel('Tempo')
axes[2].set_title('1ª sequência sintética')
axes[2].set_yticks([])

plt.tight_layout()
plt.savefig('rothp_results.png')
print("Gráfico salvo em 'rothp_results.png'")
plt.show()
