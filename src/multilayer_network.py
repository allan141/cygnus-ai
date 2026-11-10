# Cygnus AI - Fase 2
#Implementação de rede neural multicamadas

import torch
import torch.nn as nn
from pathlib import Path

# Reproduzir os mesmos resultados iniciais
torch.manual_seed(42)

# Dados de treinamento: y = x²
x = torch.linspace(-2, 2, 100).unsqueeze(1)
y = x**2

# Arquitetura: 1 entrada, duas camadas
# ocultas de 32 neuronios e 1 saida
modelo = nn.Sequential(
    nn.Linear(1, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 1)
)

# Funcao que calcula o erro
criterio = nn.MSELoss()

# Algoritmo que ajusta os pesos
otimizador = torch.optim.Adam(
    modelo.parameters(),
    lr=0.01
)

# Treinamento
for epoca in range(1001):
    modelo.train()

    previsao = modelo(x)
    perda = criterio(previsao, y)

    otimizador.zero_grad()
    perda.backward()
    otimizador.step()

    if epoca % 100 == 0:
        print(
            f"Epoca {epoca:04d} | "
            f"Perda: {perda.item():.6f}"
        )

# Teste com um valor intermediario
modelo.eval()

with torch.no_grad():
    entrada = torch.tensor([[1.5]])
    resultado = modelo(entrada)

print("\n=== CYGNUS AI - FASE 2 ===")
print(f"Entrada: {entrada.item()}")
print(f"Previsão: {resultado.item():.4f}")
print("Esperado: 2.2500")

# Salvar os pesos aprendidos

Path("checkpoints").mkdir(exist_ok=True)
torch.save(
    modelo.state_dict(),
    "checkpoints/cygnus_fase2.pt"
)

print("Pesos salvos em checkpoints/cygnus_fase2.pt")