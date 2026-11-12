# Cygnus AI - Fase 6
# Mecanismo de atencao causal desenvolvido do zero

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CygnusCausalAttention(nn.Module):
    def __init__(self, dimensao_embedding=32):
        super().__init__()

        self.dimensao_embedding = dimensao_embedding

        # Projecoes treinaveis para Query, Key e Value
        self.query = nn.Linear(dimensao_embedding, dimensao_embedding)
        self.key = nn.Linear(dimensao_embedding, dimensao_embedding)
        self.value = nn.Linear(dimensao_embedding, dimensao_embedding)

        # Projecao final
        self.projecao_saida = nn.Linear(
            dimensao_embedding, 
            dimensao_embedding
        )

    def forward(self, x, retornar_pesos=False):
        # x: [batch, sequencia, dimensao_embedding]
        if x.ndim != 3:
            raise ValueError(
                "Entrada deve ter formato [batch, sequencia, dimensao]"
            )

        batch, tamanho_sequencia, dimensao = x.shape

        if dimensao != self.dimensao_embedding:
            raise ValueError("Dimensao de embedding incompatível")

        # Calcular Q, K  e V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Calcular pontuacoes de atencao
        pontuacoes = torch.matmul(
            Q,
            K.transpose(-2, -1)    
        ) / math.sqrt(self.dimensao_embedding)

        # Criar mascara causal
        mascara = torch.tril(
            torch.ones(
                tamanho_sequencia,
                tamanho_sequencia,
                device=x.device,
                dtype=torch.bool
            )
        )

        # Bloquear tokens futuros
        pontuacoes = pontuacoes.masked_fill(
            ~mascara,
            float("-inf")
        )

        # Converter pontuacoes em pesos
        pesos_atencao = F.softmax(pontuacoes, dim=-1)

        # Combinar os valores
        contexto = torch.matmul(pesos_atencao, V)

        # Projecao final
        saida = self.projecao_saida(contexto)

        if retornar_pesos:
            return saida, pesos_atencao

        return saida

    
if __name__ == "__main__":
    torch.manual_seed(42)

    modelo = CygnusCausalAttention(dimensao_embedding=32)

    # Simular 4 tokens com embeddings de 32 dimensoes
    entrada = torch.randn(1, 4, 32)

    saida, pesos = modelo(
        entrada,
        retornar_pesos=True
    )

    print("=== CYGNUS CAUSAL ATTENTION ===")
    print("Formato da entrada:", entrada.shape)
    print("Formato da saida:", saida.shape)
    print("Formato dos pesos:", pesos.shape)

    print("\nMatriz de atencao:")
    print(pesos[0].detach())

    print(
        "\nParametros treinaveis:",
        sum(p.numel() for p in modelo.parameters())
    )
