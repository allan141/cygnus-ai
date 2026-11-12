
# Cygnus AI - Fase 7
# Multi-Head Causal Self-Attention

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CygnusMultiHeadAttention(nn.Module):
    def __init__(self, dimensao_embedding=32, numero_cabecas=4):
        super().__init__()

        if dimensao_embedding <= 0 or numero_cabecas <= 0:
            raise ValueError("As dimensoes devem ser positivas")

        if dimensao_embedding % numero_cabecas != 0:
            raise ValueError(
                "A dimensao do embedding deve ser divisivel "
                "pelo numero de cabecas"
            )

        self.dimensao_embedding = dimensao_embedding
        self.numero_cabecas = numero_cabecas
        self.dimensao_cabeca = dimensao_embedding // numero_cabecas

        # Projecoes treinaveis
        self.query = nn.Linear(
            dimensao_embedding, dimensao_embedding
        )
        self.key = nn.Linear(
            dimensao_embedding, dimensao_embedding
        )
        self.value = nn.Linear(
            dimensao_embedding, dimensao_embedding
        )

        # Combinar as informacoes das cabecas
        self.projecao_saida = nn.Linear(
            dimensao_embedding, dimensao_embedding
        )

    def forward(self, x, retornar_pesos=False):
        if x.ndim != 3:
            raise ValueError(
                "Entrada deve ter formato [batch, sequencia, dimensao]"
            )

        batch, sequencia, dimensao = x.shape

        if dimensao != self.dimensao_embedding:
            raise ValueError("Dimensao de embedding incompatível")

        # Projetar Q, K e V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Separar as cabecas
        # [batch, sequencia, dimensao]
        # -> [batch, cabecas, sequencia, dimensao_cabeca]
        Q = Q.reshape(
            batch, sequencia,
            self.numero_cabecas, self.dimensao_cabeca
        ).transpose(1, 2)

        K = K.reshape(
            batch, sequencia,
            self.numero_cabecas, self.dimensao_cabeca
        ).transpose(1, 2)

        V = V.reshape(
            batch, sequencia,
            self.numero_cabecas, self.dimensao_cabeca
        ).transpose(1, 2)

        # Calcular pontuacoes de atencao
        pontuacoes = torch.matmul(
            Q, K.transpose(-2, -1)
        ) / math.sqrt(self.dimensao_cabeca)

        # Mascara causal
        mascara = torch.tril(
            torch.ones(
                sequencia,
                sequencia,
                device=x.device,
                dtype=torch.bool
            )
        )

        pontuacoes = pontuacoes.masked_fill(
            ~mascara,
            float("-inf")
        )

        # Normalizar os pesos
        pesos_atencao = F.softmax(pontuacoes, dim=-1)

        # Aplicar atencao aos valores
        contexto = torch.matmul(pesos_atencao, V)

        # Reunir as cabecas
        contexto = contexto.transpose(1, 2).contiguous()

        contexto = contexto.reshape(
            batch, sequencia, self.dimensao_embedding
        )

        # Projecao final
        saida = self.projecao_saida(contexto)

        if retornar_pesos:
            return saida, pesos_atencao

        return saida


if __name__ == "__main__":
    torch.manual_seed(42)

    modelo = CygnusMultiHeadAttention(
        dimensao_embedding=32,
        numero_cabecas=4
    )

    entrada = torch.randn(1, 4, 32)

    saida, pesos = modelo(
        entrada,
        retornar_pesos=True
    )

    print("=== CYGNUS MULTI-HEAD ATTENTION ===")
    print("Formato da entrada:", entrada.shape)
    print("Formato da saida:", saida.shape)
    print("Formato dos pesos:", pesos.shape)

    print("\nAtencao da primeira cabeca:")
    print(pesos[0, 0].detach())

    print(
        "\nParametros treinaveis:",
        sum(p.numel() for p in modelo.parameters())
    )
